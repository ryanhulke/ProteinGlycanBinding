from jsonargparse import ArgumentParser
import torch
from pathlib import Path
import yaml
from torch import nn
from tqdm.auto import tqdm
import torch
import numpy as np
import wandb
from datamodules import GlycoProteinDataModule
from SweetBind import SweetBind
from loss import evaluate, SoftSpearmanLoss
from utils import config_logger, get_logger, set_random_seed, load_pretrained_glycan_encoder

logg = get_logger()

def add_args(parser: ArgumentParser):
    parser.add_argument("--run-id", required=True, help="Experiment ID", dest="run_id")
    parser.add_argument(
        "--config",
        help="YAML config file",
        default="config/esmc_glycan_config.yaml",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_glycan_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_spearman", type=float, default=0.0)
    parser.add_argument("--coattention", type=str, default="false")  # "true" or "false"
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="attention") # "attention" or "mlp"
    parser.add_argument("--dataset", type=str, default="glycanml")  # "glycanml" or "zscore"
    parser.add_argument("--finetune", type=int, default=0) # 0 or 1
    parser.add_argument("--seed", type=int, default=1)
    return parser

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # seed everything
    set_random_seed(args.seed or cfg.get("seed", 42))

    # ensure destination path exists in cfg
    cfg.setdefault("task", {})
    cfg["task"].setdefault("model", {})

    # your existing optional overrides
    cfg["dropout"] = float(args.dropout)
    cfg["lambda_spearman"] = float(args.lambda_spearman)

    cfg["coattention"] = str(args.coattention).strip().lower() in ("true", "1", "yes", "y")

    cfg["model_type"] = args.model_type.lower()

    cfg["dataset"] = args.dataset.lower()

    cfg["num_attention_heads"] = int(args.heads)

    config_logger(
        file=cfg.get("log_file"),
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        level=cfg.get("verbosity", 2),
        use_stdout=True,
    )

    if cfg.get("wandb_save", False):
        wandb_run = wandb.init(
                project=cfg.get("wandb_proj") or "GlycoProtPred",
                name=args.run_id,
                config=cfg,
                reinit=True,
            )

    device = torch.device(f"cuda:{cfg.get('device', 0)}" if torch.cuda.is_available() else "cpu")
    
    model = SweetBind(
        glycan_hidden_dim=args.hidden_dim,
        glycan_layers=args.num_glycan_layers,
        protein_dim=1152,  # ESM-C token embedding dim
        dropout=cfg.get("dropout", 0.1),
        model_type=args.model_type,
        coattention=cfg.get("coattention", False),
        heads=cfg.get("num_attention_heads", 4)
    )
        # model = load_pretrained_glycan_encoder(model, "./checkpoints/best_pretrain.ckpt", device)
        # for p in model.glycan_encoder.parameters():
        #     p.requires_grad = False
        # print("Glycan encoder FROZEN for first 5 epochs")

    ds = cfg.get("dataset", None)
    if ds == "glycanml":
        dataset_path = "./data/glycan_interaction.csv"
    else:
        dataset_path = "./data/LectinOracle_CFG_zscore.csv"
    pairs_dm = GlycoProteinDataModule(
        path=dataset_path,
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("num_workers", 0)),
    )

    train_loader = pairs_dm.train_dataloader()
    val_loader = pairs_dm.val_dataloader()
    test_loader = pairs_dm.test_dataloader()

     # move model to device
    model = model.to(device)
    
    lr_glycan = 1e-5
    # lr_lora = 1e-4
    lr_fusion = 1e-5
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "glycan_encoder" in n and p.requires_grad],
            "lr": lr_glycan
        },
        # {
        #     "params": [p for n, p in model.named_parameters()
        #             if "lora_" in n and p.requires_grad],
        #     "lr": lr_lora   # e.g., 3e-4
        # },
        {
            "params": [p for n, p in model.named_parameters() if "glycan_encoder" not in n and "lora_" not in n and p.requires_grad],
            "lr": lr_fusion
        },
    ]

#   print num params in glycan encoder
    n_glycan_params = sum(p.numel() for n, p in model.named_parameters() if "glycan_encoder" in n and p.requires_grad)
    n_fusion_params = sum(p.numel() for n, p in model.named_parameters() if "glycan_encoder" not in n and "lora_" not in n and p.requires_grad)
    logg.info(f"Number of trainable parameters in glycan encoder: {n_glycan_params}")
    logg.info(f"Number of trainable parameters in fusion and regression head: {n_fusion_params}")

    # OPTIMIZER AND SCHEDULER
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    # steps_per_epoch = len(train_loader)
    epochs = int(cfg.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    # TRAINING CONFIG
    eval_metrics = cfg.get("eval_metric", ["mae", "auroc", "aupr", "label_loss"])
    every_n_val = max(1, int(cfg.get("every_n_val", 1)))

    model_save_dir = Path(cfg.get("model_save_dir", "./models/best_models"))
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_save_dir / f"{args.run_id}_best.pt"
    if best_model_path.exists():
        best_model_path.unlink()

    early_cfg = cfg.get("early_stopping") or {}
    early_enabled = bool(early_cfg.get("enabled", True))
    early_metric = str(early_cfg.get("metric", "spearmanp")).lower()
    early_patience = max(1, int(early_cfg.get("patience", 10))) if early_enabled else 0
    early_min_delta = float(early_cfg.get("min_delta", 0.0)) if early_enabled else 0.0
    best_metric = float("-inf")
    epochs_since_improve = 0
    best_state_dict = None
    best_epoch = 0
    early_metric_missing_warned = False
    should_stop = False
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))

    mse_loss = nn.MSELoss()
    soft_spearman_loss = SoftSpearmanLoss(tau=0.1)
    lambda_spearman = cfg.get("lambda_spearman", 0.5)

    # model.toggle_glycan_encoder_finetune(False)

    # TRAINING LOOP
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_obs = 0
        # if epoch == 4:
        #     model.toggle_glycan_encoder_finetune(True)
        #     # set lr for glycan encoder to 1e-5 and for fusion to 1e-5
        #     optimizer.param_groups[0]['lr'] = lr_glycan
        #     optimizer.param_groups[1]['lr'] = lr_glycan
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)):
            g, p, y = batch
            y_hat = model(g, p)
            y = y.to(device, dtype=y_hat.dtype)
            loss = lambda_spearman * soft_spearman_loss(y_hat, y) + (1 - lambda_spearman) * mse_loss(y_hat, y)
            scaled_loss = loss / gradient_accumulation_steps

            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            bs = y.shape[0]
            running_loss += loss.item() * bs
            n_obs += bs

        train_loss = running_loss / max(1, n_obs)
        logg.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss})


        if (epoch % every_n_val) == 0:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                metric_names=eval_metrics,
            )
            logg.info(f"Val metrics @ epoch {epoch}: {val_metrics}")
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
            scheduler.step(val_metrics.get(early_metric))
            metric_value = val_metrics.get(early_metric)
            metric_finite = metric_value is not None and np.isfinite(metric_value)

            if metric_finite:
                metric_value = float(metric_value)
                improved = metric_value > best_metric + early_min_delta
                if improved:
                    best_metric = metric_value
                    epochs_since_improve = 0
                    best_state_dict = {
                        name: param.detach().cpu().clone()
                        for name, param in model.state_dict().items()
                    }
                    torch.save(best_state_dict, best_model_path)
                    best_epoch = epoch
                    logg.info(
                        "New best %s=%.4f at epoch %s; checkpoint saved to %s",
                        early_metric,
                        metric_value,
                        epoch,
                        best_model_path,
                    )
                    wandb.log({"early_stopping/best_metric": metric_value, "early_stopping/best_epoch": epoch})
                elif early_enabled:
                    epochs_since_improve += 1
                    wandb.log({"early_stopping/epochs_since_improve": epochs_since_improve})
                    if early_patience > 0 and epochs_since_improve >= early_patience:
                        logg.info(
                            "Early stopping triggered at epoch %s: %s=%.4f (best %.4f at epoch %s)",
                            epoch,
                            early_metric,
                            metric_value,
                            best_metric,
                            best_epoch,
                        )
                        wandb.log({"early_stopping/triggered": 1, "early_stopping/trigger_epoch": epoch})
                        should_stop = True
            elif early_enabled and not early_metric_missing_warned:
                logg.warning(
                    "Early stopping metric '%s' missing or non-finite; will keep training until it becomes available.",
                    early_metric,
                )
                early_metric_missing_warned = True

        if should_stop:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logg.info(
            "Restored best model from epoch %s with %s=%.4f",
            best_epoch,
            early_metric,
            best_metric,
        )
        wandb.log({"early_stopping/restored_epoch": best_epoch, "early_stopping/restored_metric": best_metric})

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        metric_names=eval_metrics,
    )
    if test_metrics:
        logg.info(f"Test metrics: {test_metrics}")
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    # cleanly finish wandb run if started
    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)
