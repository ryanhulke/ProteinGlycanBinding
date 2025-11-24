import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datamodules import GlycanPretrainDataModule
from GraphormerH import GlycanGraphormerEncoder, GlycanGraphormerPretrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grad_clip = 1.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain with MLM + motif count prediction")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="./models/pretrain_checkpoints") # Directory to save checkpoint
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    use_amp: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    scaler = torch.amp.GradScaler(enabled=use_amp)

    for batch in tqdm(dataloader):
        glycans = batch  # glycans: list[str]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            out = model(glycans)
            loss = out["loss"]

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    print(f"Epoch {epoch} train loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    split: str = "val",
    use_amp: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        glycans = batch  # glycans: list[str]
        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            out = model(glycans)
            loss = out["loss"]

        total_loss += loss.item()

        num_batches += 1

    denom = max(num_batches, 1)
    avg_loss = total_loss / denom

    print(f"Epoch {epoch} {split} loss: {avg_loss:.4f}")
    return avg_loss

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dm = GlycanPretrainDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=.1,
        test_fraction=.1,
        seed=args.seed,
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Model
    encoder = GlycanGraphormerEncoder().to(device)
    model = GlycanGraphormerPretrainer(glycan_encoder=encoder).to(device)
        
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=3)

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "best_pretrain_multichannel512dim.ckpt")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            use_amp=True,
        )

        val_loss = evaluate(
            model,
            val_loader,
            epoch,
            split="val",
            use_amp=True,
        )

        scheduler.step(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"Saved new best checkpoint to {best_ckpt_path}")

    # Final test eval on best checkpoint
    print("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss = evaluate(
        model,
        test_loader,
        epoch=0,
        split="test",
        use_amp=True,
    )
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
