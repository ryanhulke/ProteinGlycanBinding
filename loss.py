import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score
from scipy.stats import spearmanr

class SoftSpearmanLoss(nn.Module):
    """
    Differentiable Spearman rank-correlation loss using estimated ranks (highly accurate)

    Args
    ----
    tau: temperature of the sigmoid rank approximation\n
    descending: if True, higher y_hat means better (rank ~1)\n
    """
    def __init__(
        self,
        tau: float = 1.0,
        descending: bool = False,
    ):
        super().__init__()
        self.tau = float(tau)
        self.eps = 1e-8
        self.descending = descending

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            y_pred: [B]
            y_true: [B]
        """
        # Compute soft ranks for predictions
        r_pred = soft_rank_sigmoid(y_pred, tau=self.tau, descending=self.descending).float()  # [B, N]
        r_true = hard_rank_desc(y_true)

        # ranks per list and compute Pearson correlation
        mean = r_pred.mean(dim=-1, keepdim=True)
        std = r_pred.std(dim=-1, keepdim=True)
        r_pred = (r_pred - mean) / (std + self.eps)
        mean = r_true.mean(dim=-1, keepdim=True)
        std = r_true.std(dim=-1, keepdim=True)
        r_true = (r_true - mean) / (std + self.eps)
        rho = (r_pred * r_true).mean(dim=-1)  # [B]
        loss = 1.0 - rho  # [B]
        loss = loss.mean()
        return loss

def hard_rank_desc(y: torch.Tensor) -> torch.Tensor:
    x = y.float()
    order = torch.argsort(x, dim=-1, descending=True)
    ar = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype)
    ar = ar.expand_as(order)
    ranks = torch.zeros_like(x).scatter(-1, order, ar)
    return ranks

def soft_rank_sigmoid(
    scores: torch.Tensor,
    descending: bool = False,
    tau: float = 1.0
) -> torch.Tensor:
    """
    Differentiable ranks via pairwise sigmoids.
    For descending ranks (largest score -> rank 1):
        r_i = 1 + sum_{j != i} sigmoid((s_j - s_i)/tau)
    """
    x = scores.float()
    if descending:
        x = -x
    orig_shape = x.shape
    N = x.size(-1)
    x = x.reshape(-1, N) # [B, N]

    inv_tau = 1.0 / max(tau, 1e-6)

    # D[b, i, j] = (x[b, j] - x[b, i]) * inv_tau
    D = (x.unsqueeze(2) - x.unsqueeze(1)) * inv_tau # [B, N, N]
    P = torch.sigmoid(D) # [B, N, N]
    # Subtract self term sigmoid(0) = 0.5 once per i
    r = 1.0 + P.sum(dim=-1) - 0.5 # [B, N]
    return r.reshape(orig_shape)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, metric_names: list[str], cutoff = 1.645) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    out = {}
    # apply threshold for binarization
    y_true_bin = (y_true >= cutoff).astype(int)
    has_both_classes = (np.unique(y_true_bin).size == 2)

    for m in metric_names or []:
        key = m.lower()

        if key == "auroc":
            if has_both_classes:
                out["auroc"] = float(np.round(roc_auc_score(y_true_bin, y_prob), 4))
            else:
                out["auroc"] = float("nan")

        elif key in ("aupr", "average_precision"):
            if has_both_classes:
                prec, rec, thresh = precision_recall_curve(y_true_bin, y_prob)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                best_idx = int(np.nanargmax(f1))
                out["aupr"] = float(np.round(average_precision_score(y_true_bin, y_prob), 4))
                out["best_f1"] = float(np.round(f1[best_idx], 4))
                out["precision"] = float(np.round(prec[best_idx], 4))
                out["recall"] = float(np.round(rec[best_idx], 4))
            else:
                out["aupr"] = out["best_f1"] = out["precision"] = out["recall"] = float("nan")

        elif key == "accuracy":
            if has_both_classes:
                out["accuracy"] = float(np.round(accuracy_score(y_true_bin, (y_prob >= 0.5).astype(int)), 4))
            else:
                out["accuracy"] = float("nan")

        elif key == "label_loss":
            eps = 1e-7
            p = np.clip(y_prob, eps, 1 - eps)
            bce = -(y_true_bin * np.log(p) + (1 - y_true_bin) * np.log(1 - p))
            out["label_loss"] = float(np.round(np.mean(bce), 4))

        elif key == "spearmanp":
            spearman_corr, p_value = spearmanr(y_true, y_prob)
            out["spearmanp"] = float(np.round(spearman_corr, 4))
        elif key == "mae":
            out["mae"] = float(np.round(mean_absolute_error(y_true, y_prob), 4))
        elif key == "mse":
            out["mse"] = float(np.round(mean_squared_error(y_true, y_prob), 4))
        elif key == "rmse":
            out["rmse"] = float(np.round(np.sqrt(mean_squared_error(y_true, y_prob)), 4))
        elif key == "bs-spearmanp":
            # compute spearman only on samples with true label >= cutoff
            mask = (y_true >= cutoff)
            if np.sum(mask) >= 2:
                spearman_corr, p_value = spearmanr(y_true[mask], y_prob[mask])
                out["bs-spearmanp"] = float(np.round(spearman_corr, 4))
            else:
                out["bs-spearmanp"] = float("nan")
    return out

def evaluate(
    model: nn.Module,
    loader,
    metric_names: list[str],
) -> dict:
    if loader is None:
        return {}

    model.eval()
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    with torch.no_grad():
        for glycan_batch, protein_batch, labels in loader:
            preds = model(glycan_batch, protein_batch).squeeze(-1)              # [B]
            probs = torch.sigmoid(preds).detach().float().cpu().numpy()         # to float32 on CPU

            y_true_all.append(labels.detach().float().cpu().numpy())
            y_prob_all.append(probs)

    if not y_true_all:
        return {}

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    metrics = compute_metrics(y_true, y_prob, metric_names or [])
    return metrics