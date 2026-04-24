"""
GMT branch training script — Phase B1 baseline.

Trains any registered GMT model on the pre-built GMT cache.
Saves checkpoints and per-epoch metrics.

Usage
-----
  python src/omtf_gmt/train.py \\
      --cache-dir build/omtf_gmt/cache \\
      --datasets S1 B4 \\
      --model deepsets \\
      --hidden 64 \\
      --epochs 50 \\
      --output-dir build/omtf_gmt/checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt
from omtf_gmt.models  import build_deepsets

K_MAX = 3
ALL_DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def compute_loss(
    out:   dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    w_node: float = 1.0,
    w_cand: float = 1.0,
    w_pt:   float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    vm  = batch["valid_mask"]              # (N, 24)
    nl  = batch["node_label"]              # (N, 24)
    tid = batch["track_id"].long()         # (N, 24)
    gpt = batch["gen_pt"]                  # (N, K)
    gch = batch["gen_charge"]              # (N, K)

    # node BCE (valid stubs only)
    node_logit = out["node_logit"]
    node_loss = F.binary_cross_entropy_with_logits(
        node_logit[vm], nl[vm], reduction="mean"
    )

    # candidate BCE: slot k is positive if gen_pt[k] > 0
    cand_target = (gpt > 0).float()        # (N, K)
    cand_loss   = F.binary_cross_entropy_with_logits(
        out["candidate_logits"], cand_target, reduction="mean"
    )

    # pT regression (log-space MSE, only on signal slots)
    sig_mask  = cand_target > 0.5         # (N, K)
    pt_loss   = torch.tensor(0.0, device=node_logit.device)
    if sig_mask.any():
        log_pred = torch.log1p(out["pt_pred"][sig_mask])
        log_tgt  = torch.log1p(gpt[sig_mask])
        pt_loss  = F.mse_loss(log_pred, log_tgt)

    total = w_node * node_loss + w_cand * cand_loss + w_pt * pt_loss
    return total, {
        "node_loss": node_loss.item(),
        "cand_loss": cand_loss.item(),
        "pt_loss":   pt_loss.item(),
        "loss":      total.item(),
    }


# --------------------------------------------------------------------------- #
# Metrics (simple, for smoke-test)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def quick_metrics(out: dict, batch: dict) -> dict[str, float]:
    vm  = batch["valid_mask"]
    nl  = batch["node_label"]
    gpt = batch["gen_pt"]

    # node AUC approximation via positive-vs-negative mean logit gap
    nl_b = nl[vm].bool()
    logit = out["node_logit"][vm]
    node_gap = (logit[nl_b].mean() - logit[~nl_b].mean()).item() if nl_b.any() else 0.0

    # stub recall: fraction of signal stubs with logit > 0
    stub_recall = ((logit > 0) & nl_b).float().sum().item() / max(1, nl_b.float().sum().item())

    # candidate recovery: fraction of gen-muon slots correctly fired
    cand_logit = out["candidate_logits"]
    sig_slots  = (gpt > 0)
    cand_rec   = ((cand_logit > 0) & sig_slots).float().sum().item() / max(1, sig_slots.float().sum().item())

    return {
        "node_logit_gap": node_gap,
        "stub_recall":    stub_recall,
        "cand_recovery":  cand_rec,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GMT branch training")
    p.add_argument("--cache-dir",   type=Path, required=True)
    p.add_argument("--datasets",    nargs="+", default=["S1", "B4"])
    p.add_argument("--model",       default="deepsets", choices=["deepsets"])
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--w-node",      type=float, default=1.0)
    p.add_argument("--w-cand",      type=float, default=1.0)
    p.add_argument("--w-pt",        type=float, default=0.5)
    p.add_argument("--output-dir",  type=Path,  default=Path("build/omtf_gmt/checkpoints"))
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_ds, val_ds = [], []
    for ds in args.datasets:
        full = GMTCachedDataset(args.cache_dir, ds)
        n    = len(full)
        n_tr = int(0.85 * n)
        tr, va = torch.utils.data.random_split(full, [n_tr, n - n_tr],
                                                generator=torch.Generator().manual_seed(42))
        train_ds.append(tr); val_ds.append(va)

    train_loader = DataLoader(
        ConcatDataset(train_ds), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_gmt, num_workers=0,
    )
    val_loader   = DataLoader(
        ConcatDataset(val_ds), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_gmt, num_workers=0,
    )

    n_train = sum(len(d) for d in train_ds)
    n_val   = sum(len(d) for d in val_ds)
    print(f"Train: {n_train:,}  Val: {n_val:,}  Device: {device}")

    # --- model ---
    if args.model == "deepsets":
        model = build_deepsets(hidden=args.hidden, dropout=args.dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params={n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_loss = float("inf")
    ckpt_path = args.output_dir / f"gmt_{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # train
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["stubs"], batch["valid_mask"])
            loss, _ = compute_loss(out, batch, args.w_node, args.w_cand, args.w_pt)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # validate
        model.eval()
        val_loss = 0.0
        val_metrics: dict[str, float] = {"node_logit_gap": 0, "stub_recall": 0, "cand_recovery": 0}
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch["stubs"], batch["valid_mask"])
                loss, _ = compute_loss(out, batch, args.w_node, args.w_cand, args.w_pt)
                val_loss += loss.item()
                m = quick_metrics(out, batch)
                for k in val_metrics:
                    val_metrics[k] += m[k]
        val_loss /= max(1, len(val_loader))
        for k in val_metrics:
            val_metrics[k] /= max(1, len(val_loader))

        elapsed = time.time() - t0
        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
            f"recall={val_metrics['stub_recall']:.3f}  "
            f"cand_rec={val_metrics['cand_recovery']:.3f}  "
            f"({elapsed:.0f}s)"
        )

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, ckpt_path)

    # save history
    hist_path = args.output_dir / f"gmt_{args.model}_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\nBest val_loss={best_val_loss:.4f} at checkpoint: {ckpt_path}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    main()
