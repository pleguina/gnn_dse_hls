"""
OMTF-internal G-dataset training script.

Trains EdgeCompatG on the OMTF-internal cache (build/omtf/cache_internal_g_v1)
using the same loss setup as the TPS Phase B5 study (hn025 hard-negative loss).

The cache format is schema_version=2 (same as TPS/KMTF), so GMTCachedDataset
from src/omtf_gmt/dataset.py is used as-is for data loading.  The model comes
from src/omtf/models/edge_compat_g.py and is parametrized by n_features=11.

Usage
-----
  python src/omtf/train_g.py \\
      --cache-dir build/omtf/cache_internal_g_v1 \\
      --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \\
      --repeat B4:6 G7:4 G8:4 \\
      --hidden 64 \\
      --epochs 100 \\
      --w-hard-neg 0.25 \\
      --amp \\
      --output-dir build/omtf/checkpoints/internal_h64_hn025
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

# Data loading: reuse the GMT schema_version=2 dataset reader (not modified)
from omtf_gmt.dataset import GMTCachedDataset, collate_gmt, expand_datasets, expand_repeats

# Model: new parametric EdgeCompat in src/omtf
from omtf.models.edge_compat_g import build_edge_compat_g

K_MAX = 3


# --------------------------------------------------------------------------- #
# Loss (mirrors omtf_gmt/train.py compute_loss exactly, self-contained)
# --------------------------------------------------------------------------- #

def compute_loss(
    out:    dict[str, torch.Tensor],
    batch:  dict[str, torch.Tensor],
    w_node:           float = 1.0,
    w_cand:           float = 1.0,
    w_pt:             float = 0.5,
    w_hard_neg:       float = 0.0,
    unmatched_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    vm  = batch["valid_mask"]
    nl  = batch["node_label"]
    gpt = batch["gen_pt"]

    node_logit = out["node_logit"]
    if unmatched_weight != 1.0 and "truth_source" in batch:
        ts = batch["truth_source"]
        w  = torch.ones_like(nl, dtype=torch.float32)
        w[ts == 0] = unmatched_weight
        w  = w * vm.float()
        node_loss_raw = F.binary_cross_entropy_with_logits(
            node_logit, nl, reduction="none")
        node_loss = (node_loss_raw * w).sum() / w.sum().clamp(min=1.0)
    else:
        node_loss = F.binary_cross_entropy_with_logits(
            node_logit[vm], nl[vm])

    cand_logits = out["candidate_logits"]
    cand_target = (gpt > 0).float()
    cand_loss   = F.binary_cross_entropy_with_logits(cand_logits, cand_target)

    # pT regression in log1p-space (matches TPS train.py exactly)
    sig_mask = cand_target > 0.5
    pt_loss  = out["pt_pred"].new_tensor(0.0)
    if sig_mask.any():
        pt_loss = F.mse_loss(
            torch.log1p(out["pt_pred"][sig_mask]),
            torch.log1p(gpt[sig_mask]),
        )

    hard_neg_loss = cand_logits.new_tensor(0.0)
    if w_hard_neg > 0 and "meta_is_hard_neg" in batch:
        hn_mask = batch["meta_is_hard_neg"].bool()
        if hn_mask.any():
            hn_target = torch.zeros_like(cand_logits[hn_mask])
            hard_neg_loss = F.binary_cross_entropy_with_logits(
                cand_logits[hn_mask], hn_target)

    total = (w_node * node_loss
             + w_cand * cand_loss
             + w_pt   * pt_loss
             + w_hard_neg * hard_neg_loss)

    info = {
        "node_loss":     node_loss.item(),
        "cand_loss":     cand_loss.item(),
        "pt_loss":       pt_loss.item(),
        "hard_neg_loss": hard_neg_loss.item(),
    }
    return total, info


def quick_metrics(
    out:   dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    vm  = batch["valid_mask"]
    nl  = batch["node_label"]
    vm_sum = vm.sum().item()

    with torch.no_grad():
        node_pred  = (out["node_logit"] > 0).float()
        stub_recall = ((node_pred * nl)[vm].sum() /
                       nl[vm].sum().clamp(min=1)).item()

        cand_target = (batch["gen_pt"] > 0).float()
        cand_pred   = (out["candidate_logits"] > 0).float()
        cand_rec    = (cand_pred * cand_target).sum() / cand_target.sum().clamp(min=1)
        cand_rec    = cand_rec.item()

        zero_win_mask = (batch["gen_pt"].sum(dim=1) == 0)
        if zero_win_mask.any():
            fp = (out["candidate_logits"][zero_win_mask] > 0).any(dim=1).float().mean()
        else:
            fp = out["candidate_logits"].new_tensor(0.0)
        zero_fp = fp.item()

    return {
        "stub_recall":    stub_recall,
        "cand_recovery":  cand_rec,
        "zero_win_fp":    zero_fp,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OMTF-internal G-dataset training")
    p.add_argument("--cache-dir",    type=Path, required=True)
    p.add_argument("--datasets",     nargs="+",
                   default=["G1","G2","G3","G4","G5","G6","G7","G8","B4"])
    p.add_argument("--repeat",       nargs="*", default=["B4:6","G7:4","G8:4"],
                   metavar="DS:N")
    p.add_argument("--hidden",       type=int,   default=64)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=4096)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--w-node",       type=float, default=1.0)
    p.add_argument("--w-cand",       type=float, default=1.0)
    p.add_argument("--w-pt",         type=float, default=0.5)
    p.add_argument("--w-hard-neg",   type=float, default=0.0)
    p.add_argument("--unmatched-stub-weight", type=float, default=1.0)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--amp",          action="store_true", default=False)
    p.add_argument("--scheduler",    default="cosine", choices=["none", "cosine"])
    p.add_argument("--save-epochs",  nargs="*", type=int, default=[50, 75, 100])
    p.add_argument("--output-dir",   type=Path,
                   default=Path("build/omtf/checkpoints/internal_h64_hn025"))
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # Read n_features from the cache manifest
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    n_features = int(manifest.get("n_features", 14))

    repeats = expand_repeats(args.repeat)

    train_parts, val_parts = [], []
    ds_sizes = []
    for ds in expand_datasets(args.datasets):
        full = GMTCachedDataset(args.cache_dir, ds)
        n    = len(full)
        n_tr = int(0.85 * n)
        n_va = n - n_tr
        tr, va = torch.utils.data.random_split(
            full, [n_tr, n_va],
            generator=torch.Generator().manual_seed(42),
        )
        train_parts.append(tr)
        val_parts.append(va)
        ds_sizes.append((ds, n_tr, repeats.get(ds, 1)))

    sample_weights = []
    for _, n_tr, n_rep in ds_sizes:
        sample_weights.extend([float(n_rep)] * n_tr)

    n_train_eff = sum(n_tr * n_rep for _, n_tr, n_rep in ds_sizes)
    n_val_total = sum(len(d) for d in val_parts)

    pin = device.type == "cuda"
    pw  = args.num_workers > 0
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=n_train_eff, replacement=True,
    )
    train_loader = DataLoader(
        ConcatDataset(train_parts), batch_size=args.batch_size,
        sampler=sampler, collate_fn=collate_gmt,
        num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw,
    )
    val_loader = DataLoader(
        ConcatDataset(val_parts), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_gmt,
        num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw,
    )

    print(f"n_features={n_features}  train={n_train_eff:,}  val={n_val_total:,}  device={device}")
    for ds_name, n_tr, n_rep in ds_sizes:
        eff = n_tr * n_rep
        pct = 100.0 * eff / max(1, n_train_eff)
        print(f"  {ds_name}{'×'+str(n_rep) if n_rep>1 else ''}: {eff:,} ({pct:.1f}%)")

    model = build_edge_compat_g(
        n_features=n_features, hidden=args.hidden, dropout=args.dropout
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: EdgeCompatG  n_features={n_features}  hidden={args.hidden}"
          f"  params={n_params:,}")

    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    save_epochs     = set(args.save_epochs)
    best_val_loss   = float("inf")
    history         = []
    ckpt_best = args.output_dir / "omtf_internal_best.pt"
    ckpt_last = args.output_dir / "omtf_internal_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=pin) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.amp):
                out  = model(batch["stubs"], batch["valid_mask"])
                loss, _ = compute_loss(
                    out, batch,
                    args.w_node, args.w_cand, args.w_pt,
                    args.w_hard_neg, args.unmatched_stub_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_metrics: dict[str, float] = {"stub_recall": 0.0, "cand_recovery": 0.0, "zero_win_fp": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=pin) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out  = model(batch["stubs"], batch["valid_mask"])
                    loss, _ = compute_loss(
                        out, batch,
                        args.w_node, args.w_cand, args.w_pt,
                        args.w_hard_neg, args.unmatched_stub_weight,
                    )
                val_loss += loss.item()
                m = quick_metrics(out, batch)
                for k in val_metrics:
                    val_metrics[k] += m[k]
        val_loss /= max(1, len(val_loader))
        for k in val_metrics:
            val_metrics[k] /= max(1, len(val_loader))

        elapsed = time.time() - t0
        print(f"[{epoch:3d}/{args.epochs}] tr={tr_loss:.4f}  val={val_loss:.4f}  "
              f"recall={val_metrics['stub_recall']:.3f}  "
              f"cand_rec={val_metrics['cand_recovery']:.3f}  "
              f"zero_fp={val_metrics['zero_win_fp']:.3f}  ({elapsed:.0f}s)")

        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": val_loss, **val_metrics})

        ckpt_payload = {
            "model":         model.state_dict(),
            "epoch":         epoch,
            "best_val_loss": best_val_loss,
            "args":          vars(args),
            "n_features":    n_features,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_payload["best_val_loss"] = best_val_loss
            torch.save(ckpt_payload, ckpt_best)

        torch.save(ckpt_payload, ckpt_last)

        if epoch in save_epochs:
            snap = args.output_dir / f"omtf_internal_epoch_{epoch:04d}.pt"
            torch.save(ckpt_payload, snap)

        if scheduler:
            scheduler.step()

    (args.output_dir / "omtf_internal_history.json").write_text(
        json.dumps(history, indent=2))
    print(f"\nBest val_loss={best_val_loss:.4f}  checkpoint: {ckpt_best}")


if __name__ == "__main__":
    main()
