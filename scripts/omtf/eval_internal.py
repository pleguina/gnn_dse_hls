#!/usr/bin/env python
"""
OMTF-internal model evaluator.

Thin wrapper around scripts/omtf_gmt/eval_gmt.py that loads an OMTF-internal
checkpoint (EdgeCompatG, n_features=11) instead of the GMT EdgeCompat
(n_features=14).  All evaluation logic, metric computation, and output
formatting is delegated to eval_gmt unchanged.

Usage
-----
  python scripts/omtf/eval_internal.py \\
      --checkpoint build/omtf/checkpoints/internal_h64_hn025/omtf_internal_best.pt \\
      --cache-dir  build/omtf/cache_internal_g_v1 \\
      --datasets   G1 G2 G3 G4 G5 G6 G7 G8 B4 \\
      --threshold  0.0 \\
      --output     build/omtf/eval/internal_h64_hn025_best_eval.md \\
      --device     cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "omtf_gmt"))

# Import evaluation functions from eval_gmt (read-only, not modified)
from eval_gmt import (
    eval_dataset,
    eval_event_level,
    render_report,
    positive_pt,
)

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt, expand_datasets
from omtf.models.edge_compat_g import build_edge_compat_g

K_MAX = 3


def load_internal_model(ckpt_path: Path, device: torch.device):
    """Load an OMTF-internal EdgeCompatG checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    hidden     = int(args.get("hidden", 64))
    dropout    = float(args.get("dropout", 0.0))
    n_features = int(ckpt.get("n_features", args.get("n_features", 11)))
    epoch      = int(ckpt.get("epoch", 0))
    best_loss  = float(ckpt.get("best_val_loss", float("inf")))

    model = build_edge_compat_g(
        n_features=n_features, hidden=hidden, dropout=dropout
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return model, "edge_compat_g", hidden, dropout, epoch, best_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OMTF-internal model")
    p.add_argument("--checkpoint",  type=Path, required=True)
    p.add_argument("--cache-dir",   type=Path, required=True)
    p.add_argument("--datasets",    nargs="+",
                   default=["G1","G2","G3","G4","G5","G6","G7","G8","B4"])
    p.add_argument("--threshold",   type=float, default=0.0)
    p.add_argument("--output",      type=Path,  required=True)
    p.add_argument("--batch-size",  type=int,   default=2048)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    model, model_name, hdim, dropout, epoch, best_loss = load_internal_model(
        args.checkpoint, device
    )
    print(f"Model: {model_name} hidden={hdim} dropout={dropout} "
          f"epoch={epoch} best_val_loss={best_loss:.4f}")

    manifest = json.loads((args.cache_dir / "manifest.json").read_text())

    all_datasets = expand_datasets(args.datasets)
    pw_results  = []
    ev_results  = {}

    for ds in all_datasets:
        print(f"  [{ds}] ...", end="\r")
        dataset = GMTCachedDataset(args.cache_dir, ds)
        pw_res = eval_dataset(
            model     = model,
            dataset   = dataset,
            device    = device,
            threshold = args.threshold,
            ds_name   = ds,
            batch_size   = args.batch_size,
            num_workers  = args.num_workers,
        )
        pw_results.append(pw_res)

        ev_res = eval_event_level(
            model    = model,
            dataset  = dataset,
            device   = device,
            threshold = args.threshold,
            ds_name  = ds,
            batch_size   = args.batch_size,
            num_workers  = args.num_workers,
        )
        ev_results[ds] = ev_res
        print(f"  [{ds}] done")

    # ---- output ------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result = {"per_window": pw_results, "event_level": ev_results}
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2))

    md = render_report(
        pw_results  = pw_results,
        ev_results  = ev_results,
        threshold   = args.threshold,
        model_name  = model_name,
        hidden_dim  = hdim,
        dropout     = dropout,
        ckpt_path   = args.checkpoint,
        epoch       = epoch,
        best_loss   = best_loss,
    )
    args.output.write_text(md)

    print(f"\nReport: {args.output}")
    print(f"Data:   {json_path}")


if __name__ == "__main__":
    main()
