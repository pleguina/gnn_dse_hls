#!/usr/bin/env python
"""
Inspect per-slot attention in the slot model to diagnose multiplicity overcounting.

For each sampled window, prints:
  - true_count, candidate_logits, sigmoid(candidate_logits)
  - null_attn per slot
  - top-5 attended real stubs per slot with their track_id

Used to determine which failure case is occurring in S2 1-candidate windows:
  Case A: slot 1 attends to same muon stubs as slot 0 (duplicate attention)
  Case B: slot 1 attends to NULL but candidate head still fires
  Case C: slot 1 attends to noise stubs and fires

Usage
-----
  python scripts/omtf_gmt/inspect_slot_attention.py \\
      --checkpoint build/omtf_gmt/checkpoints/slot_model_B1a/gmt_slot_model_best.pt \\
      --cache-dir  build/omtf_gmt/cache \\
      --dataset    S2 \\
      --n-windows  20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt
from omtf_gmt.models  import build_slot_model
from omtf_gmt.features import FEATURE_NAMES


K_MAX = 3


def load_checkpoint(path: Path, device: torch.device):
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    args  = ckpt.get("args", {})
    model = build_slot_model(hidden=args.get("hidden", 64))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Loaded epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('best_val_loss', float('nan')):.4f}")
    return model


@torch.no_grad()
def inspect(model, cache_dir: Path, dataset: str, n_windows: int, device: torch.device):
    full  = GMTCachedDataset(cache_dir, dataset)
    n     = len(full)
    n_tr  = int(0.85 * n)
    _, val = torch.utils.data.random_split(
        full, [n_tr, n - n_tr],
        generator=torch.Generator().manual_seed(42),
    )

    # Pick first n_windows from validation set
    samples = [val[i] for i in range(min(n_windows, len(val)))]
    batch   = {
        "stubs":      torch.stack([s["stubs"]      for s in samples]),
        "valid_mask": torch.stack([s["valid_mask"]  for s in samples]),
        "track_id":   torch.stack([s["track_id"]    for s in samples]),
        "gen_pt":     torch.stack([s["gen_pt"]      for s in samples]),
        "node_label": torch.stack([s["node_label"]  for s in samples]),
    }

    stubs   = batch["stubs"].to(device)
    vm      = batch["valid_mask"].to(device)
    tid     = batch["track_id"].long()     # (N, 24)
    gpt     = batch["gen_pt"]              # (N, K)
    nl      = batch["node_label"]          # (N, 24)

    out = model(stubs, vm)

    cand_logits = out["candidate_logits"].cpu()   # (N, K)
    null_attn   = out["null_attn"].cpu()          # (N, K)
    attn_real   = out["attn_weights"].cpu()       # (N, K, 24)

    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset}  |  {n_windows} windows  |  threshold = 0.0")
    print(f"{'='*70}\n")

    case_counts = {"A": 0, "B": 0, "C": 0, "correct": 0}

    for i in range(len(samples)):
        true_count = int((gpt[i] > 0).sum().item())
        pred_probs = torch.sigmoid(cand_logits[i])
        pred_count = int((cand_logits[i] > 0).sum().item())

        n_valid = int(vm[i].sum().item())
        valid_ids = vm[i].nonzero(as_tuple=True)[0]  # indices of valid stubs

        print(f"--- Window {i+1:2d}  true_count={true_count}  pred_count={pred_count} "
              f"{'✓' if pred_count == true_count else '✗'} ---")
        print(f"  cand_logits: {[f'{v:.3f}' for v in cand_logits[i].tolist()]}")
        print(f"  sigmoid(logits): {[f'{v:.3f}' for v in pred_probs.tolist()]}")
        print(f"  null_attn:   {[f'{v:.3f}' for v in null_attn[i].tolist()]}")

        for k in range(K_MAX):
            slot_attn = attn_real[i, k]          # (24,)
            top5_idx  = slot_attn.argsort(descending=True)[:5]
            top5_vals = slot_attn[top5_idx]
            top5_tids = tid[i][top5_idx]
            top5_nl   = nl[i][top5_idx]

            occupied  = gpt[i, k] > 0
            fires     = cand_logits[i, k] > 0
            null_w    = null_attn[i, k].item()

            status = ""
            if not occupied and fires:
                # Classify failure case
                if top5_tids[0].item() > 0:
                    status = "  ← CASE A (duplicate: attends to real muon)"
                elif null_w > 0.5:
                    status = "  ← CASE B (attends NULL but head fires)"
                else:
                    status = "  ← CASE C (attends noise)"

            top5_str = "  ".join(
                f"stub{idx.item()}[tid={t.item()},sig={int(s.item())}]={v:.3f}"
                for idx, t, s, v in zip(top5_idx, top5_tids, top5_nl, top5_vals)
                if vm[i][idx].item()
            )
            print(f"  slot {k} (occ={'Y' if occupied else 'N'} fire={'Y' if fires else 'N'} "
                  f"null={null_w:.3f}): {top5_str}{status}")

        # Case classification for this window
        if pred_count != true_count:
            # Find first spurious slot
            for k in range(K_MAX):
                if gpt[i, k] <= 0 and cand_logits[i, k] > 0:
                    top_tid = attn_real[i, k].argsort(descending=True)
                    for idx in top_tid[:3]:
                        if vm[i][idx].item():
                            t = tid[i][idx].item()
                            nw = null_attn[i, k].item()
                            if t > 0:
                                case_counts["A"] += 1
                            elif nw > 0.5:
                                case_counts["B"] += 1
                            else:
                                case_counts["C"] += 1
                            break
                    break
        else:
            case_counts["correct"] += 1

        print()

    print(f"{'='*70}")
    print(f"  Case summary over {n_windows} windows:")
    print(f"    Correct (pred_count == true_count): {case_counts['correct']}")
    print(f"    Case A  (duplicate attention):      {case_counts['A']}")
    print(f"    Case B  (NULL attn but head fires):  {case_counts['B']}")
    print(f"    Case C  (noise attention):           {case_counts['C']}")
    print(f"{'='*70}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir",  type=Path, default=Path("build/omtf_gmt/cache"))
    p.add_argument("--dataset",    default="S2")
    p.add_argument("--n-windows",  type=int, default=20)
    p.add_argument("--device",     default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model  = load_checkpoint(args.checkpoint, device)
    inspect(model, args.cache_dir, args.dataset, args.n_windows, device)


if __name__ == "__main__":
    main()
