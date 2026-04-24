#!/usr/bin/env python
"""
GMT dataset feature audit — Phase B1 sanity check before training.

Reads the built GMT cache and reports:
  - occupancy per region (stubs/window distribution)
  - feature min/max and mean for all 11 features
  - offlineEta2 / offlineCoord2 missingness (has_eta2, has_coord2)
  - class balance: matched signal / unmatched noise / ambiguous
  - distribution of offlineCoord2 (bend proxy)

Usage
-----
  python scripts/omtf_gmt/feature_audit.py \\
      --cache-dir build/omtf_gmt/cache \\
      --datasets S1 S3 B1 B4

Output: printed report + build/omtf_gmt/FEATURE_AUDIT.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.features import FEATURE_NAMES, N_FEATURES, F_HAS_ETA2, F_HAS_COORD2, F_COORD2
from omtf_gmt.dataset  import iter_shards

ALL_DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]


def audit_dataset(cache_dir: Path, ds: str) -> dict:
    n_windows      = 0
    n_stubs_list:  list[int] = []
    feat_min       = np.full(N_FEATURES, np.inf,  dtype=np.float64)
    feat_max       = np.full(N_FEATURES, -np.inf, dtype=np.float64)
    feat_sum       = np.zeros(N_FEATURES, dtype=np.float64)
    feat_count     = 0
    n_signal       = 0   # trackId > 0
    n_noise        = 0   # trackId == 0, valid stub
    n_ambiguous    = 0   # ambiguous flag set
    has_eta2_frac: list[float] = []
    has_c2_frac:   list[float] = []
    coord2_vals:   list[float] = []

    for shard in iter_shards(cache_dir, ds):
        stubs   = shard["stubs"]       # (N, 24, 11)
        vm      = shard["valid_mask"]  # (N, 24)
        tid     = shard["track_id"]    # (N, 24) int8
        amb     = shard["ambiguous"]   # (N, 24) uint8
        mn      = shard["meta_n_stubs"]  # (N,)

        N = stubs.shape[0]
        n_windows += N
        n_stubs_list.extend(mn.tolist())

        # flatten to valid stubs only
        valid_flat = vm.reshape(-1)                    # (N*24,)
        stubs_flat = stubs.reshape(-1, N_FEATURES)     # (N*24, 11)
        tid_flat   = tid.reshape(-1).long()            # (N*24,)
        amb_flat   = amb.reshape(-1).long()            # (N*24,)

        v_stubs = stubs_flat[valid_flat].numpy()       # (M, 11)
        v_tid   = tid_flat[valid_flat].numpy()
        v_amb   = amb_flat[valid_flat].numpy()

        if len(v_stubs) > 0:
            feat_min   = np.minimum(feat_min, v_stubs.min(axis=0))
            feat_max   = np.maximum(feat_max, v_stubs.max(axis=0))
            feat_sum  += v_stubs.sum(axis=0)
            feat_count += len(v_stubs)

            n_signal    += int((v_tid > 0).sum())
            n_noise     += int((v_tid == 0).sum())
            n_ambiguous += int(v_amb.sum())

            # has_eta2, has_coord2 per window
            he2 = stubs[:, :, F_HAS_ETA2] * vm.float()   # (N, 24)
            hc2 = stubs[:, :, F_HAS_COORD2] * vm.float()
            nv  = vm.float().sum(dim=1).clamp(min=1)
            has_eta2_frac.extend((he2.sum(dim=1) / nv).tolist())
            has_c2_frac.extend((hc2.sum(dim=1) / nv).tolist())

            coord2_vals.extend(v_stubs[:, F_COORD2].tolist())

    feat_mean = feat_sum / max(1, feat_count) if feat_count > 0 else np.zeros(N_FEATURES)
    n_stubs   = np.array(n_stubs_list, dtype=np.int32)
    coord2_a  = np.array(coord2_vals)

    return {
        "n_windows":     n_windows,
        "n_stubs":       n_stubs,
        "feat_min":      feat_min,
        "feat_max":      feat_max,
        "feat_mean":     feat_mean,
        "n_signal":      n_signal,
        "n_noise":       n_noise,
        "n_ambiguous":   n_ambiguous,
        "has_eta2_mean": float(np.mean(has_eta2_frac)) if has_eta2_frac else float("nan"),
        "has_c2_mean":   float(np.mean(has_c2_frac))  if has_c2_frac  else float("nan"),
        "coord2_p5":     float(np.percentile(coord2_a, 5))  if len(coord2_a) > 0 else float("nan"),
        "coord2_p50":    float(np.percentile(coord2_a, 50)) if len(coord2_a) > 0 else float("nan"),
        "coord2_p95":    float(np.percentile(coord2_a, 95)) if len(coord2_a) > 0 else float("nan"),
    }


def render_report(results: dict[str, dict], datasets: list[str]) -> str:
    lines = ["# GMT Dataset Feature Audit\n"]

    # occupancy
    lines.append("## Stub occupancy per window\n")
    lines.append("| Dataset | Mean | p50 | p90 | p95 | p99 | Max | % zero |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        s = r["n_stubs"]
        if len(s) == 0:
            lines.append(f"| {ds} | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {ds} | {np.mean(s):.1f} | {np.percentile(s,50):.0f} | "
            f"{np.percentile(s,90):.0f} | {np.percentile(s,95):.0f} | "
            f"{np.percentile(s,99):.0f} | {s.max()} | "
            f"{100*np.mean(s==0):.1f}% |"
        )

    # class balance
    lines.append("\n## Class balance (valid stubs)\n")
    lines.append("| Dataset | Signal | Noise | Signal % | Ambiguous |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        total = r["n_signal"] + r["n_noise"]
        sig_pct = 100.0 * r["n_signal"] / max(1, total)
        lines.append(
            f"| {ds} | {r['n_signal']:,} | {r['n_noise']:,} | "
            f"{sig_pct:.1f}% | {r['n_ambiguous']:,} |"
        )

    # missingness
    lines.append("\n## Feature missingness\n")
    lines.append("| Dataset | has_eta2 (mean frac) | has_coord2 (mean frac) |")
    lines.append("| --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        lines.append(
            f"| {ds} | {r['has_eta2_mean']:.3f} | {r['has_c2_mean']:.3f} |"
        )

    # coord2 distribution
    lines.append("\n## offlineCoord2 (bend proxy) distribution\n")
    lines.append("| Dataset | p5 (rad) | p50 (rad) | p95 (rad) |")
    lines.append("| --- | --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        lines.append(
            f"| {ds} | {r['coord2_p5']:.4f} | {r['coord2_p50']:.4f} | {r['coord2_p95']:.4f} |"
        )

    # feature ranges
    lines.append("\n## Feature min/max/mean (over all valid stubs, all datasets combined)\n")
    lines.append("| Feature | Min | Mean | Max |")
    lines.append("| --- | --- | --- | --- |")
    all_min  = np.full(N_FEATURES, np.inf)
    all_max  = np.full(N_FEATURES, -np.inf)
    all_mean = np.zeros(N_FEATURES)
    n_ds = 0
    for ds in datasets:
        r = results[ds]
        all_min  = np.minimum(all_min,  r["feat_min"])
        all_max  = np.maximum(all_max,  r["feat_max"])
        all_mean += r["feat_mean"]
        n_ds += 1
    all_mean /= max(1, n_ds)
    for fi, fname in enumerate(FEATURE_NAMES):
        lines.append(
            f"| `{fname}` | {all_min[fi]:.4f} | {all_mean[fi]:.4f} | {all_max[fi]:.4f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="GMT dataset feature audit")
    p.add_argument("--cache-dir", type=Path, default=Path("build/omtf_gmt/cache"))
    p.add_argument("--datasets", nargs="+", default=["S1", "S3", "B1", "B4"])
    p.add_argument("--output", type=Path, default=Path("build/omtf_gmt/FEATURE_AUDIT.md"))
    args = p.parse_args()

    results = {}
    for ds in args.datasets:
        ds_dir = args.cache_dir / ds
        if not ds_dir.exists():
            print(f"  [{ds}] not found in cache, skipping")
            continue
        print(f"  Auditing {ds} ...", end="", flush=True)
        results[ds] = audit_dataset(args.cache_dir, ds)
        r = results[ds]
        sig_pct = 100.0 * r["n_signal"] / max(1, r["n_signal"] + r["n_noise"])
        print(f" {r['n_windows']:,} windows  "
              f"mean_stubs={np.mean(r['n_stubs']):.1f}  "
              f"signal%={sig_pct:.1f}%")

    if not results:
        print("No datasets found — build the cache first."); return

    report = render_report(results, list(results.keys()))
    print("\n" + report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
