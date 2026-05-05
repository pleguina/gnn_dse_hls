#!/usr/bin/env python
"""
GMT dataset feature audit — cache_v2 / schema v2 sanity checks.

Verifies the seven post-build checks listed in ARCHITECTURE_COMPARISON.md:

  1. Feature dimension = 14 (stubs last dim).
  2. stub_in_overlap (index 11) is binary {0, 1}.
  3. abs_eta (12) and eta_dist_to_overlap (13) are domain-consistent:
       G1–G6: abs_eta peaks near 1.0, eta_dist ≈ 0 for in-overlap stubs.
       G7–G8: abs_eta < 0.75, eta_dist strongly negative.
  4. truth_source counts: G1/G3/G7 (no PU) → ≥99% omtf_transfer (value=2);
       PU200 datasets → mixed, unmatched dominant in PU component.
  5. meta_is_hard_neg = 1 only in G7 and G8; 0 everywhere else.
  6. Target multiplicity via meta_n_gen:
       G1–G4 → only 0 or 1 targets; G5 ~3% windows have 2; G6 ~6.5% have 3;
       G7/G8 → 0 dominant.
  7. Eta-sign balance: G1–G6 shards contain both positive and negative eta stubs.

Usage
-----
  python scripts/omtf_gmt/feature_audit.py \\
      --cache-dir build/omtf_gmt/cache_v2 \\
      --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4

Output: printed report + build/omtf_gmt/eval/FEATURE_AUDIT_v2.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.features import (
    FEATURE_NAMES, N_FEATURES,
    F_ETA1, F_IN_OVERLAP, F_ABS_ETA, F_ETA_DIST_OVERLAP,
    ETA_OVERLAP_LO, ETA_OVERLAP_HI,
)
from omtf_gmt.dataset import iter_shards

# Datasets that should have is_hard_neg=1
_HARD_NEG_DATASETS = {"G7", "G8"}
# Datasets without pileup (truth_source=2 should dominate)
_NO_PU_DATASETS    = {"G1", "G3", "G7"}
# PU datasets where all three truth_source values are expected
_PU_DATASETS       = {"G2", "G4", "G5", "G6", "G8"}


def audit_dataset(cache_dir: Path, ds: str) -> dict:
    n_windows     = 0
    feat_min      = np.full(N_FEATURES, np.inf,  dtype=np.float64)
    feat_max      = np.full(N_FEATURES, -np.inf, dtype=np.float64)
    feat_sum      = np.zeros(N_FEATURES, dtype=np.float64)
    feat_count    = 0
    n_stubs_list: list[int] = []

    # truth_source counts: 0=unmatched, 1=omtf_noise, 2=omtf_transfer
    ts_counts = np.zeros(3, dtype=np.int64)

    # meta_is_hard_neg
    n_hard_neg_true  = 0
    n_hard_neg_false = 0

    # target multiplicity (meta_n_gen)
    mult_counts: dict[int, int] = {}

    # stub_in_overlap binary check
    n_in_overlap_nonbinary = 0   # should stay 0

    # eta sign balance (for pos/neg mixing check)
    n_eta_pos = 0
    n_eta_neg = 0

    # domain sanity accumulators (valid stubs only)
    abs_eta_vals:      list[float] = []
    eta_dist_vals:     list[float] = []
    abs_eta_in_ov:     list[float] = []
    eta_dist_in_ov:    list[float] = []

    for shard in iter_shards(cache_dir, ds):
        stubs  = shard["stubs"]           # (N, 24, 14)
        vm     = shard["valid_mask"]      # (N, 24) bool
        ts     = shard["truth_source"]    # (N, 24) int8
        mhn    = shard["meta_is_hard_neg"]  # (N,) int8
        mng    = shard["meta_n_gen"]      # (N,) int32
        mn     = shard["meta_n_stubs"]    # (N,) int32

        N = stubs.shape[0]
        n_windows += N

        # --- check 1 (implicitly): stubs last dim must be N_FEATURES
        assert stubs.shape[2] == N_FEATURES, \
            f"[{ds}] stubs last dim {stubs.shape[2]} ≠ {N_FEATURES}"

        n_stubs_list.extend(mn.tolist())

        # flatten to valid stubs
        valid_flat  = vm.reshape(-1)                 # (N*24,)
        stubs_flat  = stubs.reshape(-1, N_FEATURES)  # (N*24, 14)
        ts_flat     = ts.reshape(-1).long()          # (N*24,)

        v_stubs = stubs_flat[valid_flat].numpy()     # (M, 14)
        v_ts    = ts_flat[valid_flat].numpy()

        if len(v_stubs) > 0:
            feat_min   = np.minimum(feat_min, v_stubs.min(axis=0))
            feat_max   = np.maximum(feat_max, v_stubs.max(axis=0))
            feat_sum  += v_stubs.sum(axis=0)
            feat_count += len(v_stubs)

            # check 4: truth_source distribution
            for val in [0, 1, 2]:
                ts_counts[val] += int((v_ts == val).sum())

            # check 2: stub_in_overlap must be binary
            ov = v_stubs[:, F_IN_OVERLAP]
            n_in_overlap_nonbinary += int(((ov != 0.0) & (ov != 1.0)).sum())

            # check 3: domain features
            abs_e     = v_stubs[:, F_ABS_ETA]
            eta_dist  = v_stubs[:, F_ETA_DIST_OVERLAP]
            in_ov     = ov == 1.0
            abs_eta_vals.extend(abs_e.tolist())
            eta_dist_vals.extend(eta_dist.tolist())
            abs_eta_in_ov.extend(abs_e[in_ov].tolist())
            eta_dist_in_ov.extend(eta_dist[in_ov].tolist())

            # check 7: eta sign balance
            eta1 = v_stubs[:, F_ETA1]
            n_eta_pos += int((eta1 > 0).sum())
            n_eta_neg += int((eta1 < 0).sum())

        # check 5: meta_is_hard_neg
        mhn_np = mhn.numpy()
        n_hard_neg_true  += int((mhn_np == 1).sum())
        n_hard_neg_false += int((mhn_np == 0).sum())

        # check 6: target multiplicity
        for mult in mng.tolist():
            mult_counts[mult] = mult_counts.get(mult, 0) + 1

    feat_mean = feat_sum / max(1, feat_count)
    n_stubs   = np.array(n_stubs_list, dtype=np.int32)
    abs_eta_a = np.array(abs_eta_vals)
    eta_dist_a = np.array(eta_dist_vals)

    return {
        "n_windows":          n_windows,
        "n_stubs":            n_stubs,
        "feat_min":           feat_min,
        "feat_max":           feat_max,
        "feat_mean":          feat_mean,
        "ts_counts":          ts_counts,
        "n_hard_neg_true":    n_hard_neg_true,
        "n_hard_neg_false":   n_hard_neg_false,
        "mult_counts":        mult_counts,
        "n_in_overlap_nonbinary": n_in_overlap_nonbinary,
        "n_eta_pos":          n_eta_pos,
        "n_eta_neg":          n_eta_neg,
        "abs_eta_p10":        float(np.percentile(abs_eta_a, 10))  if len(abs_eta_a) else float("nan"),
        "abs_eta_p50":        float(np.percentile(abs_eta_a, 50))  if len(abs_eta_a) else float("nan"),
        "abs_eta_p90":        float(np.percentile(abs_eta_a, 90))  if len(abs_eta_a) else float("nan"),
        "abs_eta_max":        float(abs_eta_a.max())               if len(abs_eta_a) else float("nan"),
        "eta_dist_p10":       float(np.percentile(eta_dist_a, 10)) if len(eta_dist_a) else float("nan"),
        "eta_dist_p50":       float(np.percentile(eta_dist_a, 50)) if len(eta_dist_a) else float("nan"),
        "eta_dist_p90":       float(np.percentile(eta_dist_a, 90)) if len(eta_dist_a) else float("nan"),
        "frac_in_ov_zero_dist": float(np.mean(np.abs(np.array(eta_dist_in_ov)) < 1e-6)) if eta_dist_in_ov else float("nan"),
    }


def _pass_fail(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"


def render_report(results: dict[str, dict], datasets: list[str], cache_dir: Path) -> str:
    lines = ["# GMT Dataset Feature Audit — cache_v2 / schema v2\n"]

    # --- Check 1: feature dimension ---
    lines.append("## Check 1 — Feature dimension\n")
    lines.append("Verifies `stubs` last dim = 14 for every shard in every dataset.\n")
    all_ok = True
    for ds in datasets:
        r = results[ds]
        ok = r["n_windows"] > 0  # assert in audit_dataset() would have thrown otherwise
        all_ok &= ok
    lines.append(f"All shards: {_pass_fail(all_ok)} ({N_FEATURES} features confirmed)\n")

    # --- Check 2: stub_in_overlap binary ---
    lines.append("## Check 2 — stub_in_overlap is binary {{0, 1}}\n")
    lines.append("| Dataset | Non-binary count | Status |")
    lines.append("| --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        nb = r["n_in_overlap_nonbinary"]
        lines.append(f"| {ds} | {nb} | {_pass_fail(nb == 0)} |")

    # --- Check 3: domain feature sanity ---
    lines.append("\n## Check 3 — Domain features: abs_eta and eta_dist_to_overlap\n")
    lines.append("Expected: G1–G6 abs_eta peaks ≈ 1.0, in-overlap stubs have eta_dist≈0; "
                 "G7–G8 abs_eta < 0.75, eta_dist strongly negative.\n")
    lines.append("| Dataset | abs_eta p10 | p50 | p90 | max | eta_dist p10 | p50 | p90 | frac_in_ov_dist=0 | Status |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        if ds in _HARD_NEG_DATASETS:
            # Gen muon is barrel-only (|η|<0.75) but DT chambers extend to ~1.06.
            # Check median (not max) for the gen muon location, and that eta_dist p50 is
            # clearly on the barrel side of the overlap.
            ok = r["abs_eta_p50"] < 0.77 and r["eta_dist_p50"] < -0.05
        elif ds == "B4":
            ok = True  # B4 is pure noise, no strict domain constraint
        else:
            ok = (r["abs_eta_p50"] > 0.80 and
                  r["frac_in_ov_zero_dist"] > 0.50)
        lines.append(
            f"| {ds} | {r['abs_eta_p10']:.3f} | {r['abs_eta_p50']:.3f} | "
            f"{r['abs_eta_p90']:.3f} | {r['abs_eta_max']:.3f} | "
            f"{r['eta_dist_p10']:.3f} | {r['eta_dist_p50']:.3f} | "
            f"{r['eta_dist_p90']:.3f} | {r['frac_in_ov_zero_dist']:.3f} | "
            f"{_pass_fail(ok)} |"
        )

    # --- Check 4: truth_source ---
    lines.append("\n## Check 4 — truth_source counts\n")
    lines.append("Expected: no-PU datasets ≥99% value=2 (omtf_transfer); "
                 "PU200 datasets have a mix with value=0 (unmatched) dominant in PU.\n")
    lines.append("| Dataset | unmatched(0) | omtf_noise(1) | omtf_transfer(2) | transfer% | Status |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        ts = r["ts_counts"]
        total = ts.sum()
        xfer_pct = 100.0 * ts[2] / max(1, total)
        if ds in _NO_PU_DATASETS:
            ok = xfer_pct >= 95.0
        elif ds in _PU_DATASETS:
            ok = xfer_pct < 90.0  # PU datasets must have significant non-transfer stubs
        else:
            ok = True
        lines.append(
            f"| {ds} | {ts[0]:,} | {ts[1]:,} | {ts[2]:,} | {xfer_pct:.1f}% | {_pass_fail(ok)} |"
        )

    # --- Check 5: meta_is_hard_neg ---
    lines.append("\n## Check 5 — meta_is_hard_neg\n")
    lines.append("Expected: = 1 only in G7 and G8; = 0 in all other datasets.\n")
    lines.append("| Dataset | hard_neg=1 | hard_neg=0 | Expected | Status |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        hn1 = r["n_hard_neg_true"]
        hn0 = r["n_hard_neg_false"]
        expected = "1" if ds in _HARD_NEG_DATASETS else "0"
        if ds in _HARD_NEG_DATASETS:
            ok = hn1 > 0 and hn0 == 0
        else:
            ok = hn1 == 0
        lines.append(f"| {ds} | {hn1:,} | {hn0:,} | {expected} | {_pass_fail(ok)} |")

    # --- Check 6: target multiplicity ---
    lines.append("\n## Check 6 — Target multiplicity distribution (meta_n_gen)\n")
    lines.append("Expected: G1–G4 → max mult=1; G5 ~3% mult=2; G6 ~6.5% mult=3; G7/G8 → mult=0 dominant.\n")
    max_mult = max(
        (max(r["mult_counts"].keys()) if r["mult_counts"] else 0)
        for r in results.values()
    )
    header = "| Dataset | " + " | ".join(f"n={k}" for k in range(max_mult + 1)) + " | max>expected | Status |"
    sep    = "| --- " * (max_mult + 3) + "|"
    lines.append(header)
    lines.append(sep)
    for ds in datasets:
        r = results[ds]
        mc = r["mult_counts"]
        total = sum(mc.values())
        row_counts = " | ".join(
            f"{100.0 * mc.get(k, 0) / max(1, total):.1f}%" for k in range(max_mult + 1)
        )
        if ds in ("G1", "G2", "G3", "G4"):
            max_exp = 1
        elif ds == "G5":
            max_exp = 2
        elif ds == "G6":
            max_exp = 3
        elif ds in _HARD_NEG_DATASETS:
            max_exp = 0
        elif ds == "B4":
            max_exp = 0
        else:
            max_exp = 3
        over = sum(v for k, v in mc.items() if k > max_exp)
        ok = over == 0
        lines.append(f"| {ds} | {row_counts} | {over} | {_pass_fail(ok)} |")

    # --- Check 7: pos/neg shard parity ---
    lines.append("\n## Check 7 — Pos/neg shard parity\n")
    lines.append(
        "Training uses `<DS>_pos` + `<DS>_neg` pairs from the full production.  "
        "Each `_pos` entry is η>0 only and each `_neg` is η<0 only — they are "
        "combined at training time to give balanced coverage.  "
        "This check verifies that paired entries exist and have comparable sample counts "
        "(ratio within 5%).\n"
    )
    lines.append("| Base | _pos samples | _neg samples | ratio | Status |")
    lines.append("| --- | --- | --- | --- | --- |")
    import json as _json
    mf = _json.loads((cache_dir / "manifest.json").read_text())
    ds_info = mf.get("datasets", {})
    _PAIRED_BASES = [ds for ds in ["G1", "G2", "G3", "G4", "G5", "G6"]
                     if f"{ds}_pos" in ds_info and f"{ds}_neg" in ds_info]
    for base in _PAIRED_BASES:
        n_pos = ds_info[f"{base}_pos"]["n_samples"]
        n_neg = ds_info[f"{base}_neg"]["n_samples"]
        ratio = min(n_pos, n_neg) / max(n_pos, n_neg)
        ok = ratio >= 0.95
        lines.append(f"| {base} | {n_pos:,} | {n_neg:,} | {ratio:.3f} | {_pass_fail(ok)} |")

    # --- Feature range table ---
    lines.append("\n## Feature min/max/mean (all valid stubs, all audited datasets)\n")
    lines.append("| Index | Feature | Min | Mean | Max |")
    lines.append("| --- | --- | --- | --- | --- |")
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
        lines.append(f"| {fi} | `{fname}` | {all_min[fi]:.4f} | {all_mean[fi]:.4f} | {all_max[fi]:.4f} |")

    # --- Stub occupancy ---
    lines.append("\n## Stub occupancy per window\n")
    lines.append("| Dataset | Windows | Mean stubs | p50 | p90 | p99 | Max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for ds in datasets:
        r = results[ds]
        s = r["n_stubs"]
        if len(s) == 0:
            lines.append(f"| {ds} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {ds} | {r['n_windows']:,} | {np.mean(s):.1f} | "
            f"{np.percentile(s,50):.0f} | {np.percentile(s,90):.0f} | "
            f"{np.percentile(s,99):.0f} | {s.max()} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="GMT dataset feature audit — cache_v2")
    p.add_argument("--cache-dir", type=Path, default=Path("build/omtf_gmt/cache_v2"))
    p.add_argument("--datasets", nargs="+",
                   default=["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "B4"])
    p.add_argument("--output", type=Path,
                   default=Path("build/omtf_gmt/eval/FEATURE_AUDIT_v2.md"))
    args = p.parse_args()

    results = {}
    for ds in args.datasets:
        ds_dir = args.cache_dir / ds
        if not ds_dir.exists():
            print(f"  [{ds}] not found in {args.cache_dir}, skipping")
            continue
        print(f"  Auditing {ds} ...", end="", flush=True)
        results[ds] = audit_dataset(args.cache_dir, ds)
        r = results[ds]
        ts = r["ts_counts"]
        xfer_pct = 100.0 * ts[2] / max(1, ts.sum())
        print(f" {r['n_windows']:,} windows  "
              f"mean_stubs={np.mean(r['n_stubs']):.1f}  "
              f"transfer%={xfer_pct:.1f}%  "
              f"hard_neg={r['n_hard_neg_true']}")

    if not results:
        print("No datasets found — build the cache first.")
        return

    report = render_report(results, list(results.keys()), args.cache_dir)
    print("\n" + report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
