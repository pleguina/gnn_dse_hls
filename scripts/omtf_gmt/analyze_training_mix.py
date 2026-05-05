#!/usr/bin/env python
"""
GMT training mix analysis — decide dataset composition and labeling policy.

Analyses the GMT cache and produces five targeted sections:

  1. Event-level candidate-multiplicity distributions (0/1/2/3-cand windows
     per dataset), which drives candidate-head supervision quality.

  2. Stub-level composition (signal / noise / ambiguous fractions, occupancy
     stats and occupancy stratified by candidate count).

  3. Per-dataset learning-regime contribution table — shows which datasets
     are indispensable vs redundant for each supervision regime.

  4. Loss-domain supervision analysis for the candidate head and node head
     under two concrete training mixes.

  5. Candidate-head false-positive risk quantification: 0-cand window
     fraction per mix, expected batch composition, and oversampling factors
     needed to reach a target 0-cand rate.

Usage
-----
  python scripts/omtf_gmt/analyze_training_mix.py \\
      --cache-dir build/omtf_gmt/cache \\
      --datasets S1 S2 S3 S4 S5 B1 B2 B3 B4 \\
      --batch-size 512 \\
      --target-zero-frac 0.15 \\
      --output build/omtf_gmt/TRAINING_MIX_ANALYSIS.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import iter_shards

ALL_DS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]
K_MAX  = 3

# Thresholds for learning-regime classification (per non-empty window)
CLEAN_THRESH = 0.70   # sig_stub_frac > this  → "clean-signal window"
HEAVY_THRESH = 0.40   # sig_stub_frac < this  → "heavy-PU window"


# --------------------------------------------------------------------------- #
# Per-dataset statistics (streaming shards)                                   #
# --------------------------------------------------------------------------- #

def analyze_dataset(cache_dir: Path, ds: str) -> dict:
    n_windows = 0
    cand_dist = np.zeros(K_MAX + 1, dtype=np.int64)    # [0-cand .. K-cand] windows
    slot_pos  = np.zeros(K_MAX,     dtype=np.int64)    # windows where slot k is positive

    n_valid   = n_signal = n_noise = 0
    n_ambig_stub      = 0   # ambiguous valid stubs (any label)
    n_ambig_of_signal = 0   # ambiguous valid stubs that are signal

    occ_by_cand_sum   = np.zeros(K_MAX + 1, dtype=np.float64)
    occ_by_cand_count = np.zeros(K_MAX + 1, dtype=np.int64)
    occ_all: list[int] = []  # per-window stub count (for percentiles)

    clean_sig_wins = heavy_pu_wins = ambig_win_cnt = 0

    for shard in iter_shards(cache_dir, ds):
        vm   = shard["valid_mask"]   # (N, 24) bool
        nl   = shard["node_label"]   # (N, 24) float32   1.0 → signal
        tid  = shard["track_id"]     # (N, 24) int8/long
        amb  = shard["ambiguous"]    # (N, 24) uint8
        gpt  = shard["gen_pt"]       # (N, K)  float32

        N = vm.shape[0]
        n_windows += N

        # ---- candidate multiplicity per window ----
        n_cand = (gpt > 0).sum(dim=1).long().clamp(0, K_MAX)   # (N,)
        for k in range(K_MAX + 1):
            cand_dist[k] += int((n_cand == k).sum().item())

        # slot-level positive rate (for head supervision analysis)
        for k in range(K_MAX):
            slot_pos[k] += int((gpt[:, k] > 0).sum().item())

        # ---- occupancy ----
        n_stubs_w = vm.long().sum(dim=1)                        # (N,)
        occ_all.extend(n_stubs_w.tolist())
        for k in range(K_MAX + 1):
            mask_k = (n_cand == k)
            if mask_k.any():
                occ_by_cand_sum[k]   += float(n_stubs_w[mask_k].float().sum().item())
                occ_by_cand_count[k] += int(mask_k.long().sum().item())

        # ---- stub composition ----
        vm_f  = vm.reshape(-1)                                  # (N*24,)
        tid_f = tid.reshape(-1).long()
        amb_f = amb.reshape(-1).bool()

        valid_tid  = tid_f[vm_f]
        valid_amb  = amb_f[vm_f]
        sig_mask   = valid_tid > 0

        n_valid   += int(vm_f.sum().item())
        n_signal  += int(sig_mask.sum().item())
        n_noise   += int((valid_tid == 0).sum().item())
        n_ambig_stub      += int(valid_amb.sum().item())
        n_ambig_of_signal += int((valid_amb & sig_mask).sum().item())

        # ---- regime flags (per non-empty window) ----
        n_stubs_f = vm.float().sum(dim=1)                       # (N,)
        n_sig_f   = (nl * vm.float()).sum(dim=1)                # (N,)
        has_stubs = n_stubs_f > 0

        if has_stubs.any():
            sf = n_sig_f[has_stubs] / n_stubs_f[has_stubs]     # signal stub fraction
            clean_sig_wins += int((sf > CLEAN_THRESH).sum().item())
            heavy_pu_wins  += int((sf < HEAVY_THRESH).sum().item())

        amb_any = (amb.bool() & vm).any(dim=1)                  # (N,) bool
        ambig_win_cnt += int(amb_any.sum().item())

    occ_arr = np.array(occ_all, dtype=np.int32)
    occ_mean_by_cand = np.where(
        occ_by_cand_count > 0,
        occ_by_cand_sum / np.maximum(occ_by_cand_count, 1).astype(np.float64),
        np.nan,
    )

    return {
        "n_windows":         n_windows,
        "cand_dist":         cand_dist,
        "slot_pos":          slot_pos,
        "n_valid":           n_valid,
        "n_signal":          n_signal,
        "n_noise":           n_noise,
        "n_ambig_stub":      n_ambig_stub,
        "n_ambig_of_signal": n_ambig_of_signal,
        "occ_arr":           occ_arr,
        "occ_mean_by_cand":  occ_mean_by_cand,
        "clean_sig_wins":    clean_sig_wins,
        "heavy_pu_wins":     heavy_pu_wins,
        "ambig_win_cnt":     ambig_win_cnt,
    }


# --------------------------------------------------------------------------- #
# Mix aggregation                                                              #
# --------------------------------------------------------------------------- #

def mix_stats(results: dict[str, dict], datasets: list[str]) -> dict:
    cand_dist = np.zeros(K_MAX + 1, dtype=np.int64)
    slot_pos  = np.zeros(K_MAX,     dtype=np.int64)
    n_windows = n_valid = n_signal = n_noise = 0
    for d in datasets:
        r = results[d]
        cand_dist += r["cand_dist"]
        slot_pos  += r["slot_pos"]
        n_windows += r["n_windows"]
        n_valid   += r["n_valid"]
        n_signal  += r["n_signal"]
        n_noise   += r["n_noise"]
    return dict(
        n_windows=n_windows, cand_dist=cand_dist, slot_pos=slot_pos,
        n_valid=n_valid, n_signal=n_signal, n_noise=n_noise,
    )


# --------------------------------------------------------------------------- #
# Report rendering                                                             #
# --------------------------------------------------------------------------- #

def render_report(
    results: dict[str, dict],
    datasets: list[str],
    batch_size: int,
    target_zero_frac: float,
) -> str:
    lines: list[str] = []

    lines += [
        "# GMT Training Mix Analysis\n",
        f"Datasets: {', '.join(datasets)}  "
        f"| Batch size: {batch_size}  "
        f"| Target 0-cand fraction: {target_zero_frac:.0%}\n",
        f"Regime thresholds: clean-signal = sig_stub > {CLEAN_THRESH:.0%}, "
        f"heavy-PU = sig_stub < {HEAVY_THRESH:.0%} (non-empty windows only)\n",
        "---\n",
    ]

    # ------------------------------------------------------------------ #
    # Overview: dataset sizes                                              #
    # ------------------------------------------------------------------ #
    lines += ["## Overview — dataset sizes\n"]
    lines += ["| Dataset | Windows | Valid stubs | Signal stubs | Noise stubs | Signal % |"]
    lines += ["| --- | --- | --- | --- | --- | --- |"]
    total_win = 0
    for ds in datasets:
        r = results[ds]
        total_win += r["n_windows"]
        total_stubs = max(1, r["n_signal"] + r["n_noise"])
        lines.append(
            f"| {ds} | {r['n_windows']:,} | {r['n_valid']:,} | "
            f"{r['n_signal']:,} | {r['n_noise']:,} | "
            f"{100.0*r['n_signal']/total_stubs:.1f}% |"
        )
    lines.append(f"| **Total** | **{total_win:,}** | | | | |\n")

    # ------------------------------------------------------------------ #
    # Section 1: event-level candidate multiplicity                       #
    # ------------------------------------------------------------------ #
    lines += ["## 1. Event-level candidate multiplicity\n"]
    lines += [
        "Number and fraction of windows by candidate count.  "
        "Drives supervision density for each slot of the candidate head.\n"
    ]
    lines += [f"| Dataset | 0-cand | 1-cand | 2-cand | 3-cand | Total |"]
    lines += [f"| --- | --- | --- | --- | --- | --- |"]
    for ds in datasets:
        r  = results[ds]
        cd = r["cand_dist"]
        nw = r["n_windows"]
        cells = " | ".join(
            f"{cd[k]:,} ({100.0*cd[k]/max(1,nw):.1f}%)" for k in range(K_MAX + 1)
        )
        lines.append(f"| {ds} | {cells} | {nw:,} |")
    lines.append("")

    lines += ["### Fraction only (easier to scan)\n"]
    lines += ["| Dataset | 0-cand % | 1-cand % | 2-cand % | 3-cand % |"]
    lines += ["| --- | --- | --- | --- | --- |"]
    for ds in datasets:
        r  = results[ds]
        cd = r["cand_dist"]
        nw = max(1, r["n_windows"])
        cells = " | ".join(f"{100.0*cd[k]/nw:.1f}%" for k in range(K_MAX + 1))
        lines.append(f"| {ds} | {cells} |")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Section 2: stub-level composition                                   #
    # ------------------------------------------------------------------ #
    lines += ["## 2. Stub-level composition\n"]
    lines += [
        "Per-dataset stub statistics over valid stubs only.  "
        "Ambig-of-signal = ambiguous stubs that are also signal (may need masking).\n"
    ]
    lines += [
        "| Dataset | Signal % | Noise % | Ambig-of-signal % | "
        "Mean occ | p50 | p95 | p99 | % empty-win |"
    ]
    lines += ["| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds in datasets:
        r   = results[ds]
        nv  = max(1, r["n_valid"])
        ns  = max(1, r["n_signal"])
        occ = r["occ_arr"]
        lines.append(
            f"| {ds} | {100.0*r['n_signal']/nv:.1f}% "
            f"| {100.0*r['n_noise']/nv:.1f}% "
            f"| {100.0*r['n_ambig_of_signal']/ns:.1f}% "
            f"| {np.mean(occ):.1f} "
            f"| {np.percentile(occ,50):.0f} "
            f"| {np.percentile(occ,95):.0f} "
            f"| {np.percentile(occ,99):.0f} "
            f"| {100.0*np.mean(occ==0):.1f}% |"
        )
    lines.append("")

    # ------------------------------------------------------------------ #
    # Section 3: occupancy by candidate count                             #
    # ------------------------------------------------------------------ #
    lines += ["## 3. Occupancy stratified by candidate count\n"]
    lines += [
        "Mean stubs per window, split by the number of true candidates in that window.  "
        "Reveals how detector occupancy correlates with reconstruction difficulty.\n"
    ]
    lines += ["| Dataset | Occ@0-cand | Occ@1-cand | Occ@2-cand | Occ@3-cand |"]
    lines += ["| --- | --- | --- | --- | --- |"]
    for ds in datasets:
        r = results[ds]
        cells = []
        for k in range(K_MAX + 1):
            v = r["occ_mean_by_cand"][k]
            cells.append("—" if np.isnan(v) else f"{v:.1f}")
        lines.append(f"| {ds} | {' | '.join(cells)} |")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Section 4: learning-regime contribution table                       #
    # ------------------------------------------------------------------ #
    lines += ["## 4. Learning-regime contribution\n"]
    lines += [
        "| Dataset | 0-cand % | 1-cand % | 2-cand % | 3-cand % "
        "| Clean-sig % | Heavy-PU % | Ambig-win % |"
    ]
    lines += ["| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds in datasets:
        r  = results[ds]
        cd = r["cand_dist"]
        nw = max(1, r["n_windows"])
        c  = " | ".join(f"{100.0*cd[k]/nw:.1f}%" for k in range(K_MAX + 1))
        lines.append(
            f"| {ds} | {c} "
            f"| {100.0*r['clean_sig_wins']/nw:.1f}% "
            f"| {100.0*r['heavy_pu_wins']/nw:.1f}% "
            f"| {100.0*r['ambig_win_cnt']/nw:.1f}% |"
        )
    lines.append("")

    lines += ["### Regime key\n"]
    lines += [
        "| Regime | Definition | Main learning signal |",
        "| --- | --- | --- |",
        f"| Clean-signal | sig_stub_frac > {CLEAN_THRESH:.0%} | Pure track reconstruction |",
        f"| Heavy-PU | sig_stub_frac < {HEAVY_THRESH:.0%}, ≥1 stub | PU rejection |",
        "| Ambig-win | ≥1 ambiguous stub | Ambiguity resolution |",
        "| 0-cand | cand_target = [0,0,0] | False-positive suppression |\n",
    ]

    # ------------------------------------------------------------------ #
    # Section 5: loss-domain supervision analysis                         #
    # ------------------------------------------------------------------ #
    lines += ["## 5. Loss-domain supervision analysis\n"]

    proposed_mix = [d for d in ["S1", "S3", "B1", "B4"] if d in results]
    all_mix      = datasets

    MIXES: dict[str, list[str]] = {}
    if proposed_mix:
        MIXES[f"Original proposal ({'+'.join(proposed_mix)})"] = proposed_mix
    MIXES[f"All datasets ({'+'.join(all_mix)})"] = all_mix

    # 5a: candidate head — cand multiplicity
    lines += ["### 5a. Candidate head — target multiplicity distribution\n"]
    lines += [
        "| Mix | Windows | 0-cand | 1-cand | 2-cand | 3-cand | 0-cand % |"
    ]
    lines += ["| --- | --- | --- | --- | --- | --- | --- |"]
    for name, mix in MIXES.items():
        ms = mix_stats(results, mix)
        cd = ms["cand_dist"]
        nw = max(1, ms["n_windows"])
        cells = " | ".join(f"{cd[k]:,}" for k in range(K_MAX + 1))
        lines.append(
            f"| {name} | {nw:,} | {cells} | {100.0*cd[0]/nw:.1f}% |"
        )
    lines.append("")

    # 5b: candidate head — per-slot positive rate
    lines += ["### 5b. Candidate head — per-slot positive rate\n"]
    lines += [
        "Fraction of training windows where each output slot has a positive target.  "
        "Slot imbalance here directly sets the BCE positive-class rate.\n"
    ]
    lines += ["| Mix | Slot 0 (%) | Slot 1 (%) | Slot 2 (%) |"]
    lines += ["| --- | --- | --- | --- |"]
    for name, mix in MIXES.items():
        ms = mix_stats(results, mix)
        nw = max(1, ms["n_windows"])
        cells = " | ".join(f"{100.0*ms['slot_pos'][k]/nw:.1f}%" for k in range(K_MAX))
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    # 5c: node head — signal/noise balance
    lines += ["### 5c. Node head — signal/noise stub balance\n"]
    lines += [
        "| Mix | Signal stubs | Noise stubs | Noise:Signal |"
    ]
    lines += ["| --- | --- | --- | --- |"]
    for name, mix in MIXES.items():
        ms = mix_stats(results, mix)
        ns = max(1, ms["n_signal"])
        lines.append(
            f"| {name} | {ms['n_signal']:,} | {ms['n_noise']:,} "
            f"| {ms['n_noise']/ns:.2f}:1 |"
        )
    lines.append("")

    # 5d: edge head note
    lines += [
        "### 5d. Edge head\n",
        "Not applicable for Phase B1 (DeepSets baseline has no edge construction).  "
        "Edge-head supervision analysis will be needed before implementing the "
        "edge-compatibility model.\n",
    ]

    # ------------------------------------------------------------------ #
    # Section 6: false-positive risk (B4 / 0-cand underrepresentation)   #
    # ------------------------------------------------------------------ #
    lines += ["## 6. False-positive risk analysis\n"]
    lines += [
        f"With batch_size={batch_size}, the probability that a random batch contains "
        f"**at least one 0-cand window** is 1 − (1 − p)^{batch_size}.\n"
    ]
    lines += [
        "| Mix | 0-cand windows | 0-cand % | "
        "Expected 0-cand/batch | P(any 0-cand in batch) |"
    ]
    lines += ["| --- | --- | --- | --- | --- |"]
    mix_zero_fracs: dict[str, float] = {}
    for name, mix in MIXES.items():
        ms  = mix_stats(results, mix)
        cd  = ms["cand_dist"]
        nw  = max(1, ms["n_windows"])
        p   = cd[0] / nw
        mix_zero_fracs[name] = p
        exp = p * batch_size
        p_any = 1.0 - (1.0 - p) ** batch_size
        lines.append(
            f"| {name} | {cd[0]:,} | {100.0*p:.2f}% "
            f"| {exp:.1f} | {100.0*p_any:.1f}% |"
        )
    lines.append("")

    lines += ["### Oversampling factors to reach target 0-cand fraction\n"]
    lines += [
        f"Target: **{target_zero_frac:.0%}** of training windows should be 0-candidate.  "
        f"Factor applies to the pure-0-cand dataset(s) (e.g. B4).\n"
    ]
    for name, mix in MIXES.items():
        ms = mix_stats(results, mix)
        cd = ms["cand_dist"]
        nw = max(1, ms["n_windows"])
        p  = cd[0] / nw
        n0 = cd[0]
        n_pos = nw - n0

        if p >= target_zero_frac:
            lines.append(
                f"- **{name}**: current 0-cand rate = {100.0*p:.1f}% — "
                f"already at or above target. No oversampling needed."
            )
        else:
            # Solve: (n0 * x) / (n0 * x + n_pos) = target
            # → n0*x = target * (n0*x + n_pos)
            # → n0*x*(1-target) = target*n_pos
            # → x = target*n_pos / (n0*(1-target))
            x = (target_zero_frac * n_pos) / max(1, n0 * (1.0 - target_zero_frac))
            lines.append(
                f"- **{name}**: current = {100.0*p:.1f}%, target = {100.0*target_zero_frac:.0f}% → "
                f"oversample 0-cand dataset(s) by **{x:.1f}×**."
            )
    lines.append("")

    # ------------------------------------------------------------------ #
    # Section 7: recommendations                                          #
    # ------------------------------------------------------------------ #
    lines += ["## 7. Recommendations\n"]

    # Indispensability
    pure_zero = [d for d in datasets if results[d]["cand_dist"][0] == results[d]["n_windows"]]
    mixed_zero = [d for d in datasets
                  if 0 < results[d]["cand_dist"][0] < results[d]["n_windows"]]
    three_cand = [d for d in datasets if results[d]["cand_dist"][3] > 0]
    two_cand   = [d for d in datasets if results[d]["cand_dist"][2] > 0]

    lines += ["### Indispensable datasets\n"]
    if pure_zero:
        lines.append(
            f"**Pure 0-cand datasets** (the only source of all-zero cand_target): "
            f"{', '.join(pure_zero)}.  Must always be included."
        )
    if mixed_zero:
        lines.append(
            f"**Mixed 0-cand datasets** (have some 0-cand windows): "
            f"{', '.join(mixed_zero)}.  Provide 0-cand signal only if "
            f"the window-builder does not filter them out."
        )
    if three_cand:
        lines.append(
            f"**3-cand datasets** (only training signal for slot-2 of the "
            f"candidate head): {', '.join(three_cand)}.  Removing them leaves "
            f"slot 2 with zero positive training examples."
        )
    if not two_cand:
        lines.append(
            "**No dataset provides 2-cand windows.** Slot 1 of the candidate head "
            "will only see positive targets from close-dimuon events."
        )
    lines.append("")

    # Ambiguity
    lines += ["### Labeling policy — ambiguous stubs\n"]
    flagged = [(d, 100.0 * results[d]["n_ambig_of_signal"] / max(1, results[d]["n_signal"]))
               for d in datasets
               if results[d]["n_signal"] > 0]
    high_ambig = [(d, pct) for d, pct in flagged if pct > 5.0]
    low_ambig  = [(d, pct) for d, pct in flagged if pct <= 5.0]

    if high_ambig:
        lines += [
            f"Datasets with > 5% ambiguous-of-signal stubs (consider masking from node_loss):\n"
        ]
        for d, pct in sorted(high_ambig, key=lambda x: -x[1]):
            lines.append(f"- {d}: {pct:.1f}%")
        lines.append("")
    if low_ambig:
        ds_str = ", ".join(f"{d} ({pct:.1f}%)" for d, pct in sorted(low_ambig, key=lambda x: -x[1]))
        lines += [f"Low ambiguity (< 5%): {ds_str} — masking not critical.\n"]

    # Regime redundancy note
    lines += ["### Regime redundancy\n"]
    lines += [
        "Examine the regime table (section 4) to identify redundant pairs.  "
        "Datasets with nearly identical (0-cand%, clean-sig%, heavy-PU%) distributions "
        "offer diminishing returns when both are included.  "
        "The primary differentiators are candidate multiplicity and occupancy regime — "
        "prioritise datasets that are unique in either dimension.\n"
    ]

    # Slot-2 warning
    if three_cand:
        lines += ["### Slot-2 supervision warning\n"]
        for d in three_cand:
            r  = results[d]
            cd = r["cand_dist"]
            nw = max(1, r["n_windows"])
            lines.append(
                f"- {d}: {cd[3]:,} 3-cand windows ({100.0*cd[3]/nw:.1f}%) — "
                f"sole provider of slot-2 positives."
            )
        lines.append(
            "\nIf these datasets are excluded, slot 2 of the candidate head sees "
            "only negative targets and will learn to always predict 'no third candidate', "
            "silently breaking multi-muon reconstruction.\n"
        )

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="GMT training mix analysis")
    p.add_argument("--cache-dir",         type=Path,  default=Path("build/omtf_gmt/cache"))
    p.add_argument("--datasets",          nargs="+",  default=ALL_DS)
    p.add_argument("--batch-size",        type=int,   default=512)
    p.add_argument("--target-zero-frac",  type=float, default=0.15)
    p.add_argument("--output",            type=Path,
                   default=Path("build/omtf_gmt/TRAINING_MIX_ANALYSIS.md"))
    args = p.parse_args()

    results: dict[str, dict] = {}
    for ds in args.datasets:
        ds_dir = args.cache_dir / ds
        if not ds_dir.exists():
            print(f"  [{ds}] not in cache — skipping")
            continue
        print(f"  Analysing {ds} ...", end="", flush=True)
        results[ds] = analyze_dataset(args.cache_dir, ds)
        r  = results[ds]
        cd = r["cand_dist"]
        nw = r["n_windows"]
        print(
            f" {nw:,} windows | "
            + "  ".join(f"{k}-cand={100.0*cd[k]/nw:.1f}%" for k in range(K_MAX + 1))
        )

    if not results:
        print("No datasets found — build the cache first.")
        return

    report = render_report(results, list(results.keys()), args.batch_size, args.target_zero_frac)
    print("\n" + report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
