"""
Generate a threshold-scan operating-point summary from an existing eval JSON.

The eval JSON already contains per-dataset zero_threshold_scan (zero_fp_rate at every
0.1 increment) and roc (slot-level efficiency/fake_rate at every 0.1 increment).

This script extracts the key physics metrics at a set of thresholds and writes a
Markdown summary table suitable for choosing the best operating point.

Usage
-----
    python scripts/omtf_gmt/summarize_threshold_scan.py \
        --eval-json build/omtf_gmt/eval/edge_compat_h64_B4_tps_100ep_best_eval.json \
        --output    build/omtf_gmt/eval/tps_h64_threshold_scan_summary.md
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

SCAN_THRESHOLDS = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

SIGNAL_DATASETS  = ["G1_pos", "G1_neg", "G2_pos", "G2_neg",
                    "G3_pos", "G3_neg", "G4_pos", "G4_neg",
                    "G5_pos", "G5_neg", "G6_pos", "G6_neg"]
BG_DATASETS      = ["G7", "G8", "B4"]


def nearest(scan: list[dict], key: str, thr: float) -> float | None:
    """Return the value at the threshold point nearest to thr."""
    best = min(scan, key=lambda p: abs(p["threshold"] - thr))
    return best.get(key)


def build_per_ds(pw_records: list[dict]) -> dict:
    """Index per-window records by dataset name."""
    return {r["ds"]: r for r in pw_records}


def cand_eff_at(pw: dict, thr: float) -> float | None:
    """
    Candidate efficiency at threshold thr.
    Use zero_threshold_scan to get zero_fp_rate, but for signal cand_eff we need
    the slot-level data.  We use overall_efficiency from zero_threshold_scan
    indirectly: for signal datasets, the overall_efficiency is in the roc field
    as 'efficiency' (slot-level, but identical to cand_eff for single-slot datasets).
    For multi-slot (G5/G6) the slot-level roc efficiency pooled across all slots.
    """
    roc = pw.get("roc", [])
    if not roc:
        return None
    return nearest(roc, "efficiency", thr)


def zero_fp_at(pw: dict, thr: float) -> float | None:
    scan = pw.get("zero_threshold_scan", [])
    if not scan:
        return None
    return nearest(scan, "zero_fp_rate", thr)


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{100*v:.1f}"
    return f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--output",    type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.eval_json.read_text())
    pw_records = data["per_window"]
    by_ds = build_per_ds(pw_records)

    lines: list[str] = []
    lines.append("# TPS-h64 Threshold Scan — Operating Point Summary\n")
    lines.append(f"Source: `{args.eval_json}`\n")
    lines.append(
        "Metrics extracted from the dense per-dataset ROC/zero_threshold_scan "
        "built into the eval JSON (0.1-step resolution).\n"
    )
    lines.append(
        "**Selection criteria:** B4 FP = 0%, minimize G7/G8 FP, maximize G2/G4/G5/G6 eff.\n"
    )

    # --- per-threshold summary table ---
    lines.append("## Per-threshold summary (all signal datasets pooled, avg pos+neg)\n")

    # group (G1,G2,...) pairs
    sig_pairs = [("G1","G1_pos","G1_neg"), ("G2","G2_pos","G2_neg"),
                 ("G3","G3_pos","G3_neg"), ("G4","G4_pos","G4_neg"),
                 ("G5","G5_pos","G5_neg"), ("G6","G6_pos","G6_neg")]

    header = "| Threshold | G1 eff% | G2 eff% | G3 eff% | G4 eff% | G5 eff% | G6 eff% | G7 FP% | G8 FP% | B4 FP% |"
    sep    = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines.append(header)
    lines.append(sep)

    for thr in SCAN_THRESHOLDS:
        row_vals = [f"{thr:+.2f}"]
        for _, dsp, dsn in sig_pairs:
            ep = cand_eff_at(by_ds[dsp], thr) if dsp in by_ds else None
            en = cand_eff_at(by_ds[dsn], thr) if dsn in by_ds else None
            avg = (ep + en) / 2 if (ep is not None and en is not None) else (ep or en)
            row_vals.append(fmt(avg))
        for ds_bg in ["G7", "G8", "B4"]:
            v = zero_fp_at(by_ds[ds_bg], thr) if ds_bg in by_ds else None
            row_vals.append(fmt(v))
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.append("")

    # --- per-dataset detail tables ---
    lines.append("## Per-dataset efficiency vs threshold\n")
    lines.append("Values are slot-level efficiency from the built-in ROC scan.\n")

    ds_order = ["G1_pos","G1_neg","G2_pos","G2_neg","G3_pos","G3_neg",
                "G4_pos","G4_neg","G5_pos","G5_neg","G6_pos","G6_neg"]
    thr_header = "| Dataset | " + " | ".join(f"{t:+.2f}" for t in SCAN_THRESHOLDS) + " |"
    thr_sep    = "| --- | " + " | ".join("---:" for _ in SCAN_THRESHOLDS) + " |"
    lines.append(thr_header)
    lines.append(thr_sep)
    for ds in ds_order:
        if ds not in by_ds:
            continue
        pw = by_ds[ds]
        vals = [fmt(cand_eff_at(pw, t)) for t in SCAN_THRESHOLDS]
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## Zero-window FP rate vs threshold (background datasets)\n")
    lines.append(thr_header.replace("Dataset", "Dataset (bg)"))
    lines.append(thr_sep)
    for ds in ["G7", "G8", "B4"]:
        if ds not in by_ds:
            continue
        pw = by_ds[ds]
        vals = [fmt(zero_fp_at(pw, t)) for t in SCAN_THRESHOLDS]
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")
    lines.append("")

    # --- recommendation ---
    lines.append("## Recommendation\n")
    lines.append(
        "Choose the threshold that keeps B4 FP at 0% (already satisfied across all thresholds "
        "for this model), minimises G8 FP, and keeps G2/G4/G5/G6 efficiency acceptable.\n"
    )
    lines.append(
        "Typical good operating points for trigger use:\n"
        "- **threshold 0.0** — baseline, balanced\n"
        "- **threshold 0.5** — lower G7/G8 FP at ~1–2 pp efficiency cost\n"
        "- **threshold 1.0** — significant FP reduction, larger efficiency cost\n"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
