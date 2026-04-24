"""
Report renderer for the dataset audit pipeline.

Produces:
  - DATASET_DECISION_REPORT.md  (filled tables + decision rules)
  - audit_results.json          (raw numbers for programmatic use)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from omtf.features import PAIR_FEATURE_NAMES


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #

def _row(cells: list, widths: list[int]) -> str:
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"


def _table(headers: list[str], rows: list[list]) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    lines = [
        _row(headers, widths),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for row in rows:
        lines.append(_row(row, widths))
    return "\n".join(lines)


def _fmt(v: Any, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


# --------------------------------------------------------------------------- #
# JSON serialisation helper
# --------------------------------------------------------------------------- #

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


# --------------------------------------------------------------------------- #
# Section renderers
# --------------------------------------------------------------------------- #

DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]


def _section_phase0(results: dict) -> str:
    p0 = results.get("phase0", {})
    if not p0:
        return "## Phase 0 — Integrity\n\n_not run_\n"

    rows = []
    for ds in DATASETS:
        r = p0.get(ds)
        if r is None:
            rows.append([ds, "—", "—", "—"])
            continue
        rows.append([ds, f"{r.n_entries:,}", f"{r.bad_nstubs + r.bad_label + r.bad_padding}",
                     f"{r.bad_pct:.3f}%"])

    t0 = _table(["Dataset", "Entries", "Bad entries", "Bad %"], rows)

    # NanoAOD join
    rows2 = []
    for ds in DATASETS:
        r = p0.get(ds)
        if r is None:
            rows2.append([ds, "—", "—", "—", "—"])
            continue
        rows2.append([ds, f"{r.n_entries:,}", f"{r.nano_matched:,}",
                      f"{r.nano_unmatched:,}", f"{r.nano_partial:,}"])

    t1 = _table(["Dataset", "OMTF entries", "NanoAOD matched", "Unmatched", "Partial join"], rows2)

    decisions = "\n".join(
        f"- **{ds}**: {p0[ds].decision()}" for ds in DATASETS if ds in p0
    )

    return (
        "## Phase 0 — Dataset trustworthiness\n\n"
        "### 0.1 Branch integrity\n\n" + t0 + "\n\n"
        "### 0.2 NanoAOD join validation\n\n" + t1 + "\n\n"
        "### Decisions\n\n" + decisions + "\n"
    )


def _section_phase1(results: dict) -> str:
    p1 = results.get("phase1", {})
    if not p1:
        return "## Phase 1 — Occupancy\n\n_not run_\n"

    rows_occ = []
    rows_mult = []
    for ds in DATASETS:
        r = p1.get(ds)
        if r is None:
            rows_occ.append([ds] + ["—"] * 7)
            rows_mult.append([ds] + ["—"] * 7)
            continue
        occ  = r.occupancy_stats()
        mult = r.multiplicity_stats()
        rows_occ.append([
            ds,
            _fmt(occ["mean"], 1), _fmt(occ["p50"], 1), _fmt(occ["p90"], 1),
            _fmt(occ["p95"], 1),  _fmt(occ["p99"], 1), str(occ["max"]),
            _fmt(occ["pct_over_nmax"], 2) + "%",
        ])
        rows_mult.append([
            ds,
            _fmt(mult["mean_total"], 1),  _fmt(mult["mean_signal"], 1),
            _fmt(mult["mean_noise"], 1),  _fmt(mult["zero_signal_pct"], 1) + "%",
            _fmt(mult["one_track_pct"], 1) + "%",
            _fmt(mult["two_track_pct"], 1) + "%",
            _fmt(mult["three_track_pct"], 1) + "%",
        ])

    t0 = _table(["Dataset","Mean","p50","p90","p95","p99","Max","% > 24"], rows_occ)
    t1 = _table(["Dataset","Mean total","Mean signal","Mean noise",
                 "Zero-signal %","1-track %","2-track %","3-track %"], rows_mult)

    decisions = "\n".join(
        f"- **{ds}**: {p1[ds].nmax_decision()}" for ds in DATASETS if ds in p1
    )

    return (
        "## Phase 1 — Occupancy and firmware bounds\n\n"
        "### 1.1 Raw stub occupancy (Nmax=24)\n\n" + t0 + "\n\n"
        "### 1.2 Signal / noise / track multiplicity\n\n" + t1 + "\n\n"
        "### Decisions\n\n" + decisions + "\n"
    )


def _section_phase2(results: dict) -> str:
    p2 = results.get("phase2", {})
    if not p2:
        return "## Phase 2 — Edge problem strength\n\n_not run_\n"

    rows_edge = []
    for ds in DATASETS:
        r = p2.get(ds)
        if r is None:
            rows_edge.append([ds] + ["—"] * 4)
            continue
        rows_edge.append([
            ds,
            f"{r.total_edges:,}",
            f"{r.positive_edges:,}",
            _fmt(r.positive_edge_pct, 2) + "%",
            _fmt(r.zero_positive_window_pct, 1) + "%",
        ])

    t0 = _table(["Dataset","Total edges","Positive edges","Positive %","Zero-pos windows %"],
                rows_edge)

    # per-feature separability
    feat_rows = []
    for fi, fname in enumerate(PAIR_FEATURE_NAMES):
        row = [fname]
        for ds in DATASETS:
            r = p2.get(ds)
            if r is None or r.feature_auc is None:
                row.append("—")
            else:
                auc = float(r.feature_auc[fi])
                row.append(f"{auc:.3f} ({r.separability_label(auc)})")
        feat_rows.append(row)

    t1 = _table(["Feature"] + DATASETS, feat_rows)

    decisions = "\n".join(
        f"- **{ds}**: {p2[ds].edge_class_decision()}"
        for ds in DATASETS if ds in p2 and p2[ds] is not None
    )

    return (
        "## Phase 2 — Edge problem strength\n\n"
        "### 2.1 Legal edge statistics\n\n" + t0 + "\n\n"
        "### 2.2 Feature separability (AUC, same-track vs different-track)\n\n" + t1 + "\n\n"
        "### Decisions\n\n" + decisions + "\n"
    )


def _section_phase3(results: dict) -> str:
    p3d = results.get("phase3_dist", {})
    p3c = results.get("phase3_corr", {})
    p3b = results.get("phase3_baseline", {})

    parts = ["## Phase 3 — pT learnability\n"]

    # distribution
    if p3d:
        rows = []
        for ds in DATASETS:
            r = p3d.get(ds)
            if r is None:
                rows.append([ds] + ["—"] * 9)
                continue
            s = r.stats()
            rows.append([
                ds, _fmt(s.get("mean"), 1),
                _fmt(s.get("p10"), 1), _fmt(s.get("p50"), 1), _fmt(s.get("p90"), 1),
                _fmt(s.get("p99"), 1),
                _fmt(s.get("pct_gt_5"), 1) + "%",
                _fmt(s.get("pct_gt_10"), 1) + "%",
                _fmt(s.get("pct_gt_20"), 1) + "%",
                _fmt(s.get("pct_gt_22"), 1) + "%",
            ])
        t = _table(["Dataset","Mean pT","p10","p50","p90","p99","% > 5","%>10","%>20","%>22"], rows)
        parts.append("\n### 3.1 pT distribution (single-track windows)\n\n" + t)

        # threshold bins
        bin_labels = ["2-4", "4-6", "8-12", "13-17", "18-24", ">24"]
        rows2 = []
        for ds in DATASETS:
            r = p3d.get(ds)
            if r is None:
                rows2.append([ds] + ["—"] * len(bin_labels))
                continue
            bins = r.threshold_bins()
            rows2.append([ds] + [str(bins.get(lb, "—")) for lb in bin_labels])
        t2 = _table(["Dataset"] + [f"{lb} GeV" for lb in bin_labels], rows2)
        parts.append("\n#### pT threshold-bin counts\n\n" + t2)

        decisions = "\n".join(
            f"- **{ds}**: {p3d[ds].imbalance_decision()}" for ds in DATASETS if ds in p3d
        )
        parts.append("\n#### Decisions\n\n" + decisions)

    # correlations
    if p3c:
        feat_names = ["n_sig","mean_phiB","mean_eta","mean_r","mean_quality",
                      "mean_layer","best_abs_kappa","mean_abs_kappa","mean_abs_dphi"]
        rows = []
        for fname in feat_names:
            row = [f"`{fname}`"]
            for ds in DATASETS:
                r = p3c.get(ds)
                if r is None or fname not in r.pearson:
                    row.append("—")
                else:
                    pr = r.pearson[fname]
                    sp = r.spearman[fname]
                    row.append(f"{_fmt(pr, 3)} / {_fmt(sp, 3)}")
            rows.append(row)
        t = _table(["Feature (Pearson/Spearman)"] + DATASETS, rows)
        parts.append("\n\n### 3.2 Curvature-proxy correlations with q/pT\n\n" + t)
        decisions = "\n".join(
            f"- **{ds}**: {p3c[ds].correlation_decision()}" for ds in DATASETS if ds in p3c
        )
        parts.append("\n#### Decisions\n\n" + decisions)

    # baseline
    if p3b:
        rows = []
        for ds in DATASETS:
            r = p3b.get(ds)
            if r is None:
                rows.append([ds] + ["—"] * 5)
                continue
            rows.append([
                ds,
                f"{r.n_train:,} / {r.n_test:,}",
                _fmt(r.linear_pearson, 3),
                _fmt(r.linear_mae, 4),
                _fmt(r.mlp_pearson, 3),
                _fmt(r.cls_auc.get("pT>10", float("nan")), 3),
            ])
        t = _table(["Dataset","Train/Test","Lin Pearson","Lin MAE","MLP Pearson","Cls AUC@10"], rows)
        parts.append("\n\n### 3.3 Tiny pT baseline\n\n" + t)
        decisions = "\n".join(
            f"- **{ds}**: {p3b[ds].baseline_decision()}" for ds in DATASETS if ds in p3b
        )
        parts.append("\n#### Decisions\n\n" + decisions)

    return "\n".join(parts) + "\n"


def _section_phase5(results: dict) -> str:
    r = results.get("phase5")
    if r is None:
        return "## Phase 5 — B4 false-positive audit\n\n_not run (no checkpoint provided)_\n"

    rows = [
        ["n_windows",         f"{r.n_windows:,}"],
        ["n_accepted",        f"{r.n_accepted:,}"],
        ["bg_accept %",       f"{r.accept_rate:.2f}%"],
        ["mean cand/accepted",f"{r.mean_candidates_per_accepted:.2f}"],
        ["accepted mean stubs", _fmt(r.accepted_mean_stubs, 2)],
        ["rejected mean stubs", _fmt(r.rejected_mean_stubs, 2)],
        ["accepted mean cand score", _fmt(r.accepted_mean_cand_score, 3)],
        ["rejected mean cand score", _fmt(r.rejected_mean_cand_score, 3)],
    ]
    t = _table(["Quantity", "Value"], rows)
    return (
        "## Phase 5 — B4 false-positive audit\n\n" + t + "\n\n"
        "### Decision\n\n" + r.decision() + "\n"
    )


def _section_phase6(results: dict) -> str:
    p6 = results.get("phase6", {})
    if not p6:
        return "## Phase 6 — SlotModel fairness\n\n_not run_\n"

    MT_DS = ["S3", "S4", "S5", "B3"]
    rows_order = []
    rows_conf  = []
    for ds in MT_DS:
        pair = p6.get(ds)
        if pair is None:
            rows_order.append([ds] + ["—"] * 4)
            rows_conf.append([ds]  + ["—"] * 4)
            continue
        so, mt = pair
        rows_order.append([
            ds,
            f"{so.n_two_track_windows:,}",
            _fmt(so.k1_higher_pt_pct, 1) + "%",
            _fmt(so.k1_lower_phi_pct, 1)  + "%",
            _fmt(so.tid1_highest_pt_pct, 1) + "%",
        ])
        rows_conf.append([
            ds,
            _fmt(mt.mean_dphi_tracks, 1),
            _fmt(mt.close_track_pct, 1)  + "%",
            _fmt(mt.mean_ambig_pct, 2)   + "%",
            _fmt(mt.layer_conflict_pct, 1) + "%",
        ])

    t0 = _table(["Dataset","2-track windows","k1>k2 pT %","k1 lower phi %","k1 is highest pT %"], rows_order)
    t1 = _table(["Dataset","Mean Δphi (hw)","Close-track %","Ambiguous stub %","Layer conflict %"], rows_conf)

    decisions = "\n".join(
        f"- **{ds}**: {p6[ds][0].ordering_decision()}\n  {p6[ds][1].confusion_decision()}"
        for ds in MT_DS if ds in p6
    )

    return (
        "## Phase 6 — SlotModel fairness\n\n"
        "### 6.1 trackId ordering stability\n\n" + t0 + "\n\n"
        "### 6.2 Multi-track confusion\n\n" + t1 + "\n\n"
        "### Decisions\n\n" + decisions + "\n"
    )


def _section_phase7(results: dict) -> str:
    p7 = results.get("phase7", {})
    if not p7:
        return "## Phase 7 — dXY visibility\n\n_not run_\n"

    DS7 = ["S2", "S5", "B2"]
    feat_names = ["mean_phiB_signal","mean_abs_phiB","n_signal_stubs",
                  "mean_r_signal","mean_quality","mean_eta_signal",
                  "best_abs_kappa","mean_abs_kappa"]

    rows = []
    for fname in feat_names:
        row = [f"`{fname}`"]
        for ds in DS7:
            r = p7.get(ds)
            if r is None or fname not in r.pearson:
                row.append("—")
            else:
                pr = r.pearson[fname]
                row.append(_fmt(pr, 3))
        rows.append(row)
    t = _table(["Feature (Pearson)"] + DS7, rows)

    stats_rows = []
    for ds in DS7:
        r = p7.get(ds)
        if r is None:
            stats_rows.append([ds] + ["—"] * 4)
            continue
        s = r.dxy_stats()
        stats_rows.append([ds, _fmt(s.get("p50"),1), _fmt(s.get("p90"),1),
                           _fmt(s.get("p99"),1), _fmt(s.get("max"),1)])
    t_stats = _table(["Dataset","dXY p50 (cm)","p90","p99","max"], stats_rows)

    decisions = "\n".join(
        f"- **{ds}**: {p7[ds].visibility_decision()}" for ds in DS7 if ds in p7
    )

    return (
        "## Phase 7 — dXY visibility\n\n"
        "### dXY distribution\n\n" + t_stats + "\n\n"
        "### Feature correlations with |dXY|\n\n" + t + "\n\n"
        "### Decisions\n\n" + decisions + "\n"
    )


# --------------------------------------------------------------------------- #
# Main entry points
# --------------------------------------------------------------------------- #

def render_report(results: dict, output_path: Path) -> None:
    """Write the filled markdown report."""
    date = datetime.now().strftime("%Y-%m-%d")
    sections = [
        f"# OMTF Dataset Decision Report\n\n**Date**: {date}\n",
        _section_phase0(results),
        _section_phase1(results),
        _section_phase2(results),
        _section_phase3(results),
        _section_phase5(results),
        _section_phase6(results),
        _section_phase7(results),
    ]
    output_path.write_text("\n---\n\n".join(sections))
    print(f"  Report written: {output_path}")


def save_json(results: dict, output_path: Path) -> None:
    """Serialise results to JSON (arrays → lists, dataclasses → dicts)."""
    def _serialise(obj):
        if hasattr(obj, "__dataclass_fields__"):
            d = {}
            for k in obj.__dataclass_fields__:
                v = getattr(obj, k)
                if k.startswith("_"):
                    continue
                d[k] = _serialise(v)
            return d
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output_path.write_text(json.dumps(_serialise(results), indent=2, cls=_NpEncoder))
    print(f"  JSON saved:   {output_path}")
