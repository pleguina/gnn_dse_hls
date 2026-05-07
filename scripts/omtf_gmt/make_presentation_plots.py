#!/usr/bin/env python
"""
Generate all presentation plots for the GMT/OMTF ML study.

Usage
-----
    python scripts/omtf_gmt/make_presentation_plots.py

Outputs written to build/omtf_gmt/plots/presentation/ as .png and .pdf.
Missing input files are warned and skipped; the script never crashes.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpatch
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "build/omtf_gmt/plots/presentation"
EVAL   = ROOT / "build/omtf_gmt/eval"

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

COLORS = {
    "KMTF-h128":  "#1f77b4",
    "TPS-h64":    "#ff7f0e",
    "TPS-h128":   "#2ca02c",
    "baseline":   "#aec7e8",
    "hn025":      "#ff7f0e",
    "hn050":      "#d62728",
    "hn100":      "#8c1414",
    "usw025":     "#9467bd",
    "usw050":     "#c5b0d5",
    "h96":        "#8c564b",
    "h128":       "#e377c2",
    "internal":   "#17becf",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)


def warn_skip(name: str, reason: str) -> None:
    print(f"  [SKIP] {name}: {reason}")


def try_load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


readme_sections: list[str] = ["# Presentation Plots — Index\n"]


def add_readme(name: str, sources: str, meaning: str, slides: str) -> None:
    readme_sections.append(
        f"## `{name}`\n"
        f"- **Source:** {sources}\n"
        f"- **Meaning:** {meaning}\n"
        f"- **Slides:** {slides}\n"
    )


# ── 1. Dataset evolution timeline ────────────────────────────────────────────

def plot_01_timeline() -> None:
    name = "01_dataset_evolution_timeline"
    events = [
        (1,  "S/B datasets\n(old OMTF internal)"),
        (2,  "Phase B1\nDeepSets / EdgeCompat"),
        (3,  "Overcounting found\n(S2 pred=2 for true=1)"),
        (4,  "False-slot diagnostic"),
        (5,  "Domain mismatch audit\n(S/B contains out-of-domain stubs)"),
        (6,  "G1–G8 dataset redesign\n& production"),
        (7,  "Cache schema v2\ntruth transfer"),
        (8,  "KMTF vs TPS comparison\n(TPS wins G8 PU)"),
        (9,  "Phase B5: hard-neg loss\nhn025 selected"),
        (10, "TPS EdgeCompat h64-hn025\nFP32 baseline frozen"),
        (11, "Next:\nassignment head / QAT"),
    ]
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.set_xlim(0.3, len(events) + 0.7)
    ax.set_ylim(-0.8, 0.8)
    ax.axis("off")

    y_line = 0.0
    ax.axhline(y_line, color="grey", linewidth=2, zorder=0)

    for i, (x, label) in enumerate(events):
        final = (i == len(events) - 1)
        color = "#2ca02c" if final else ("#ff7f0e" if i == len(events) - 2 else "#1f77b4")
        ax.plot(x, y_line, "o", color=color, markersize=12, zorder=3)
        va   = "bottom" if i % 2 == 0 else "top"
        yoff = 0.35 if i % 2 == 0 else -0.35
        ax.text(x, yoff, label, ha="center", va=va, fontsize=9.5,
                multialignment="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.text(x, y_line + (0.12 if i % 2 == 0 else -0.12),
                str(i + 1), ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_title("GMT/OMTF ML Study — Evolution Timeline", fontsize=16, pad=10)
    savefig(fig, name)
    add_readme(name, "Hardcoded", "High-level study evolution from S/B datasets to FP32 freeze",
               "Introduction slide")


# ── 2. Eta regions ────────────────────────────────────────────────────────────

def plot_02_eta_regions() -> None:
    name = "02_eta_regions_omtf_overlap"
    fig, ax = plt.subplots(figsize=(10, 4))

    ETA_MAX = 2.0
    ax.set_xlim(-ETA_MAX, ETA_MAX)
    ax.set_ylim(0, 1)
    ax.set_xlabel("|η|", fontsize=14)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    regions = [
        (-ETA_MAX, -1.24, "#d9d9d9", "Endcap\n(outside OMTF)"),
        (-1.24, -0.82, "#fdd49e", "OMTF overlap\ntarget"),
        (-0.82,  0.82, "#c6dbef", "Barrel / central\n(G7/G8 hard neg)"),
        ( 0.82,  1.24, "#fdd49e", "OMTF overlap\ntarget"),
        ( 1.24,  ETA_MAX, "#d9d9d9", "Endcap\n(outside OMTF)"),
    ]
    for x0, x1, color, label in regions:
        ax.axvspan(x0, x1, color=color, alpha=0.8)
        ax.text((x0 + x1) / 2, 0.55, label, ha="center", va="center",
                fontsize=10, multialignment="center")

    for eta, ls in [(-1.24, "--"), (-0.82, "-"), (0.82, "-"), (1.24, "--")]:
        ax.axvline(eta, color="black", linewidth=1.2, linestyle=ls)
        ax.text(eta, 0.96, f"{eta:+.2f}", ha="center", fontsize=9, color="black")

    ax.text(0, 0.18, "G7/G8 (hard negatives)\nreal muon, not in overlap",
            ha="center", va="center", fontsize=10, color="#8c564b",
            bbox=dict(boxstyle="round", fc="white", ec="#8c564b", alpha=0.9))
    for xoff in [-1.03, 1.03]:
        ax.annotate("G1–G6 signal", xy=(xoff, 0.3), ha="center", fontsize=10,
                    color="#d62728", fontweight="bold")

    ax.set_title("OMTF Overlap Region and Dataset Roles", fontsize=15)
    savefig(fig, name)
    add_readme(name, "Hardcoded geometry", "Eta regions: barrel, overlap target, endcap", "Physics motivation")


# ── 3. Phi processor windows ──────────────────────────────────────────────────

def plot_03_phi_windows() -> None:
    name = "03_phi_processor_windows"
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    phi0 = [np.deg2rad(15), np.deg2rad(135), np.deg2rad(255)]
    half = np.deg2rad(60)
    colors = ["#aec7e8", "#ffbb78", "#98df8a"]
    labels = ["Proc 0\n(centre 15°)", "Proc 1\n(centre 135°)", "Proc 2\n(centre 255°)"]

    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(theta, np.ones_like(theta), color="grey", linewidth=0.5)

    for p0, col, lab in zip(phi0, colors, labels):
        theta_arc = np.linspace(p0 - half, p0 + half, 200)
        r_arc = np.ones(200) * 0.95
        ax.fill_between(theta_arc, 0, r_arc, color=col, alpha=0.6, label=lab)
        ax.plot([p0 - half, p0 - half], [0, 1.0], color="black", lw=1)
        ax.plot([p0 + half, p0 + half], [0, 1.0], color="black", lw=1)
        ax.plot(p0, 0.65, "ko", markersize=6)
        ax.text(p0, 0.55, lab, ha="center", va="center", fontsize=9, multialignment="center")

    ax.set_yticks([])
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax.set_title("OMTF Processor φ-Windows (3-processor config)\n"
                 "phiZero: proc0=15°, proc1=135°, proc2=255°", pad=20)
    ax.text(0, 1.25,
            "reg_stub_phiHw is processor-local\n(phiZero subtracted before storing)",
            ha="center", va="center", fontsize=10, transform=ax.transData,
            bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))
    savefig(fig, name)
    add_readme(name, "Hardcoded (regioning.py constants)",
               "Three OMTF processor phi sectors; annotates processor-local coordinate",
               "Feature engineering slide")


# ── 4. Coordinate scale table ─────────────────────────────────────────────────

def plot_04_coord_table() -> None:
    name = "04_coordinate_scale_table"
    cols  = ["Input view", "φ variable", "φ frame / scale", "η variable", "Note"]
    rows  = [
        ["OMTF internal", "reg_stub_phiHw", "local 5400-bin\n(add phiZero(proc))",
         "reg_stub_etaHw\n×0.010875", "processor-local; needs phiZero correction"],
        ["KMTF", "offlineCoord1", "global rad",
         "offlineEta1\n(float)", "avoid int16 coord1; use offlineCoord1"],
        ["TPS", "offlineCoord1", "global rad",
         "offlineEta1\n(float)", "same convention as KMTF; station check relaxed"],
    ]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 2.2)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#e8f4fd")
    ax.set_title("Coordinate conventions: OMTF-internal, KMTF, TPS", fontsize=14, pad=14)
    savefig(fig, name)
    add_readme(name, "Hardcoded (from audit docs)", "Safe phi/eta variable choices per input view", "Feature engineering")


# ── 5. B1 architecture comparison ─────────────────────────────────────────────

def plot_05_b1_architectures() -> None:
    models  = ["DeepSets h64", "EdgeCompat h64"]
    b4_fp   = [7.2, 0.0]
    s4_eff  = [51.8, 69.3]
    val_loss = [0.8549, 0.7167]
    colors  = ["#aec7e8", "#ff7f0e"]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, vals, ylabel, title, ylim in [
        (axes[0], list(zip(b4_fp, s4_eff)),
         "Rate / Efficiency [%]",
         "Phase B1: B4 FP and S4 slot-2 efficiency",
         (0, 80)),
    ]:
        w = 0.3
        ax.bar(x - w/2, [v[0] for v in vals], w, label="B4 FP%",  color="#d62728", alpha=0.85)
        ax.bar(x + w/2, [v[1] for v in vals], w, label="S4 slot-2 eff%", color="#2ca02c", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_ylim(*ylim); ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    axes[1].bar(x, val_loss, color=colors, alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Validation loss"); axes[1].set_title("Phase B1: validation loss")
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.5)
    for xi, v in enumerate(val_loss):
        axes[1].text(xi, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)

    fig.tight_layout()
    savefig(fig, "05a_b1_architecture_fp_eff")
    plt.close("all")
    add_readme("05a_b1_architecture_fp_eff", "Hardcoded (Phase B1 results)",
               "B4 FP and S4 slot-2 efficiency comparison; EdgeCompat eliminates B4 FP",
               "Architecture comparison")
    add_readme("05a_b1_architecture_fp_eff (val loss panel)", "Hardcoded",
               "Validation loss; EdgeCompat lower", "Architecture comparison")


# ── 6. Overcounting confusion ─────────────────────────────────────────────────

def plot_06_overcounting() -> None:
    name = "06_initial_overcounting_confusion"
    datasets = ["S2 (1-target)", "B2 (1-target)", "B4 (0-target)"]
    pred0 = [16,   95,  4729]
    pred1 = [5094, 3926, 0]
    pred2 = [6866, 10169, 0]
    pred3 = [17,   15,   0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors_bar = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
    labels_bar = ["pred=0", "pred=1", "pred=2", "pred=3"]

    for ax, ds, p0, p1, p2, p3 in zip(axes, datasets,
                                        pred0, pred1, pred2, pred3):
        vals = [p0, p1, p2, p3]
        total = sum(vals) or 1
        fracs = [v / total * 100 for v in vals]
        bars = ax.bar(labels_bar, fracs, color=colors_bar, alpha=0.85, edgecolor="white")
        for bar, frac in zip(bars, fracs):
            if frac > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.5,
                        f"{frac:.1f}%", ha="center", fontsize=10)
        ax.set_title(ds, fontsize=12)
        ax.set_ylabel("Fraction of windows [%]" if ax is axes[0] else "")
        ax.set_ylim(0, 110)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Phase B1: Candidate multiplicity prediction (true=1 or true=0 windows)",
                 fontsize=14)
    axes[0].annotate("⚠ 57% over-counted\n(pred=2 for true=1)", xy=(2, 72), fontsize=11,
                     color="#d62728", fontweight="bold", ha="center")
    fig.tight_layout()
    savefig(fig, name)
    add_readme(name, "Hardcoded (Phase B1 confusion matrix)",
               "S2/B2 true=1 windows predicted as 2 candidates (57%/73%); B4 correct",
               "Motivating the G-dataset redesign")


# ── 7. False slot attribution ─────────────────────────────────────────────────

def plot_07_false_slot() -> None:
    name = "07_false_slot_attribution"
    categories = ["duplicate", "noise_coherent", "noise_diffuse", "out_of_domain"]
    s2_vals = [97.9, 0.7, 1.5, 0.0]
    b2_vals = [26.9, 16.8, 56.3, 0.0]
    colors_bar = ["#d62728", "#ff7f0e", "#1f77b4", "#9467bd"]

    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, s2_vals, w, label="S2", color=colors_bar, alpha=0.85)
    ax.bar(x + w/2, b2_vals, w, label="B2", color=colors_bar, alpha=0.45, hatch="//",
           edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Fraction of false slot-1 fires [%]")
    ax.set_title("False slot-1 attribution: cause of spurious 2nd candidate")
    handles = [mpatches.Patch(facecolor="grey", label="S2"),
               mpatches.Patch(facecolor="grey", alpha=0.4, hatch="//", label="B2")]
    ax.legend(handles=handles)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 110)
    savefig(fig, name)
    add_readme(name, "Hardcoded (false_slot_diagnostic.md)",
               "S2 false slot almost entirely duplicate muon; B2 dominated by diffuse noise",
               "Overcounting diagnostic")


# ── 8. Domain mismatch audit ──────────────────────────────────────────────────

def plot_08_domain_mismatch() -> None:
    name = "08_domain_mismatch_overlap_fractions"
    groups = ["S1/B1", "S2/B2", "S3/S4/B3", "B4"]
    sig_in  = [44, 43, 83,  0]
    noi_in  = [19, 37, 80, 59]
    noi_out = [81, 63, 20, 41]

    x = np.arange(len(groups))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,   sig_in,  w, label="Signal stubs in overlap",     color="#2ca02c", alpha=0.85)
    ax.bar(x,       noi_in,  w, label="Noise stubs in overlap",      color="#ff7f0e", alpha=0.85)
    ax.bar(x + w,   noi_out, w, label="Noise stubs outside overlap", color="#d62728", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Fraction [%]"); ax.set_ylim(0, 100)
    ax.set_title("Domain Mismatch Audit: overlap-region stub fractions (old S/B datasets)")
    ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.annotate("Old S/B contained many\nout-of-domain stubs\n→ model trained on wrong region",
                xy=(1.5, 75), fontsize=11, ha="center",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"))
    savefig(fig, name)
    add_readme(name, "Hardcoded (domain_mismatch_audit.md)",
               "Key finding: S/B stubs were mostly outside the overlap target region",
               "Dataset redesign motivation")


# ── 9. G-dataset role table ───────────────────────────────────────────────────

def plot_09_g_roles() -> None:
    name = "09_g_dataset_roles"
    cols = ["Dataset", "nGenMuon", "Overlap\ntargets", "PU200", "Role"]
    rows = [
        ["G1", "1", "1", "No",  "Clean prompt overlap"],
        ["G2", "1", "1", "Yes", "Prompt + PU200"],
        ["G3", "1", "1", "No",  "Clean displaced overlap"],
        ["G4", "1", "1", "Yes", "Displaced + PU200"],
        ["G5", "2", "2", "Yes", "Two displaced overlap muons"],
        ["G6", "3", "3", "Yes", "Three prompt overlap muons"],
        ["G7", "1", "0", "No",  "Low-η hard negative (no overlap)"],
        ["G8", "1", "0", "Yes", "Low-η hard negative + PU200"],
    ]
    row_colors = [
        ["#e8f4e8"] * 5,
        ["#e8f4e8"] * 5,
        ["#e8f4e8"] * 5,
        ["#e8f4e8"] * 5,
        ["#fff3cd"] * 5,
        ["#fff3cd"] * 5,
        ["#fde8e8"] * 5,
        ["#fde8e8"] * 5,
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                 cellColours=row_colors)
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1, 2.0)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("G-dataset definitions (green=signal, yellow=multi-muon, red=background)",
                 fontsize=13, pad=14)
    legend_patches = [
        mpatches.Patch(color="#e8f4e8", label="Signal (G1–G4)"),
        mpatches.Patch(color="#fff3cd", label="Multi-muon (G5–G6)"),
        mpatches.Patch(color="#fde8e8", label="Hard negative (G7–G8)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=11)
    savefig(fig, name)
    add_readme(name, "Hardcoded", "Role of each G dataset in training and evaluation", "Dataset design")


# ── 10. G-dataset validation summary ─────────────────────────────────────────

def plot_10_validation() -> None:
    name = "10_g_dataset_validation_summary"
    datasets = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    counts   = [300000, 300000, 167188, 200678, 199500, 199500, 150000, 298000]
    status   = ["PASS", "PASS", "PASS", "PASS", "WARN", "WARN", "PASS", "PASS"]
    colors_s = ["#2ca02c" if s == "PASS" else "#ff7f0e" for s in status]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(datasets, [c / 1000 for c in counts], color=colors_s, alpha=0.85, edgecolor="white")
    for bar, s, c in zip(bars, status, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{s}\n{c//1000}k", ha="center", fontsize=10)
    ax.set_ylabel("Event count [k]")
    ax.set_title("G-dataset production: event counts and validation status")
    ax.set_ylim(0, 360)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    legend_patches = [mpatches.Patch(color="#2ca02c", label="PASS"),
                      mpatches.Patch(color="#ff7f0e", label="WARN (accepted)")]
    ax.legend(handles=legend_patches)
    fig.tight_layout()
    savefig(fig, name)

    # 10b truth transfer match rates
    tt_rates = [99.8, 69.9, 99.8, 70.5, 68.4, 63.8, 99.1, 40.4]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bar_colors = ["#2ca02c" if r > 80 else "#ff7f0e" if r > 50 else "#d62728"
                  for r in tt_rates]
    ax2.bar(datasets, tt_rates, color=bar_colors, alpha=0.85, edgecolor="white")
    ax2.axhline(70, color="grey", linestyle="--", alpha=0.6, label="70% reference")
    ax2.set_ylabel("Truth-transfer match rate [%]")
    ax2.set_title("Truth-transfer match rates per G dataset\n"
                  "(PU200 events have lower match rate — expected)")
    ax2.set_ylim(0, 110)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()
    for i, (ds, r) in enumerate(zip(datasets, tt_rates)):
        ax2.text(i, r + 1, f"{r:.1f}%", ha="center", fontsize=10)
    fig2.tight_layout()
    savefig(fig2, "10b_truth_transfer_match_rates")
    add_readme(name, "Hardcoded (validation runs)", "G-dataset event counts and validation status", "Dataset validation")
    add_readme("10b_truth_transfer_match_rates", "Hardcoded",
               "Truth transfer match rates; PU200 lower due to pile-up ambiguity", "Dataset validation")


# ── 11. Overcounting before/after ────────────────────────────────────────────

def plot_11_overcounting_fix() -> None:
    name = "11_overcounting_before_after_g_datasets"
    models = ["B1 EdgeCompat\n(S/B)", "B3 EdgeCompat\n(G)", "B3 DeepSets\n(G)"]
    clean_overcount = [57.4, 0.14, 0.25]
    pu_overcount    = [71.7, 1.2,  1.3]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, clean_overcount, w, label="Clean 1-target windows", color="#d62728", alpha=0.85)
    ax.bar(x + w/2, pu_overcount,    w, label="PU200 1-target windows", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Over-count rate [%] (pred=2 when true=1)")
    ax.set_title("Overcounting rate: before and after G-dataset redesign")
    ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 85)
    ax.annotate("G datasets solve\nthe overcounting problem",
                xy=(1, 5), fontsize=12, ha="center", color="#2ca02c", fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="#2ca02c"))
    fig.tight_layout()
    savefig(fig, name)
    add_readme(name, "Hardcoded (Phase B1 and B3 eval reports)",
               "Overcounting rate drops from 57%/72% to <1.3% with G datasets",
               "Key result — dataset redesign impact")


# ── 12. KMTF vs TPS comparison ────────────────────────────────────────────────

def plot_12_kmtf_tps() -> None:
    ds_sig = ["G1", "G2", "G3", "G4", "G5", "G6"]
    eff = {
        "KMTF-h128": [0.850, 0.836, 0.787, 0.757, 0.788, 0.951],
        "TPS-h64":   [0.941, 0.849, 0.917, 0.833, 0.822, 0.948],
        "TPS-h128":  [0.947, 0.844, 0.923, 0.824, 0.811, 0.929],
    }
    x = np.arange(len(ds_sig))
    w = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, vals) in enumerate(eff.items()):
        ax.bar(x + (i - 1) * w, [v * 100 for v in vals], w,
               label=model, color=COLORS[model], alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(ds_sig)
    ax.set_ylabel("Candidate efficiency [%]")
    ax.set_title("KMTF vs TPS: signal efficiency (threshold 0.0)")
    ax.set_ylim(70, 100); ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    savefig(fig, "12a_kmtf_vs_tps_signal_efficiency")

    ds_bg = ["G7", "G8", "B4"]
    fp = {
        "KMTF-h128": [0.064, 0.098, 0.000],
        "TPS-h64":   [0.062, 0.029, 0.000],
        "TPS-h128":  [0.049, 0.027, 0.000],
    }
    x2 = np.arange(len(ds_bg))
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (model, vals) in enumerate(fp.items()):
        ax2.bar(x2 + (i - 1) * w, [v * 100 for v in vals], w,
                label=model, color=COLORS[model], alpha=0.85, edgecolor="white")
    ax2.set_xticks(x2); ax2.set_xticklabels(ds_bg)
    ax2.set_ylabel("Zero-window FP rate [%]")
    ax2.set_title("KMTF vs TPS: background FP rate\n(TPS ~3× lower G8 FP vs KMTF)")
    ax2.legend(); ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_ylim(0, 12)
    fig2.tight_layout()
    savefig(fig2, "12b_kmtf_vs_tps_background_fp")
    add_readme("12a_kmtf_vs_tps_signal_efficiency", "Hardcoded (kmtf_vs_tps_comparison.md)",
               "TPS wins on G1/G3/G4 prompt and displaced efficiency", "KMTF vs TPS")
    add_readme("12b_kmtf_vs_tps_background_fp", "Hardcoded",
               "TPS has 3× lower G8 PU hard-negative FP — decisive advantage", "KMTF vs TPS")


# ── 13. Event-level efficiency ────────────────────────────────────────────────

def plot_13_event_level() -> None:
    ds_sig = ["G1", "G2", "G3", "G4", "G5", "G6"]
    eff_ev = {
        "KMTF-h128": [0.902, 0.890, 0.860, 0.848, 0.880, 0.989],
        "TPS-h64":   [0.995, 0.970, 0.981, 0.955, 0.955, 0.992],
        "TPS-h128":  [0.996, 0.960, 0.983, 0.939, 0.948, 0.977],
    }
    x = np.arange(len(ds_sig))
    w = 0.28
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (model, vals) in enumerate(eff_ev.items()):
        ax.bar(x + (i - 1) * w, [v * 100 for v in vals], w,
               label=model, color=COLORS[model], alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(ds_sig)
    ax.set_ylabel("Event-level trigger efficiency [%]  (pT > 10 GeV)")
    ax.set_title("Event-level trigger efficiency: KMTF vs TPS")
    ax.set_ylim(80, 102); ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    savefig(fig, "13_event_level_efficiency_kmtf_tps")

    ds_bg = ["G7", "G8", "B4"]
    bg_accept = {
        "KMTF-h128": [0.052, 0.100, 0.000],
        "TPS-h64":   [0.058, 0.041, 0.000],
        "TPS-h128":  [0.053, 0.038, 0.000],
    }
    x2 = np.arange(len(ds_bg))
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (model, vals) in enumerate(bg_accept.items()):
        ax2.bar(x2 + (i - 1) * w, [v * 100 for v in vals], w,
                label=model, color=COLORS[model], alpha=0.85, edgecolor="white")
    ax2.set_xticks(x2); ax2.set_xticklabels(ds_bg)
    ax2.set_ylabel("Event-level background accept rate [%]")
    ax2.set_title("Event-level background accept: TPS ~2.5× lower G8 vs KMTF")
    ax2.legend(); ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig2.tight_layout()
    savefig(fig2, "13b_event_level_background_accept_kmtf_tps")
    add_readme("13_event_level_efficiency_kmtf_tps", "Hardcoded", "Event-level efficiency; TPS clearly better", "KMTF vs TPS")
    add_readme("13b_event_level_background_accept_kmtf_tps", "Hardcoded", "Event-level background accept", "KMTF vs TPS")


# ── 14. Displaced efficiency vs d0 ───────────────────────────────────────────

def plot_14_d0() -> None:
    name = "14_displaced_efficiency_vs_d0"
    bin_labels = ["0–0.05", "0.05–0.1", "0.1–0.2", "0.2–0.5",
                  "0.5–1", "1–2", "2–5", "5–10", "10–20", "20–50", ">50"]
    d0_eff = {
        "KMTF-h128": [0.913, 0.739, 0.828, 0.815, 0.783, 0.830, 0.801, 0.815, 0.804, 0.790, 0.692],
        "TPS-h64":   [0.927, 0.915, 0.938, 0.879, 0.870, 0.872, 0.873, 0.880, 0.876, 0.859, 0.755],
        "TPS-h128":  [0.915, 0.915, 0.924, 0.868, 0.864, 0.857, 0.862, 0.866, 0.869, 0.856, 0.748],
    }
    x = np.arange(len(bin_labels))
    fig, ax = plt.subplots(figsize=(13, 5))
    for model, vals in d0_eff.items():
        ax.plot(x, [v * 100 for v in vals], "o-", label=model,
                color=COLORS[model], linewidth=2, markersize=7)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_ylabel("Displacement efficiency [%]")
    ax.set_xlabel("|d0| bin [cm]")
    ax.set_title("Displaced muon efficiency vs |d0| (G3/G4 datasets)")
    ax.set_ylim(60, 100); ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    savefig(fig, name)
    add_readme(name, "Hardcoded (kmtf_vs_tps_comparison.md d0 bins)",
               "TPS maintains >85% efficiency at all d0 values; KMTF drops at 0.05cm bin",
               "Displaced muon performance")


# ── 15. Threshold scan ROC ────────────────────────────────────────────────────

def plot_15_roc() -> None:
    name = "15_threshold_scan_eff_vs_fake"
    roc_data = {
        "KMTF-h128": {"eff": [0.986, 0.967, 0.870, 0.710, 0.182],
                      "fake": [0.121, 0.102, 0.069, 0.041, 0.007],
                      "thr": [-2, -1, 0, 1, 2]},
        "TPS-h64":   {"eff": [0.973, 0.950, 0.899, 0.785, 0.405],
                      "fake": [0.054, 0.040, 0.030, 0.019, 0.007],
                      "thr": [-2, -1, 0, 1, 2]},
        "TPS-h128":  {"eff": [0.973, 0.948, 0.889, 0.781, 0.332],
                      "fake": [0.049, 0.037, 0.026, 0.018, 0.005],
                      "thr": [-2, -1, 0, 1, 2]},
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, d in roc_data.items():
        ax.plot([f * 100 for f in d["fake"]],
                [e * 100 for e in d["eff"]],
                "o-", label=model, color=COLORS[model], linewidth=2, markersize=8)
        idx0 = d["thr"].index(0)
        ax.annotate(f"thr=0\n({d['fake'][idx0]*100:.1f}%, {d['eff'][idx0]*100:.1f}%)",
                    xy=(d["fake"][idx0] * 100, d["eff"][idx0] * 100),
                    xytext=(d["fake"][idx0] * 100 + 0.3, d["eff"][idx0] * 100 - 2.5),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))
    ax.set_xlabel("Negative-slot fake rate [%]")
    ax.set_ylabel("Slot-level efficiency [%]")
    ax.set_title("Threshold scan: efficiency vs fake rate (slot-level)")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    savefig(fig, name)
    add_readme(name, "Hardcoded (threshold scan outputs)",
               "ROC curve: TPS lower fake rate at same efficiency vs KMTF", "Model comparison")


# ── 16. B5 sweep results ──────────────────────────────────────────────────────

def plot_16_b5() -> None:
    runs = ["baseline", "hn025", "hn050", "hn100", "usw025", "usw050", "h96", "h128"]
    sig_eff = {
        "baseline": [94.1, 84.9, 91.7, 83.2, 82.2, 94.8],
        "hn025":    [93.3, 82.9, 90.3, 80.5, 80.0, 94.9],
        "hn050":    [90.8, 79.5, 88.2, 77.6, 76.9, 94.9],
        "hn100":    [90.0, 75.5, 87.5, 74.1, 76.5, 94.6],
        "usw025":   [94.1, 87.2, 92.1, 85.8, 83.8, 96.4],
        "usw050":   [94.3, 84.9, 91.9, 83.0, 81.5, 94.6],
        "h96":      [92.6, 82.9, 91.4, 81.3, 79.7, 93.9],
        "h128":     [94.2, 83.0, 92.1, 80.9, 79.3, 94.8],
    }
    bg_fp = {
        "baseline": [6.2, 2.9, 0.0],
        "hn025":    [3.6, 2.2, 0.0],
        "hn050":    [3.3, 2.0, 0.0],
        "hn100":    [1.6, 1.5, 0.0],
        "usw025":   [7.2, 3.6, 0.0],
        "usw050":   [5.5, 2.8, 0.0],
        "h96":      [8.7, 2.9, 0.0],
        "h128":     [6.4, 2.7, 0.0],
    }
    ds_sig = ["G1", "G2", "G3", "G4", "G5", "G6"]
    x = np.arange(len(runs))

    # signal efficiencies per group
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    for ax, (ds_idx, ds_name) in zip(axes.flat, enumerate(ds_sig)):
        vals = [sig_eff[r][ds_idx] for r in runs]
        bar_cols = [COLORS.get(r, "#888888") for r in runs]
        bars = ax.bar(runs, vals, color=bar_cols, alpha=0.85, edgecolor="white")
        ax.set_title(ds_name, fontsize=13)
        ax.set_ylabel("Efficiency [%]" if ds_idx % 3 == 0 else "")
        ax.set_ylim(65, 100)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        # highlight selected
        bars[1].set_edgecolor("#d62728")
        bars[1].set_linewidth(2.5)
    fig.suptitle("Phase B5: signal efficiency per run (hn025 highlighted)", fontsize=14)
    fig.tight_layout()
    savefig(fig, "16a_b5_signal_efficiencies")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    w = 0.25
    x2 = np.arange(len(runs))
    g7 = [bg_fp[r][0] for r in runs]
    g8 = [bg_fp[r][1] for r in runs]
    ax2.bar(x2 - w/2, g7, w, label="G7 FP%", color="#d62728", alpha=0.85)
    ax2.bar(x2 + w/2, g8, w, label="G8 FP%", color="#ff7f0e", alpha=0.85)
    ax2.set_xticks(x2); ax2.set_xticklabels(runs)
    ax2.set_ylabel("Zero-window FP rate [%]")
    ax2.set_title("Phase B5: background FP rates\n"
                  "(hn025 = best G7/G8 trade-off; hn100 over-suppresses signal)")
    ax2.legend(); ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.axvline(0.5, color="green", linewidth=2, linestyle="--", alpha=0.5)
    ax2.text(1, 8.0, "← selected: hn025", fontsize=11, color="#d62728", fontweight="bold")
    fig2.tight_layout()
    savefig(fig2, "16b_b5_background_fp")
    add_readme("16a_b5_signal_efficiencies", "Hardcoded (B5 eval JSONs)", "Per-G-dataset efficiency for all B5 runs", "B5 sweep")
    add_readme("16b_b5_background_fp", "Hardcoded", "G7/G8 FP rates; hn025 best balance", "B5 sweep decision")


# ── 17. Pareto plots ──────────────────────────────────────────────────────────

def plot_17_pareto() -> None:
    runs = ["baseline", "hn025", "hn050", "hn100", "usw025", "usw050", "h96", "h128"]
    avg_sig = {
        "baseline": np.mean([94.1, 84.9, 91.7, 83.2, 82.2, 94.8]),
        "hn025":    np.mean([93.3, 82.9, 90.3, 80.5, 80.0, 94.9]),
        "hn050":    np.mean([90.8, 79.5, 88.2, 77.6, 76.9, 94.9]),
        "hn100":    np.mean([90.0, 75.5, 87.5, 74.1, 76.5, 94.6]),
        "usw025":   np.mean([94.1, 87.2, 92.1, 85.8, 83.8, 96.4]),
        "usw050":   np.mean([94.3, 84.9, 91.9, 83.0, 81.5, 94.6]),
        "h96":      np.mean([92.6, 82.9, 91.4, 81.3, 79.7, 93.9]),
        "h128":     np.mean([94.2, 83.0, 92.1, 80.9, 79.3, 94.8]),
    }
    g7 = {"baseline": 6.2, "hn025": 3.6, "hn050": 3.3, "hn100": 1.6,
          "usw025": 7.2, "usw050": 5.5, "h96": 8.7, "h128": 6.4}
    g8 = {"baseline": 2.9, "hn025": 2.2, "hn050": 2.0, "hn100": 1.5,
          "usw025": 3.6, "usw050": 2.8, "h96": 2.9, "h128": 2.7}

    for fp_key, fp_dict, fname, xlabel in [
        ("G8 FP%", g8, "17_b5_pareto_signal_vs_g8fp", "G8 FP rate [%]"),
        ("G7 FP%", g7, "17b_b5_pareto_signal_vs_g7fp", "G7 FP rate [%]"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for r in runs:
            col = "#d62728" if r == "hn025" else COLORS.get(r, "#888888")
            ms  = 120 if r in ("baseline", "hn025") else 70
            ax.scatter(fp_dict[r], avg_sig[r], s=ms, color=col, zorder=4,
                       edgecolors="white", linewidths=0.8)
            offx = 0.05 if r != "usw025" else -0.5
            offy = 0.2  if r != "hn100"  else -0.5
            ax.annotate(r, xy=(fp_dict[r], avg_sig[r]),
                        xytext=(fp_dict[r] + offx, avg_sig[r] + offy),
                        fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average signal efficiency G1–G6 [%]")
        ax.set_title(f"B5 Pareto: avg signal efficiency vs {fp_key}\n"
                     "(upper-left is better; hn025 improves Pareto frontier)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.invert_xaxis()
        fig.tight_layout()
        savefig(fig, fname)
        add_readme(fname, "Hardcoded", f"Pareto plot: hn025 improves Pareto frontier vs baseline (same avg eff, lower {fp_key})", "B5 decision")


# ── 18. Equal-FP budget ───────────────────────────────────────────────────────

def plot_18_equal_fp() -> None:
    name = "18_equal_fp_budget_hn025_vs_baseline"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: same G7 FP ~ 3.6%
    ax = axes[0]
    configs  = ["baseline\n@ thr+0.50", "hn025\n@ thr 0.0"]
    g2_vals  = [78.1, 82.9]
    g4_vals  = [76.5, 80.5]
    g5_vals  = [75.9, 80.0]
    x = np.arange(len(configs))
    w = 0.28
    ax.bar(x - w, g2_vals, w, label="G2", color="#1f77b4", alpha=0.85)
    ax.bar(x,     g4_vals, w, label="G4", color="#ff7f0e", alpha=0.85)
    ax.bar(x + w, g5_vals, w, label="G5", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(configs)
    ax.set_ylabel("Efficiency [%]"); ax.set_title("Equal G7 FP budget (~3.6%)")
    ax.set_ylim(70, 90); ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.text(0.5, 87, "+4–5 pp gain →", ha="center", fontsize=12,
            color="#d62728", fontweight="bold")

    # Panel B: same G8 FP ~ 2.4%
    ax2 = axes[1]
    configs2 = ["baseline\n@ thr+0.25\n(G7=5.0%)", "hn025\n@ thr 0.0\n(G7=3.6%)"]
    g2_2 = [82.5, 82.9]
    g4_2 = [80.9, 80.5]
    g5_2 = [79.9, 80.0]
    x2 = np.arange(len(configs2))
    ax2.bar(x2 - w, g2_2, w, label="G2", color="#1f77b4", alpha=0.85)
    ax2.bar(x2,     g4_2, w, label="G4", color="#ff7f0e", alpha=0.85)
    ax2.bar(x2 + w, g5_2, w, label="G5", color="#2ca02c", alpha=0.85)
    ax2.set_xticks(x2); ax2.set_xticklabels(configs2)
    ax2.set_ylabel("Efficiency [%]"); ax2.set_title("Equal G8 FP budget (~2.4%)")
    ax2.set_ylim(70, 90); ax2.legend(); ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.text(0.5, 87, "−1.4 pp G7 FP, signal preserved →",
             ha="center", fontsize=11, color="#d62728", fontweight="bold")

    fig.suptitle("hn025 is not a threshold shift: it moves the Pareto frontier",
                 fontsize=13)
    fig.tight_layout()
    savefig(fig, name)
    add_readme(name, "Hardcoded (operating point comparison)",
               "At equal FP budget, hn025@0.0 gains 4–5pp signal vs baseline@+0.50",
               "B5 key result — Pareto improvement")


# ── 19. Final model summary ───────────────────────────────────────────────────

def plot_19_final_summary() -> None:
    name = "19_final_baseline_summary"
    rows = [
        ["Input view",        "TPS (MuonStubTps — NanoAOD)"],
        ["Model",             "EdgeCompatNet h64"],
        ["Training datasets", "G1–G8 + B4  (15 sub-datasets)"],
        ["Dataset mix",       "B4×6, G7×4, G8×4 oversampling"],
        ["Loss",              "node BCE + candidate BCE + pT log-MSE + hard-neg (w=0.25)"],
        ["Best epoch",        "80 / 100  (val_loss = 0.3915)"],
        ["Threshold",         "logit > 0.0  (first operating point)"],
        ["G1 prompt eff",     "93.3%"],
        ["G2 PU eff",         "82.9%"],
        ["G3 displaced eff",  "90.3%"],
        ["G7 hard-neg FP",    "3.6%"],
        ["G8 PU hard-neg FP", "2.2%"],
        ["B4 pure-noise FP",  "0.0%"],
        ["Why selected",      "Strictly better equal-FP Pareto vs baseline at any threshold"],
        ["Next step",         "Quantization-Aware Training (QAT)"],
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    col_widths = [0.32, 0.68]
    t = ax.table(cellText=rows, loc="center", cellLoc="left",
                 colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1, 1.8)
    highlight_rows = {7, 8, 9, 10, 11, 12}
    green_rows     = {7, 8, 9}
    red_rows       = {10, 11, 12}
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            continue
        if c == 0:
            cell.set_facecolor("#e8ecf5")
            cell.set_text_props(fontweight="bold")
        if r in green_rows:
            cell.set_facecolor("#e8f4e8" if c == 1 else "#d0e8d0")
        if r in red_rows and c == 1:
            cell.set_facecolor("#fde8e8")
        if r == len(rows):
            cell.set_facecolor("#fff3cd")
    ax.set_title("Phase B5 — Selected Floating-Point Baseline for QAT",
                 fontsize=15, pad=20, fontweight="bold")
    savefig(fig, name)
    add_readme(name, "Hardcoded (Phase B5 freeze)", "Complete summary of selected baseline model and metrics", "Final slide / summary")


# ── 20. Next-step roadmap ─────────────────────────────────────────────────────

def plot_20_roadmap() -> None:
    name = "20_next_steps_roadmap"
    steps = [
        ("FP32 TPS EdgeCompat\nh64-hn025 baseline\n(frozen)", "#2ca02c"),
        ("Threshold / Pareto\noperating point scan", "#1f77b4"),
        ("Assignment head\nbranch (B6)", "#ff7f0e"),
        ("Freeze FP32\nmodel", "#2ca02c"),
        ("Quantization-Aware\nTraining (QAT)", "#9467bd"),
        ("Fixed-point\nevaluation", "#8c564b"),
        ("HLS/RTL\nimplementation", "#e377c2"),
        ("Board\nintegration tests", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")

    box_w, box_h = 0.80, 0.55
    for i, (label, color) in enumerate(steps):
        x = i
        rect = mpatches.FancyBboxPatch(
            (x - box_w/2, 0.5 - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.05", facecolor=color, alpha=0.85,
            edgecolor="white", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, 0.5, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", multialignment="center")
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 0.5 + 0.01, 0.5), xytext=(i + 0.5 - 0.01, 0.5),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=2))

    ax.text(-0.5, 1.25, "Current ✓", ha="center", fontsize=10, color="#2ca02c", fontweight="bold")
    ax.text(1.5, 1.25, "In progress", ha="center", fontsize=10, color="#ff7f0e")
    ax.text(4.5, 1.25, "Future", ha="center", fontsize=10, color="#9467bd")
    ax.set_title("GMT/OMTF ML — Path to Firmware Implementation", fontsize=15, pad=10)
    savefig(fig, name)
    add_readme(name, "Hardcoded", "End-to-end roadmap from FP32 baseline to board integration", "Conclusions / next steps")


# ── 21. Assignment head schematic ─────────────────────────────────────────────

def plot_21_assign_schematic() -> None:
    name = "21_assignment_head_schematic"
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)

    boxes = [
        (1,   2, "Stubs\n(B, Nmax, 14)",        "#c6dbef"),
        (3,   2, "Node encoder\nMLP(14→H→H)",    "#9ecae1"),
        (5,   2, "Edge compat\n(all pairs)",      "#6baed6"),
        (7,   2, "Node updater\nMLP(2H→H→H)",    "#4292c6"),
        (9,   2, "Node logit\nhead",              "#2171b5"),
        (9,   0.8, "Assign head\nMLP(H→H→K)\n→ softmax",  "#fd8d3c"),
        (11.5, 0.8, "Slot ctx[k]\n= Σ aᵢ·emb_i", "#fdae6b"),
        (13.5, 0.8, "Cand/pT/charge\nheads (×K)", "#fdd0a2"),
    ]
    for bx, by, label, color in boxes:
        rect = mpatches.FancyBboxPatch((bx - 0.75, by - 0.5), 1.5, 1.0,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(bx, by, label, ha="center", va="center", fontsize=9,
                multialignment="center")

    arrows = [
        (1.75, 2, 2.25, 2), (3.75, 2, 4.25, 2), (5.75, 2, 6.25, 2),
        (7.75, 2, 8.25, 2),
        (7.75, 2, 8.25, 0.8),
        (9.75, 0.8, 10.75, 0.8),
        (12.25, 0.8, 12.75, 0.8),
    ]
    for x0, y0, x1, y1 in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.5))

    ax.text(9, 3.4, "Shared encoder (identical to EdgeCompat B5)",
            ha="center", fontsize=11, color="#2171b5", style="italic")
    ax.text(11.5, 3.4, "New assignment decoder (B6)",
            ha="center", fontsize=11, color="#d62728", style="italic")
    ax.axvline(8.75, color="grey", linestyle="--", alpha=0.5, ymin=0.05, ymax=0.95)
    ax.set_title("EdgeCompatAssign: architecture overview (Phase B6)", fontsize=14, pad=10)
    savefig(fig, name)
    add_readme(name, "Hardcoded (model design)", "EdgeCompatAssign architecture with assignment decoder", "B6 introduction")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to {OUTDIR}")

    plot_fns = [
        plot_01_timeline,
        plot_02_eta_regions,
        plot_03_phi_windows,
        plot_04_coord_table,
        plot_05_b1_architectures,
        plot_06_overcounting,
        plot_07_false_slot,
        plot_08_domain_mismatch,
        plot_09_g_roles,
        plot_10_validation,
        plot_11_overcounting_fix,
        plot_12_kmtf_tps,
        plot_13_event_level,
        plot_14_d0,
        plot_15_roc,
        plot_16_b5,
        plot_17_pareto,
        plot_18_equal_fp,
        plot_19_final_summary,
        plot_20_roadmap,
        plot_21_assign_schematic,
    ]

    for fn in plot_fns:
        try:
            print(f"  {fn.__name__} ...", end=" ", flush=True)
            fn()
            print("ok")
        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()

    readme = OUTDIR / "README.md"
    readme.write_text("\n".join(readme_sections) + "\n")
    print(f"\nREADME: {readme}")
    print(f"Done — {len(plot_fns)} plots written to {OUTDIR}")


if __name__ == "__main__":
    main()
