"""
Domain-mismatch audit for the GMT-visible-stub dataset.

Answers the question: are B2/S2 false second candidates real PU fakes,
or real-looking KMTF structures outside the OMTF-overlap task domain?

For each dataset the script computes:

  Stub-level:
    - fraction of valid stubs inside/outside overlap (0.83 ≤ |η| ≤ 1.24)
    - same breakdown for signal stubs (track_id > 0) vs noise stubs (track_id == 0)

  Window-level:
    - n_overlap_targets     : (gen_pt > 0).sum()  — official training target
    - n_signal_eta_clusters : distinct η bands with ≥2 signal stubs (proxy for
                              in-domain visible objects)
    - n_noise_eta_clusters  : distinct η bands with ≥2 noise stubs (proxy for
                              coherent noise / out-of-domain structures)
    - overlap_signal_frac   : fraction of signal stubs inside overlap
    - overlap_noise_frac    : fraction of noise stubs inside overlap

  Key table (S2/B2 only):
    For each combination of (n_overlap_targets, n_noise_eta_clusters):
      how many windows fall in this cell?
    This reveals whether "target=1, but 2 coherent structures visible" is common.

Usage
-----
  python scripts/omtf_gmt/audit_domain_mismatch.py \
      --cache-dir build/omtf_gmt/cache \
      --datasets  S1 S2 S3 S4 S5 B1 B2 B3 B4 \
      --output    build/omtf_gmt/eval/domain_mismatch_audit.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt
from omtf_gmt.features import F_ETA1

# OMTF overlap acceptance in |η|
ETA_OVERLAP_LO = 0.83
ETA_OVERLAP_HI = 1.24

# η band width for coherent-cluster detection
ETA_BAND_WIDTH = 0.15


def eta_band(eta_val: float) -> int:
    """Map eta to a discrete band index (width ETA_BAND_WIDTH)."""
    return int(eta_val / ETA_BAND_WIDTH)


def count_eta_clusters(eta_vals: torch.Tensor, min_stubs: int = 2) -> int:
    """Number of distinct η bands with ≥ min_stubs stubs."""
    bands: dict[int, int] = defaultdict(int)
    for e in eta_vals.tolist():
        bands[eta_band(abs(e))] += 1
    return sum(1 for c in bands.values() if c >= min_stubs)


def audit_dataset(ds_name: str, cache_dir: Path, batch_size: int = 512) -> dict:
    ds     = GMTCachedDataset(cache_dir, ds_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_gmt, num_workers=2)

    # stub-level accumulators
    n_total_stubs = 0
    n_signal_stubs = 0
    n_noise_stubs = 0
    n_signal_in_overlap = 0
    n_noise_in_overlap = 0
    n_signal_out_overlap = 0
    n_noise_out_overlap = 0

    # window-level accumulators
    n_windows = 0
    # (n_targets, n_noise_clusters) → count
    target_vs_noise_clusters: dict[tuple[int, int], int] = defaultdict(int)
    overlap_signal_fracs = []
    overlap_noise_fracs  = []

    for batch in loader:
        stubs = batch["stubs"]       # (N, Nmax, 11)
        vm    = batch["valid_mask"]  # (N, Nmax) bool
        tids  = batch["track_id"]    # (N, Nmax) int
        gpt   = batch["gen_pt"]      # (N, K)

        eta1 = stubs[:, :, F_ETA1]  # (N, Nmax)
        abs_eta = eta1.abs()
        in_overlap = (abs_eta >= ETA_OVERLAP_LO) & (abs_eta <= ETA_OVERLAP_HI)

        is_signal = tids > 0         # (N, Nmax)
        is_noise  = (tids == 0) & vm # (N, Nmax)
        is_signal_vm = is_signal & vm

        n_total_stubs     += int(vm.sum())
        n_signal_stubs    += int(is_signal_vm.sum())
        n_noise_stubs     += int(is_noise.sum())
        n_signal_in_overlap  += int((is_signal_vm & in_overlap).sum())
        n_noise_in_overlap   += int((is_noise & in_overlap).sum())
        n_signal_out_overlap += int((is_signal_vm & ~in_overlap).sum())
        n_noise_out_overlap  += int((is_noise & ~in_overlap).sum())

        n_targets = (gpt > 0).sum(dim=1)  # (N,)

        for b in range(stubs.shape[0]):
            vm_b  = vm[b]
            t_b   = tids[b]
            eta_b = eta1[b]

            sig_eta   = eta_b[vm_b & (t_b > 0)]
            noise_eta = eta_b[vm_b & (t_b == 0)]

            n_sig_clust   = count_eta_clusters(sig_eta)
            n_noise_clust = count_eta_clusters(noise_eta)
            nt            = int(n_targets[b].item())

            target_vs_noise_clusters[(nt, n_noise_clust)] += 1

            # per-window overlap fractions
            if len(sig_eta) > 0:
                overlap_signal_fracs.append(
                    float(((sig_eta.abs() >= ETA_OVERLAP_LO) &
                           (sig_eta.abs() <= ETA_OVERLAP_HI)).float().mean())
                )
            if len(noise_eta) > 0:
                overlap_noise_fracs.append(
                    float(((noise_eta.abs() >= ETA_OVERLAP_LO) &
                           (noise_eta.abs() <= ETA_OVERLAP_HI)).float().mean())
                )

        n_windows += stubs.shape[0]

    return {
        "ds_name":           ds_name,
        "n_windows":         n_windows,
        "n_total_stubs":     n_total_stubs,
        "n_signal_stubs":    n_signal_stubs,
        "n_noise_stubs":     n_noise_stubs,
        "n_signal_in_overlap":   n_signal_in_overlap,
        "n_signal_out_overlap":  n_signal_out_overlap,
        "n_noise_in_overlap":    n_noise_in_overlap,
        "n_noise_out_overlap":   n_noise_out_overlap,
        "target_vs_noise_clusters": dict(target_vs_noise_clusters),
        "mean_overlap_signal_frac": sum(overlap_signal_fracs) / max(1, len(overlap_signal_fracs)),
        "mean_overlap_noise_frac":  sum(overlap_noise_fracs)  / max(1, len(overlap_noise_fracs)),
    }


def format_report(results: list[dict]) -> str:
    lines = ["# GMT Domain Mismatch Audit\n"]
    lines.append(f"Overlap acceptance: {ETA_OVERLAP_LO} ≤ |η| ≤ {ETA_OVERLAP_HI}")
    lines.append(f"Noise cluster threshold: ≥2 stubs within |Δη| < {ETA_BAND_WIDTH}\n")

    # ---- stub-level table ----
    lines.append("## Stub-level overlap fractions\n")
    lines.append("| Dataset | total stubs | signal stubs | sig in overlap | noise stubs | noise in overlap | noise out overlap |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        s_tot  = max(1, r["n_signal_stubs"])
        n_tot  = max(1, r["n_noise_stubs"])
        lines.append(
            f"| {r['ds_name']} "
            f"| {r['n_total_stubs']:,} "
            f"| {r['n_signal_stubs']:,} "
            f"| {100*r['n_signal_in_overlap']/s_tot:.1f}% "
            f"| {r['n_noise_stubs']:,} "
            f"| {100*r['n_noise_in_overlap']/n_tot:.1f}% "
            f"| {100*r['n_noise_out_overlap']/n_tot:.1f}% |"
        )

    # ---- window-level table ----
    lines.append("\n## Mean per-window overlap fraction\n")
    lines.append("| Dataset | mean signal-stub overlap frac | mean noise-stub overlap frac |")
    lines.append("| --- | --- | --- |")
    for r in results:
        lines.append(
            f"| {r['ds_name']} "
            f"| {r['mean_overlap_signal_frac']:.3f} "
            f"| {r['mean_overlap_noise_frac']:.3f} |"
        )

    # ---- key table: target count vs noise clusters ----
    lines.append("\n## Key table: n_overlap_targets vs n_coherent_noise_clusters\n")
    lines.append("(Each cell = fraction of windows in that dataset)\n")

    for r in results:
        if r["ds_name"] not in ("S2", "B2", "S1", "S4", "B1", "B4"):
            continue
        tvc   = r["target_vs_noise_clusters"]
        total = r["n_windows"]
        max_nc = max((nc for _, nc in tvc.keys()), default=0)
        max_nc = min(max_nc, 4)

        lines.append(f"### {r['ds_name']}  ({total:,} windows)\n")
        header = "| true targets \\ noise clusters | " + " | ".join(str(i) for i in range(max_nc + 1)) + " |"
        sep    = "| --- " * (max_nc + 2) + "|"
        lines.append(header)
        lines.append(sep)

        for nt in range(4):
            row_total = sum(v for (t, _), v in tvc.items() if t == nt)
            if row_total == 0:
                continue
            cells = []
            for nc in range(max_nc + 1):
                c = tvc.get((nt, nc), 0)
                cells.append(f"{100*c/total:.1f}%")
            lines.append(f"| targets={nt} | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- interpretation note ----
    lines.append("## Interpretation\n")
    lines.append("- **sig in overlap < 100%**: signal muon stubs outside OMTF η acceptance — model sees them but target labels don't reflect them.")
    lines.append("- **noise out overlap > 0%**: noise stubs with η outside overlap — may be real KMTF barrel muons labelled noise.")
    lines.append("- **targets=1, noise_clusters≥1**: window has 1 official target but ≥1 coherent noise stub cluster — model is asked to suppress a real-looking second object.")

    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--datasets",  nargs="+",
                   default=["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"])
    p.add_argument("--output",    type=Path,
                   default=Path("build/omtf_gmt/eval/domain_mismatch_audit.md"))
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    results = []
    for ds_name in args.datasets:
        print(f"Auditing {ds_name}...", flush=True)
        results.append(audit_dataset(ds_name, args.cache_dir, args.batch_size))

    report = format_report(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
