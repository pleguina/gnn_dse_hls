#!/usr/bin/env python
"""
OMTF Dataset Decision Audit — CLI entry point.

Runs the analysis phases defined in docs/omtf/DATASET_DECISION_PLAN.md and
writes a filled DATASET_DECISION_REPORT.md plus audit_results.json.

Usage
-----
  # Full audit (all phases, all datasets, graph cache)
  python scripts/omtf/dataset_audit.py \\
      --cache-dir build/omtf/cache/schema_v1_graph \\
      --output-dir build/omtf/audit

  # Skip slow baseline training; audit specific datasets only
  python scripts/omtf/dataset_audit.py \\
      --cache-dir build/omtf/cache/schema_v1_graph \\
      --datasets S1 S2 B1 B4 \\
      --phases 0 1 2 3d 3c 6 7 \\
      --output-dir build/omtf/audit

  # Include B4 false-positive audit (requires checkpoint)
  python scripts/omtf/dataset_audit.py \\
      --cache-dir build/omtf/cache/schema_v1_graph \\
      --checkpoint build/omtf/checkpoints/edge_compat_best_trig.pt \\
      --output-dir build/omtf/audit

Available phase codes
---------------------
  0   branch integrity + NanoAOD join validation
  1   stub occupancy + track multiplicity
  2   legal edge stats + feature separability  (requires graph cache)
  3d  pT distribution + threshold bins
  3c  pT-curvature correlations               (faster with graph cache)
  3b  tiny pT baseline (sklearn)
  5   B4 false-positive audit                 (requires --checkpoint)
  6   trackId ordering + multi-track confusion
  7   dXY visibility                          (displaced datasets only)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── repo-root / src on path ──────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
# ─────────────────────────────────────────────────────────────────────────────

from omtf.analysis import (
    run_phase0, run_phase1, run_phase2,
    run_phase3_distribution, run_phase3_correlations, run_phase3_baseline,
    run_phase5, run_phase6, run_phase7,
    render_report, save_json,
    ALL_DATASETS, MULTI_TRACK_DATASETS, DISPLACED_DATASETS,
    available_datasets, has_graph, validate_schema,
)

ALL_PHASE_CODES = ["0", "1", "2", "3d", "3c", "3b", "5", "6", "7"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OMTF dataset decision audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--cache-dir", required=True, type=Path,
        help="Path to pre-built .pt shard cache (manifest.json must exist)",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to audit (default: all available in cache)",
    )
    p.add_argument(
        "--phases", nargs="+", default=ALL_PHASE_CODES,
        choices=ALL_PHASE_CODES + ["all"],
        help="Phase codes to run (default: all)",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Model checkpoint for phase 5 (B4 audit)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("build/omtf/audit"),
        help="Directory for report output (default: build/omtf/audit)",
    )
    p.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Device for phase 5 inference (default: cuda)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-shard progress lines",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir = args.cache_dir.resolve()
    if not cache_dir.exists():
        print(f"ERROR: cache directory not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    validate_schema(cache_dir)

    avail = available_datasets(cache_dir)
    if args.datasets:
        datasets = [d for d in args.datasets if d in avail]
        missing = set(args.datasets) - set(avail)
        if missing:
            print(f"WARN: datasets not in cache, skipping: {sorted(missing)}")
    else:
        datasets = avail

    phases = set(args.phases)
    if "all" in phases:
        phases = set(ALL_PHASE_CODES)

    verbose = not args.quiet
    gp = has_graph(cache_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cache:    {cache_dir}")
    print(f"Graph:    {gp}")
    print(f"Datasets: {datasets}")
    print(f"Phases:   {sorted(phases)}")
    print()

    results: dict = {}
    t0 = time.time()

    if "0" in phases:
        print("=== Phase 0: Integrity ===")
        results["phase0"] = run_phase0(cache_dir, datasets, verbose=verbose)
        print()

    if "1" in phases:
        print("=== Phase 1: Occupancy ===")
        results["phase1"] = run_phase1(cache_dir, datasets, verbose=verbose)
        print()

    if "2" in phases:
        print("=== Phase 2: Edge stats ===")
        if not gp:
            print("  SKIP — cache has no graph data; rebuild with --include-graph")
        else:
            results["phase2"] = run_phase2(cache_dir, datasets, verbose=verbose)
        print()

    if "3d" in phases:
        print("=== Phase 3d: pT distribution ===")
        results["phase3_dist"] = run_phase3_distribution(cache_dir, datasets, verbose=verbose)
        print()

    if "3c" in phases:
        print("=== Phase 3c: pT correlations ===")
        results["phase3_corr"] = run_phase3_correlations(cache_dir, datasets, verbose=verbose)
        print()

    if "3b" in phases:
        print("=== Phase 3b: Tiny pT baseline ===")
        # only run on single-track-relevant datasets (skip B4 which has no gen muons)
        baseline_ds = [d for d in datasets if d != "B4"]
        results["phase3_baseline"] = run_phase3_baseline(cache_dir, baseline_ds, verbose=verbose)
        print()

    if "5" in phases:
        print("=== Phase 5: B4 false-positive audit ===")
        if args.checkpoint is None:
            print("  SKIP — no --checkpoint provided")
        elif "B4" not in datasets:
            print("  SKIP — B4 not in selected datasets")
        else:
            import torch
            device_str = args.device if torch.cuda.is_available() else "cpu"
            results["phase5"] = run_phase5(
                cache_dir, args.checkpoint,
                device_str=device_str, verbose=verbose,
            )
        print()

    if "6" in phases:
        print("=== Phase 6: SlotModel fairness ===")
        mt_ds = [d for d in datasets if d in MULTI_TRACK_DATASETS]
        if not mt_ds:
            print("  SKIP — no multi-track datasets selected")
        else:
            results["phase6"] = run_phase6(cache_dir, mt_ds, verbose=verbose)
        print()

    if "7" in phases:
        print("=== Phase 7: dXY visibility ===")
        dxy_ds = [d for d in datasets if d in DISPLACED_DATASETS]
        if not dxy_ds:
            print("  SKIP — no displaced datasets selected")
        else:
            results["phase7"] = run_phase7(cache_dir, dxy_ds, verbose=verbose)
        print()

    elapsed = time.time() - t0
    print(f"All phases done in {elapsed:.0f}s")
    print()

    report_path = args.output_dir / "DATASET_DECISION_REPORT.md"
    json_path   = args.output_dir / "audit_results.json"
    render_report(results, report_path)
    save_json(results, json_path)


if __name__ == "__main__":
    main()
