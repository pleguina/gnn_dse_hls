"""
Master dataset audit runner — Stage 1.

Runs all checks in order and produces:
  - build/audit/audit_report.md
  - build/audit/occupancy_summary.csv
  - build/audit/feature_ranges.json
  - build/audit/plots/ (if --plot)

Usage:
    python scripts/audit/run_dataset_audit.py
    python scripts/audit/run_dataset_audit.py --max-files 5 --plot
    python scripts/audit/run_dataset_audit.py --dataset S1 --max-files 3
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "build" / "audit"
SIGNOFF_PATH = PROJECT_ROOT / "docs" / "omtf" / "DATASET_SIGNOFF.md"

DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]


def run_step(label: str, cmd: list) -> tuple:
    """Run a subprocess step, return (ok, stdout+stderr)."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,  # let output stream to terminal
        text=True,
    )
    ok = result.returncode == 0
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run full OMTF dataset audit")
    parser.add_argument("--dataset", type=str, help="Single dataset (default: all)")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-entries", type=int, default=500)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip-joins", action="store_true",
                        help="Skip join check (faster, nano files not required)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds_flag = ["--dataset", args.dataset] if args.dataset else ["--all"]
    common = ds_flag + ["--max-files", str(args.max_files),
                        "--max-entries", str(args.max_entries)]
    python = sys.executable

    steps = [
        ("Branch integrity check",
         [python, "src/audit/check_branches.py"] + common),
        ("Label integrity check",
         [python, "src/audit/check_labels.py"] + common),
        ("Occupancy analysis",
         [python, "src/audit/occupancy.py"] + common + (["--plot"] if args.plot else [])),
        ("Feature range analysis",
         [python, "src/audit/feature_ranges.py"] + common + (["--plot"] if args.plot else [])),
    ]

    if not args.skip_joins:
        steps.insert(2, (
            "NanoAOD join check",
            [python, "src/audit/check_joins.py"] + ds_flag +
            ["--max-files", str(min(args.max_files, 3)),
             "--max-entries", str(min(args.max_entries, 200))]
        ))

    results = {}
    for label, cmd in steps:
        ok = run_step(label, cmd)
        results[label] = ok

    # Write audit report
    report_path = OUTPUT_DIR / "audit_report.md"
    with open(report_path, "w") as f:
        f.write(f"# OMTF Dataset Audit Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Datasets**: {args.dataset or 'all'}\n")
        f.write(f"**Max files per dataset**: {args.max_files}\n\n")
        f.write(f"## Step Results\n\n")
        f.write(f"| Step | Status |\n|---|---|\n")
        for label, ok in results.items():
            status = "PASS" if ok else "FAIL"
            f.write(f"| {label} | {status} |\n")
        f.write(f"\n## Artifacts\n\n")
        f.write(f"- `build/audit/occupancy_summary.csv` — stubs/window statistics\n")
        f.write(f"- `build/audit/feature_ranges.json` — feature min/max/mean/std\n")
        if args.plot:
            f.write(f"- `build/audit/plots/` — occupancy and feature distribution plots\n")
        f.write(f"\n## Next Steps\n\n")
        f.write(f"Fill in `docs/omtf/DATASET_SIGNOFF.md` using the artifacts above.\n")
        f.write(f"Update `configs/model_config.yaml` `omtf.Nmax` from occupancy_summary.csv.\n")

    print(f"\n{'='*60}")
    print(f"  Audit complete")
    print(f"{'='*60}")
    all_ok = all(results.values())
    for label, ok in results.items():
        icon = "OK" if ok else "FAIL"
        print(f"  [{icon}] {label}")
    print(f"\nReport: {report_path}")
    print(f"DATASET_SIGNOFF: {SIGNOFF_PATH}")

    if not all_ok:
        print("\nSome checks failed. Review errors above before proceeding to Stage 2.")
        return 1

    print("\nAll checks passed. Fill DATASET_SIGNOFF.md before starting Stage 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
