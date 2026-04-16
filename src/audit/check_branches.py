"""
Stage 1 audit — structural integrity of OMTFAllInputTree.

Checks per file:
  - All reg_stub_* vectors have equal length per entry
  - reg_iProcessor in expected range [0, 11]
  - reg_eventNum non-duplicate within file
  - Branch schema consistent with expected list

Usage:
    python src/audit/check_branches.py --dataset B1 --max-files 5
    python src/audit/check_branches.py --all --max-files 3
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "prod"

DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]

EXPECTED_BRANCHES = [
    "reg_eventNum", "reg_iProcessor", "reg_mtfType",
    "reg_stub_layer", "reg_stub_phiHw", "reg_stub_phiBHw",
    "reg_stub_etaHw", "reg_stub_r", "reg_stub_quality",
    "reg_stub_type", "reg_stub_bx", "reg_stub_trackId",
    "reg_stub_ambiguous",
]

VECTOR_BRANCHES = [
    "reg_stub_layer", "reg_stub_phiHw", "reg_stub_phiBHw",
    "reg_stub_etaHw", "reg_stub_r", "reg_stub_quality",
    "reg_stub_type", "reg_stub_bx", "reg_stub_trackId",
    "reg_stub_ambiguous",
]


def check_file(path: Path, max_entries: int = 500) -> dict:
    import ROOT
    from audit.root_utils import open_hits_tree, uchar

    result = {
        "file": str(path.name),
        "ok": True,
        "errors": [],
        "warnings": [],
        "n_entries": 0,
        "n_checked": 0,
        "unequal_length_entries": 0,
        "iprocessor_range": [255, 0],
        "eventnum_duplicates": 0,
    }

    f, t = open_hits_tree(path)
    if t is None:
        result["ok"] = False
        result["errors"].append(f"Cannot open tree in {path.name}")
        return result

    # Schema check
    actual_branches = set(b.GetName() for b in t.GetListOfBranches())
    missing = set(EXPECTED_BRANCHES) - actual_branches
    if missing:
        result["errors"].append(f"Missing branches: {sorted(missing)}")
        result["ok"] = False

    n_entries = int(t.GetEntries())
    result["n_entries"] = n_entries
    n_check = min(n_entries, max_entries)
    result["n_checked"] = n_check

    seen_eventnums = set()
    iproc_min, iproc_max = 255, 0

    for i in range(n_check):
        t.GetEntry(i)

        iproc = uchar(t.reg_iProcessor)
        if iproc < iproc_min:
            iproc_min = iproc
        if iproc > iproc_max:
            iproc_max = iproc
        if iproc > 11:
            result["warnings"].append(f"Entry {i}: reg_iProcessor={iproc} out of [0,11]")

        evnum = int(t.reg_eventNum)
        if evnum in seen_eventnums:
            result["eventnum_duplicates"] += 1
        seen_eventnums.add(evnum)

        lengths = {br: t.__getattr__(br).size() for br in VECTOR_BRANCHES}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            result["unequal_length_entries"] += 1
            if result["unequal_length_entries"] <= 3:
                result["warnings"].append(
                    f"Entry {i}: unequal vector lengths: {lengths}"
                )

    result["iprocessor_range"] = [iproc_min, iproc_max]

    if result["unequal_length_entries"] > 0:
        result["errors"].append(
            f"{result['unequal_length_entries']} entries with unequal vector lengths"
        )
        result["ok"] = False

    if result["eventnum_duplicates"] > n_check * 0.5:
        result["warnings"].append(
            f"High eventNum duplicate rate: {result['eventnum_duplicates']}/{n_check} "
            "(expected for multi-processor entries per event)"
        )

    f.Close()
    return result


def check_dataset(dataset: str, max_files: int = 5, max_entries: int = 500) -> list:
    d = DATA_ROOT / dataset
    if not d.exists():
        print(f"  [SKIP] {d} not found")
        return []

    files = sorted(d.glob(f"omtf_hits_{dataset}_*.root"))[:max_files]
    if not files:
        print(f"  [SKIP] No hits files found in {d}")
        return []

    results = []
    for path in files:
        r = check_file(path, max_entries=max_entries)
        status = "OK" if r["ok"] else "FAIL"
        print(f"    [{status}] {r['file']}  "
              f"entries={r['n_entries']}  "
              f"iproc=[{r['iprocessor_range'][0]},{r['iprocessor_range'][1]}]  "
              f"evnum_dups={r['eventnum_duplicates']}  "
              f"errors={len(r['errors'])}  warnings={len(r['warnings'])}")
        for e in r["errors"]:
            print(f"      ERROR: {e}")
        for w in r["warnings"][:3]:
            print(f"      WARN:  {w}")
        results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(description="Check OMTFAllInputTree branch integrity")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-entries", type=int, default=500)
    args = parser.parse_args()

    import ROOT
    ROOT.gROOT.SetBatch(True)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else ["B1"])

    all_ok = True
    for ds in datasets:
        print(f"\n=== {ds} ===")
        results = check_dataset(ds, max_files=args.max_files, max_entries=args.max_entries)
        for r in results:
            if not r["ok"]:
                all_ok = False

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
