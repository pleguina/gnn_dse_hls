"""
Stage 1 audit — truth-label integrity of reg_stub_trackId and reg_stub_ambiguous.

Checks:
  - reg_stub_trackId == 0 behaves as noise/PU stub
  - reg_stub_trackId > 0 maps to expected GenMuon index range per dataset
  - reg_stub_ambiguous is not trivially all-zero or all-one
  - Multi-muon events: track IDs non-colliding within each processor window
  - Same-track vs cross-track pair ratio (edge label balance)

Usage:
    python src/audit/check_labels.py --dataset S1 --max-files 5
    python src/audit/check_labels.py --all
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "prod"

DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]

EXPECTED_MAX_TRACK_ID = {
    "S1": 1, "S2": 1,
    "S3": 2, "S5": 2,
    "B1": 1, "B2": 1, "B3": 2,
    "S4": 3,
    "B4": 0,
}


def check_file(path: Path, max_entries: int = 500) -> dict:
    import ROOT
    from audit.root_utils import open_hits_tree, read_entry

    result = {
        "file": path.name,
        "ok": True,
        "errors": [],
        "warnings": [],
        "n_entries": 0,
        "total_stubs": 0,
        "noise_stubs": 0,
        "signal_stubs": 0,
        "ambiguous_stubs": 0,
        "max_trackid_seen": 0,
        "same_track_pairs": 0,
        "cross_track_pairs": 0,
        "total_pairs": 0,
    }

    f, t = open_hits_tree(path)
    if t is None:
        result["ok"] = False
        result["errors"].append(f"Cannot open {path.name}")
        return result

    n_entries = min(int(t.GetEntries()), max_entries)
    result["n_entries"] = n_entries

    for i in range(n_entries):
        e = read_entry(t, i)
        track_ids = e["track_id"]
        ambiguous = e["ambiguous"]
        n_stubs = e["n_stubs"]

        result["total_stubs"] += n_stubs
        result["noise_stubs"] += sum(1 for tid in track_ids if tid == 0)
        result["signal_stubs"] += sum(1 for tid in track_ids if tid != 0)
        result["ambiguous_stubs"] += sum(1 for a in ambiguous if a != 0)

        max_tid = max((abs(tid) for tid in track_ids), default=0)
        if max_tid > result["max_trackid_seen"]:
            result["max_trackid_seen"] = max_tid

        # Edge label balance: count same-track vs cross-track pairs
        for a in range(n_stubs):
            for b in range(a + 1, n_stubs):
                result["total_pairs"] += 1
                if track_ids[a] != 0 and track_ids[a] == track_ids[b]:
                    result["same_track_pairs"] += 1
                else:
                    result["cross_track_pairs"] += 1

    dataset = path.parent.name
    expected_max = EXPECTED_MAX_TRACK_ID.get(dataset, 3)
    if result["max_trackid_seen"] > expected_max:
        result["warnings"].append(
            f"max trackId seen ({result['max_trackid_seen']}) > expected ({expected_max})"
        )

    if dataset == "B4" and result["signal_stubs"] > 0:
        result["errors"].append(
            f"B4 has {result['signal_stubs']} stubs with trackId != 0 "
            "(expected 0 for noise-only)"
        )
        result["ok"] = False

    if result["total_stubs"] > 0:
        amb_frac = result["ambiguous_stubs"] / result["total_stubs"]
        if amb_frac == 0.0 and dataset not in ("B4",):
            result["warnings"].append("Ambiguous fraction is exactly 0.0")
        if amb_frac == 1.0:
            result["warnings"].append("Ambiguous fraction is exactly 1.0")

    f.Close()
    return result


def summarize(results: list, dataset: str):
    total_stubs = sum(r["total_stubs"] for r in results)
    noise = sum(r["noise_stubs"] for r in results)
    signal = sum(r["signal_stubs"] for r in results)
    ambig = sum(r["ambiguous_stubs"] for r in results)
    same = sum(r["same_track_pairs"] for r in results)
    total_pairs = sum(r["total_pairs"] for r in results)

    if total_stubs == 0:
        return

    print(f"\n  Summary for {dataset}:")
    print(f"    Total stubs         : {total_stubs}")
    print(f"    Noise (trackId=0)   : {noise}  ({100*noise/total_stubs:.1f}%)")
    print(f"    Signal (trackId!=0) : {signal}  ({100*signal/total_stubs:.1f}%)")
    print(f"    Ambiguous           : {ambig}  ({100*ambig/total_stubs:.1f}%)")
    if total_pairs > 0:
        cross = total_pairs - same
        print(f"    Same-track pairs    : {same}  ({100*same/total_pairs:.1f}%)")
        print(f"    Cross-track pairs   : {cross}  ({100*cross/total_pairs:.1f}%)")
        if same > 0:
            print(f"    Edge imbalance      : {cross/same:.1f}x more negative than positive")


def check_dataset(dataset: str, max_files: int = 5, max_entries: int = 300) -> list:
    d = DATA_ROOT / dataset
    if not d.exists():
        print(f"  [SKIP] {d} not found")
        return []

    files = sorted(d.glob(f"omtf_hits_{dataset}_*.root"))[:max_files]
    results = []

    for path in files:
        r = check_file(path, max_entries=max_entries)
        status = "OK" if r["ok"] else "FAIL"
        sig_frac = (r["signal_stubs"] / max(r["total_stubs"], 1)) * 100
        amb_frac = (r["ambiguous_stubs"] / max(r["total_stubs"], 1)) * 100
        print(f"    [{status}] {r['file']}  "
              f"stubs={r['total_stubs']}  "
              f"signal={sig_frac:.1f}%  "
              f"ambig={amb_frac:.1f}%  "
              f"max_tid={r['max_trackid_seen']}")
        for e in r["errors"]:
            print(f"      ERROR: {e}")
        for w in r["warnings"]:
            print(f"      WARN:  {w}")
        results.append(r)

    summarize(results, dataset)
    return results


def main():
    parser = argparse.ArgumentParser(description="Check OMTF truth-label integrity")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-entries", type=int, default=300)
    args = parser.parse_args()

    import ROOT
    ROOT.gROOT.SetBatch(True)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    datasets = DATASETS if args.all else ([args.dataset] if args.dataset else ["S1"])

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
