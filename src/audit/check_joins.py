"""
Stage 1 audit — validate join between OMTFAllInputTree and NanoAOD Events tree.

Join key: reg_eventNum (UInt_t) <-> event (ULong64_t) via uint32 cast.
Index: reg_stub_trackId value k > 0 should index GenMuon_pt[k-1].

Usage:
    python src/audit/check_joins.py --dataset S1 --max-files 3
    python src/audit/check_joins.py --all
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "prod"

DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]


def check_file_pair(hits_path: Path, nano_path: Path, max_entries: int = 200) -> dict:
    import ROOT
    from audit.root_utils import open_hits_tree, open_nano_tree, read_entry

    result = {
        "hits_file": hits_path.name,
        "ok": True,
        "errors": [],
        "warnings": [],
        "hits_entries": 0,
        "nano_entries": 0,
        "join_attempted": 0,
        "join_success": 0,
        "join_failed": 0,
        "trackid_index_ok": 0,
        "trackid_index_oob": 0,
    }

    fh, th = open_hits_tree(hits_path)
    if th is None:
        result["ok"] = False
        result["errors"].append(f"Cannot open hits: {hits_path.name}")
        return result

    fn, tn = open_nano_tree(nano_path)
    if tn is None:
        result["ok"] = False
        result["errors"].append(f"Cannot open nano: {nano_path.name}")
        fh.Close()
        return result

    result["hits_entries"] = int(th.GetEntries())
    result["nano_entries"] = int(tn.GetEntries())

    # Build event key -> nano entry index from NanoAOD
    nano_event_map = {}
    for i in range(tn.GetEntries()):
        tn.GetEntry(i)
        key = int(tn.event) & 0xFFFFFFFF  # uint32 cast
        if key in nano_event_map:
            result["warnings"].append(f"Duplicate NanoAOD key {key} at entry {i}")
        nano_event_map[key] = i

    n_check = min(int(th.GetEntries()), max_entries)
    for i in range(n_check):
        e = read_entry(th, i)
        hits_key = e["event_num"]  # already uint32 from UInt_t
        result["join_attempted"] += 1

        if hits_key not in nano_event_map:
            result["join_failed"] += 1
            if result["join_failed"] <= 3:
                result["warnings"].append(
                    f"Hits entry {i}: eventNum={hits_key} not in NanoAOD"
                )
            continue

        result["join_success"] += 1
        nano_idx = nano_event_map[hits_key]
        tn.GetEntry(nano_idx)
        n_gen = int(tn.nGenMuon)

        for tid in e["track_id"]:
            if tid == 0:
                continue
            gen_idx = tid - 1
            if gen_idx < 0 or gen_idx >= n_gen:
                result["trackid_index_oob"] += 1
                if result["trackid_index_oob"] <= 3:
                    result["warnings"].append(
                        f"Entry {i}: trackId={tid} -> gen_idx={gen_idx} "
                        f"out of range [0, {n_gen})"
                    )
            else:
                result["trackid_index_ok"] += 1

    if result["join_attempted"] > 0:
        join_rate = result["join_success"] / result["join_attempted"]
        if join_rate < 0.95:
            result["errors"].append(
                f"Low join rate: {join_rate:.2%} "
                f"({result['join_success']}/{result['join_attempted']})"
            )
            result["ok"] = False

    total_tid = result["trackid_index_ok"] + result["trackid_index_oob"]
    if total_tid > 0 and result["trackid_index_oob"] / total_tid > 0.01:
        result["errors"].append(
            f"High trackId OOB rate: {result['trackid_index_oob']}/{total_tid}"
        )
        result["ok"] = False

    fh.Close()
    fn.Close()
    return result


def check_dataset(dataset: str, max_files: int = 3, max_entries: int = 200) -> list:
    d = DATA_ROOT / dataset
    if not d.exists():
        print(f"  [SKIP] {d} not found")
        return []

    hits_files = sorted(d.glob(f"omtf_hits_{dataset}_*.root"))[:max_files]
    results = []

    for hf in hits_files:
        idx = hf.stem.split("_")[-1]
        nf = hf.parent / f"omtf_nano_{dataset}_{idx}.root"
        if not nf.exists():
            print(f"    [SKIP] No nano file for {hf.name}")
            continue

        r = check_file_pair(hf, nf, max_entries=max_entries)
        status = "OK" if r["ok"] else "FAIL"
        join_rate = (r["join_success"] / max(r["join_attempted"], 1)) * 100
        print(f"    [{status}] {r['hits_file']}  "
              f"join={join_rate:.1f}%  "
              f"tid_ok={r['trackid_index_ok']}  "
              f"tid_oob={r['trackid_index_oob']}")
        for e in r["errors"]:
            print(f"      ERROR: {e}")
        for w in r["warnings"][:3]:
            print(f"      WARN:  {w}")
        results.append(r)

    if results:
        total_joined = sum(r["join_success"] for r in results)
        total_attempted = sum(r["join_attempted"] for r in results)
        total_oob = sum(r["trackid_index_oob"] for r in results)
        print(f"\n  {dataset} summary: join_rate="
              f"{100*total_joined/max(total_attempted,1):.1f}%  "
              f"trackId_oob={total_oob}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate NanoAOD join integrity")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--max-entries", type=int, default=200)
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
