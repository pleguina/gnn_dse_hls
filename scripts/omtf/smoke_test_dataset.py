"""
Vertical slice smoke test — Stage 2.

Checks all 8 criteria for the V1 milestone:
  1. OMTFDataset loads without errors
  2. stubs shape == (Nmax, 8)
  3. valid_mask shape == (Nmax,); sum matches n_stubs
  4. No real stubs in padding rows (stubs[~valid_mask] all zero)
  5. node_label consistent with track_id != 0
  6. Graph output present when include_graph=True
  7. edge_index endpoints are within [0, Nmax)
  8. edge_label consistent with track_id matching

Usage:
    python scripts/omtf/smoke_test_dataset.py --dataset S1 --max-entries 50
    python scripts/omtf/smoke_test_dataset.py --dataset S1 --max-entries 50 --graph
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_ROOT = PROJECT_ROOT / "data" / "prod"
NMAX = 24


def run_smoke_test(dataset: str, max_entries: int, include_graph: bool) -> bool:
    import ROOT
    ROOT.gROOT.SetBatch(True)

    from omtf.dataset import OMTFDataset

    files = sorted((DATA_ROOT / dataset).glob(f"omtf_hits_{dataset}_*.root"))[:1]
    if not files:
        print(f"[SKIP] No files found for {dataset} under {DATA_ROOT}")
        return True

    print(f"Testing {files[0].name}  Nmax={NMAX}  include_graph={include_graph}")

    ds = OMTFDataset(
        files=files,
        Nmax=NMAX,
        include_graph=include_graph,
        max_entries=max_entries,
    )

    if len(ds) == 0:
        print("  [FAIL] Dataset is empty")
        return False

    print(f"  Loaded {len(ds)} samples")

    passes = 0
    fails = 0

    for idx in range(min(len(ds), max_entries)):
        s = ds[idx]

        # 1. Keys present
        for key in ("stubs", "valid_mask", "track_id", "ambiguous", "node_label", "meta"):
            if key not in s:
                print(f"  [FAIL] sample {idx}: missing key '{key}'")
                fails += 1
                continue

        # 2. stubs shape
        if s["stubs"].shape != (NMAX, 8):
            print(f"  [FAIL] sample {idx}: stubs shape {s['stubs'].shape} != ({NMAX}, 8)")
            fails += 1
            continue

        # 3. valid_mask shape and sum
        vm = s["valid_mask"]
        n_real = int(vm.sum().item())
        if vm.shape != (NMAX,):
            print(f"  [FAIL] sample {idx}: valid_mask shape {vm.shape}")
            fails += 1
            continue
        if n_real != s["meta"]["n_stubs"]:
            print(f"  [FAIL] sample {idx}: valid_mask sum {n_real} != n_stubs {s['meta']['n_stubs']}")
            fails += 1
            continue

        # 4. Padding rows are zero
        if n_real < NMAX:
            pad_stubs = s["stubs"][~vm]
            if not torch.all(pad_stubs == 0):
                print(f"  [FAIL] sample {idx}: non-zero values in padding rows")
                fails += 1
                continue

        # 5. node_label consistent with track_id
        expected_label = (s["track_id"] != 0).float()
        if not torch.all(s["node_label"] == expected_label):
            print(f"  [FAIL] sample {idx}: node_label/track_id mismatch")
            fails += 1
            continue

        # 6 & 7 & 8: graph checks
        if include_graph:
            if "edge_index" not in s:
                print(f"  [FAIL] sample {idx}: edge_index missing with include_graph=True")
                fails += 1
                continue

            ei = s["edge_index"]
            ea = s["edge_attr"]
            el = s["edge_label"]

            # 6. edge_attr has 7 features
            if ea.ndim == 2 and ea.shape[0] > 0 and ea.shape[1] != 7:
                print(f"  [FAIL] sample {idx}: edge_attr width {ea.shape[1]} != 7")
                fails += 1
                continue

            # 7. endpoints in range
            if ei.shape[1] > 0:
                if ei.max().item() >= NMAX or ei.min().item() < 0:
                    print(f"  [FAIL] sample {idx}: edge_index out of range [0, {NMAX})")
                    fails += 1
                    continue

            # 8. edge_label matches track_id
            if ei.shape[1] > 0:
                src, dst = ei[0], ei[1]
                tid = s["track_id"]
                expected_el = ((tid[src] == tid[dst]) & (tid[src] != 0)).float()
                if not torch.all(el == expected_el):
                    print(f"  [FAIL] sample {idx}: edge_label/track_id mismatch")
                    fails += 1
                    continue

        passes += 1

    print(f"\n  Results: {passes} PASS, {fails} FAIL out of {passes + fails} samples")
    return fails == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="S1")
    parser.add_argument("--max-entries", type=int, default=50)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()

    ok = run_smoke_test(args.dataset, args.max_entries, args.graph)
    print("\nSMOKE TEST: " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
