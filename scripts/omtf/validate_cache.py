#!/usr/bin/env python
"""
Validate OMTF cache against the original ROOT dataset.

Loads N samples from both OMTFDataset (ROOT path) and CachedOMTFDataset,
checks exact tensor equality field by field, and compares aggregate statistics.

Usage
-----
  python scripts/omtf/validate_cache.py \
    --dataset S1 \
    --cache-dir build/omtf/cache/schema_v1

  # Include graph fields
  python scripts/omtf/validate_cache.py \
    --dataset S1 \
    --cache-dir build/omtf/cache/schema_v1 \
    --include-graph \
    --n-samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audit.root_utils import set_root_batch_mode
from omtf.dataset import OMTFDataset
from omtf.cache_dataset import CachedOMTFDataset

DATA_ROOT = PROJECT_ROOT / "data" / "prod"
NMAX = 24

# Fields that must match exactly (tensor equality)
EXACT_FIELDS = [
    "stubs", "valid_mask", "track_id", "ambiguous", "node_label",
    "gen_pt", "gen_charge", "gen_dxy", "gen_phi",
]
GRAPH_FIELDS = ["edge_index", "edge_attr", "edge_label", "edge_ambig"]
META_FIELDS  = ["event_num", "i_processor", "n_stubs", "n_gen_tracks"]


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        # Allow int8/uint8 comparison via int32 upcast
        return (a.to(torch.int32) == b.to(torch.int32)).all().item()
    return torch.equal(a, b)


def validate(
    dataset: str,
    cache_dir: Path,
    n_samples: int,
    include_graph: bool,
    max_files: int | None,
) -> bool:
    set_root_batch_mode()

    # Load ROOT dataset
    ds_dir = DATA_ROOT / dataset
    files = sorted(ds_dir.glob(f"omtf_hits_{dataset}_*.root"))
    if not files:
        print(f"[ERROR] No ROOT files for {dataset}")
        return False
    if max_files:
        files = files[:max_files]

    print(f"Loading ROOT dataset ({dataset}, {len(files)} files)...")
    root_ds = OMTFDataset(
        files, Nmax=NMAX, include_graph=include_graph, include_nano=True,
    )
    print(f"  {len(root_ds):,} samples")

    print(f"Loading cache dataset ({cache_dir / dataset})...")
    cache_ds = CachedOMTFDataset(cache_dir, dataset)
    print(f"  {len(cache_ds):,} samples")

    if len(root_ds) != len(cache_ds):
        print(f"[WARN] Sample count mismatch: ROOT={len(root_ds):,}  cache={len(cache_ds):,}")
        print("       This is expected if cache was built with --max-files or --max-entries.")

    n = min(n_samples, len(root_ds), len(cache_ds))
    print(f"\nComparing first {n} samples field by field...")

    all_pass = True
    mismatches: list[str] = []

    for i in range(n):
        rs = root_ds[i]
        cs = cache_ds[i]

        for field in EXACT_FIELDS:
            if not _tensor_equal(rs[field], cs[field]):
                mismatches.append(f"sample {i}: {field} mismatch  "
                                  f"root={rs[field].shape}/{rs[field].dtype}  "
                                  f"cache={cs[field].shape}/{cs[field].dtype}")
                all_pass = False

        for field in META_FIELDS:
            rv = rs["meta"].get(field)
            cv = cs["meta"].get(field)
            if rv != cv:
                mismatches.append(f"sample {i}: meta.{field}  root={rv}  cache={cv}")
                all_pass = False

        if include_graph and "edge_index" in rs:
            for field in GRAPH_FIELDS:
                if not _tensor_equal(rs[field], cs[field]):
                    mismatches.append(f"sample {i}: {field} mismatch")
                    all_pass = False

        if mismatches and len(mismatches) >= 10:
            print("  [stopping early — too many mismatches]")
            break

    if mismatches:
        print(f"\n[FAIL] {len(mismatches)} field mismatches:")
        for m in mismatches[:20]:
            print(f"  {m}")
    else:
        print(f"  All {n} samples match exactly.")

    # Aggregate statistics on the full validated prefix
    print(f"\nAggregate statistics (first {n} samples):")
    root_vm  = torch.stack([root_ds[i]["valid_mask"]  for i in range(n)])
    cache_vm = torch.stack([cache_ds[i]["valid_mask"] for i in range(n)])
    root_nl  = torch.stack([root_ds[i]["node_label"]  for i in range(n)])
    cache_nl = torch.stack([cache_ds[i]["node_label"] for i in range(n)])
    root_tid = torch.stack([root_ds[i]["track_id"].to(torch.int32) for i in range(n)])
    cache_tid = torch.stack([cache_ds[i]["track_id"].to(torch.int32) for i in range(n)])

    stats = {
        "total_stubs":   (int(root_vm.sum()),  int(cache_vm.sum())),
        "signal_stubs":  (int(root_nl.sum()),  int(cache_nl.sum())),
        "trackid_sum":   (int(root_tid.sum()), int(cache_tid.sum())),
    }
    if include_graph:
        n_root_edges  = sum(root_ds[i]["edge_index"].shape[1]  for i in range(n))
        n_cache_edges = sum(cache_ds[i]["edge_index"].shape[1] for i in range(n))
        stats["total_edges"] = (n_root_edges, n_cache_edges)

    stat_pass = True
    for name, (rv, cv) in stats.items():
        ok = rv == cv
        if not ok:
            stat_pass = False
            all_pass = False
        status = "OK" if ok else "MISMATCH"
        print(f"  {name:20s}  root={rv:>10,}  cache={cv:>10,}  [{status}]")

    result = "PASS" if all_pass else "FAIL"
    print(f"\nValidation result: [{result}]")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Validate OMTF cache vs ROOT dataset")
    parser.add_argument("--dataset",       required=True)
    parser.add_argument("--cache-dir",     required=True, type=Path)
    parser.add_argument("--n-samples",     type=int, default=200)
    parser.add_argument("--include-graph", action="store_true")
    parser.add_argument("--max-files",     type=int, default=None,
                        help="Limit ROOT files loaded (speeds up loading for validation)")
    args = parser.parse_args()

    ok = validate(
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        n_samples=args.n_samples,
        include_graph=args.include_graph,
        max_files=args.max_files,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
