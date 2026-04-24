#!/usr/bin/env python
"""
OMTF ROOT → .pt shard cache converter.

Reads ROOT hits files using uproot (bulk read per file — not entry-by-entry),
applies the same preprocessing as OMTFDataset._process_entry_fn, and writes
sharded .pt files plus a manifest.json.

Usage
-----
  # Build S1 only, no graph, 8 worker threads
  python scripts/omtf/build_cache.py \
    --datasets S1 \
    --output-dir build/omtf/cache/schema_v1

  # All datasets with graph precomputation
  python scripts/omtf/build_cache.py \
    --datasets S1 S2 S3 S4 S5 B1 B2 B3 B4 \
    --include-graph \
    --output-dir build/omtf/cache/schema_v1 \
    --num-workers 8

  # Debug run (first 2 files, 100 entries each)
  python scripts/omtf/build_cache.py \
    --datasets S1 \
    --max-files 2 --max-entries 100 \
    --output-dir /tmp/omtf_cache_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omtf.dataset import _process_entry_fn
from omtf.features import RAW_FEATURE_NAMES, PAIR_FEATURE_NAMES
from omtf.cache_dataset import SCHEMA_VERSION, _KAPPA_CLIP, _TRUNC_POLICY, _EDGE_POLICY
from audit.root_utils import TREE_PATH, NANO_TREE

DATA_ROOT = PROJECT_ROOT / "data" / "prod"

HITS_BRANCHES = [
    "reg_eventNum",
    "reg_iProcessor",
    "reg_stub_phiHw",
    "reg_stub_phiBHw",
    "reg_stub_r",
    "reg_stub_etaHw",
    "reg_stub_quality",
    "reg_stub_type",
    "reg_stub_layer",
    "reg_stub_trackId",
    "reg_stub_ambiguous",
]


# ---------------------------------------------------------------------------
# uproot bulk readers
# ---------------------------------------------------------------------------

def _read_nano_bulk(nano_path: Path) -> dict:
    """Bulk-read nano gen data into an event_num → gen_targets dict."""
    import uproot

    if not nano_path.exists():
        return {}

    try:
        with uproot.open(str(nano_path)) as f:
            if NANO_TREE not in f:
                return {}
            tree = f[NANO_TREE]
            n = int(tree.num_entries)
            if n == 0:
                return {}

            avail = set(tree.keys())
            branches = ["event", "nGenMuon", "GenMuon_pt"]
            if "GenMuon_charge" in avail:
                branches.append("GenMuon_charge")
            if "GenMuon_dXY" in avail:
                branches.append("GenMuon_dXY")
            if "GenMuon_phi" in avail:
                branches.append("GenMuon_phi")

            arrays = tree.arrays(branches, library="np")

        has_charge = "GenMuon_charge" in arrays
        has_dxy    = "GenMuon_dXY"    in arrays
        has_phi    = "GenMuon_phi"    in arrays

        event_map: dict = {}
        for i in range(n):
            key = int(arrays["event"][i]) & 0xFFFFFFFF
            n_gen = int(arrays["nGenMuon"][i])
            if n_gen == 0:
                event_map[key] = {"pt": [], "charge": [], "dxy": [], "phi": []}
            else:
                pts = arrays["GenMuon_pt"][i]
                event_map[key] = {
                    "pt":     [float(pts[k])                             for k in range(n_gen)],
                    "charge": [int(arrays["GenMuon_charge"][i][k])       for k in range(n_gen)] if has_charge else [0]   * n_gen,
                    "dxy":    [float(arrays["GenMuon_dXY"][i][k])        for k in range(n_gen)] if has_dxy    else [0.0] * n_gen,
                    "phi":    [float(arrays["GenMuon_phi"][i][k])        for k in range(n_gen)] if has_phi    else [0.0] * n_gen,
                }
        return event_map

    except Exception as e:
        print(f"  [WARN] nano read failed ({nano_path.name}): {e}")
        return {}


def _process_hits_file(
    hits_path: Path,
    include_nano: bool,
    include_graph: bool,
    nmax: int,
    kappa_clip: float,
    max_entries: int | None,
) -> list[dict]:
    """
    Bulk-read one ROOT hits file with uproot and process all entries.

    This is the key speedup over the current per-entry GetEntry approach:
    all branches are read in a single uproot call, then we loop over the
    resulting numpy arrays.
    """
    import uproot

    nano_map: dict = {}
    if include_nano:
        nano_path = hits_path.parent / hits_path.name.replace("omtf_hits_", "omtf_nano_")
        nano_map = _read_nano_bulk(nano_path)

    try:
        with uproot.open(str(hits_path)) as f:
            if TREE_PATH not in f:
                return []
            tree = f[TREE_PATH]
            n = int(tree.num_entries)
            if max_entries is not None:
                n = min(n, max_entries)
            # Single bulk read — much faster than N × GetEntry calls
            arrays = tree.arrays(HITS_BRANCHES, entry_stop=n, library="np")
    except Exception as e:
        print(f"  [WARN] failed to read {hits_path.name}: {e}")
        return []

    results = []
    for i in range(n):
        entry = {
            "event_num":   int(arrays["reg_eventNum"][i]),
            "i_processor": int(arrays["reg_iProcessor"][i]),
            "n_stubs":     len(arrays["reg_stub_phiHw"][i]),
            "phi":         arrays["reg_stub_phiHw"][i].tolist(),
            "phiB":        arrays["reg_stub_phiBHw"][i].tolist(),
            "r":           arrays["reg_stub_r"][i].tolist(),
            "eta":         arrays["reg_stub_etaHw"][i].tolist(),
            "quality":     arrays["reg_stub_quality"][i].tolist(),
            "type":        arrays["reg_stub_type"][i].tolist(),
            "layer":       arrays["reg_stub_layer"][i].tolist(),
            "track_id":    arrays["reg_stub_trackId"][i].tolist(),
            "ambiguous":   arrays["reg_stub_ambiguous"][i].tolist(),
        }
        nano_gen = nano_map.get(entry["event_num"])
        results.append(_process_entry_fn(entry, nano_gen, nmax, include_graph, kappa_clip))

    return results


# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------

def _write_shard(samples: list[dict], path: Path, include_graph: bool) -> None:
    data = {
        "stubs":      torch.stack([s["stubs"]      for s in samples]),
        "valid_mask": torch.stack([s["valid_mask"] for s in samples]),
        "track_id":   torch.stack([s["track_id"]   for s in samples]),
        "ambiguous":  torch.stack([s["ambiguous"]  for s in samples]),
        "node_label": torch.stack([s["node_label"] for s in samples]),
        "gen_pt":     torch.stack([s["gen_pt"]     for s in samples]),
        "gen_charge": torch.stack([s["gen_charge"] for s in samples]),
        "gen_dxy":    torch.stack([s["gen_dxy"]    for s in samples]),
        "gen_phi":    torch.stack([s["gen_phi"]    for s in samples]),
        "meta_event_num": torch.tensor([s["meta"]["event_num"]   for s in samples], dtype=torch.int32),
        "meta_i_proc":   torch.tensor([s["meta"]["i_processor"] for s in samples], dtype=torch.int32),
        "meta_n_stubs":  torch.tensor([s["meta"]["n_stubs"]      for s in samples], dtype=torch.int32),
        "meta_n_gen":    torch.tensor([s["meta"]["n_gen_tracks"] for s in samples], dtype=torch.int32),
    }
    if include_graph:
        data["edge_index"] = [s["edge_index"] for s in samples]
        data["edge_attr"]  = [s["edge_attr"]  for s in samples]
        data["edge_label"] = [s["edge_label"] for s in samples]
        data["edge_ambig"] = [s["edge_ambig"] for s in samples]

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


# ---------------------------------------------------------------------------
# Per-dataset conversion
# ---------------------------------------------------------------------------

def convert_dataset(
    dataset: str,
    output_dir: Path,
    nmax: int,
    kappa_clip: float,
    include_nano: bool,
    include_graph: bool,
    shard_size: int,
    max_files: int | None,
    max_entries: int | None,
    num_workers: int,
) -> dict:
    """Convert one dataset. Returns info dict for manifest."""
    ds_dir = DATA_ROOT / dataset
    files = sorted(ds_dir.glob(f"omtf_hits_{dataset}_*.root"))
    if not files:
        print(f"  [SKIP] no files found for {dataset}")
        return {}
    if max_files:
        files = files[:max_files]

    out_ds_dir = output_dir / dataset
    out_ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {dataset}: {len(files)} files → {out_ds_dir}")
    t0 = time.perf_counter()

    buffer: list[dict] = []
    shard_idx = 0
    total_samples = 0

    def _process(p):
        return _process_hits_file(p, include_nano, include_graph, nmax, kappa_clip, max_entries)

    n_workers = min(num_workers, len(files))
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # map preserves file order (unlike as_completed)
            for samples in ex.map(_process, files):
                buffer.extend(samples)
                while len(buffer) >= shard_size:
                    _write_shard(buffer[:shard_size],
                                 out_ds_dir / f"shard_{shard_idx:04d}.pt",
                                 include_graph)
                    shard_idx += 1
                    total_samples += shard_size
                    buffer = buffer[shard_size:]
    else:
        for p in files:
            buffer.extend(_process(p))
            while len(buffer) >= shard_size:
                _write_shard(buffer[:shard_size],
                             out_ds_dir / f"shard_{shard_idx:04d}.pt",
                             include_graph)
                shard_idx += 1
                total_samples += shard_size
                buffer = buffer[shard_size:]

    # Flush remaining
    if buffer:
        _write_shard(buffer, out_ds_dir / f"shard_{shard_idx:04d}.pt", include_graph)
        total_samples += len(buffer)
        shard_idx += 1

    elapsed = time.perf_counter() - t0
    rate = total_samples / elapsed if elapsed > 0 else 0
    print(f"    → {total_samples:,} samples, {shard_idx} shards  ({elapsed:.1f}s, {rate:.0f} samples/s)")

    return {
        "n_samples":    total_samples,
        "n_shards":     shard_idx,
        "source_files": [str(p) for p in files],
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    output_dir: Path,
    datasets_info: dict,
    nmax: int,
    kappa_clip: float,
    include_graph: bool,
) -> None:
    manifest = {
        "schema_version":    SCHEMA_VERSION,
        "nmax":              nmax,
        "raw_feature_names": RAW_FEATURE_NAMES,
        "pair_feature_names": PAIR_FEATURE_NAMES,
        "kappa_clip":        kappa_clip,
        "truncation_policy": _TRUNC_POLICY,
        "include_graph":     include_graph,
        "graph_edge_policy": _EDGE_POLICY if include_graph else None,
        "converter_version": "1",
        "created_at":        datetime.now().isoformat(),
        "datasets":          datasets_info,
    }
    path = output_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build OMTF .pt shard cache from ROOT")
    parser.add_argument("--datasets",     nargs="+",
                        default=["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"])
    parser.add_argument("--output-dir",   required=True, type=Path)
    parser.add_argument("--nmax",         type=int,   default=24)
    parser.add_argument("--kappa-clip",   type=float, default=500.0)
    parser.add_argument("--include-graph", action="store_true",
                        help="Precompute and store graph outputs (edge_index etc.)")
    parser.add_argument("--no-nano",      action="store_true",
                        help="Skip NanoAOD gen matching (faster, no gen_pt targets)")
    parser.add_argument("--shard-size",   type=int, default=5000,
                        help="Samples per shard file (default 5000)")
    parser.add_argument("--max-files",    type=int, default=None,
                        help="Limit files per dataset (for testing)")
    parser.add_argument("--max-entries",  type=int, default=None,
                        help="Limit entries per file (for testing)")
    parser.add_argument("--num-workers",  type=int, default=4,
                        help="Parallel file-loading threads")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    include_nano = not args.no_nano

    print(f"Building cache: {args.output_dir}")
    print(f"  datasets={args.datasets}  nmax={args.nmax}  "
          f"kappa_clip={args.kappa_clip}  graph={args.include_graph}  "
          f"nano={include_nano}  shard_size={args.shard_size}  "
          f"workers={args.num_workers}\n")

    t_total = time.perf_counter()
    datasets_info = {}
    for ds in args.datasets:
        info = convert_dataset(
            dataset=ds,
            output_dir=args.output_dir,
            nmax=args.nmax,
            kappa_clip=args.kappa_clip,
            include_nano=include_nano,
            include_graph=args.include_graph,
            shard_size=args.shard_size,
            max_files=args.max_files,
            max_entries=args.max_entries,
            num_workers=args.num_workers,
        )
        if info:
            datasets_info[ds] = info

    write_manifest(args.output_dir, datasets_info, args.nmax, args.kappa_clip, args.include_graph)
    elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
