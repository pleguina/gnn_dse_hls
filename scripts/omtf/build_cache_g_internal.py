#!/usr/bin/env python
"""
OMTF-internal G-dataset cache builder.

Reads reg_stub_* directly from OMTFAllInputTree (no NanoAOD stub collection,
no truth transfer) and writes sharded .pt files compatible with the GMT
schema_version=2 cache format used by src/omtf_gmt/dataset.py.

Adapted from scripts/omtf/build_cache.py (old S/B datasets, schema_version=1)
for the new G-dataset format.

Key differences from the TPS builder (make_gmt_dataset_tps.py):
  * Uses reg_stub_* as input features directly (no MuonStubTps selection).
  * No phi-window filtering — each OMTFAllInputTree entry IS one window.
  * Native truth: reg_stub_trackId == 0 → noise, > 0 → signal.
  * NanoAOD still used for gen_pt / gen_charge / gen_dxy targets.
  * 11 features (see src/omtf/features_g.py).

Usage
-----
  python scripts/omtf/build_cache_g_internal.py \\
      --datasets G1_pos G1_neg G7 G8 B4 \\
      --data-dir data/prod \\
      --output-dir build/omtf/cache_internal_g_v1 \\
      --max-files 5

  python scripts/omtf/build_cache_g_internal.py \\
      --data-dir data/prod \\
      --output-dir build/omtf/cache_internal_g_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf.features_g import (
    build_node_features_g,
    N_FEATURES_G,
    FEATURE_NAMES_G,
)

try:
    import uproot
    import awkward as ak
except ImportError:
    print("uproot and awkward are required.  pip install uproot awkward")
    sys.exit(1)

# ---- constants -------------------------------------------------------------

ALL_DATASETS = [
    "G1_pos", "G1_neg",
    "G2_pos", "G2_neg",
    "G3_pos", "G3_neg",
    "G4_pos", "G4_neg",
    "G5_pos", "G5_neg",
    "G6_pos", "G6_neg",
    "G7", "G8", "B4",
]
HARD_NEG_DATASETS = {"G7", "G8"}

SCHEMA_VERSION = 2
SHARD_SIZE     = 5_000
_NMAX          = 24
_K_MAX         = 3

OMTF_HITS_BRANCHES = [
    "reg_eventNum", "reg_iProcessor",
    "reg_stub_phiHw",  "reg_stub_phiBHw",
    "reg_stub_etaHw",  "reg_stub_r",
    "reg_stub_quality","reg_stub_type",
    "reg_stub_layer",  "reg_stub_bx",
    "reg_stub_trackId","reg_stub_ambiguous",
]

NANO_BRANCHES_REQUIRED = ["event", "nGenMuon", "GenMuon_pt", "GenMuon_charge"]
NANO_BRANCHES_OPTIONAL = ["GenMuon_dXY"]


# ---- NanoAOD gen-target loader (no stub reading) ---------------------------

def _load_nano_gen_map(nano_path: Path) -> dict[int, dict]:
    """Return dict: uint32(event) → {gen_pt_full, gen_charge_full, gen_dxy_full}."""
    tree   = uproot.open(str(nano_path))["Events"]
    avail  = set(tree.keys())
    branches = NANO_BRANCHES_REQUIRED + [b for b in NANO_BRANCHES_OPTIONAL if b in avail]
    has_dxy  = "GenMuon_dXY" in avail
    arr    = tree.arrays(branches, library="ak")

    event_map: dict[int, dict] = {}
    for i in range(len(arr)):
        ev     = int(arr["event"][i]) & 0xFFFFFFFF
        n_gen  = int(arr["nGenMuon"][i])
        gen_pt = ak.to_numpy(arr["GenMuon_pt"][i]).astype(np.float32)
        gen_ch = ak.to_numpy(arr["GenMuon_charge"][i]).astype(np.float32)
        gen_dxy = (ak.to_numpy(arr["GenMuon_dXY"][i]).astype(np.float32)
                   if has_dxy else np.zeros(n_gen, dtype=np.float32))
        event_map[ev] = {
            "gen_pt_full":     gen_pt,
            "gen_charge_full": gen_ch,
            "gen_dxy_full":    gen_dxy,
        }
    return event_map


# ---- per-window processor --------------------------------------------------

def _process_entry(
    event_num:   int,
    proc:        int,
    phi_hw:      np.ndarray,
    phib_hw:     np.ndarray,
    eta_hw:      np.ndarray,
    r:           np.ndarray,
    quality:     np.ndarray,
    type_:       np.ndarray,
    layer:       np.ndarray,
    bx:          np.ndarray,
    track_id:    np.ndarray,
    ambiguous:   np.ndarray,
    nano_ev:     dict | None,
    is_hard_neg: bool,
) -> dict | None:
    n_real = len(phi_hw)
    if n_real == 0:
        return None

    # Build features from reg_stub_* (no truth transfer)
    X = build_node_features_g(phi_hw, phib_hw, eta_hw, r, quality, type_, layer, bx)

    # Native truth labels
    tr_tid = track_id.astype(np.int8)
    tr_amb = ambiguous.astype(np.uint8)
    # truth_source: 2=signal, 1=noise (all valid stubs have confirmed identity)
    tr_src = np.where(tr_tid > 0, np.int8(2), np.int8(1)).astype(np.int8)

    # Truncate by quality descending if needed
    if n_real > _NMAX:
        order  = np.argsort(-quality.astype(np.float32))[:_NMAX]
        X      = X[order]
        tr_tid = tr_tid[order]
        tr_amb = tr_amb[order]
        tr_src = tr_src[order]
        n_real = _NMAX

    # Candidate targets from unique positive track_ids in this window
    positive_ids = sorted(set(int(x) for x in tr_tid[:n_real] if int(x) > 0))

    gen_pt     = np.zeros(_K_MAX, dtype=np.float32)
    gen_charge = np.zeros(_K_MAX, dtype=np.float32)
    gen_dxy    = np.zeros(_K_MAX, dtype=np.float32)
    tgt_tid    = np.zeros(_K_MAX, dtype=np.int8)

    if not is_hard_neg and nano_ev is not None:
        pt_full  = nano_ev["gen_pt_full"]
        ch_full  = nano_ev["gen_charge_full"]
        dxy_full = nano_ev["gen_dxy_full"]
        for slot, tid_val in enumerate(positive_ids[:_K_MAX]):
            idx = tid_val - 1
            if idx < len(pt_full):
                gen_pt[slot]     = pt_full[idx]
                gen_charge[slot] = ch_full[idx]
                gen_dxy[slot]    = dxy_full[idx]
            tgt_tid[slot] = tid_val

    # Pad to NMAX
    stubs_pad = np.zeros((_NMAX, N_FEATURES_G), dtype=np.float32)
    stubs_pad[:n_real] = X
    vm  = np.zeros(_NMAX, dtype=bool); vm[:n_real]  = True
    tid = np.zeros(_NMAX, dtype=np.int8);  tid[:n_real] = tr_tid
    amb = np.zeros(_NMAX, dtype=np.uint8); amb[:n_real] = tr_amb
    nl  = (tid != 0).astype(np.float32) * vm.astype(np.float32)
    ts  = np.zeros(_NMAX, dtype=np.int8); ts[:n_real] = tr_src

    return {
        "stubs":           torch.from_numpy(stubs_pad),
        "valid_mask":      torch.from_numpy(vm),
        "track_id":        torch.from_numpy(tid),
        "ambiguous":       torch.from_numpy(amb),
        "node_label":      torch.from_numpy(nl),
        "truth_source":    torch.from_numpy(ts),
        "gen_pt":          torch.from_numpy(gen_pt),
        "gen_charge":      torch.from_numpy(gen_charge),
        "gen_dxy":         torch.from_numpy(gen_dxy),
        "target_track_id": torch.from_numpy(tgt_tid),
        "meta_event_num":   torch.tensor(event_num,                                dtype=torch.int32),
        "meta_i_proc":      torch.tensor(proc,                                     dtype=torch.int32),
        "meta_n_stubs":     torch.tensor(n_real,                                   dtype=torch.int32),
        "meta_n_gen":       torch.tensor(0 if is_hard_neg else len(positive_ids),  dtype=torch.int32),
        "meta_is_hard_neg": torch.tensor(int(is_hard_neg),                         dtype=torch.int8),
    }


def _stack_shard(samples: list[dict]) -> dict:
    keys = [
        "stubs", "valid_mask", "track_id", "ambiguous",
        "node_label", "truth_source",
        "gen_pt", "gen_charge", "gen_dxy", "target_track_id",
        "meta_event_num", "meta_i_proc", "meta_n_stubs",
        "meta_n_gen", "meta_is_hard_neg",
    ]
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


# ---- file-pair processor ---------------------------------------------------

def process_file_pair(
    hits_path:   Path,
    nano_path:   Path,
    is_hard_neg: bool,
    verbose:     bool,
) -> list[dict]:
    nano_map = _load_nano_gen_map(nano_path) if nano_path.exists() else {}

    tree = uproot.open(str(hits_path))["simOmtfPhase2Digis/OMTFAllInputTree"]
    arr  = tree.arrays(OMTF_HITS_BRANCHES, library="ak")

    samples: list[dict] = []
    n_skipped = 0

    for i in range(len(arr)):
        ev   = int(arr["reg_eventNum"][i]) & 0xFFFFFFFF
        proc = int(arr["reg_iProcessor"][i])

        nano_ev = nano_map.get(ev)
        if nano_ev is None and not is_hard_neg:
            n_skipped += 1
            continue

        s = _process_entry(
            event_num   = ev,
            proc        = proc,
            phi_hw      = ak.to_numpy(arr["reg_stub_phiHw"][i]).astype(np.int32),
            phib_hw     = ak.to_numpy(arr["reg_stub_phiBHw"][i]).astype(np.int32),
            eta_hw      = ak.to_numpy(arr["reg_stub_etaHw"][i]).astype(np.int32),
            r           = ak.to_numpy(arr["reg_stub_r"][i]).astype(np.float32),
            quality     = ak.to_numpy(arr["reg_stub_quality"][i]).astype(np.int16),
            type_       = ak.to_numpy(arr["reg_stub_type"][i]).astype(np.int16),
            layer       = ak.to_numpy(arr["reg_stub_layer"][i]).astype(np.int16),
            bx          = ak.to_numpy(arr["reg_stub_bx"][i]).astype(np.int8),
            track_id    = ak.to_numpy(arr["reg_stub_trackId"][i]).astype(np.int8),
            ambiguous   = ak.to_numpy(arr["reg_stub_ambiguous"][i]).astype(np.uint8),
            nano_ev     = nano_ev,
            is_hard_neg = is_hard_neg,
        )
        if s is not None:
            samples.append(s)

    if verbose and n_skipped:
        print(f"    skipped {n_skipped} entries (event not in NanoAOD)")
    return samples


# ---- main ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OMTF-internal G-dataset cache")
    p.add_argument("--datasets",   nargs="+", default=ALL_DATASETS)
    p.add_argument("--data-dir",   type=Path, default=Path("data/prod"))
    p.add_argument("--output-dir", type=Path, default=Path("build/omtf/cache_internal_g_v1"))
    p.add_argument("--max-files",  type=int,  default=None)
    p.add_argument("--shard-size", type=int,  default=SHARD_SIZE)
    p.add_argument("--quiet",      action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    outdir  = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    manifest: dict = {
        "schema_version":        SCHEMA_VERSION,
        "branch":                "omtf_internal",
        "stage":                 "C_internal_g_v1",
        "stub_source":           "OMTFAllInputTree",
        "nmax":                  _NMAX,
        "k_max":                 _K_MAX,
        "n_features":            N_FEATURES_G,
        "feature_names":         FEATURE_NAMES_G,
        "truth_source_encoding": {"2": "omtf_signal", "1": "omtf_noise", "0": "padding"},
        "datasets":              {},
    }

    for ds in args.datasets:
        ds_dir     = args.data_dir / ds
        hits_files = sorted(ds_dir.glob("omtf_hits_*.root"))
        if args.max_files:
            hits_files = hits_files[:args.max_files]
        if not hits_files:
            print(f"  [{ds}] no hits files in {ds_dir}, skipping")
            continue

        ds_out = outdir / ds
        ds_out.mkdir(exist_ok=True)
        for old in ds_out.glob("shard_*.pt"):
            old.unlink()

        t0          = time.time()
        base_ds     = ds.removesuffix("_pos").removesuffix("_neg")
        is_hard_neg = base_ds in HARD_NEG_DATASETS
        pending:   list[dict] = []
        shard_idx = 0
        n_total   = 0
        n_files   = 0

        for hits_path in hits_files:
            nano_path = hits_path.parent / hits_path.name.replace("omtf_hits_", "omtf_nano_")
            samples = process_file_pair(hits_path, nano_path, is_hard_neg, verbose)
            pending.extend(samples)
            n_files += 1

            while len(pending) >= args.shard_size:
                torch.save(_stack_shard(pending[:args.shard_size]),
                           ds_out / f"shard_{shard_idx:04d}.pt")
                shard_idx += 1
                n_total   += args.shard_size
                pending    = pending[args.shard_size:]

            if verbose:
                print(f"  [{ds}] {n_files}/{len(hits_files)} files  "
                      f"samples={n_total + len(pending):,}", end="\r")

        if pending:
            torch.save(_stack_shard(pending), ds_out / f"shard_{shard_idx:04d}.pt")
            n_total   += len(pending)
            shard_idx += 1

        elapsed = time.time() - t0
        manifest["datasets"][ds] = {
            "n_samples":    n_total,
            "n_shards":     shard_idx,
            "n_files":      n_files,
            "is_hard_neg":  is_hard_neg,
            "base_dataset": base_ds,
        }
        if verbose:
            print(f"\n  [{ds}] done — {n_total:,} samples in {elapsed:.0f}s")

    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing["datasets"].update(manifest["datasets"])
        manifest = existing
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
