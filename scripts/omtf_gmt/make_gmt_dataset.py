#!/usr/bin/env python
"""
GMT-visible-stub dataset builder (Phase B1 — barrel KMTF, offline truth transfer).

For each entry in OMTFAllInputTree (= one OMTF processor window per event):
  1. Collect KMTF barrel stubs from NanoAOD whose offlineCoord1 falls inside
     the processor's 120° phi window.
  2. Transfer truth labels (trackId, ambiguous) from OMTF stubs via
     composite phi+BX+station matching.
  3. Attach gen-muon kinematics from NanoAOD GenMuon collection.
  4. Pad/truncate to Nmax stubs and write sharded .pt output.

Output schema (one shard = dict of stacked tensors):
  stubs            (N, Nmax, 10) float32
  valid_mask       (N, Nmax)     bool
  track_id         (N, Nmax)     int8
  ambiguous        (N, Nmax)     uint8
  node_label       (N, Nmax)     float32
  gen_pt           (N, K)        float32
  gen_charge       (N, K)        float32
  gen_dxy          (N, K)        float32
  meta_event_num   (N,)          int32
  meta_i_proc      (N,)          int32
  meta_n_stubs     (N,)          int32
  meta_n_gen       (N,)          int32

Usage
-----
  python scripts/omtf_gmt/make_gmt_dataset.py \\
      --datasets S1 B1 B4 \\
      --data-dir data/prod \\
      --output-dir build/omtf_gmt/cache \\
      --max-files 50

  # All datasets, all files
  python scripts/omtf_gmt/make_gmt_dataset.py \\
      --data-dir data/prod \\
      --output-dir build/omtf_gmt/cache
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

from omtf_gmt.regioning  import stubs_in_window, phi_rel, omtf_phi_to_global_rad
from omtf_gmt.features   import build_node_features, N_FEATURES
from omtf_gmt.truth_transfer import transfer

try:
    import uproot
    import awkward as ak
except ImportError:
    print("uproot and awkward are required.  Run: pip install uproot awkward")
    sys.exit(1)

# ----- constants -----------------------------------------------------------

ALL_DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]
SCHEMA_VERSION = 1
SHARD_SIZE     = 5_000
_NMAX          = 24
_K_MAX         = 3

OMTF_HITS_BRANCHES = [
    "reg_eventNum", "reg_iProcessor",
    "reg_stub_phiHw", "reg_stub_layer", "reg_stub_bx",
    "reg_stub_trackId", "reg_stub_ambiguous",
]
NANO_BRANCHES = [
    "event", "nGenMuon",
    "GenMuon_pt", "GenMuon_charge", "GenMuon_dXY",
    "nMuonStubKmtf",
    "MuonStubKmtf_isBarrel",
    "MuonStubKmtf_offlineCoord1", "MuonStubKmtf_offlineCoord2",
    "MuonStubKmtf_offlineEta1",   "MuonStubKmtf_offlineEta2",
    "MuonStubKmtf_quality",       "MuonStubKmtf_etaQuality",
    "MuonStubKmtf_bxNum",
    "MuonStubKmtf_tfLayer",       "MuonStubKmtf_depthRegion",
]

# ----- file-level builder --------------------------------------------------

def _load_nano_event_map(nano_path: Path) -> dict[int, dict]:
    """Return dict: uint32(event) → {gen_pt, gen_charge, gen_dxy, kmtf_stubs}."""
    tree = uproot.open(str(nano_path))["Events"]
    arr  = tree.arrays(NANO_BRANCHES, library="ak")

    event_map: dict[int, dict] = {}
    for i in range(len(arr)):
        ev = int(arr["event"][i]) & 0xFFFFFFFF   # cast to uint32

        # gen muons
        n_gen  = int(arr["nGenMuon"][i])
        gen_pt = ak.to_numpy(arr["GenMuon_pt"][i]).astype(np.float32)[:_K_MAX]
        gen_ch = ak.to_numpy(arr["GenMuon_charge"][i]).astype(np.float32)[:_K_MAX]
        gen_d  = ak.to_numpy(arr["GenMuon_dXY"][i]).astype(np.float32)[:_K_MAX]
        # pad to K_MAX
        def _pad(a): return np.pad(a, (0, max(0, _K_MAX - len(a))))
        gen_pt = _pad(gen_pt); gen_ch = _pad(gen_ch); gen_d = _pad(gen_d)

        # KMTF barrel stubs
        is_bar = ak.to_numpy(arr["MuonStubKmtf_isBarrel"][i]).astype(bool)
        c1  = ak.to_numpy(arr["MuonStubKmtf_offlineCoord1"][i])[is_bar].astype(np.float64)
        c2  = ak.to_numpy(arr["MuonStubKmtf_offlineCoord2"][i])[is_bar].astype(np.float32)
        e1  = ak.to_numpy(arr["MuonStubKmtf_offlineEta1"][i])[is_bar].astype(np.float32)
        e2  = ak.to_numpy(arr["MuonStubKmtf_offlineEta2"][i])[is_bar].astype(np.float32)
        q   = ak.to_numpy(arr["MuonStubKmtf_quality"][i])[is_bar].astype(np.int16)
        eq  = ak.to_numpy(arr["MuonStubKmtf_etaQuality"][i])[is_bar].astype(np.int16)
        bx  = ak.to_numpy(arr["MuonStubKmtf_bxNum"][i])[is_bar].astype(np.int8)
        lay = ak.to_numpy(arr["MuonStubKmtf_tfLayer"][i])[is_bar].astype(np.int8)
        dep = ak.to_numpy(arr["MuonStubKmtf_depthRegion"][i])[is_bar].astype(np.int8)

        event_map[ev] = {
            "n_gen": min(n_gen, _K_MAX),
            "gen_pt": gen_pt, "gen_charge": gen_ch, "gen_dxy": gen_d,
            "kmtf": {
                "c1": c1, "c2": c2, "e1": e1, "e2": e2,
                "q": q, "eq": eq, "bx": bx, "lay": lay, "dep": dep,
            },
        }
    return event_map


def _process_omtf_entry(
    event_num: int,
    proc:      int,
    # OMTF stubs for this window
    phi_hw:    np.ndarray,
    layer:     np.ndarray,
    bx_omtf:   np.ndarray,
    track_id_omtf: np.ndarray,
    ambig_omtf:    np.ndarray,
    # NanoAOD lookup
    nano_ev:   dict,
) -> dict | None:
    """Build one GMT sample from an OMTF processor-window entry."""
    kmtf = nano_ev["kmtf"]
    c1   = kmtf["c1"]

    if len(c1) == 0:
        return None   # no KMTF stubs in event at all

    # --- 1. select KMTF stubs in this processor's phi window ---
    mask = stubs_in_window(c1, proc)
    if not mask.any():
        return None   # no KMTF stubs in this region

    k_c1  = c1[mask].astype(np.float64)
    k_c2  = kmtf["c2"][mask]
    k_e1  = kmtf["e1"][mask]
    k_e2  = kmtf["e2"][mask]
    k_q   = kmtf["q"][mask]
    k_eq  = kmtf["eq"][mask]
    k_bx  = kmtf["bx"][mask]
    k_lay = kmtf["lay"][mask]
    k_dep = kmtf["dep"][mask]

    n_real = len(k_c1)

    # --- 2. truth transfer ---
    tr = transfer(
        kmtf_phi_rad  = k_c1,
        kmtf_depth    = k_dep,
        kmtf_bx       = k_bx,
        omtf_phi_hw   = phi_hw,
        omtf_layer    = layer,
        omtf_bx       = bx_omtf,
        omtf_track_id = track_id_omtf,
        omtf_ambiguous= ambig_omtf,
        omtf_proc     = proc,
    )

    # --- 3. build node features (processor-centred phi) ---
    phi_r = phi_rel(k_c1, proc).astype(np.float32)
    X = build_node_features(phi_r, k_c2, k_e1, k_e2, k_q, k_eq, k_bx, k_lay, k_dep)

    # --- 4. truncate by quality (descending) if needed ---
    if n_real > _NMAX:
        order = np.argsort(-k_q.astype(float))[:_NMAX]
        X       = X[order]
        tr_tid  = tr.track_id[order]
        tr_amb  = tr.ambiguous[order]
        n_real  = _NMAX
    else:
        tr_tid = tr.track_id
        tr_amb = tr.ambiguous

    # --- 5. pad to Nmax ---
    stubs_pad = np.zeros((_NMAX, N_FEATURES), dtype=np.float32)
    stubs_pad[:n_real] = X
    vm   = np.zeros(_NMAX, dtype=bool)
    vm[:n_real] = True
    tid  = np.zeros(_NMAX, dtype=np.int8)
    tid[:n_real] = tr_tid
    amb  = np.zeros(_NMAX, dtype=np.uint8)
    amb[:n_real] = tr_amb
    nl   = (tid != 0).astype(np.float32) * vm.astype(np.float32)

    return {
        "stubs":      torch.from_numpy(stubs_pad),
        "valid_mask": torch.from_numpy(vm),
        "track_id":   torch.from_numpy(tid),
        "ambiguous":  torch.from_numpy(amb),
        "node_label": torch.from_numpy(nl),
        "gen_pt":     torch.from_numpy(nano_ev["gen_pt"]),
        "gen_charge": torch.from_numpy(nano_ev["gen_charge"]),
        "gen_dxy":    torch.from_numpy(nano_ev["gen_dxy"]),
        "meta_event_num": torch.tensor(event_num, dtype=torch.int32),
        "meta_i_proc":    torch.tensor(proc,      dtype=torch.int32),
        "meta_n_stubs":   torch.tensor(n_real,    dtype=torch.int32),
        "meta_n_gen":     torch.tensor(nano_ev["n_gen"], dtype=torch.int32),
    }


def _stack_shard(samples: list[dict]) -> dict:
    keys_scalar = ["meta_event_num", "meta_i_proc", "meta_n_stubs", "meta_n_gen"]
    keys_tensor = ["stubs","valid_mask","track_id","ambiguous","node_label",
                   "gen_pt","gen_charge","gen_dxy"]
    shard = {}
    for k in keys_tensor:
        shard[k] = torch.stack([s[k] for s in samples])
    for k in keys_scalar:
        shard[k] = torch.stack([s[k] for s in samples])
    return shard


def process_file_pair(
    hits_path: Path,
    nano_path: Path,
    verbose: bool = False,
) -> list[dict]:
    """Process one (hits, nano) file pair → list of sample dicts."""
    nano_map = _load_nano_event_map(nano_path)

    hits_tree = uproot.open(str(hits_path))["simOmtfPhase2Digis/OMTFAllInputTree"]
    arr = hits_tree.arrays(OMTF_HITS_BRANCHES, library="ak")

    samples: list[dict] = []
    n_skipped = 0

    for i in range(len(arr)):
        ev   = int(arr["reg_eventNum"][i]) & 0xFFFFFFFF
        proc = int(arr["reg_iProcessor"][i])

        if ev not in nano_map:
            n_skipped += 1
            continue

        phi_hw  = ak.to_numpy(arr["reg_stub_phiHw"][i]).astype(np.int32)
        layer   = ak.to_numpy(arr["reg_stub_layer"][i]).astype(np.int8)
        bx_o    = ak.to_numpy(arr["reg_stub_bx"][i]).astype(np.int8)
        tid_o   = ak.to_numpy(arr["reg_stub_trackId"][i]).astype(np.int8)
        amb_o   = ak.to_numpy(arr["reg_stub_ambiguous"][i]).astype(np.uint8)

        sample = _process_omtf_entry(
            ev, proc, phi_hw, layer, bx_o, tid_o, amb_o, nano_map[ev]
        )
        if sample is not None:
            samples.append(sample)

    if verbose and n_skipped:
        print(f"    skipped {n_skipped} OMTF entries (event not in NanoAOD)")
    return samples


# ----- main ----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GMT-visible-stub dataset")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--data-dir", type=Path, default=Path("data/prod"),
                   help="Root of data directory (contains <DATASET>/ subdirs)")
    p.add_argument("--output-dir", type=Path, default=Path("build/omtf_gmt/cache"),
                   help="Output cache directory")
    p.add_argument("--max-files", type=int, default=None,
                   help="Max hits files per dataset (None = all)")
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "branch": "omtf_gmt",
        "stage": "B1_barrel_kmtf",
        "nmax": _NMAX,
        "k_max": _K_MAX,
        "n_features": N_FEATURES,
        "datasets": {},
    }

    for ds in args.datasets:
        ds_data_dir = args.data_dir / ds
        hits_files  = sorted(ds_data_dir.glob("omtf_hits_*.root"))
        if args.max_files:
            hits_files = hits_files[:args.max_files]
        if not hits_files:
            print(f"  [{ds}] no hits files found in {ds_data_dir}, skipping")
            continue

        ds_out = outdir / ds
        ds_out.mkdir(exist_ok=True)
        t0 = time.time()

        pending: list[dict] = []
        shard_idx = 0
        n_total   = 0
        n_files   = 0

        for hits_path in hits_files:
            nano_path = hits_path.parent / hits_path.name.replace("omtf_hits_", "omtf_nano_")
            if not nano_path.exists():
                if verbose:
                    print(f"  [{ds}] missing nano file for {hits_path.name}, skipping")
                continue

            samples = process_file_pair(hits_path, nano_path, verbose=False)
            pending.extend(samples)
            n_files += 1

            while len(pending) >= args.shard_size:
                shard = _stack_shard(pending[:args.shard_size])
                torch.save(shard, ds_out / f"shard_{shard_idx:04d}.pt")
                shard_idx += 1
                n_total   += args.shard_size
                pending    = pending[args.shard_size:]

            if verbose:
                print(f"  [{ds}] {n_files}/{len(hits_files)} files  "
                      f"samples_so_far={n_total + len(pending):,}", end="\r")

        # flush remaining
        if pending:
            shard = _stack_shard(pending)
            torch.save(shard, ds_out / f"shard_{shard_idx:04d}.pt")
            n_total += len(pending)

        elapsed = time.time() - t0
        manifest["datasets"][ds] = {
            "n_samples": n_total,
            "n_shards":  shard_idx + (1 if pending else 0),
            "n_files":   n_files,
        }
        if verbose:
            print(f"\n  [{ds}] done — {n_total:,} samples in {elapsed:.0f}s")

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {outdir / 'manifest.json'}")


if __name__ == "__main__":
    main()
