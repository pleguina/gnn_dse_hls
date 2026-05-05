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
  stubs            (N, Nmax, 14) float32  — per-stub node features (schema v2)
  valid_mask       (N, Nmax)     bool
  track_id         (N, Nmax)     int8     — OMTF trackId per stub (0 = noise)
  ambiguous        (N, Nmax)     uint8
  node_label       (N, Nmax)     float32  — 1 if stub belongs to an overlap target
  truth_source     (N, Nmax)     int8     — 2=omtf_transfer, 1=omtf_noise, 0=unmatched
  gen_pt           (N, K)        float32  — pT of overlap candidate targets (window-level)
  gen_charge       (N, K)        float32  — charge of overlap candidate targets
  gen_dxy          (N, K)        float32  — dXY of overlap candidate targets
  target_track_id  (N, K)        int8     — OMTF trackId of each candidate slot (0 = empty)
  meta_event_num   (N,)          int32
  meta_i_proc      (N,)          int32
  meta_n_stubs     (N,)          int32
  meta_n_gen       (N,)          int32    — number of overlap targets in this window
  meta_is_hard_neg (N,)          int8     — 1 for G7/G8 (real muon, no overlap target)

Per-stub labels come from truth transfer (track_id > 0 → signal stub).
Per-candidate labels are the overlap targets present in this processor window,
indexed by OMTF trackId.  Hard negatives (G7/G8) have zero candidate targets
even though real KMTF barrel stubs are present.

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
from omtf_gmt.features   import build_node_features, N_FEATURES, FEATURE_NAMES
from omtf_gmt.truth_transfer import transfer

try:
    import uproot
    import awkward as ak
except ImportError:
    print("uproot and awkward are required.  Run: pip install uproot awkward")
    sys.exit(1)

# ----- constants -----------------------------------------------------------

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

# truth_source encoding: omtf_transfer=2, omtf_noise=1, unmatched=0
_TRUTH_SRC_ENC: dict[str, int] = {
    "omtf_transfer": 2,
    "omtf_noise":    1,
    "unmatched":     0,
}

OMTF_HITS_BRANCHES = [
    "reg_eventNum", "reg_iProcessor",
    "reg_stub_phiHw", "reg_stub_layer", "reg_stub_bx",
    "reg_stub_trackId", "reg_stub_ambiguous",
]
NANO_BRANCHES_REQUIRED = [
    "event", "nGenMuon",
    "GenMuon_pt", "GenMuon_charge",
    "nMuonStubKmtf",
    "MuonStubKmtf_isBarrel",
    "MuonStubKmtf_offlineCoord1", "MuonStubKmtf_offlineCoord2",
    "MuonStubKmtf_offlineEta1",   "MuonStubKmtf_offlineEta2",
    "MuonStubKmtf_quality",       "MuonStubKmtf_etaQuality",
    "MuonStubKmtf_bxNum",
    "MuonStubKmtf_tfLayer",       "MuonStubKmtf_depthRegion",
]
# optional: absent in some production files (e.g. certain S3 files)
NANO_BRANCHES_OPTIONAL = ["GenMuon_dXY"]

# ----- file-level builder --------------------------------------------------

def _load_nano_event_map(nano_path: Path) -> dict[int, dict]:
    """Return dict: uint32(event) → {gen_pt, gen_charge, gen_dxy, kmtf_stubs}."""
    tree = uproot.open(str(nano_path))["Events"]
    available = set(tree.keys())
    branches  = NANO_BRANCHES_REQUIRED + [b for b in NANO_BRANCHES_OPTIONAL if b in available]
    has_dxy   = "GenMuon_dXY" in available
    arr  = tree.arrays(branches, library="ak")

    event_map: dict[int, dict] = {}
    for i in range(len(arr)):
        ev = int(arr["event"][i]) & 0xFFFFFFFF   # cast to uint32

        # gen muons — keep full arrays; window-level targets are selected in
        # _process_omtf_entry using the transferred track_ids (1-based index).
        n_gen         = int(arr["nGenMuon"][i])
        gen_pt_full   = ak.to_numpy(arr["GenMuon_pt"][i]).astype(np.float32)
        gen_ch_full   = ak.to_numpy(arr["GenMuon_charge"][i]).astype(np.float32)
        gen_dxy_full  = (ak.to_numpy(arr["GenMuon_dXY"][i]).astype(np.float32)
                         if has_dxy else np.zeros(n_gen, dtype=np.float32))

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
            "gen_pt_full":    gen_pt_full,
            "gen_charge_full": gen_ch_full,
            "gen_dxy_full":   gen_dxy_full,
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
    is_hard_neg: bool = False,
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
    tr_src = np.array(tr.truth_source, dtype=object)   # list → array for indexing
    if n_real > _NMAX:
        order  = np.argsort(-k_q.astype(float))[:_NMAX]
        X      = X[order]
        tr_tid = tr.track_id[order]
        tr_amb = tr.ambiguous[order]
        tr_src = tr_src[order]
        n_real = _NMAX
    else:
        tr_tid = tr.track_id
        tr_amb = tr.ambiguous

    # --- 5. window-level overlap candidate targets ---
    # positive_ids: sorted unique OMTF track_ids > 0 found among the (possibly
    # truncated) stubs.  track_id is 1-based → index into GenMuon arrays.
    positive_ids = sorted(set(int(x) for x in tr_tid[:n_real] if int(x) > 0))

    gen_pt     = np.zeros(_K_MAX, dtype=np.float32)
    gen_charge = np.zeros(_K_MAX, dtype=np.float32)
    gen_dxy    = np.zeros(_K_MAX, dtype=np.float32)
    tgt_tid    = np.zeros(_K_MAX, dtype=np.int8)

    if not is_hard_neg:
        pt_full  = nano_ev["gen_pt_full"]
        ch_full  = nano_ev["gen_charge_full"]
        dxy_full = nano_ev["gen_dxy_full"]
        for slot, tid_val in enumerate(positive_ids[:_K_MAX]):
            idx = tid_val - 1   # 0-based
            if idx < len(pt_full):
                gen_pt[slot]     = pt_full[idx]
                gen_charge[slot] = ch_full[idx]
                gen_dxy[slot]    = dxy_full[idx]
            tgt_tid[slot] = tid_val
    # hard negatives: gen_pt/charge/dxy/tgt_tid all stay zero (forced)

    # --- 6. pad to Nmax ---
    stubs_pad = np.zeros((_NMAX, N_FEATURES), dtype=np.float32)
    stubs_pad[:n_real] = X
    vm   = np.zeros(_NMAX, dtype=bool)
    vm[:n_real] = True
    tid  = np.zeros(_NMAX, dtype=np.int8)
    tid[:n_real] = tr_tid
    amb  = np.zeros(_NMAX, dtype=np.uint8)
    amb[:n_real] = tr_amb
    nl   = (tid != 0).astype(np.float32) * vm.astype(np.float32)

    # Encode truth_source per stub: omtf_transfer=2, omtf_noise=1, unmatched=0.
    # Padding positions keep 0 (unmatched) — harmless since valid_mask=False there.
    ts = np.zeros(_NMAX, dtype=np.int8)
    for j in range(n_real):
        ts[j] = _TRUTH_SRC_ENC.get(str(tr_src[j]), 0)

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
        "meta_event_num":    torch.tensor(event_num,            dtype=torch.int32),
        "meta_i_proc":       torch.tensor(proc,                 dtype=torch.int32),
        "meta_n_stubs":      torch.tensor(n_real,               dtype=torch.int32),
        "meta_n_gen":        torch.tensor(0 if is_hard_neg else len(positive_ids), dtype=torch.int32),
        "meta_is_hard_neg":  torch.tensor(int(is_hard_neg),     dtype=torch.int8),
    }


def _stack_shard(samples: list[dict]) -> dict:
    keys_scalar = [
        "meta_event_num", "meta_i_proc", "meta_n_stubs",
        "meta_n_gen", "meta_is_hard_neg",
    ]
    keys_tensor = [
        "stubs", "valid_mask", "track_id", "ambiguous",
        "node_label", "truth_source",
        "gen_pt", "gen_charge", "gen_dxy", "target_track_id",
    ]
    shard = {}
    for k in keys_tensor:
        shard[k] = torch.stack([s[k] for s in samples])
    for k in keys_scalar:
        shard[k] = torch.stack([s[k] for s in samples])
    return shard


def process_file_pair(
    hits_path: Path,
    nano_path: Path,
    is_hard_neg: bool = False,
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
            ev, proc, phi_hw, layer, bx_o, tid_o, amb_o, nano_map[ev],
            is_hard_neg=is_hard_neg,
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
        "stage": "B2_barrel_kmtf_with_overlap_meta",
        "nmax": _NMAX,
        "k_max": _K_MAX,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "truth_source_encoding": {v: k for k, v in _TRUTH_SRC_ENC.items()},
        "datasets": {},
    }

    for ds in args.datasets:
        ds_dir     = args.data_dir / ds
        hits_files = sorted(ds_dir.glob("omtf_hits_*.root"))
        if args.max_files:
            hits_files = hits_files[:args.max_files]
        if not hits_files:
            print(f"  [{ds}] no hits files found in {ds_dir}, skipping")
            continue

        ds_out = outdir / ds
        ds_out.mkdir(exist_ok=True)
        # remove stale shards before writing so no old files linger
        for old in ds_out.glob("shard_*.pt"):
            old.unlink()
        t0 = time.time()
        # strip _pos/_neg suffix to resolve the base dataset name for HARD_NEG check
        base_ds = ds.removesuffix("_pos").removesuffix("_neg")
        is_hard_neg = base_ds in HARD_NEG_DATASETS

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

            samples = process_file_pair(hits_path, nano_path,
                                        is_hard_neg=is_hard_neg, verbose=False)
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
            "n_samples":   n_total,
            "n_shards":    shard_idx + (1 if pending else 0),
            "n_files":     n_files,
            "is_hard_neg": is_hard_neg,
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
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
