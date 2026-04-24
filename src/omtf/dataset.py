"""
OMTF ROOT → PyTorch Dataset.

Reads OMTFAllInputTree entries from ROOT hits files, applies the agreed
padding/truncation policy, and returns per-window samples as dicts of tensors.

Padding  : zero-pad stub arrays to Nmax with valid_mask
Truncation: by quality (descending) when n_stubs > Nmax

Output per sample (all tensors on CPU, batching done by DataLoader)
------------------------------------------------------------------
  stubs       : (Nmax, 7) float32  — [phi, phiB, eta, r, quality, type, layer]
                                      bx excluded (identically 0 in all current samples)
  valid_mask  : (Nmax,)   bool     — True for real stubs, False for padding
  track_id    : (Nmax,)   int8     — raw trackId per stub (0 = noise)
  ambiguous   : (Nmax,)   uint8    — ambiguity flag per stub
  node_label  : (Nmax,)   float32  — 1.0 if stub is signal (trackId != 0)
  meta        : dict               — event_num, i_processor, n_stubs, n_gen_tracks

NanoAOD gen targets (always present; zeros when no nano data or noise-only)
---------------------------------------------------------------------------
  gen_pt      : (3,)  float32  — GenMuon_pt per track slot (0 if absent)
  gen_charge  : (3,)  float32  — GenMuon_charge per slot (+1/-1, 0 if absent)
  gen_dxy     : (3,)  float32  — GenMuon_dXY transverse impact parameter
  gen_phi     : (3,)  float32  — GenMuon_phi

Index k in these arrays corresponds to track_id = k+1 (the gen-muon-to-stub
assignment guaranteed by the data: trackId - 1 indexes GenMuon arrays).
All-zeros for B4 (noise-only) and windows with no NanoAOD match.

Optional graph output (requires include_graph=True)
----------------------------------------------------
  edge_index  : (2, E) int64
  edge_attr   : (E, 7) float32 — PAIR_FEATURE_NAMES order, kappa_hat clipped
  edge_label  : (E,)   float32 — 1.0 for same-track pair
  edge_ambig  : (E,)   bool

Usage
-----
    from omtf.dataset import OMTFDataset

    ds = OMTFDataset(
        files=["data/prod/S1/omtf_hits_S1_0001.root"],
        Nmax=24,
        include_graph=True,
    )
    sample = ds[0]
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omtf.features import (
    RAW_FEATURE_NAMES,
    compute_pair_features_np,
    clip_kappa_hat,
    build_edge_labels,
)

# Column indices in the stubs tensor
_COL = {name: i for i, name in enumerate(RAW_FEATURE_NAMES)}

# Default kappa_hat clip value (see DATASET_SIGNOFF.md §5)
_KAPPA_CLIP = 500.0
_GEN_K = 3  # max gen muons per window (S4: 3 prompt muons)


# ---------------------------------------------------------------------------
# Module-level helpers — must be at module scope to be picklable for Pool.map
# ---------------------------------------------------------------------------

def _process_entry_fn(
    entry: dict,
    nano_gen: dict | None,
    Nmax: int,
    include_graph: bool,
    kappa_clip: float,
) -> dict:
    """Process one ROOT entry into a sample dict. Pure function, no instance state."""
    n = entry["n_stubs"]

    phi     = np.array(entry["phi"],     dtype=np.int32)
    phiB    = np.array(entry["phiB"],    dtype=np.int32)
    eta     = np.array(entry["eta"],     dtype=np.int32)
    r       = np.array(entry["r"],       dtype=np.int32)
    quality = np.array(entry["quality"], dtype=np.int32)
    typ     = np.array(entry["type"],    dtype=np.int32)
    layer   = np.array(entry["layer"],   dtype=np.int32)
    track_id  = np.array(entry["track_id"],  dtype=np.int8)
    ambiguous = np.array(entry["ambiguous"], dtype=np.uint8)

    if n > Nmax:
        order = np.argsort(-quality)[:Nmax]
        phi, phiB, eta, r = phi[order], phiB[order], eta[order], r[order]
        quality, typ, layer = quality[order], typ[order], layer[order]
        track_id  = track_id[order]
        ambiguous = ambiguous[order]
        n = Nmax

    stubs = np.zeros((Nmax, len(RAW_FEATURE_NAMES)), dtype=np.float32)
    stubs[:n, _COL["phi"]]     = phi
    stubs[:n, _COL["phiB"]]    = phiB
    stubs[:n, _COL["eta"]]     = eta
    stubs[:n, _COL["r"]]       = r
    stubs[:n, _COL["quality"]] = quality
    stubs[:n, _COL["type"]]    = typ
    stubs[:n, _COL["layer"]]   = layer

    valid_mask = np.zeros(Nmax, dtype=bool)
    valid_mask[:n] = True

    tid_padded = np.zeros(Nmax, dtype=np.int8)
    tid_padded[:n] = track_id
    amb_padded = np.zeros(Nmax, dtype=np.uint8)
    amb_padded[:n] = ambiguous

    node_label = (tid_padded != 0).astype(np.float32)

    gen_pt     = np.zeros(_GEN_K, dtype=np.float32)
    gen_charge = np.zeros(_GEN_K, dtype=np.float32)
    gen_dxy    = np.zeros(_GEN_K, dtype=np.float32)
    gen_phi    = np.zeros(_GEN_K, dtype=np.float32)
    n_gen_tracks = 0

    if nano_gen is not None:
        pts     = nano_gen.get("pt",     [])
        charges = nano_gen.get("charge", [])
        dxys    = nano_gen.get("dxy",    [])
        phis    = nano_gen.get("phi",    [])
        n_gen_tracks = min(len(pts), _GEN_K)
        for k in range(n_gen_tracks):
            gen_pt[k]     = float(pts[k])
            gen_charge[k] = float(charges[k]) if k < len(charges) else 0.0
            gen_dxy[k]    = float(dxys[k])    if k < len(dxys)    else 0.0
            gen_phi[k]    = float(phis[k])     if k < len(phis)    else 0.0

    sample = {
        "stubs":      torch.from_numpy(stubs),
        "valid_mask": torch.from_numpy(valid_mask),
        "track_id":   torch.from_numpy(tid_padded),
        "ambiguous":  torch.from_numpy(amb_padded),
        "node_label": torch.from_numpy(node_label),
        "gen_pt":     torch.from_numpy(gen_pt),
        "gen_charge": torch.from_numpy(gen_charge),
        "gen_dxy":    torch.from_numpy(gen_dxy),
        "gen_phi":    torch.from_numpy(gen_phi),
        "meta": {
            "event_num":    entry["event_num"],
            "i_processor":  entry["i_processor"],
            "n_stubs":      n,
            "n_gen_tracks": n_gen_tracks,
        },
    }

    if include_graph:
        edge_index, edge_attr = compute_pair_features_np(
            phi=phi, phiB=phiB, eta=eta, r=r,
            valid_mask=valid_mask[:n],
            layer=layer,
            exclude_same_layer=True,
        )
        edge_attr = clip_kappa_hat(edge_attr, kappa_clip)

        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            elabel, eambig = build_edge_labels(
                track_id=tid_padded, src=src, dst=dst, ambiguous=amb_padded,
            )
        else:
            elabel = np.zeros(0, dtype=np.float32)
            eambig = np.zeros(0, dtype=bool)

        sample["edge_index"] = torch.from_numpy(edge_index)
        sample["edge_attr"]  = torch.from_numpy(edge_attr)
        sample["edge_label"] = torch.from_numpy(elabel)
        sample["edge_ambig"] = torch.from_numpy(eambig)

    return sample


def _load_file_worker(args: tuple) -> list:
    """Load and process all entries from one ROOT file. Module-level for Pool.map."""
    path_str, include_nano, include_graph, Nmax, kappa_clip, max_entries = args
    from audit.root_utils import open_hits_tree, read_entry, load_nano_event_map

    path = Path(path_str)
    f, t = open_hits_tree(path)
    if t is None:
        return []

    nano_map: dict = {}
    if include_nano:
        nano_path = path.parent / path.name.replace("omtf_hits_", "omtf_nano_")
        nano_map = load_nano_event_map(nano_path)

    n = int(t.GetEntries())
    if max_entries is not None:
        n = min(n, max_entries)

    results = []
    for ei in range(n):
        entry = read_entry(t, ei)
        nano_gen = nano_map.get(entry["event_num"])
        results.append(_process_entry_fn(entry, nano_gen, Nmax, include_graph, kappa_clip))
    f.Close()
    return results


class OMTFDataset(Dataset):
    """
    PyTorch Dataset over a list of ROOT hits files.

    All ROOT I/O happens lazily in __getitem__ to avoid holding TFile handles
    across DataLoader worker processes.  An entry index table is built once
    at construction time by scanning entry counts without reading branch data.

    Parameters
    ----------
    files        : list of ROOT hits file paths
    Nmax         : fixed stub budget per window (default 24)
    include_graph: if True, compute and return edge_index/edge_attr/edge_label
    kappa_clip   : clip value for kappa_hat pair feature (default 500)
    max_entries  : cap per file (for fast debug runs; None = all)
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        Nmax: int = 24,
        include_graph: bool = False,
        include_nano: bool = True,
        kappa_clip: float = _KAPPA_CLIP,
        max_entries: int | None = None,
        num_workers: int = 0,
    ):
        self.files = [Path(f) for f in files]
        self.Nmax = Nmax
        self.include_graph = include_graph
        self.include_nano = include_nano
        self.kappa_clip = kappa_clip
        self.max_entries = max_entries
        self._num_workers = num_workers

        # Pre-load and process all entries at construction time so each epoch
        # is a pure in-memory iteration (no ROOT I/O in __getitem__).
        self._data: list[dict] = []
        self._load_all()

    # ------------------------------------------------------------------
    # Eager loading — read each file once, process all entries
    # ------------------------------------------------------------------

    @staticmethod
    def _hits_to_nano_path(hits_path: Path) -> Path:
        return hits_path.parent / hits_path.name.replace("omtf_hits_", "omtf_nano_")

    def _load_all(self) -> None:
        args_list = [
            (str(p), self.include_nano, self.include_graph,
             self.Nmax, self.kappa_clip, self.max_entries)
            for p in self.files
        ]

        if self._num_workers > 0:
            n_workers = min(self._num_workers, len(self.files))
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for file_entries in executor.map(_load_file_worker, args_list):
                    self._data.extend(file_entries)
        else:
            for file_entries in map(_load_file_worker, args_list):
                self._data.extend(file_entries)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]

    # ------------------------------------------------------------------
    # Per-entry processing
    # ------------------------------------------------------------------

    def _process_entry(self, entry: dict, nano_gen: dict | None = None) -> dict:
        return _process_entry_fn(entry, nano_gen, self.Nmax, self.include_graph, self.kappa_clip)


# ---------------------------------------------------------------------------
# Collation helper for DataLoader
# ---------------------------------------------------------------------------

def collate_omtf(batch: list[dict]) -> dict:
    """
    Default collate for OMTFDataset batches.

    Stacks fixed-size tensors normally. For graph outputs (edge_index, edge_attr,
    edge_label, edge_ambig) — returns a list of per-sample tensors since edge counts
    vary. Models that need batched graphs should use torch_geometric's batching instead.
    """
    out: dict = {}

    # Fixed-size tensors: stack normally
    for key in ("stubs", "valid_mask", "track_id", "ambiguous", "node_label",
                "gen_pt", "gen_charge", "gen_dxy", "gen_phi"):
        if key in batch[0]:
            out[key] = torch.stack([s[key] for s in batch])

    # Meta: list of dicts
    out["meta"] = [s["meta"] for s in batch]

    # Variable-size graph tensors: keep as list
    for key in ("edge_index", "edge_attr", "edge_label", "edge_ambig"):
        if key in batch[0]:
            out[key] = [s[key] for s in batch]

    return out
