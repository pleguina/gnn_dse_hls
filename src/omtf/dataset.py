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
  meta        : dict               — event_num, i_processor, n_stubs (Python scalars)

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
        kappa_clip: float = _KAPPA_CLIP,
        max_entries: int | None = None,
    ):
        self.files = [Path(f) for f in files]
        self.Nmax = Nmax
        self.include_graph = include_graph
        self.kappa_clip = kappa_clip
        self.max_entries = max_entries

        # Build flat index: list of (file_idx, entry_idx_within_file)
        self._index: list[tuple[int, int]] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction (no branch reads — entry counts only)
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        import ROOT
        from audit.root_utils import open_hits_tree

        for fi, path in enumerate(self.files):
            f, t = open_hits_tree(path)
            if t is None:
                continue
            n = int(t.GetEntries())
            if self.max_entries is not None:
                n = min(n, self.max_entries)
            for ei in range(n):
                self._index.append((fi, ei))
            f.Close()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        import ROOT
        from audit.root_utils import open_hits_tree, read_entry

        fi, ei = self._index[idx]
        f, t = open_hits_tree(self.files[fi])
        entry = read_entry(t, ei)
        f.Close()

        return self._process_entry(entry)

    # ------------------------------------------------------------------
    # Per-entry processing
    # ------------------------------------------------------------------

    def _process_entry(self, entry: dict) -> dict:
        n = entry["n_stubs"]

        # --- Raw arrays (Python lists → numpy) ---
        phi     = np.array(entry["phi"],     dtype=np.int32)
        phiB    = np.array(entry["phiB"],    dtype=np.int32)
        eta     = np.array(entry["eta"],     dtype=np.int32)
        r       = np.array(entry["r"],       dtype=np.int32)
        quality = np.array(entry["quality"], dtype=np.int32)
        typ     = np.array(entry["type"],    dtype=np.int32)
        layer   = np.array(entry["layer"],   dtype=np.int32)
        # bx is read from ROOT but NOT stored in the stub tensor — it is
        # identically 0 in all current samples (uninformative).
        track_id   = np.array(entry["track_id"],  dtype=np.int8)
        ambiguous  = np.array(entry["ambiguous"], dtype=np.uint8)

        # --- Truncation: keep top-quality stubs when n > Nmax ---
        if n > self.Nmax:
            order = np.argsort(-quality)[:self.Nmax]
            phi, phiB, eta, r = phi[order], phiB[order], eta[order], r[order]
            quality, typ, layer = quality[order], typ[order], layer[order]
            track_id   = track_id[order]
            ambiguous  = ambiguous[order]
            n = self.Nmax

        # --- Build stub tensor (Nmax, 7) float32 with zero padding ---
        stubs = np.zeros((self.Nmax, len(RAW_FEATURE_NAMES)), dtype=np.float32)
        stubs[:n, _COL["phi"]]     = phi
        stubs[:n, _COL["phiB"]]    = phiB
        stubs[:n, _COL["eta"]]     = eta
        stubs[:n, _COL["r"]]       = r
        stubs[:n, _COL["quality"]] = quality
        stubs[:n, _COL["type"]]    = typ
        stubs[:n, _COL["layer"]]   = layer

        valid_mask = np.zeros(self.Nmax, dtype=bool)
        valid_mask[:n] = True

        # Padded track_id and ambiguous arrays
        tid_padded = np.zeros(self.Nmax, dtype=np.int8)
        tid_padded[:n] = track_id
        amb_padded = np.zeros(self.Nmax, dtype=np.uint8)
        amb_padded[:n] = ambiguous

        node_label = (tid_padded != 0).astype(np.float32)

        sample = {
            "stubs":      torch.from_numpy(stubs),
            "valid_mask": torch.from_numpy(valid_mask),
            "track_id":   torch.from_numpy(tid_padded),
            "ambiguous":  torch.from_numpy(amb_padded),
            "node_label": torch.from_numpy(node_label),
            "meta": {
                "event_num":   entry["event_num"],
                "i_processor": entry["i_processor"],
                "n_stubs":     n,
            },
        }

        if self.include_graph:
            edge_index, edge_attr = compute_pair_features_np(
                phi=phi, phiB=phiB, eta=eta, r=r,
                valid_mask=valid_mask[:n],
            )
            edge_attr = clip_kappa_hat(edge_attr, self.kappa_clip)

            # Build edge labels from padded tid/amb arrays
            if edge_index.shape[1] > 0:
                src, dst = edge_index[0], edge_index[1]
                # src/dst index into the real (unpadded) stubs; tid/amb already sliced
                elabel, eambig = build_edge_labels(
                    track_id=tid_padded,
                    src=src,
                    dst=dst,
                    ambiguous=amb_padded,
                )
            else:
                elabel = np.zeros(0, dtype=np.float32)
                eambig = np.zeros(0, dtype=bool)

            sample["edge_index"] = torch.from_numpy(edge_index)
            sample["edge_attr"]  = torch.from_numpy(edge_attr)
            sample["edge_label"] = torch.from_numpy(elabel)
            sample["edge_ambig"] = torch.from_numpy(eambig)

        return sample


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
    for key in ("stubs", "valid_mask", "track_id", "ambiguous", "node_label"):
        if key in batch[0]:
            out[key] = torch.stack([s[key] for s in batch])

    # Meta: list of dicts
    out["meta"] = [s["meta"] for s in batch]

    # Variable-size graph tensors: keep as list
    for key in ("edge_index", "edge_attr", "edge_label", "edge_ambig"):
        if key in batch[0]:
            out[key] = [s[key] for s in batch]

    return out
