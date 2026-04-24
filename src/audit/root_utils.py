"""
Shared ROOT reading utilities for OMTF scripts.

Supports two backends:
1) PyROOT (if available)
2) uproot fallback (default in environments without ROOT bindings)

ROOT's Python bindings may return `str` of length 1 for C++ char/UChar_t values.
These helpers normalize values to proper Python ints with correct sign handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TREE_PATH = "simOmtfPhase2Digis/OMTFAllInputTree"
NANO_TREE = "Events"


def set_root_batch_mode() -> None:
    """Enable ROOT batch mode when PyROOT is available; no-op otherwise."""
    try:
        import ROOT
        ROOT.gROOT.SetBatch(True)
    except Exception:
        pass


def schar(v) -> int:
    """Convert ROOT signed char (str of length 1) to Python int [-128, 127]."""
    x = ord(v) if isinstance(v, str) else int(v)
    return x if x < 128 else x - 256


def uchar(v) -> int:
    """Convert ROOT unsigned char (str of length 1) to Python int [0, 255]."""
    return ord(v) if isinstance(v, str) else int(v)


def read_vec_schar(vec) -> list:
    """Read a signed-char vector-like branch into a list of ints."""
    return [schar(v) for v in vec]


def read_vec_uchar(vec) -> list:
    """Read an unsigned-char vector-like branch into a list of ints."""
    return [uchar(v) for v in vec]


def read_vec_short(vec) -> list:
    """Read a short vector-like branch into a Python list of ints."""
    return [int(v) for v in vec]


class _UprootFileAdapter:
    """Minimal adapter exposing Close() like ROOT TFile."""

    def __init__(self, fh):
        self._fh = fh

    def Close(self):
        self._fh.close()


class _UprootTreeAdapter:
    """Minimal adapter exposing GetEntries/GetEntry + branch attrs like TTree."""

    def __init__(self, tree):
        self._tree = tree
        self._num_entries = int(tree.num_entries)

    def GetEntries(self):
        return self._num_entries

    def GetEntry(self, i: int):
        arrays = self._tree.arrays(entry_start=i, entry_stop=i + 1, library="np")
        for name, values in arrays.items():
            value = values[0]
            if isinstance(value, np.generic):
                value = value.item()
            setattr(self, name, value)
        return 1


def _open_with_pyroot(path, tree_path):
    import ROOT

    f = ROOT.TFile.Open(str(path))
    if not f or f.IsZombie():
        return None, None
    t = f.Get(tree_path)
    if not t:
        f.Close()
        return None, None
    return f, t


def _open_with_uproot(path, tree_path):
    import uproot

    f = uproot.open(str(path))
    if tree_path not in f:
        f.close()
        return None, None
    t = f[tree_path]
    return _UprootFileAdapter(f), _UprootTreeAdapter(t)


def open_hits_tree(path):
    """Open a hits file and return (File, Tree) or (None, None) on failure."""
    path = Path(path)
    try:
        return _open_with_pyroot(path, TREE_PATH)
    except Exception:
        return _open_with_uproot(path, TREE_PATH)


def open_nano_tree(path):
    """Open a nano file and return (File, Tree) or (None, None) on failure."""
    path = Path(path)
    try:
        return _open_with_pyroot(path, NANO_TREE)
    except Exception:
        return _open_with_uproot(path, NANO_TREE)


def _vec_len(vec) -> int:
    if hasattr(vec, "size") and callable(vec.size):
        return int(vec.size())
    return len(vec)


def load_nano_event_map(path) -> dict:
    """
    Read all GenMuon entries from a NanoAOD file into an in-memory event map.

    Returns {event_num(uint32): {'pt': list, 'charge': list, 'dxy': list, 'phi': list}}
    Returns {} if the file does not exist or cannot be opened.
    """
    path = Path(path)
    if not path.exists():
        return {}
    f, t = open_nano_tree(path)
    if t is None:
        return {}

    n = int(t.GetEntries())
    if n == 0:
        f.Close()
        return {}

    # Probe optional branches on first entry
    t.GetEntry(0)
    has_dxy = hasattr(t, "GenMuon_dXY")
    has_phi = hasattr(t, "GenMuon_phi")

    event_map: dict = {}
    for i in range(n):
        t.GetEntry(i)
        key = int(t.event) & 0xFFFFFFFF
        n_gen = int(t.nGenMuon)
        if n_gen == 0:
            event_map[key] = {"pt": [], "charge": [], "dxy": [], "phi": []}
            continue
        event_map[key] = {
            "pt":     [float(t.GenMuon_pt[k])     for k in range(n_gen)],
            "charge": [int(t.GenMuon_charge[k])   for k in range(n_gen)],
            "dxy":    [float(t.GenMuon_dXY[k])    for k in range(n_gen)] if has_dxy else [0.0] * n_gen,
            "phi":    [float(t.GenMuon_phi[k])    for k in range(n_gen)] if has_phi else [0.0] * n_gen,
        }

    f.Close()
    return event_map


def read_entry(t, i: int) -> dict:
    """
    Read one entry from OMTFAllInputTree and return a dict of Python lists.
    All signed char vectors are sign-corrected.
    """
    t.GetEntry(i)
    n = _vec_len(t.reg_stub_phiHw)
    return {
        "event_num": int(t.reg_eventNum),
        "i_processor": uchar(t.reg_iProcessor),
        "mtf_type": schar(t.reg_mtfType),
        "n_stubs": n,
        "phi":     read_vec_short(t.reg_stub_phiHw),
        "phiB":    read_vec_short(t.reg_stub_phiBHw),
        "r":       read_vec_short(t.reg_stub_r),
        "eta":     read_vec_schar(t.reg_stub_etaHw),
        "quality": read_vec_schar(t.reg_stub_quality),
        "type":    read_vec_schar(t.reg_stub_type),
        "layer":   read_vec_schar(t.reg_stub_layer),
        "bx":      read_vec_schar(t.reg_stub_bx),
        "track_id":  read_vec_schar(t.reg_stub_trackId),
        "ambiguous": read_vec_uchar(t.reg_stub_ambiguous),
    }
