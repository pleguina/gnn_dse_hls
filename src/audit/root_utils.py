"""
Shared ROOT reading utilities for OMTF audit scripts.

ROOT's Python bindings return `str` of length 1 for C++ char/UChar_t values.
These helpers convert them to proper Python ints with correct sign handling.
"""

TREE_PATH = "simOmtfPhase2Digis/OMTFAllInputTree"
NANO_TREE = "Events"


def schar(v) -> int:
    """Convert ROOT signed char (str of length 1) to Python int [-128, 127]."""
    x = ord(v) if isinstance(v, str) else int(v)
    return x if x < 128 else x - 256


def uchar(v) -> int:
    """Convert ROOT unsigned char (str of length 1) to Python int [0, 255]."""
    return ord(v) if isinstance(v, str) else int(v)


def read_vec_schar(vec) -> list:
    """Read a ROOT vector<signed char> branch into a Python list of ints."""
    return [schar(vec[j]) for j in range(vec.size())]


def read_vec_uchar(vec) -> list:
    """Read a ROOT vector<unsigned char> branch into a Python list of ints."""
    return [uchar(vec[j]) for j in range(vec.size())]


def read_vec_short(vec) -> list:
    """Read a ROOT vector<short> branch into a Python list of ints."""
    return [int(vec[j]) for j in range(vec.size())]


def open_hits_tree(path):
    """Open a hits file and return (TFile, TTree) or (None, None) on failure."""
    import ROOT
    f = ROOT.TFile.Open(str(path))
    if not f or f.IsZombie():
        return None, None
    t = f.Get(TREE_PATH)
    if not t:
        f.Close()
        return None, None
    return f, t


def open_nano_tree(path):
    """Open a nano file and return (TFile, TTree) or (None, None) on failure."""
    import ROOT
    f = ROOT.TFile.Open(str(path))
    if not f or f.IsZombie():
        return None, None
    t = f.Get(NANO_TREE)
    if not t:
        f.Close()
        return None, None
    return f, t


def read_entry(t, i: int) -> dict:
    """
    Read one entry from OMTFAllInputTree and return a dict of Python lists.
    All signed char vectors are sign-corrected.
    """
    t.GetEntry(i)
    n = t.reg_stub_phiHw.size()
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
