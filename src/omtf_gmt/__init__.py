"""
omtf_gmt — GMT-visible-stub model branch (Branch B).

Parallel development line to the OMTF-internal branch.
Inference inputs: GMT-visible KMTF barrel stubs only.
Truth: transferred offline from OMTFAllInputTree during dataset construction.

Stage B1: barrel KMTF stubs, OMTF processor-region extraction.
"""

from .regioning      import (processor_phi_center, processor_phi_window,
                              stubs_in_window, phi_rel,
                              omtf_phi_to_global_rad, phiZero_hw,
                              NPROCESSORS, WINDOW_HALF, PHI_BINS)
from .features       import build_node_features, FEATURE_NAMES, N_FEATURES
from .truth_transfer import transfer, TransferResult, PHI_MATCH_THRESH
from .dataset        import GMTCachedDataset, collate_gmt

__all__ = [
    "processor_phi_center", "processor_phi_window",
    "stubs_in_window", "phi_rel",
    "omtf_phi_to_global_rad", "phiZero_hw", "NPROCESSORS", "WINDOW_HALF", "PHI_BINS",
    "build_node_features", "FEATURE_NAMES", "N_FEATURES",
    "transfer", "TransferResult", "PHI_MATCH_THRESH",
    "GMTCachedDataset", "collate_gmt",
]
