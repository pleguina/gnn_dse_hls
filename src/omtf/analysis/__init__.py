"""
OMTF dataset audit and decision analysis package.

Each phase corresponds to a section of docs/omtf/DATASET_DECISION_PLAN.md.
All phase functions read from a pre-built cache (build/omtf/cache/schema_v1_graph)
and work with any dataset that matches the schema produced by build_cache.py.

Quick usage
-----------
    from omtf.analysis import run_phase1, run_phase3_distribution
    from pathlib import Path
    cache = Path("build/omtf/cache/schema_v1_graph")
    p1 = run_phase1(cache, ["S1", "B1"])
    p3 = run_phase3_distribution(cache, ["S1", "B1"])
"""

from .phase0 import run_phase0, IntegrityResult
from .phase1 import run_phase1, OccupancyResult
from .phase2 import run_phase2, EdgeResult
from .phase3 import (
    run_phase3_distribution, PTDistResult,
    run_phase3_correlations, CorrResult,
    run_phase3_baseline, BaselineResult,
)
from .phase5 import run_phase5, B4AuditResult
from .phase6 import run_phase6, SlotOrderResult, MultiTrackResult
from .phase7 import run_phase7, DxyResult
from .report import render_report, save_json
from .loader import (
    ALL_DATASETS, SIGNAL_DATASETS, BACKGROUND_DATASETS,
    DISPLACED_DATASETS, MULTI_TRACK_DATASETS, PU200_DATASETS,
    available_datasets, has_graph, validate_schema,
)

__all__ = [
    "run_phase0", "IntegrityResult",
    "run_phase1", "OccupancyResult",
    "run_phase2", "EdgeResult",
    "run_phase3_distribution", "PTDistResult",
    "run_phase3_correlations", "CorrResult",
    "run_phase3_baseline", "BaselineResult",
    "run_phase5", "B4AuditResult",
    "run_phase6", "SlotOrderResult", "MultiTrackResult",
    "run_phase7", "DxyResult",
    "render_report", "save_json",
    "ALL_DATASETS", "SIGNAL_DATASETS", "BACKGROUND_DATASETS",
    "DISPLACED_DATASETS", "MULTI_TRACK_DATASETS", "PU200_DATASETS",
    "available_datasets", "has_graph", "validate_schema",
]
