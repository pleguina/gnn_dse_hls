"""
Shard-level data access for the dataset audit pipeline.

Iterates over pre-built .pt shards without loading all data into memory at once.
Works with any cache produced by build_cache.py that matches the schema defined
in cache_dataset.py (schema_version == 1).  New datasets with the same schema
work automatically — add them to the cache and pass their name here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch

from omtf.features import RAW_FEATURE_NAMES, PAIR_FEATURE_NAMES

SCHEMA_VERSION = 1

ALL_DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]
SIGNAL_DATASETS = ["S1", "S2", "S3", "S4", "S5"]
BACKGROUND_DATASETS = ["B1", "B2", "B3", "B4"]
DISPLACED_DATASETS = ["S2", "S5", "B2"]
MULTI_TRACK_DATASETS = ["S3", "S4", "S5", "B3"]
PU200_DATASETS = ["B1", "B2", "B3"]


def load_manifest(cache_dir: Path) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest.json in {cache_dir}")
    return json.loads(path.read_text())


def validate_schema(cache_dir: Path) -> None:
    """Raise if the cache schema is incompatible with the current feature definitions."""
    m = load_manifest(cache_dir)
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema version {m.get('schema_version')} != expected {SCHEMA_VERSION}"
        )
    if m.get("raw_feature_names") != RAW_FEATURE_NAMES:
        raise ValueError(
            f"Cache raw features {m.get('raw_feature_names')} != {RAW_FEATURE_NAMES}"
        )
    if m.get("pair_feature_names") != PAIR_FEATURE_NAMES:
        raise ValueError(
            f"Cache pair features {m.get('pair_feature_names')} != {PAIR_FEATURE_NAMES}"
        )


def has_graph(cache_dir: Path) -> bool:
    return load_manifest(cache_dir).get("include_graph", False)


def available_datasets(cache_dir: Path) -> list[str]:
    return list(load_manifest(cache_dir).get("datasets", {}).keys())


def dataset_n_samples(cache_dir: Path, dataset: str) -> int:
    return load_manifest(cache_dir)["datasets"][dataset]["n_samples"]


def iter_shards(
    cache_dir: Path,
    dataset: str,
    device: str = "cpu",
) -> Iterator[dict[str, object]]:
    """Yield one dict of stacked tensors per shard.

    Fixed-size fields (stubs, valid_mask, …) are stacked tensors of shape
    (N, …).  Variable-length graph fields (edge_index, edge_attr, edge_label,
    edge_ambig) are Python lists of per-sample tensors when present.
    """
    ds_dir = Path(cache_dir) / dataset
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset directory missing: {ds_dir}")
    shards = sorted(ds_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards in {ds_dir}")
    for path in shards:
        yield torch.load(path, map_location=device, weights_only=False)
