"""
File-level train/val/test split generation for OMTF ROOT datasets.

Splits are generated deterministically at file granularity (not event granularity)
to avoid data leakage between events from the same run/file.

Usage
-----
    from omtf.splits import build_splits, load_split

    splits = build_splits("data/prod", datasets=["S1", "B1", "B4"],
                          train_frac=0.7, val_frac=0.15, seed=42)
    # splits["train"] -> list of Path
    save_splits(splits, "build/splits/run0.json")
    splits = load_splits("build/splits/run0.json")
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


def _discover_files(data_root: str | Path, datasets: Sequence[str]) -> list[Path]:
    root = Path(data_root)
    files = []
    for ds in datasets:
        d = root / ds
        if not d.exists():
            continue
        found = sorted(d.glob(f"omtf_hits_{ds}_*.root"))
        files.extend(found)
    return files


def build_splits(
    data_root: str | Path,
    datasets: Sequence[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """
    Randomly shuffle all ROOT hits files for the given datasets and split
    into train/val/test at file level.

    test_frac is derived as (1 - train_frac - val_frac).
    """
    assert 0 < train_frac < 1
    assert 0 < val_frac < 1
    assert train_frac + val_frac < 1

    files = _discover_files(data_root, datasets)
    if not files:
        raise FileNotFoundError(
            f"No ROOT hits files found for datasets {datasets} under {data_root}"
        )

    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))
    # test gets the remainder
    n_test  = max(0, n - n_train - n_val)

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }


def save_splits(splits: dict[str, list[Path]], output_path: str | Path) -> None:
    """Persist splits to JSON (paths stored as strings)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: [str(p) for p in v] for k, v in splits.items()}
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)


def load_splits(path: str | Path) -> dict[str, list[Path]]:
    """Load splits from a JSON file produced by save_splits."""
    with open(path) as f:
        raw = json.load(f)
    return {k: [Path(p) for p in v] for k, v in raw.items()}
