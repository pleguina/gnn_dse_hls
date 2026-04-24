"""
Phase 1 — Occupancy and firmware bounds.

1.1  Raw stub occupancy per dataset (percentiles, % > Nmax)
1.2  Signal / noise / track-multiplicity breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .loader import iter_shards

_K_MAX = 3   # maximum gen muon slots


@dataclass
class OccupancyResult:
    dataset: str
    n_entries: int = 0
    _n_stubs:  list = field(default_factory=list, repr=False)
    _n_signal: list = field(default_factory=list, repr=False)
    _n_noise:  list = field(default_factory=list, repr=False)
    _n_tracks: list = field(default_factory=list, repr=False)

    # filled by finalize()
    n_stubs:  np.ndarray = field(default=None, repr=False)
    n_signal: np.ndarray = field(default=None, repr=False)
    n_noise:  np.ndarray = field(default=None, repr=False)
    n_tracks: np.ndarray = field(default=None, repr=False)

    def finalize(self) -> None:
        self.n_stubs  = np.asarray(self._n_stubs,  dtype=np.int32)
        self.n_signal = np.asarray(self._n_signal, dtype=np.int32)
        self.n_noise  = np.asarray(self._n_noise,  dtype=np.int32)
        self.n_tracks = np.asarray(self._n_tracks, dtype=np.int32)

    def occupancy_stats(self, nmax: int = 24) -> dict:
        s = self.n_stubs
        return {
            "mean":    float(np.mean(s)),
            "p50":     float(np.percentile(s, 50)),
            "p90":     float(np.percentile(s, 90)),
            "p95":     float(np.percentile(s, 95)),
            "p99":     float(np.percentile(s, 99)),
            "max":     int(np.max(s)),
            "pct_over_nmax": float(100.0 * np.mean(s > nmax)),
        }

    def multiplicity_stats(self) -> dict:
        sig = self.n_signal
        return {
            "mean_total":      float(np.mean(self.n_stubs)),
            "mean_signal":     float(np.mean(sig)),
            "mean_noise":      float(np.mean(self.n_noise)),
            "zero_signal_pct": float(100.0 * np.mean(sig == 0)),
            "one_track_pct":   float(100.0 * np.mean(self.n_tracks == 1)),
            "two_track_pct":   float(100.0 * np.mean(self.n_tracks == 2)),
            "three_track_pct": float(100.0 * np.mean(self.n_tracks >= 3)),
        }

    def nmax_decision(self, nmax: int = 24) -> str:
        pct = self.occupancy_stats(nmax)["pct_over_nmax"]
        if pct < 0.01:
            return f"PASS — Nmax={nmax} covers 100% of windows"
        elif pct < 1.0:
            return f"WARN — {pct:.2f}% windows exceed Nmax={nmax}; check truncation"
        else:
            return f"FAIL — {pct:.2f}% windows exceed Nmax={nmax}; increase or change policy"


def run_phase1(
    cache_dir: Path,
    datasets: Sequence[str],
    nmax: int = 24,
    verbose: bool = True,
) -> dict[str, OccupancyResult]:
    results: dict[str, OccupancyResult] = {}

    for ds in datasets:
        r = OccupancyResult(dataset=ds)
        if verbose:
            print(f"  [phase1] {ds} ...", end="", flush=True)

        for shard in iter_shards(cache_dir, ds):
            vm  = shard["valid_mask"]         # (N, 24) bool
            tid = shard["track_id"].long()    # (N, 24)
            N   = int(vm.shape[0])
            r.n_entries += N

            n_stubs = vm.sum(dim=1)                         # (N,)
            signal_mask = (tid != 0) & vm
            n_signal = signal_mask.sum(dim=1)               # (N,)
            n_noise  = n_stubs - n_signal                   # (N,)

            # count unique nonzero trackIds using K=3 known slots (vectorized)
            n_tracks = sum(
                ((tid == k) & vm).any(dim=1).long()
                for k in range(1, _K_MAX + 1)
            )                                               # (N,)

            r._n_stubs.extend(n_stubs.tolist())
            r._n_signal.extend(n_signal.tolist())
            r._n_noise.extend(n_noise.tolist())
            r._n_tracks.extend(n_tracks.tolist())

        r.finalize()
        results[ds] = r

        if verbose:
            occ = r.occupancy_stats(nmax)
            mult = r.multiplicity_stats()
            print(
                f" {r.n_entries:,} — mean_stubs={occ['mean']:.1f}  "
                f"signal%={100.0 - mult['zero_signal_pct']:.1f}  "
                f">{nmax}%={occ['pct_over_nmax']:.2f}"
            )

    return results
