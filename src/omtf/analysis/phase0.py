"""
Phase 0 — Dataset trustworthiness.

0.1  Branch integrity: checks internal cache consistency
       - meta_n_stubs == valid_mask.sum()
       - node_label == (track_id != 0) for valid stubs
       - padding rows are all-zero in stubs tensor

0.2  NanoAOD join completeness:
       - fraction of windows with at least one populated gen_pt slot
       - fraction of windows where filled gen slots == meta_n_gen
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .loader import iter_shards


@dataclass
class IntegrityResult:
    dataset: str
    n_entries: int = 0
    bad_nstubs: int = 0          # meta_n_stubs != valid_mask.sum()
    bad_label: int = 0           # node_label inconsistent with track_id
    bad_padding: int = 0         # padding rows with non-zero stub values
    nano_matched: int = 0        # windows with >= 1 filled gen_pt slot
    nano_unmatched: int = 0      # windows with no filled gen_pt slot
    nano_partial: int = 0        # filled slots < meta_n_gen (join gap)

    @property
    def bad_pct(self) -> float:
        n = max(1, self.n_entries)
        return 100.0 * (self.bad_nstubs + self.bad_label + self.bad_padding) / n

    @property
    def nano_matched_pct(self) -> float:
        return 100.0 * self.nano_matched / max(1, self.n_entries)

    @property
    def nano_partial_pct(self) -> float:
        return 100.0 * self.nano_partial / max(1, self.n_entries)

    def decision(self) -> str:
        if self.bad_pct > 0:
            return f"FAIL — {self.bad_pct:.2f}% bad entries; fix dataset before training"
        return "PASS"


def run_phase0(
    cache_dir: Path,
    datasets: Sequence[str],
    verbose: bool = True,
) -> dict[str, IntegrityResult]:
    results: dict[str, IntegrityResult] = {}
    for ds in datasets:
        r = IntegrityResult(dataset=ds)
        if verbose:
            print(f"  [phase0] {ds} ...", end="", flush=True)

        for shard in iter_shards(cache_dir, ds):
            vm     = shard["valid_mask"]         # (N, 24) bool
            tid    = shard["track_id"].long()    # (N, 24)
            nl     = shard["node_label"]         # (N, 24) float
            stubs  = shard["stubs"]              # (N, 24, 7) float
            mn     = shard["meta_n_stubs"]       # (N,) int32
            mg     = shard["meta_n_gen"]         # (N,) int32
            gen_pt = shard["gen_pt"]             # (N, 3) float

            N = int(vm.shape[0])
            r.n_entries += N

            # 0.1a meta_n_stubs must equal valid_mask column sum
            vm_sum = vm.sum(dim=1).int()
            r.bad_nstubs += int((vm_sum != mn).sum())

            # 0.1b node_label == (track_id != 0) masked to valid stubs
            expected = (tid != 0).float() * vm.float()
            r.bad_label += int(((nl - expected).abs() > 0.01).any(dim=1).sum())

            # 0.1c padding rows (valid_mask==False) must have zero stubs
            pad_mask = (~vm).unsqueeze(-1).float()          # (N, 24, 1)
            r.bad_padding += int((stubs.abs() * pad_mask > 1e-6).any(dim=-1).any(dim=-1).sum())

            # 0.2 NanoAOD join completeness
            filled = (gen_pt > 0).sum(dim=1).int()          # (N,) slots with real pT
            has_any = (filled > 0)
            r.nano_matched    += int(has_any.sum())
            r.nano_unmatched  += int((~has_any).sum())
            r.nano_partial    += int((filled < mg).sum())

        results[ds] = r
        if verbose:
            status = r.decision()
            print(f" {r.n_entries:,} entries — {status}")

    return results
