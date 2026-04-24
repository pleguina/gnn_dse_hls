"""
Phase 6 — SlotModel fairness analysis.

6.1  trackId ordering vs pT: is trackId=1 the highest-pT muon?
6.2  Multi-track confusion: phi separation, ambiguous stubs, layer conflicts

Relevant datasets: S3 (2 prompt µ), S4 (3 prompt µ), S5 (2 displaced µ), B3 (PU200 multi-µ)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .loader import iter_shards

_K_MAX = 3


@dataclass
class SlotOrderResult:
    dataset: str
    n_multi_track: int = 0      # windows with >= 2 gen tracks

    # trackId=1 vs highest-pT match
    n_tid1_is_highest_pt: int = 0  # fraction of multi-track windows where trackId=1 stub
                                    # belongs to the highest-pT gen muon

    # pT correlation: for 2-track windows, is pt[trackId=1] > pt[trackId=2]?
    n_two_track_windows: int = 0
    n_k1_higher_pt: int = 0        # trackId=1 muon has higher pT than trackId=2

    # phi ordering
    n_k1_lower_phi: int = 0        # trackId=1 muon has lower phi than trackId=2

    @property
    def tid1_highest_pt_pct(self) -> float:
        return 100.0 * self.n_tid1_is_highest_pt / max(1, self.n_multi_track)

    @property
    def k1_higher_pt_pct(self) -> float:
        return 100.0 * self.n_k1_higher_pt / max(1, self.n_two_track_windows)

    @property
    def k1_lower_phi_pct(self) -> float:
        return 100.0 * self.n_k1_lower_phi / max(1, self.n_two_track_windows)

    def ordering_decision(self) -> str:
        if self.n_two_track_windows < 10:
            return "INSUFFICIENT DATA"
        pt_corr = self.k1_higher_pt_pct
        # if close to 50%, ordering is random
        if abs(pt_corr - 50.0) < 10.0:
            return (
                f"ARBITRARY — trackId ordering uncorrelated with pT ({pt_corr:.1f}% k1>k2); "
                "SlotModel fixed-index supervision is unfair; try permutation-invariant matching"
            )
        elif pt_corr > 65.0:
            return f"ORDERED (pT-descending) — {pt_corr:.1f}% k1>k2; fixed-index may be fair"
        else:
            return f"PARTIALLY ORDERED — {pt_corr:.1f}% k1>k2; investigate generation convention"


@dataclass
class MultiTrackResult:
    dataset: str
    n_windows: int = 0
    n_two_track: int = 0

    # stub-level track separation
    mean_dphi_tracks: float = float("nan")    # mean |phi_track1 - phi_track2| for 2-track windows
    close_track_pct:  float = float("nan")    # % windows where |Δphi| < threshold (e.g. 50 hw units)

    # ambiguous stubs
    mean_ambig_pct: float = float("nan")      # mean fraction of stubs flagged ambiguous

    # same-layer conflicts between different tracks
    layer_conflict_pct: float = float("nan")  # % windows with >=1 same-layer pair from different tracks

    def confusion_decision(self) -> str:
        if self.n_two_track < 10:
            return "INSUFFICIENT DATA"
        close = self.close_track_pct
        if close is not None and close > 20.0:
            return f"CLOSE-TRACK COMMON ({close:.1f}% windows) — edge model needed for disambiguation"
        return f"TRACKS WELL SEPARATED ({close:.1f}% close) — simpler builder may be sufficient"


def run_phase6(
    cache_dir: Path,
    datasets: Sequence[str],
    close_track_threshold_hw: float = 50.0,
    verbose: bool = True,
) -> dict[str, tuple[SlotOrderResult, MultiTrackResult]]:
    results = {}

    for ds in datasets:
        so = SlotOrderResult(dataset=ds)
        mt = MultiTrackResult(dataset=ds)
        if verbose:
            print(f"  [phase6] {ds} ...", end="", flush=True)

        dphi_list: list[float] = []
        ambig_list: list[float] = []
        conflict_count: int = 0

        for shard in iter_shards(cache_dir, ds):
            vm     = shard["valid_mask"]          # (N, 24) bool
            tid    = shard["track_id"].long()     # (N, 24)
            stubs  = shard["stubs"]              # (N, 24, 7)
            gen_pt = shard["gen_pt"]             # (N, 3)
            mg     = shard["meta_n_gen"].long()  # (N,)
            ambig  = shard["ambiguous"]          # (N, 24) uint8

            N = int(vm.shape[0])
            mt.n_windows += N

            # multi-track windows
            multi_mask = mg >= 2
            so.n_multi_track += int(multi_mask.sum())

            # two-track windows
            two_mask = mg == 2
            so.n_two_track_windows += int(two_mask.sum())
            mt.n_two_track += int(two_mask.sum())

            if multi_mask.any():
                # trackId ordering vs pT  (for 2-track windows)
                two_idx = two_mask.nonzero(as_tuple=True)[0]
                if len(two_idx) > 0:
                    pt1 = gen_pt[two_idx, 0]    # pT of track with trackId=1
                    pt2 = gen_pt[two_idx, 1]    # pT of track with trackId=2
                    so.n_k1_higher_pt += int((pt1 > pt2).sum())

                    # phi ordering proxy: mean phi of stubs with trackId=1 vs trackId=2
                    for ii in two_idx.tolist():
                        vm_i   = vm[ii]                           # (24,)
                        tid_i  = tid[ii]                          # (24,)
                        phi_i  = stubs[ii, :, 0].numpy()         # (24,) phi col

                        k1_phi = phi_i[(tid_i == 1) & vm_i]
                        k2_phi = phi_i[(tid_i == 2) & vm_i]
                        if len(k1_phi) > 0 and len(k2_phi) > 0:
                            dphi = abs(float(np.mean(k1_phi)) - float(np.mean(k2_phi)))
                            dphi_list.append(dphi)
                            if dphi < close_track_threshold_hw:
                                pass   # counted below via dphi_list
                            if np.mean(k1_phi) < np.mean(k2_phi):
                                so.n_k1_lower_phi += 1

                # trackId=1 is highest-pT?
                multi_idx = multi_mask.nonzero(as_tuple=True)[0]
                pt_all = gen_pt[multi_idx]   # (M, 3)
                # highest pT slot index
                best_slot = pt_all.argmax(dim=1)   # (M,) — 0=trackId1, 1=trackId2, ...
                so.n_tid1_is_highest_pt += int((best_slot == 0).sum())

            # ambiguous stub fraction per window
            ambig_np  = ambig.float()      # (N, 24)
            vm_np     = vm.float()
            n_valid   = vm_np.sum(dim=1)   # (N,)
            n_ambig   = (ambig_np * vm_np).sum(dim=1)
            frac = (n_ambig / n_valid.clamp(min=1)).numpy()
            ambig_list.extend(frac.tolist())

            # same-layer conflicts between different tracks (for two-track windows)
            two_idx = two_mask.nonzero(as_tuple=True)[0]
            for ii in two_idx.tolist():
                vm_i  = vm[ii]
                tid_i = tid[ii]
                lay_i = stubs[ii, :, 6].long()    # col 6 = layer
                # layers of trackId=1 stubs
                lay_k1 = set(lay_i[(tid_i == 1) & vm_i].tolist())
                lay_k2 = set(lay_i[(tid_i == 2) & vm_i].tolist())
                if lay_k1 & lay_k2:
                    conflict_count += 1

        if dphi_list:
            mt.mean_dphi_tracks = float(np.mean(dphi_list))
            mt.close_track_pct  = float(100.0 * np.mean(np.array(dphi_list) < close_track_threshold_hw))
        if ambig_list:
            mt.mean_ambig_pct = float(100.0 * np.mean(ambig_list))
        if mt.n_two_track > 0:
            mt.layer_conflict_pct = float(100.0 * conflict_count / mt.n_two_track)

        results[ds] = (so, mt)
        if verbose:
            k1_pct = so.k1_higher_pt_pct
            print(
                f" {mt.n_windows:,} — 2-track={mt.n_two_track:,}  "
                f"k1_higher_pt={k1_pct:.1f}%  "
                f"mean_dphi={mt.mean_dphi_tracks:.1f}"
            )

    return results
