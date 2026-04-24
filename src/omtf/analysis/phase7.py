"""
Phase 7 — dXY visibility analysis.

Checks whether displacement (GenMuon_dXY) is detectable from current stub
features: phiB, phiB residual vs expected prompt, kappa_hat, layer pattern.

Only meaningful for displaced datasets: S2, S5, B2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .loader import iter_shards, has_graph


@dataclass
class DxyResult:
    dataset: str
    n_single_track: int = 0
    dxy_values: np.ndarray = field(default=None, repr=False)

    # Pearson correlations between dXY and per-window stub features
    pearson: dict[str, float] = field(default_factory=dict)
    spearman: dict[str, float] = field(default_factory=dict)

    def dxy_stats(self) -> dict:
        d = self.dxy_values
        if d is None or len(d) == 0:
            return {}
        return {
            "mean":  float(np.mean(d)),
            "p50":   float(np.percentile(d, 50)),
            "p90":   float(np.percentile(d, 90)),
            "p99":   float(np.percentile(d, 99)),
            "max":   float(np.max(d)),
        }

    def best_feature(self) -> str:
        if not self.pearson:
            return "N/A"
        return max(self.pearson, key=lambda k: abs(self.pearson.get(k, 0) or 0))

    def visibility_decision(self) -> str:
        if not self.pearson:
            return "NO DATA"
        best = self.best_feature()
        r = abs(self.pearson.get(best, 0) or 0)
        if r >= 0.2:
            return f"VISIBLE — {best} correlates with dXY (|r|={r:.3f}); dXY head may learn with reweighting"
        elif r >= 0.05:
            return f"WEAK — marginal signal (|r|={r:.3f}); consider phiB residual or displacement-specific features"
        return f"NOT VISIBLE — no feature correlates with dXY (|r|={r:.3f}); redesign features before training dXY head"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def run_phase7(
    cache_dir: Path,
    datasets: Sequence[str] = ("S2", "S5", "B2"),
    verbose: bool = True,
) -> dict[str, DxyResult]:
    gp = has_graph(cache_dir)
    results: dict[str, DxyResult] = {}

    for ds in datasets:
        r = DxyResult(dataset=ds)
        if verbose:
            print(f"  [phase7] {ds} ...", end="", flush=True)

        dxy_vals:  list[float] = []
        feat_dict: dict[str, list[float]] = {
            "mean_phiB_signal":   [],
            "mean_abs_phiB":      [],
            "n_signal_stubs":     [],
            "mean_r_signal":      [],
            "mean_quality":       [],
            "mean_eta_signal":    [],
        }
        if gp:
            feat_dict["best_abs_kappa"] = []
            feat_dict["mean_abs_kappa"] = []

        for shard in iter_shards(cache_dir, ds):
            import torch as _t
            vm      = shard["valid_mask"]         # (N, 24)
            tid     = shard["track_id"].long()    # (N, 24)
            stubs   = shard["stubs"]             # (N, 24, 7)
            gen_dxy = shard["gen_dxy"]           # (N, 3)
            gen_pt  = shard["gen_pt"]            # (N, 3)
            K_MAX   = gen_pt.shape[1]

            # stub-based single-track filter (same convention as phase3)
            k_pres = _t.stack([((tid == k) & vm).any(dim=1) for k in range(1, K_MAX + 1)], dim=1)
            n_tr   = k_pres.long().sum(dim=1)
            single = n_tr == 1
            active = k_pres.long().argmax(dim=1)   # 0-based slot index
            if not single.any():
                continue

            idx   = single.nonzero(as_tuple=True)[0]
            slots = active[idx]
            pts   = _t.gather(gen_pt[idx], 1, slots.unsqueeze(1)).squeeze(1)
            valid_pt = pts > 0
            if not valid_pt.any():
                continue

            idx2   = idx[valid_pt]
            slots2 = slots[valid_pt]
            vm_s    = vm[idx2]
            tid_s   = tid[idx2]
            stubs_s = stubs[idx2]
            dxy_s   = _t.gather(gen_dxy[idx2], 1, slots2.unsqueeze(1)).squeeze(1).numpy()

            r.n_single_track += int(len(idx2))
            dxy_vals.extend(dxy_s.tolist())

            # signal stub mask: trackId == slot+1 for each window
            sig_k    = slots2.unsqueeze(1) + 1
            sig_mask = ((tid_s == sig_k) & vm_s).float()   # (M, 24)
            sig_count = sig_mask.sum(dim=1).numpy()     # (M,)
            denom = np.maximum(sig_count, 1.0)

            weighted = (stubs_s * sig_mask.unsqueeze(-1)).sum(dim=1).numpy()  # (M, 7)
            # cols: phi=0, phiB=1, eta=2, r=3, quality=4, type=5, layer=6
            feat_dict["mean_phiB_signal"].extend((weighted[:, 1] / denom).tolist())
            feat_dict["mean_abs_phiB"].extend((np.abs(weighted[:, 1]) / denom).tolist())
            feat_dict["n_signal_stubs"].extend(sig_count.tolist())
            feat_dict["mean_r_signal"].extend((weighted[:, 3] / denom).tolist())
            feat_dict["mean_quality"].extend((weighted[:, 4] / denom).tolist())
            feat_dict["mean_eta_signal"].extend((np.abs(weighted[:, 2]) / denom).tolist())

            if gp and "edge_attr" in shard and "edge_label" in shard:
                ea_list = shard["edge_attr"]
                el_list = shard["edge_label"]
                for j, orig_i in enumerate(idx2.tolist()):
                    ea = ea_list[orig_i]
                    el = el_list[orig_i]
                    if ea.shape[0] == 0:
                        feat_dict["best_abs_kappa"].append(0.0)
                        feat_dict["mean_abs_kappa"].append(0.0)
                        continue
                    pos = el.numpy() > 0.5
                    kappa = ea[:, 3].numpy()
                    if pos.sum() > 0:
                        feat_dict["best_abs_kappa"].append(float(np.max(np.abs(kappa[pos]))))
                        feat_dict["mean_abs_kappa"].append(float(np.mean(np.abs(kappa[pos]))))
                    else:
                        feat_dict["best_abs_kappa"].append(0.0)
                        feat_dict["mean_abs_kappa"].append(0.0)

        r.dxy_values = np.asarray(dxy_vals, dtype=np.float32)

        if len(dxy_vals) > 10:
            dxy_arr = np.asarray(dxy_vals)
            for fname, vals in feat_dict.items():
                if len(vals) != len(dxy_arr):
                    continue
                v = np.asarray(vals)
                if v.std() < 1e-9:
                    r.pearson[fname]  = float("nan")
                    r.spearman[fname] = float("nan")
                else:
                    r.pearson[fname]  = float(np.corrcoef(v, dxy_arr)[0, 1])
                    r.spearman[fname] = _spearman(v, dxy_arr)

        results[ds] = r
        if verbose:
            best = r.best_feature()
            pr   = abs(r.pearson.get(best, float("nan")) or float("nan"))
            dxy_p50 = float(np.median(r.dxy_values)) if len(dxy_vals) > 0 else float("nan")
            print(f" {r.n_single_track:,} — median_dXY={dxy_p50:.1f}cm  best={best}(|r|={pr:.3f})")

    return results
