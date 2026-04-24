"""
Phase 2 — Edge problem strength.

2.1  Legal edge statistics: counts, class balance, windows with no positive edge
2.2  Feature separability: per-feature AUC (same-track vs different-track)

Requires a graph cache (include_graph=True).  Returns None for each dataset
if graph data is absent from the shard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .loader import iter_shards, has_graph
from omtf.features import PAIR_FEATURE_NAMES


@dataclass
class EdgeResult:
    dataset: str
    n_entries: int = 0
    # edge class balance
    total_edges: int = 0
    positive_edges: int = 0       # same-track (edge_label == 1)
    zero_positive_windows: int = 0

    # per-feature separability (AUC of single-feature classifier for each dim)
    # shape (6,) — indexed by PAIR_FEATURE_NAMES
    feature_auc: np.ndarray = field(default=None, repr=False)

    @property
    def positive_edge_pct(self) -> float:
        return 100.0 * self.positive_edges / max(1, self.total_edges)

    @property
    def zero_positive_window_pct(self) -> float:
        return 100.0 * self.zero_positive_windows / max(1, self.n_entries)

    def edge_class_decision(self) -> str:
        pct = self.positive_edge_pct
        zpct = self.zero_positive_window_pct
        parts = []
        if pct < 1.0:
            parts.append(f"positive edge% very low ({pct:.2f}%) — use class weighting or focal loss")
        if zpct > 20.0:
            parts.append(f"{zpct:.1f}% windows have no positive edge — legal mask may be too strict")
        if not parts:
            parts.append(f"positive edges common ({pct:.1f}%) — EdgeCompatNet well motivated")
        return "; ".join(parts)

    def separability_label(self, auc: float) -> str:
        if auc >= 0.70:
            return "strong"
        elif auc >= 0.60:
            return "medium"
        else:
            return "weak"


def _single_feature_auc(pos_vals: np.ndarray, neg_vals: np.ndarray) -> float:
    """AUC of a single-feature threshold classifier (Mann-Whitney U / AUC)."""
    from sklearn.metrics import roc_auc_score
    n_pos = len(pos_vals)
    n_neg = len(neg_vals)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # subsample for speed if arrays are very large
    max_each = 200_000
    if n_pos > max_each:
        pos_vals = pos_vals[np.random.default_rng(0).choice(n_pos, max_each, replace=False)]
    if n_neg > max_each:
        neg_vals = neg_vals[np.random.default_rng(1).choice(n_neg, max_each, replace=False)]
    y = np.concatenate([np.ones(len(pos_vals)), np.zeros(len(neg_vals))])
    scores = np.concatenate([pos_vals, neg_vals])
    auc = roc_auc_score(y, scores)
    return float(max(auc, 1.0 - auc))   # reflect so AUC >= 0.5


def run_phase2(
    cache_dir: Path,
    datasets: Sequence[str],
    verbose: bool = True,
) -> dict[str, EdgeResult | None]:
    if not has_graph(cache_dir):
        if verbose:
            print("  [phase2] cache has no graph data — skipping")
        return {ds: None for ds in datasets}

    n_feats = len(PAIR_FEATURE_NAMES)
    results: dict[str, EdgeResult | None] = {}

    for ds in datasets:
        r = EdgeResult(dataset=ds)
        if verbose:
            print(f"  [phase2] {ds} ...", end="", flush=True)

        # accumulate positive/negative edge feature vectors
        pos_feats: list[np.ndarray] = []
        neg_feats: list[np.ndarray] = []

        for shard in iter_shards(cache_dir, ds):
            if "edge_label" not in shard:
                break

            ei_list = shard["edge_index"]    # list of (2, E_i)
            ea_list = shard["edge_attr"]     # list of (E_i, 6)
            el_list = shard["edge_label"]    # list of (E_i,)

            N = len(el_list)
            r.n_entries += N

            for i in range(N):
                el = el_list[i]              # (E_i,)
                ea = ea_list[i]              # (E_i, 6)
                n_edges = int(el.shape[0])
                n_pos   = int((el > 0.5).sum())

                r.total_edges    += n_edges
                r.positive_edges += n_pos
                if n_pos == 0 and n_edges > 0:
                    r.zero_positive_windows += 1

                if n_edges > 0:
                    ea_np = ea.numpy()
                    pos_mask = el.numpy() > 0.5
                    if pos_mask.sum() > 0:
                        pos_feats.append(ea_np[pos_mask])
                    if (~pos_mask).sum() > 0:
                        neg_feats.append(ea_np[~pos_mask])

        # compute per-feature AUC
        if pos_feats and neg_feats:
            pos_all = np.concatenate(pos_feats, axis=0)   # (P, 6)
            neg_all = np.concatenate(neg_feats, axis=0)   # (Q, 6)
            auc = np.zeros(n_feats)
            for f in range(n_feats):
                auc[f] = _single_feature_auc(pos_all[:, f], neg_all[:, f])
            r.feature_auc = auc

        results[ds] = r
        if verbose:
            pct = r.positive_edge_pct
            auc_str = ""
            if r.feature_auc is not None:
                best = PAIR_FEATURE_NAMES[int(np.argmax(r.feature_auc))]
                auc_str = f"  best_feat={best}({r.feature_auc.max():.3f})"
            print(f" {r.n_entries:,} — pos_edge%={pct:.1f}{auc_str}")

    return results
