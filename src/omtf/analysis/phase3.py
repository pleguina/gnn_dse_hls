"""
Phase 3 — Is pT/rank learnable from current inputs?

3.1  pT distribution and threshold-bin statistics
3.2  Pearson / Spearman correlations between curvature proxies and 1/pT
3.3  Tiny classical pT baseline (linear + small MLP on per-window summaries)

All analyses run on single-track windows only (n_gen_tracks == 1) to get
clean 1-to-1 label → feature alignment.  Multi-track datasets are included
but filtered to single-track windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .loader import iter_shards, has_graph
from omtf.features import PAIR_FEATURE_NAMES

# L1 pT thresholds in GeV
PT_THRESHOLDS = [3, 5, 10, 15, 20, 22]
# pT bins for distribution table (GeV)
PT_BINS = [(0, 3), (3, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 50), (50, 9999)]


# --------------------------------------------------------------------------- #
# 3.1  Distribution
# --------------------------------------------------------------------------- #

@dataclass
class PTDistResult:
    dataset: str
    n_windows: int = 0          # all windows
    n_single_track: int = 0     # windows with exactly 1 gen track
    pt_values: np.ndarray = field(default=None, repr=False)   # gen_pt[0] for single-track

    def stats(self) -> dict:
        p = self.pt_values
        if p is None or len(p) == 0:
            return {}
        out = {
            "mean": float(np.mean(p)),
            "p10":  float(np.percentile(p, 10)),
            "p50":  float(np.percentile(p, 50)),
            "p90":  float(np.percentile(p, 90)),
            "p99":  float(np.percentile(p, 99)),
        }
        for thr in PT_THRESHOLDS:
            out[f"pct_gt_{thr}"] = float(100.0 * np.mean(p > thr))
        return out

    def threshold_bins(self) -> dict:
        """Count muons in each pT bin."""
        p = self.pt_values
        if p is None or len(p) == 0:
            return {}
        out = {}
        for lo, hi in PT_BINS:
            label = f"{lo}-{hi}" if hi < 9000 else f">{lo}"
            out[label] = int(np.sum((p >= lo) & (p < hi)))
        return out

    def imbalance_decision(self) -> str:
        bins = self.threshold_bins()
        total = sum(bins.values())
        if total == 0:
            return "NO DATA"
        # check if any threshold bin has < 1% of total
        thin = [k for k, v in bins.items() if v / total < 0.01]
        if thin:
            return f"WARN — bins underpopulated: {thin}; consider reweighting or new samples"
        return "PASS — threshold bins populated"


# --------------------------------------------------------------------------- #
# 3.2  Correlations
# --------------------------------------------------------------------------- #

@dataclass
class CorrResult:
    dataset: str
    n_pairs: int = 0
    pearson:  dict[str, float] = field(default_factory=dict)
    spearman: dict[str, float] = field(default_factory=dict)

    def best_feature(self) -> str:
        if not self.pearson:
            return "N/A"
        return max(self.pearson, key=lambda k: abs(self.pearson[k]))

    def correlation_decision(self) -> str:
        if not self.pearson:
            return "NO DATA"
        best = self.best_feature()
        r = abs(self.pearson[best])
        if r >= 0.3:
            return f"PASS — curvature proxy visible (best={best}, |r|={r:.3f}); pT failure is loss/head related"
        elif r >= 0.1:
            return f"WARN — weak signal (best={best}, |r|={r:.3f}); check feature normalisation"
        return f"FAIL — no curvature signal (best={best}, |r|={r:.3f}); inspect feature construction"


# --------------------------------------------------------------------------- #
# 3.3  Tiny baseline
# --------------------------------------------------------------------------- #

@dataclass
class BaselineResult:
    dataset: str
    n_train: int = 0
    n_test:  int = 0
    # linear regression on q/pT
    linear_pearson:  float = float("nan")
    linear_mae:      float = float("nan")
    # MLP regression on q/pT
    mlp_pearson:     float = float("nan")
    mlp_mae:         float = float("nan")
    # logistic classifier AUC per threshold
    cls_auc: dict[str, float] = field(default_factory=dict)

    def baseline_decision(self) -> str:
        if np.isnan(self.linear_pearson):
            return "NO DATA"
        if abs(self.linear_pearson) > 0.3 or any(v > 0.65 for v in self.cls_auc.values()):
            return "PASS — information present; EdgeCompat pT failure is training/loss related"
        return "WARN — even simple baseline struggles; consider feature engineering or new samples"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _extract_window_features(shard: dict, graph_present: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, q_over_pt, pt) for single-track windows in shard.

    "Single-track" means exactly one unique nonzero trackId is present in the
    stubs — determined from the stub data, not from meta_n_gen.  This correctly
    handles datasets like S1 where meta_n_gen=2 (both gen muons stored) but only
    one passes through each processor window.

    Features per window:
      [n_signal_stubs, mean_phiB_signal, mean_eta_signal, mean_r_signal,
       mean_quality_signal, n_unique_layers,
       best_abs_kappa (if graph), mean_abs_kappa (if graph), mean_abs_dphi (if graph)]
    """
    import torch as _torch

    vm     = shard["valid_mask"]          # (N, 24)
    tid    = shard["track_id"].long()     # (N, 24)
    stubs  = shard["stubs"]              # (N, 24, 7) [phi,phiB,eta,r,quality,type,layer]
    gen_pt = shard["gen_pt"]             # (N, 3)
    gen_ch = shard["gen_charge"]         # (N, 3)
    K_MAX  = gen_pt.shape[1]             # usually 3

    # stub-based track count: exactly one nonzero trackId present
    k_present = _torch.stack(
        [((tid == k) & vm).any(dim=1) for k in range(1, K_MAX + 1)], dim=1
    )                                    # (N, K_MAX) bool
    n_tracks = k_present.long().sum(dim=1)  # (N,)
    # which trackId is the unique one (only valid when n_tracks==1)
    active_slot = k_present.long().argmax(dim=1)  # (N,) index 0..K_MAX-1 → trackId=slot+1

    single = n_tracks == 1
    if not single.any():
        return np.empty((0, 9)), np.empty(0), np.empty(0)

    idx  = single.nonzero(as_tuple=True)[0]
    N_s  = int(idx.shape[0])
    slot = active_slot[idx]              # (N_s,) — 0-based slot index for gen_pt

    vm_s    = vm[idx]                   # (N_s, 24)
    tid_s   = tid[idx]                  # (N_s, 24)
    stubs_s = stubs[idx]               # (N_s, 24, 7)

    # pT and charge for the active gen muon slot
    pt_s = _torch.gather(gen_pt[idx], 1, slot.unsqueeze(1)).squeeze(1).numpy()
    ch_s = _torch.gather(gen_ch[idx], 1, slot.unsqueeze(1)).squeeze(1).numpy()

    # filter out windows where the matched gen_pt is zero (join miss)
    valid_pt = pt_s > 0
    if valid_pt.sum() == 0:
        return np.empty((0, 9)), np.empty(0), np.empty(0)
    idx_v   = np.where(valid_pt)[0]
    pt_s    = pt_s[valid_pt]
    ch_s    = ch_s[valid_pt]
    vm_s    = vm_s[idx_v]
    tid_s   = tid_s[idx_v]
    stubs_s = stubs_s[idx_v]
    slot    = slot[idx_v]
    idx     = idx[idx_v]
    N_s     = int(len(pt_s))

    q_pt = ch_s / np.maximum(pt_s, 1e-3)

    # signal stub mask: trackId == slot+1
    sig_k = slot.unsqueeze(1) + 1       # (N_s, 1) trackId of the signal track
    sig = ((tid_s == sig_k) & vm_s).float().unsqueeze(-1)  # (N_s, 24, 1)
    sig_count = sig.squeeze(-1).sum(dim=1).numpy()       # (N_s,) — n_signal_stubs

    # weighted mean of stub columns over signal stubs
    # cols: [phi=0, phiB=1, eta=2, r=3, quality=4, type=5, layer=6]
    denom = np.maximum(sig_count, 1.0)
    weighted = (stubs_s * sig).sum(dim=1).numpy()        # (N_s, 7)
    mean_phiB    = weighted[:, 1] / denom
    mean_eta     = weighted[:, 2] / denom
    mean_r       = weighted[:, 3] / denom
    mean_quality = weighted[:, 4] / denom
    mean_layer   = weighted[:, 6] / denom                # proxy for n_unique_layers

    X = np.stack([
        sig_count, mean_phiB, mean_eta, mean_r, mean_quality, mean_layer,
        np.zeros(N_s), np.zeros(N_s), np.zeros(N_s),   # kappa placeholders
    ], axis=1)                                           # (N_s, 9)

    # fill kappa columns from graph if available
    if graph_present and "edge_attr" in shard and "edge_label" in shard:
        ea_list = shard["edge_attr"]
        el_list = shard["edge_label"]
        idx_arr = idx.tolist() if hasattr(idx, "tolist") else list(idx)
        for j, orig_i in enumerate(idx_arr):
            ea = ea_list[orig_i]   # (E_i, 6): col3=kappa_hat, col0=delta_phi
            el = el_list[orig_i]   # (E_i,)
            if ea.shape[0] == 0:
                continue
            pos = el.numpy() > 0.5
            kappa = ea[:, 3].numpy()
            dphi  = ea[:, 0].numpy()
            if pos.sum() > 0:
                X[j, 6] = float(np.max(np.abs(kappa[pos])))
                X[j, 7] = float(np.mean(np.abs(kappa[pos])))
                X[j, 8] = float(np.mean(np.abs(dphi[pos])))

    return X, q_pt, pt_s


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


# --------------------------------------------------------------------------- #
# Public runners
# --------------------------------------------------------------------------- #

def run_phase3_distribution(
    cache_dir: Path,
    datasets: Sequence[str],
    verbose: bool = True,
) -> dict[str, PTDistResult]:
    results: dict[str, PTDistResult] = {}

    for ds in datasets:
        r = PTDistResult(dataset=ds)
        if verbose:
            print(f"  [phase3-dist] {ds} ...", end="", flush=True)

        pt_vals: list[float] = []

        for shard in iter_shards(cache_dir, ds):
            gen_pt = shard["gen_pt"]            # (N, 3)
            vm     = shard["valid_mask"]         # (N, 24)
            tid    = shard["track_id"].long()    # (N, 24)
            K_MAX  = gen_pt.shape[1]
            N      = int(gen_pt.shape[0])
            r.n_windows += N

            # stub-based single-track filter (see _extract_window_features)
            import torch as _t
            k_pres = _t.stack([((tid == k) & vm).any(dim=1) for k in range(1, K_MAX + 1)], dim=1)
            n_tr   = k_pres.long().sum(dim=1)
            single = n_tr == 1
            active = k_pres.long().argmax(dim=1)          # 0-based slot index
            r.n_single_track += int(single.sum())

            if single.any():
                idx_s = single.nonzero(as_tuple=True)[0]
                slots = active[idx_s]
                pts   = _t.gather(gen_pt[idx_s], 1, slots.unsqueeze(1)).squeeze(1)
                valid_pt = pts > 0
                pt_vals.extend(pts[valid_pt].tolist())

        r.pt_values = np.asarray(pt_vals, dtype=np.float32)
        results[ds] = r

        if verbose:
            st = r.stats()
            med = st.get("p50", float("nan"))
            gt10 = st.get("pct_gt_10", float("nan"))
            print(f" {r.n_single_track:,} single-track — median_pT={med:.1f} GeV  %>10={gt10:.1f}%")

    return results


def run_phase3_correlations(
    cache_dir: Path,
    datasets: Sequence[str],
    verbose: bool = True,
) -> dict[str, CorrResult]:
    gp = has_graph(cache_dir)
    results: dict[str, CorrResult] = {}

    FEAT_NAMES = ["n_sig", "mean_phiB", "mean_eta", "mean_r", "mean_quality",
                  "mean_layer", "best_abs_kappa", "mean_abs_kappa", "mean_abs_dphi"]

    for ds in datasets:
        r = CorrResult(dataset=ds)
        if verbose:
            print(f"  [phase3-corr] {ds} ...", end="", flush=True)

        Xs, Qs, Ps = [], [], []
        for shard in iter_shards(cache_dir, ds):
            X, q_pt, pt = _extract_window_features(shard, gp)
            if len(X) > 0:
                Xs.append(X); Qs.append(q_pt); Ps.append(pt)

        if not Xs:
            results[ds] = r
            if verbose: print(" (no single-track windows)")
            continue

        X_all  = np.concatenate(Xs, axis=0)    # (M, 9)
        q_all  = np.concatenate(Qs, axis=0)    # (M,) — signed curvature target
        inv_pt = np.concatenate(Ps, axis=0)    # (M,)
        inv_pt = 1.0 / np.maximum(inv_pt, 1e-3)

        r.n_pairs = int(len(q_all))

        for fi, fname in enumerate(FEAT_NAMES):
            col = X_all[:, fi]
            if col.std() < 1e-9:
                r.pearson[fname]  = float("nan")
                r.spearman[fname] = float("nan")
                continue
            r.pearson[fname]  = float(np.corrcoef(col, q_all)[0, 1])
            r.spearman[fname] = _spearman(col, q_all)

        results[ds] = r
        if verbose:
            best = r.best_feature()
            pr   = r.pearson.get(best, float("nan"))
            print(f" {r.n_pairs:,} windows — best={best}  |pearson|={abs(pr):.3f}")

    return results


def run_phase3_baseline(
    cache_dir: Path,
    datasets: Sequence[str],
    verbose: bool = True,
) -> dict[str, BaselineResult]:
    """Train linear + MLP baselines on per-window summary features → q/pT."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    gp = has_graph(cache_dir)
    results: dict[str, BaselineResult] = {}

    for ds in datasets:
        r = BaselineResult(dataset=ds)
        if verbose:
            print(f"  [phase3-baseline] {ds} ...", end="", flush=True)

        Xs, Qs, Ps = [], [], []
        for shard in iter_shards(cache_dir, ds):
            X, q_pt, pt = _extract_window_features(shard, gp)
            if len(X) > 0:
                Xs.append(X); Qs.append(q_pt); Ps.append(pt)

        if not Xs:
            results[ds] = r
            if verbose: print(" (no data)")
            continue

        X_all = np.concatenate(Xs, axis=0)
        q_all = np.concatenate(Qs, axis=0)
        pt_all = np.concatenate(Ps, axis=0)

        # train/test split (80/20)
        n = len(q_all)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        n_train = int(0.8 * n)
        tr, te = perm[:n_train], perm[n_train:]
        r.n_train, r.n_test = int(n_train), int(n - n_train)

        scaler = StandardScaler().fit(X_all[tr])
        X_tr = scaler.transform(X_all[tr])
        X_te = scaler.transform(X_all[te])

        # linear regression on q/pT
        lr = LinearRegression().fit(X_tr, q_all[tr])
        pred_lr = lr.predict(X_te)
        r.linear_pearson = float(np.corrcoef(pred_lr, q_all[te])[0, 1])
        r.linear_mae     = float(np.mean(np.abs(pred_lr - q_all[te])))

        # small MLP on q/pT (hidden_layer_sizes=(32, 16))
        mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200,
                           random_state=0, early_stopping=True).fit(X_tr, q_all[tr])
        pred_mlp = mlp.predict(X_te)
        r.mlp_pearson = float(np.corrcoef(pred_mlp, q_all[te])[0, 1])
        r.mlp_mae     = float(np.mean(np.abs(pred_mlp - q_all[te])))

        # threshold classifiers: pT > threshold
        for thr in PT_THRESHOLDS:
            y_cls = (pt_all[te] > thr).astype(int)
            if y_cls.sum() == 0 or (1 - y_cls).sum() == 0:
                r.cls_auc[f"pT>{thr}"] = float("nan")
                continue
            lr_cls = LogisticRegression(max_iter=200).fit(X_tr, (pt_all[tr] > thr).astype(int))
            r.cls_auc[f"pT>{thr}"] = float(roc_auc_score(y_cls, lr_cls.predict_proba(X_te)[:, 1]))

        results[ds] = r
        if verbose:
            print(
                f" n={n:,} — lin_r={r.linear_pearson:.3f} "
                f"mlp_r={r.mlp_pearson:.3f} "
                f"cls@10={r.cls_auc.get('pT>10', float('nan')):.3f}"
            )

    return results
