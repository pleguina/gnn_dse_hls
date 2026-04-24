"""
Phase 5 — B4 false-positive audit.

5.1  Compare accepted vs rejected B4 windows (stub count, layer pattern,
     edge scores, candidate scores)
5.2  B4 fake multiplicity: candidates per window and per event

Requires a model checkpoint.  The model is loaded via the same load_model()
used in eval.py, so any supported architecture works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .loader import iter_shards


@dataclass
class B4AuditResult:
    n_windows: int = 0
    n_accepted: int = 0         # windows with at least one candidate logit > 0
    n_total_candidates: int = 0  # sum of accepted candidates across all windows

    # per-window stats split by accepted/rejected
    accepted_mean_stubs:   float = float("nan")
    rejected_mean_stubs:   float = float("nan")
    accepted_mean_cand_score: float = float("nan")
    rejected_mean_cand_score: float = float("nan")

    # candidate score distribution (binned)
    score_bins: np.ndarray = field(default=None, repr=False)
    score_hist: np.ndarray = field(default=None, repr=False)

    @property
    def accept_rate(self) -> float:
        return 100.0 * self.n_accepted / max(1, self.n_windows)

    @property
    def mean_candidates_per_accepted(self) -> float:
        return self.n_total_candidates / max(1, self.n_accepted)

    def decision(self) -> str:
        rate = self.accept_rate
        if rate < 5.0:
            return f"PASS — bg_accept={rate:.1f}%; threshold tuning likely sufficient"
        elif rate < 20.0:
            return f"WARN — bg_accept={rate:.1f}%; inspect window topology"
        return f"FAIL — bg_accept={rate:.1f}%; need stronger suppression or hard-negative mining"


def run_phase5(
    cache_dir: Path,
    checkpoint_path: str | Path,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> B4AuditResult:
    """Run B4 false-positive audit against a saved model checkpoint."""
    import sys
    from pathlib import Path as _Path
    _repo = _Path(__file__).resolve().parents[4]   # gnn_dse_hls/
    if str(_repo / "src") not in sys.path:
        sys.path.insert(0, str(_repo / "src"))

    from omtf.eval import load_model

    device = torch.device(device_str)
    if verbose:
        print(f"  [phase5] loading {Path(checkpoint_path).name} on {device_str}")

    model, model_name = load_model(str(checkpoint_path), device)

    r = B4AuditResult()
    if verbose:
        print("  [phase5] B4 audit ...", end="", flush=True)

    # per-window stats accumulation
    acc_stubs_list: list[float] = []
    rej_stubs_list: list[float] = []
    acc_scores_list: list[float] = []
    rej_scores_list: list[float] = []
    all_cand_scores: list[float] = []

    graph_present = "edge_label" in _peek_shard(cache_dir, "B4")

    with torch.no_grad():
        for shard in iter_shards(cache_dir, "B4", device=device_str):
            stubs  = shard["stubs"].to(device)     # (N, 24, 7)
            vm     = shard["valid_mask"].to(device) # (N, 24)
            N      = stubs.shape[0]
            r.n_windows += N
            n_stubs_arr = vm.sum(dim=1).float().cpu().numpy()  # (N,)

            # run model in batches of 256
            batch_size = 256
            cand_logits_list: list[torch.Tensor] = []
            for b_start in range(0, N, batch_size):
                b_end = min(b_start + batch_size, N)
                batch = {
                    "stubs":      stubs[b_start:b_end],
                    "valid_mask": vm[b_start:b_end],
                }
                if graph_present:
                    batch["edge_index"] = shard["edge_index"][b_start:b_end]
                    batch["edge_attr"]  = shard["edge_attr"][b_start:b_end]
                # model forward — expect (N_b, K) candidate logits
                out = _safe_forward(model, model_name, batch, device)
                cand_logits_list.append(out.cpu())

            cand_logits = torch.cat(cand_logits_list, dim=0)  # (N, K)
            max_logit   = cand_logits.max(dim=1).values       # (N,)

            accepted = max_logit > 0.0  # (N,) bool
            n_cand_per_window = (cand_logits > 0.0).sum(dim=1).float()  # (N,)

            r.n_accepted += int(accepted.sum())
            r.n_total_candidates += int(n_cand_per_window[accepted].sum())

            max_logit_np = max_logit.numpy()
            n_cand_np    = n_cand_per_window.numpy()
            accepted_np  = accepted.numpy()

            acc_stubs_list.extend(n_stubs_arr[accepted_np].tolist())
            rej_stubs_list.extend(n_stubs_arr[~accepted_np].tolist())
            acc_scores_list.extend(max_logit_np[accepted_np].tolist())
            rej_scores_list.extend(max_logit_np[~accepted_np].tolist())
            all_cand_scores.extend(max_logit_np.tolist())

    if acc_stubs_list:
        r.accepted_mean_stubs    = float(np.mean(acc_stubs_list))
        r.accepted_mean_cand_score = float(np.mean(acc_scores_list))
    if rej_stubs_list:
        r.rejected_mean_stubs    = float(np.mean(rej_stubs_list))
        r.rejected_mean_cand_score = float(np.mean(rej_scores_list))

    # score histogram
    if all_cand_scores:
        hist, bins = np.histogram(all_cand_scores, bins=50, range=(-5.0, 5.0))
        r.score_hist = hist
        r.score_bins = 0.5 * (bins[:-1] + bins[1:])

    if verbose:
        print(
            f" {r.n_windows:,} windows — bg_accept={r.accept_rate:.1f}%  "
            f"cand/accepted={r.mean_candidates_per_accepted:.2f}"
        )
    return r


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _peek_shard(cache_dir: Path, dataset: str) -> dict:
    """Return the first shard (for key inspection without loading everything)."""
    ds_dir = Path(cache_dir) / dataset
    shards = sorted(ds_dir.glob("shard_*.pt"))
    if not shards:
        return {}
    shard = torch.load(shards[0], map_location="cpu", weights_only=False)
    return {k: None for k in shard.keys()}


def _safe_forward(model, model_name: str, batch: dict, device) -> torch.Tensor:
    """Call model.forward and extract candidate logits, handling both model APIs."""
    try:
        out = model(
            stubs      = batch["stubs"],
            valid_mask = batch["valid_mask"],
            edge_index = batch.get("edge_index"),
            edge_attr  = batch.get("edge_attr"),
        )
    except TypeError:
        # fallback: positional
        out = model(batch["stubs"], batch["valid_mask"])

    if isinstance(out, dict):
        cand = out.get("candidate_logits", out.get("cand_logits"))
        if cand is None:
            # last resort: take any (N, 3) tensor in the output dict
            for v in out.values():
                if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == 3:
                    cand = v; break
    elif isinstance(out, (tuple, list)):
        # assume first element of appropriate shape is candidate logits
        for item in out:
            if isinstance(item, torch.Tensor) and item.ndim == 2 and item.shape[1] == 3:
                cand = item; break
        else:
            cand = out[0]
    else:
        cand = out

    if cand is None:
        raise RuntimeError(f"Cannot extract candidate logits from model output: {type(out)}")
    return cand
