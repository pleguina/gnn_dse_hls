"""
OMTF evaluation entrypoint.

Loads one or two checkpoints and evaluates on specified datasets, reporting
the full set of trigger-quality metrics needed for the Architecture Decision Gate.

Usage
-----
    # Single model
    python src/omtf/eval.py --checkpoint build/omtf/checkpoints/slot_model_best.pt \
                            --datasets S1 S3 B1 B4

    # Side-by-side comparison (ADG)
    python src/omtf/eval.py \
        --checkpoint  build/omtf/checkpoints/edge_compat_best.pt \
        --checkpoint2 build/omtf/checkpoints/slot_model_best.pt \
        --datasets S1 S3 B4

    # All nine datasets
    python src/omtf/eval.py --checkpoint ... --all-datasets

    # Use pre-built .pt cache (much faster than ROOT)
    python src/omtf/eval.py --checkpoint ... \
        --cache-dir build/omtf/cache/schema_v1 --all-datasets

    # Save results JSON
    python src/omtf/eval.py --checkpoint ... --save

Metrics reported
----------------
Stub/node  : node_auc, stub_eff, cand_rec_proxy
Window     : trigger_eff@pt5, trigger_eff@pt10, trigger_eff@pt20,
             cand_recovery (slot-ordered), dimuon_sep (multi-muon datasets)
Background : bg_accept (B4 only)
pT heads   : pt_mae, pt_sigma68 (per-slot models with gen_pt)
Edge       : edge_auc (edge_compat, edge_mlp)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omtf.dataset import OMTFDataset, collate_omtf
from omtf.cache_dataset import CachedOMTFDataset
from omtf.metrics import (
    node_auc,
    stub_recovery_efficiency,
    candidate_recovery_proxy,
    candidate_recovery_slot,
    background_acceptance,
    background_acceptance_slots,
    trigger_efficiency_vs_pt,
    dimuon_separation_efficiency,
    pt_resolution_slots,
    dxy_resolution_slots,
    edge_auc,
    print_metrics_table,
    EVAL_DISCLAIMER,
)
from torch.utils.data import DataLoader
from audit.root_utils import set_root_batch_mode

DATA_ROOT = PROJECT_ROOT / "data" / "prod"
EVAL_DIR  = PROJECT_ROOT / "build" / "omtf" / "eval"
NMAX = 24

_PROFILE = False  # set to True via --profile flag


@contextmanager
def _timer(label: str):
    if not _PROFILE:
        yield
        return
    t0 = time.perf_counter()
    yield
    print(f"  [time] {label}: {time.perf_counter() - t0:.2f}s")

# Datasets with 2+ gen muons per window (dimuon separation is meaningful)
MULTI_MUON_DATASETS = {"S3", "S4", "S5", "B3"}
# Datasets where background acceptance is the primary metric
NOISE_DATASETS = {"B4"}
# Default PT thresholds for the efficiency scan
PT_THRESHOLDS = [0.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    """Load any OMTF model from a checkpoint. Auto-detects model type."""
    from omtf.models.baselines import build_baseline
    from omtf.models.edge_compat import build_edge_compat
    from omtf.models.slot_model import build_slot_model

    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})
    model_name = args.get("model", "deepsets")
    hidden     = args.get("hidden", 64)
    dropout    = args.get("dropout", 0.0)

    if model_name == "deepsets":
        model = build_baseline("deepsets", hidden=hidden, dropout=dropout)
    elif model_name == "edge_mlp":
        model = build_baseline("edge_mlp", hidden=hidden, dropout=dropout)
    elif model_name == "edge_compat":
        model = build_edge_compat(hidden=hidden, K=3, dropout=dropout)
    elif model_name == "slot_model":
        model = build_slot_model(hidden=hidden, K=3, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_name!r}")

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "unexpected key" in str(e).lower():
            missing = [k for k in model.state_dict() if k not in ckpt["model"]]
            print(f"  [WARN] partial load — checkpoint predates current arch "
                  f"(missing: {missing[:3]}{'...' if len(missing) > 3 else ''})")
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            raise
    model = model.to(device)
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_name}  epoch={epoch}  val_loss={val_loss:.4f}  "
          f"params={n_params:,}")
    return model, model_name


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loader(
    dataset: str,
    include_graph: bool,
    max_files: int | None,
    max_entries: int | None,
    batch_size: int = 512,
    num_workers: int = 4,
    cache_dir: Path | None = None,
    pin_memory: bool = False,
) -> DataLoader | None:
    ds = None

    if cache_dir is not None and (cache_dir / dataset).exists():
        try:
            with _timer("cache load"):
                ds = CachedOMTFDataset(
                    cache_dir, dataset,
                    include_graph=include_graph,
                    max_entries=max_entries,
                )
            print(f"  [cache] {len(ds):,} samples from {cache_dir / dataset}")
        except ValueError as e:
            print(f"  [cache] fallback to ROOT ({e})")

    if ds is None:
        d = DATA_ROOT / dataset
        files = sorted(d.glob(f"omtf_hits_{dataset}_*.root"))
        if max_files:
            files = files[:max_files]
        if not files:
            return None
        with _timer("ROOT load"):
            ds = OMTFDataset(
                files, Nmax=NMAX,
                include_graph=include_graph,
                include_nano=True,
                max_entries=max_entries,
                num_workers=num_workers,
            )
        print(f"  [root]  {len(ds):,} samples from {len(files)} files")

    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_omtf, num_workers=0,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# Per-model forward pass helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_deepsets(model, batch, device):
    return model(batch["stubs"].to(device), batch["valid_mask"].to(device))


@torch.no_grad()
def _forward_edge_compat(model, batch, device):
    return model(
        batch["stubs"].to(device),
        batch["valid_mask"].to(device),
        [e.to(device) for e in batch["edge_index"]],
        [e.to(device) for e in batch["edge_attr"]],
    )


@torch.no_grad()
def _forward_slot_model(model, batch, device):
    return model(
        batch["stubs"].to(device),
        batch["valid_mask"].to(device),
    )


_FORWARD = {
    "deepsets":    _forward_deepsets,
    "edge_mlp":    None,          # handled separately
    "edge_compat": _forward_edge_compat,
    "slot_model":  _forward_slot_model,
}


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------

def _empty_acc() -> dict:
    return {
        "node_logits": [], "node_labels": [], "valid": [], "ambig": [], "tid": [],
        "cand_logits": [],
        "gen_pt": [], "gen_charge": [], "gen_dxy": [],
        "pt_pred": [], "dxy_pred": [],
        "edge_logits": [], "edge_labels": [], "edge_ambig": [],
        "slot_attn": [],   # (B, K, Nmax) from slot-capable models
        "n_windows": 0,
    }


def _accumulate_batch(acc: dict, model, model_name: str, batch: dict, device: torch.device) -> None:
    """Process one batch and accumulate results into acc."""
    fwd = _FORWARD.get(model_name)

    if model_name == "edge_mlp":
        if "edge_attr" not in batch:
            return
        B = batch["stubs"].shape[0]
        acc["n_windows"] += B
        for ea, el, eamb in zip(batch["edge_attr"], batch["edge_label"],
                                 batch["edge_ambig"]):
            if ea.shape[0] == 0:
                continue
            logit = model(ea.to(device)).cpu()
            acc["edge_logits"].append(logit)
            acc["edge_labels"].append(el)
            acc["edge_ambig"].append(eamb)
        return

    acc["n_windows"] += batch["stubs"].shape[0]
    out = fwd(model, batch, device)

    acc["node_logits"].append(out["node_logits"].cpu())
    acc["node_labels"].append(batch["node_label"])
    acc["valid"].append(batch["valid_mask"])
    acc["ambig"].append(batch["ambiguous"].bool())
    acc["tid"].append(batch["track_id"])

    if "candidate_logits" in out:
        acc["cand_logits"].append(out["candidate_logits"].cpu())
    if "attn" in out:
        acc["slot_attn"].append(out["attn"].cpu())

    if "gen_pt"     in batch: acc["gen_pt"].append(batch["gen_pt"])
    if "gen_charge" in batch: acc["gen_charge"].append(batch["gen_charge"])
    if "gen_dxy"    in batch: acc["gen_dxy"].append(batch["gen_dxy"])

    _GEN_K = 3
    if "pt_pred" in out and out["pt_pred"].dim() == 2 and out["pt_pred"].shape[-1] == _GEN_K:
        acc["pt_pred"].append(out["pt_pred"].cpu())
    if "dxy_pred" in out and out["dxy_pred"].dim() == 2 and out["dxy_pred"].shape[-1] == _GEN_K:
        acc["dxy_pred"].append(out["dxy_pred"].cpu())

    if "edge_logits" in out:
        for logit, label, ambig in zip(
            out["edge_logits"],
            batch.get("edge_label", []),
            batch.get("edge_ambig", []),
        ):
            if logit.shape[0] > 0:
                acc["edge_logits"].append(logit.cpu())
                acc["edge_labels"].append(label)
                acc["edge_ambig"].append(ambig)


@torch.inference_mode()
def evaluate_dataset(
    model,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str,
) -> dict:
    model.eval()
    acc = _empty_acc()

    with _timer("forward+accumulate"):
        for batch in loader:
            _accumulate_batch(acc, model, model_name, batch, device)

    if model_name == "edge_mlp":
        return _compute_edge_only(acc, dataset_name)
    return _compute_metrics(acc, dataset_name, model_name)


@torch.inference_mode()
def evaluate_dataset_pair(
    model_a, name_a: str,
    model_b, name_b: str,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str,
) -> tuple[dict, dict]:
    """One-pass dual evaluation: both models run on each batch. Halves I/O time."""
    model_a.eval()
    model_b.eval()
    acc_a, acc_b = _empty_acc(), _empty_acc()

    with _timer("forward+accumulate (dual)"):
        for batch in loader:
            _accumulate_batch(acc_a, model_a, name_a, batch, device)
            _accumulate_batch(acc_b, model_b, name_b, batch, device)

    def _finish(acc, name):
        if name == "edge_mlp":
            return _compute_edge_only(acc, dataset_name)
        return _compute_metrics(acc, dataset_name, name)

    return _finish(acc_a, name_a), _finish(acc_b, name_b)


def _compute_edge_only(acc: dict, dataset_name: str) -> dict:
    result = {}
    if acc["edge_logits"]:
        result["edge_auc"] = edge_auc(
            torch.cat(acc["edge_logits"]),
            torch.cat(acc["edge_labels"]),
            torch.cat(acc["edge_ambig"]) if acc["edge_ambig"] else None,
        )
    if dataset_name in NOISE_DATASETS:
        # Use total evaluated windows (acc["n_windows"]), not just windows-with-edges,
        # so windows with zero edges are correctly counted as non-firing.
        n_windows = acc.get("n_windows", len(acc["edge_logits"]))
        n_any = sum(1 for l in acc["edge_logits"] if (l > 0).any())
        result["bg_accept"] = n_any / max(n_windows, 1)
    return result


# ---------------------------------------------------------------------------
# Slot specialization diagnostics
# ---------------------------------------------------------------------------

def _slot_diagnostics(attn_list: list) -> dict:
    """
    Given accumulated attention tensors (each (B, K, Nmax)), compute:

      slot_entropy      : mean attention entropy per slot (K,)
                          H[k] = -sum_i attn[k,i] * log(attn[k,i]+eps)
                          Max entropy = log(Nmax). Low entropy = focused.
      pairwise_overlap  : mean cosine similarity between slot attention maps.
                          High overlap means slots are not specializing.
      top_collision_rate: fraction of windows where any two slots share their
                          highest-weighted stub. Measures hard collapse.
      slot_occupancy    : fraction of windows where each slot has the highest
                          candidate-logit — computed separately if cand_logits
                          available; left out here (only attn available).
    """
    import torch.nn.functional as _F

    attn = torch.cat(attn_list, dim=0)   # (N, K, Nmax)
    N, K, Nmax = attn.shape
    eps = 1e-9

    # Entropy per window per slot
    ent = -(attn * (attn + eps).log()).sum(dim=-1)    # (N, K)
    entropy_mean = ent.mean(dim=0).tolist()           # [K]
    entropy_std  = ent.std(dim=0).tolist()            # [K]

    # Pairwise cosine overlap
    attn_norm = _F.normalize(attn, p=2, dim=-1)       # (N, K, Nmax)
    sim = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # (N, K, K)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    if pairs:
        overlap_vals = [sim[:, i, j].mean().item() for i, j in pairs]
        pairwise_overlap = float(sum(overlap_vals) / len(overlap_vals))
    else:
        pairwise_overlap = float("nan")

    # Top-stub collision
    top_stub = attn.argmax(dim=-1)                    # (N, K)
    if pairs:
        collision_per_pair = [
            (top_stub[:, i] == top_stub[:, j]).float().mean().item()
            for i, j in pairs
        ]
        top_collision_rate = float(sum(collision_per_pair) / len(collision_per_pair))
    else:
        top_collision_rate = float("nan")

    return {
        "slot_entropy_mean":    entropy_mean,
        "slot_entropy_std":     entropy_std,
        "slot_pairwise_overlap":   pairwise_overlap,
        "slot_top_collision_rate": top_collision_rate,
        "n_windows": N,
        "K": K,
        "Nmax": Nmax,
    }


def _compute_metrics(acc: dict, dataset_name: str, model_name: str) -> dict:
    result = {}

    # --- Flatten stub-level tensors ---
    logits_flat = torch.cat([x.reshape(-1) for x in acc["node_logits"]])
    labels_flat = torch.cat([x.reshape(-1) for x in acc["node_labels"]])
    valid_flat  = torch.cat([x.reshape(-1) for x in acc["valid"]])
    ambig_flat  = torch.cat([x.reshape(-1) for x in acc["ambig"]])
    tid_flat    = torch.cat([x.reshape(-1) for x in acc["tid"]])

    result["node_auc"]  = node_auc(logits_flat, labels_flat, valid_flat, ambig_flat)
    result["stub_eff"]  = stub_recovery_efficiency(logits_flat, labels_flat, valid_flat)
    result["n_windows"] = acc["n_windows"]

    # Candidate recovery proxy (stub-level, all models with node_logits)
    if dataset_name not in NOISE_DATASETS:
        result["cand_rec_proxy"] = candidate_recovery_proxy(
            logits_flat, labels_flat, tid_flat, valid_flat
        )

    # --- Slot / candidate head metrics ---
    if acc["cand_logits"] and acc["gen_pt"]:
        cand = torch.cat(acc["cand_logits"])   # (N_total, K)
        gpt  = torch.cat(acc["gen_pt"])        # (N_total, K)

        if dataset_name not in NOISE_DATASETS:
            # Slot-based candidate recovery (fixed-index assignment)
            result["cand_recovery"] = candidate_recovery_slot(cand, gpt)

            # Trigger efficiency curve
            eff_curve = trigger_efficiency_vs_pt(cand, gpt, PT_THRESHOLDS)
            result["trig_eff_curve"] = eff_curve

            # Spot values at canonical thresholds
            for pt_cut in (5.0, 10.0, 20.0):
                if pt_cut in eff_curve["thresholds"]:
                    idx = eff_curve["thresholds"].index(pt_cut)
                    result[f"trig_eff@{int(pt_cut)}"] = eff_curve["efficiency"][idx]
                    result[f"trig_eff@{int(pt_cut)}_n"] = eff_curve["n_windows"][idx]

            # Dimuon separation (multi-muon datasets)
            if dataset_name in MULTI_MUON_DATASETS:
                result["dimuon_sep"] = dimuon_separation_efficiency(cand, gpt)

        # Background acceptance for noise datasets
        if dataset_name in NOISE_DATASETS:
            result["bg_accept"] = background_acceptance_slots(cand)

    elif acc["cand_logits"] and dataset_name in NOISE_DATASETS:
        # No gen_pt for B4 (expected), but can still compute bg_accept
        cand = torch.cat(acc["cand_logits"])
        result["bg_accept"] = background_acceptance_slots(cand)

    elif not acc["cand_logits"] and dataset_name in NOISE_DATASETS:
        # Node-logit based bg_accept fallback (deepsets)
        logits_2d = torch.cat(acc["node_logits"])  # (B, Nmax)
        valid_2d  = torch.cat(acc["valid"])
        result["bg_accept"] = background_acceptance(logits_2d, valid_2d)

    # --- pT resolution (per-slot models with gen_pt) ---
    if acc["pt_pred"] and acc["gen_pt"] and dataset_name not in NOISE_DATASETS:
        pt_p = torch.cat(acc["pt_pred"])
        gpt  = torch.cat(acc["gen_pt"])
        pt_res = pt_resolution_slots(pt_p, gpt)
        result["pt_mae"]     = pt_res["mae"]
        result["pt_sigma68"] = pt_res["sigma68"]
        result["pt_n_muons"] = pt_res["n_muons"]

    # --- dXY resolution (per-slot models with gen_dxy) ---
    if acc["dxy_pred"] and acc["gen_pt"] and acc["gen_dxy"] and dataset_name not in NOISE_DATASETS:
        dxy_p = torch.cat(acc["dxy_pred"])
        gpt   = torch.cat(acc["gen_pt"])
        gdxy  = torch.cat(acc["gen_dxy"])
        dxy_res = dxy_resolution_slots(dxy_p, gpt, gdxy)
        result["dxy_mae"]           = dxy_res["mae"]
        result["dxy_sigma68"]       = dxy_res["sigma68"]
        result["dxy_mae_displaced"] = dxy_res["mae_displaced"]

    # --- Edge metrics ---
    if acc["edge_logits"]:
        result["edge_auc"] = edge_auc(
            torch.cat(acc["edge_logits"]),
            torch.cat(acc["edge_labels"]),
            torch.cat(acc["edge_ambig"]) if acc["edge_ambig"] else None,
        )

    # --- Candidate threshold scan ---
    if acc["cand_logits"]:
        result["threshold_scan"] = _candidate_threshold_scan(
            acc["cand_logits"], acc["gen_pt"], dataset_name
        )

    # --- Slot specialization diagnostics ---
    if acc["slot_attn"]:
        result["slot_diagnostics"] = _slot_diagnostics(acc["slot_attn"])

    # --- Binomial confidence intervals for proportion metrics ---
    _n_total = acc["n_windows"]
    for key in ("stub_eff", "bg_accept", "cand_recovery", "dimuon_sep"):
        if key in result and _n_total > 0:
            result[f"{key}_ci95"] = list(_wilson_ci(result[key], _n_total))
    for pt_cut in (5, 10, 20):
        k = f"trig_eff@{pt_cut}"
        nk = result.get(f"{k}_n", 0)
        if k in result and nk > 0:
            result[f"{k}_ci95"] = list(_wilson_ci(result[k], nk))

    return result


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion p observed over n trials."""
    if p != p or n == 0:  # nan or zero
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# Thresholds swept for the candidate operating-point scan
_CAND_THRESHOLDS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def _candidate_threshold_scan(
    cand_logits_list: list,
    gen_pt_list: list,
    dataset_name: str,
) -> dict:
    """
    Sweep candidate thresholds and return per-threshold efficiency (signal) or
    bg_accept (noise).  Returns a dict suitable for JSON serialization.
    """
    cand = torch.cat(cand_logits_list)  # (N, K)

    if gen_pt_list:
        gpt = torch.cat(gen_pt_list)    # (N, K)
        is_signal = (gpt > 0).any(dim=1)
    else:
        is_signal = torch.zeros(cand.shape[0], dtype=torch.bool)

    is_noise = ~is_signal

    efficiencies: list[float] = []
    bg_accepts:   list[float] = []

    for thr in _CAND_THRESHOLDS:
        fires = (cand > thr).any(dim=1)
        eff = float(fires[is_signal].float().mean()) if is_signal.any() else float("nan")
        bga = float(fires[is_noise].float().mean())  if is_noise.any()  else float("nan")
        efficiencies.append(eff)
        bg_accepts.append(bga)

    return {
        "thresholds":    _CAND_THRESHOLDS,
        "efficiency":    efficiencies,
        "bg_accept":     bg_accepts,
        "is_noise_only": bool(dataset_name in NOISE_DATASETS),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

# Which metrics go into the compact ADG comparison table (ordered)
_TABLE_KEYS = [
    "node_auc", "stub_eff", "cand_rec_proxy", "cand_recovery",
    "trig_eff@5", "trig_eff@10", "trig_eff@20",
    "dimuon_sep", "bg_accept",
    "pt_mae", "pt_sigma68",
    "dxy_mae", "dxy_sigma68", "dxy_mae_displaced",
    "edge_auc",
]


def print_comparison_table(
    results_a: dict,
    results_b: dict,
    name_a: str,
    name_b: str,
):
    """Print a side-by-side ADG comparison table."""
    all_datasets = sorted(set(list(results_a) + list(results_b)))
    all_keys = []
    for ds in all_datasets:
        for k in _TABLE_KEYS:
            if (k in results_a.get(ds, {}) or k in results_b.get(ds, {})) and k not in all_keys:
                all_keys.append(k)

    col_w = 9
    ds_w  = 6
    name_w = max(len(name_a), len(name_b), 12) + 2

    print(f"\n{'='*80}")
    print(f"  ADG Comparison: {name_a}  vs  {name_b}")
    print('='*80)

    for ds in all_datasets:
        ra = results_a.get(ds, {})
        rb = results_b.get(ds, {})
        keys_present = [k for k in all_keys if k in ra or k in rb]
        if not keys_present:
            continue

        print(f"\n  Dataset: {ds}")
        hdr = f"  {'Model':{name_w}s}" + "".join(f"  {k:{col_w}s}" for k in keys_present)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for name, res in [(name_a, ra), (name_b, rb)]:
            row = f"  {name:{name_w}s}"
            for k in keys_present:
                v = res.get(k, float("nan"))
                if isinstance(v, float):
                    row += f"  {v:{col_w}.4f}"
                else:
                    row += f"  {'—':{col_w}s}"
            print(row)

    print()
    print(EVAL_DISCLAIMER)
    print('='*80)


def save_results(
    results: dict,
    model_name: str,
    checkpoint_path: str,
    *,
    split_file: str | None = None,
    split_name: str | None = None,
    epoch: int | str | None = None,
    candidate_threshold: float = 0.0,
) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"{model_name}_{ts}.json"

    serializable = {}
    for ds, metrics in results.items():
        serializable[ds] = {}
        for k, v in metrics.items():
            if isinstance(v, (float, int, str)):
                serializable[ds][k] = v
            elif isinstance(v, dict):
                serializable[ds][k] = v

    # Git provenance (best-effort)
    git_hash = None
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    payload = {
        "model":               model_name,
        "checkpoint":          str(checkpoint_path),
        "epoch":               epoch,
        "split_file":          str(split_file) if split_file else None,
        "split":               split_name,
        "candidate_threshold": candidate_threshold,
        "timestamp":           ts,
        "git_hash":            git_hash,
        "results":             serializable,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _PROFILE

    parser = argparse.ArgumentParser(description="OMTF evaluation")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--checkpoint2", default=None,
                        help="Second checkpoint for side-by-side ADG comparison")
    parser.add_argument("--datasets", nargs="+",
                        default=["S1", "S3", "B1", "B4"],
                        help="Datasets to evaluate")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Evaluate all 9 datasets")
    parser.add_argument("--cache-dir",   type=Path, default=None,
                        help="Pre-built .pt shard cache (from build_cache.py). "
                             "Falls back to ROOT if a dataset is not in the cache.")
    parser.add_argument("--split-file",  type=Path, default=None,
                        help="Path to a splits JSON saved by train.py. "
                             "When provided, --split selects which partition to evaluate. "
                             "Forces ROOT-based loading (cache is not split-aware).")
    parser.add_argument("--split",       default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate when --split-file is given (default: test)")
    parser.add_argument("--max-files",   type=int, default=None)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--batch-size",  type=int, default=512)
    parser.add_argument("--save",        action="store_true",
                        help="Save results to build/omtf/eval/")
    parser.add_argument("--profile",     action="store_true",
                        help="Print timing for each phase")
    parser.add_argument("--cpu",         action="store_true")
    args = parser.parse_args()

    _PROFILE = args.profile
    set_root_batch_mode()

    # Device
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        try:
            torch.empty(1, device="cuda")
        except Exception:
            use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    pin_memory = use_cuda

    datasets = (["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]
                if args.all_datasets else args.datasets)

    # Load model(s)
    print("\nLoading checkpoint(s)...")
    model_a, name_a = load_model(args.checkpoint, device)
    model_b, name_b = None, None
    if args.checkpoint2:
        model_b, name_b = load_model(args.checkpoint2, device)

    need_graph_a = name_a in ("edge_mlp", "edge_compat")
    need_graph_b = name_b in ("edge_mlp", "edge_compat") if name_b else False
    need_graph   = need_graph_a or need_graph_b

    # Held-out split: load explicit file list per dataset
    split_files_by_ds: dict[str, list[Path]] | None = None
    if args.split_file is not None:
        from omtf.splits import load_splits
        all_splits = load_splits(args.split_file)
        chosen = all_splits.get(args.split, [])
        if not chosen:
            print(f"[WARN] --split-file has no '{args.split}' partition, "
                  f"keys: {list(all_splits)}")
        split_files_by_ds = {}
        for p in chosen:
            ds_name = Path(p).parent.name
            split_files_by_ds.setdefault(ds_name, []).append(Path(p))
        print(f"  Split '{args.split}' from {args.split_file}: "
              f"{sum(len(v) for v in split_files_by_ds.values())} files across "
              f"{list(split_files_by_ds)} (ROOT-only, cache ignored)")

    # Evaluate
    results_a, results_b = {}, {}

    for ds in datasets:
        print(f"\n--- {ds} ---")
        with _timer("make_loader"):
            # When a split file is given, use ROOT-only loading for that file subset
            if split_files_by_ds is not None:
                ds_files = split_files_by_ds.get(ds, [])
                if not ds_files:
                    print(f"  [SKIP] no {args.split} files for {ds}")
                    continue
                from omtf.dataset import OMTFDataset
                split_ds = OMTFDataset(
                    ds_files, Nmax=NMAX,
                    include_graph=need_graph,
                    include_nano=True,
                    max_entries=args.max_entries,
                )
                print(f"  [split:{args.split}] {len(split_ds):,} samples "
                      f"from {len(ds_files)} files")
                loader = DataLoader(
                    split_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_omtf, num_workers=0,
                    pin_memory=pin_memory,
                )
            else:
                loader = make_loader(
                    ds, need_graph, args.max_files, args.max_entries,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                    pin_memory=pin_memory,
                )
        if loader is None:
            print(f"  [SKIP] no files found")
            continue

        if model_b is not None:
            # One-pass: both models on the same batches
            print(f"  {name_a} + {name_b} (one-pass):", end=" ", flush=True)
            results_a[ds], results_b[ds] = evaluate_dataset_pair(
                model_a, name_a, model_b, name_b, loader, device, ds
            )
            _print_inline_summary(results_a[ds])
            print(f"  {name_b}:", end=" ", flush=True)
            _print_inline_summary(results_b[ds])
        else:
            print(f"  {name_a}:", end=" ", flush=True)
            results_a[ds] = evaluate_dataset(model_a, name_a, loader, device, ds)
            _print_inline_summary(results_a[ds])

    # Print results
    if model_b is not None:
        print_comparison_table(results_a, results_b, name_a, name_b)
    else:
        display = {ds: {k: v for k, v in m.items() if k != "trig_eff_curve"}
                   for ds, m in results_a.items()}
        print_metrics_table(display, title=f"OMTF Evaluation — {name_a}")

    # Save
    if args.save:
        _ckpt_a = torch.load(args.checkpoint, map_location="cpu")
        _epoch_a = _ckpt_a.get("epoch", None)
        _sf = str(args.split_file) if args.split_file else None
        _sn = args.split if args.split_file else None
        save_results(results_a, name_a, args.checkpoint,
                     split_file=_sf, split_name=_sn, epoch=_epoch_a)
        if model_b is not None:
            _ckpt_b = torch.load(args.checkpoint2, map_location="cpu")
            _epoch_b = _ckpt_b.get("epoch", None)
            save_results(results_b, name_b, args.checkpoint2,
                         split_file=_sf, split_name=_sn, epoch=_epoch_b)


def _print_inline_summary(metrics: dict):
    parts = []
    for k in ("node_auc", "stub_eff", "cand_recovery", "trig_eff@10", "bg_accept", "edge_auc"):
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float) and not (v != v):  # not nan
                parts.append(f"{k}={v:.4f}")
    print("  ".join(parts) if parts else "(no metrics)")


if __name__ == "__main__":
    main()
