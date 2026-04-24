"""
OMTF training entrypoint — Stage 3 baselines.

Supports Baseline A (DeepSets) and Baseline B (pure edge MLP).

Usage
-----
    # Baseline A on S1+B1+B4, 50 epochs
    python src/omtf/train.py --model deepsets --datasets S1 B1 B4 --epochs 50

    # Baseline B on S1
    python src/omtf/train.py --model edge_mlp --datasets S1 --epochs 30 --graph

    # Resume from checkpoint
    python src/omtf/train.py --model deepsets --datasets S1 B1 B4 --resume build/omtf/checkpoints/deepsets_best.pt

Checkpoints saved to: build/omtf/checkpoints/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omtf.dataset import OMTFDataset, collate_omtf
from omtf.cache_dataset import CachedOMTFDataset
from omtf.splits import build_splits, save_splits
from omtf.losses import node_bce_loss, edge_bce_loss, combined_loss
from omtf.metrics import stub_recovery_efficiency, edge_auc, node_auc
from omtf.models.baselines import build_baseline
from omtf.models.edge_compat import build_edge_compat
from omtf.models.slot_model import build_slot_model, compute_slot_signal_mass
from audit.root_utils import set_root_batch_mode

CHECKPOINT_DIR = PROJECT_ROOT / "build" / "omtf" / "checkpoints"
DATA_ROOT = PROJECT_ROOT / "data" / "prod"
NMAX = 24


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------

def _build_loaders(
    datasets: list[str],
    batch_size: int,
    include_graph: bool,
    max_files_per_ds: int | None,
    max_entries: int | None,
    num_workers: int,
    pin_memory: bool,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from file-level splits or .pt cache."""
    from torch.utils.data import ConcatDataset, Subset

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        print(f"  Using .pt cache: {cache_dir}")

        all_ds = [
            CachedOMTFDataset(cache_dir, ds, include_graph=include_graph,
                              max_entries=max_entries)
            for ds in datasets
        ]

        # Deterministic 75/15/10 split by index within each dataset
        import random as _random
        rng = _random.Random(seed)

        train_subsets, val_subsets = [], []
        for ds_obj in all_ds:
            n = len(ds_obj)
            indices = list(range(n))
            rng.shuffle(indices)
            n_train = int(0.75 * n)
            n_val   = int(0.15 * n)
            train_subsets.append(Subset(ds_obj, indices[:n_train]))
            val_subsets.append(Subset(ds_obj, indices[n_train:n_train + n_val]))

        train_ds = ConcatDataset(train_subsets)
        val_ds   = ConcatDataset(val_subsets)
    else:
        set_root_batch_mode()

        splits = build_splits(DATA_ROOT, datasets, train_frac=0.75, val_frac=0.15, seed=seed)

        if max_files_per_ds is not None:
            from collections import defaultdict
            by_ds: dict[str, list[Path]] = defaultdict(list)
            for p in splits["train"]:
                by_ds[p.parent.name].append(p)
            limited_train = []
            for ds, files in by_ds.items():
                limited_train.extend(files[:max_files_per_ds])
            splits["train"] = limited_train

            by_ds_val: dict[str, list[Path]] = defaultdict(list)
            for p in splits["val"]:
                by_ds_val[p.parent.name].append(p)
            limited_val = []
            for ds, files in by_ds_val.items():
                limited_val.extend(files[:max(1, max_files_per_ds // 2)])
            splits["val"] = limited_val

        train_ds = OMTFDataset(splits["train"], Nmax=NMAX, include_graph=include_graph,
                               max_entries=max_entries)
        val_ds   = OMTFDataset(splits["val"],   Nmax=NMAX, include_graph=include_graph,
                               max_entries=max_entries)

    print(f"  Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_omtf, num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_omtf, num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    return train_loader, val_loader


# --------------------------------------------------------------------------
# Loss computation per model type
# --------------------------------------------------------------------------

def _compute_loss_deepsets(model, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    stubs      = batch["stubs"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    node_label = batch["node_label"].to(device)
    ambiguous  = batch["ambiguous"].to(device).bool()

    out = model(stubs, valid_mask)
    node_logits = out["node_logits"]

    node_loss = node_bce_loss(
        node_logits.reshape(-1),
        node_label.reshape(-1),
        valid_mask=valid_mask.reshape(-1),
        ambiguous_mask=ambiguous.reshape(-1),
    )

    parts = {"node_bce": node_loss, "total": node_loss}
    return node_loss, parts


def _slot_head_losses(
    out: dict,
    batch: dict,
    device: torch.device,
    slot_signal_mass: "torch.Tensor | None" = None,
) -> tuple[torch.Tensor, dict]:
    """
    Shared supervision for the K slot output heads (candidate, pT, charge, dXY).
    Works for both EdgeCompatNet and SlotModel.

    Candidate BCE uses hard gen_pt targets when available (correct for both signal
    and noise windows — B4 windows have gen_pt=0 everywhere, which pushes all
    candidate logits toward zero and directly suppresses background acceptance).
    Falls back to slot_signal_mass proxy when gen_pt not in batch.
    slot_signal_mass must be computed externally (compute_slot_signal_mass) and
    passed in; it is no longer read from out[].
    """
    parts: dict = {}
    total = torch.tensor(0.0, device=device)

    gen_pt     = batch.get("gen_pt")
    gen_charge = batch.get("gen_charge")
    gen_dxy    = batch.get("gen_dxy")

    # --- Candidate BCE ---
    if gen_pt is not None:
        cand_target = (gen_pt.to(device) > 0).float()          # (B, K) hard binary
        cand_loss = nn.functional.binary_cross_entropy_with_logits(
            out["candidate_logits"], cand_target
        )
    elif slot_signal_mass is not None:
        cand_loss = nn.functional.binary_cross_entropy_with_logits(
            out["candidate_logits"], slot_signal_mass
        )
    else:
        cand_loss = torch.tensor(0.0, device=device)
    total = total + 0.5 * cand_loss
    parts["cand_bce"] = cand_loss

    # --- pT regression (log scale MSE, occupied slots only) ---
    if gen_pt is not None:
        gpt = gen_pt.to(device)
        occupied = gpt > 0
        if occupied.any():
            pt_loss = nn.functional.mse_loss(
                out["pt_pred"][occupied],
                torch.log(gpt[occupied].clamp(min=1e-3)),
            )
            total = total + 0.1 * pt_loss
            parts["pt_reg"] = pt_loss

    # --- Charge BCE (+1/-1 → binary, occupied slots only) ---
    if gen_charge is not None:
        gc = gen_charge.to(device)
        occ_c = gc != 0
        if occ_c.any():
            charge_loss = nn.functional.binary_cross_entropy_with_logits(
                out["charge_pred"][occ_c],
                (gc[occ_c] > 0).float(),
            )
            total = total + 0.1 * charge_loss
            parts["charge_bce"] = charge_loss

    # --- dXY regression (Huber delta=10 cm, occupied slots only) ---
    if gen_dxy is not None and gen_pt is not None:
        gd = gen_dxy.to(device)
        occupied = gen_pt.to(device) > 0
        if occupied.any():
            dxy_loss = nn.functional.huber_loss(
                out["dxy_pred"][occupied], gd[occupied], delta=10.0
            )
            total = total + 0.01 * dxy_loss
            parts["dxy_reg"] = dxy_loss

    return total, parts


def _compute_loss_edge_compat(model, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    stubs      = batch["stubs"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    node_label = batch["node_label"].to(device)
    ambiguous  = batch["ambiguous"].to(device).bool()

    edge_index = [ei.to(device) for ei in batch.get("edge_index", [])]
    edge_attr  = [ea.to(device) for ea in batch.get("edge_attr",  [])]
    edge_label = batch.get("edge_label", [])
    edge_ambig = batch.get("edge_ambig", [])

    out = model(stubs, valid_mask, edge_index, edge_attr)

    # Node BCE
    node_loss = node_bce_loss(
        out["node_logits"].reshape(-1),
        node_label.reshape(-1),
        valid_mask=valid_mask.reshape(-1),
        ambiguous_mask=ambiguous.reshape(-1),
    )

    # Edge BCE — average across samples in batch
    edge_loss = torch.tensor(0.0, device=device)
    n_edge_terms = 0
    for el_logit, el_label, el_ambig in zip(
            out["edge_logits"], edge_label, edge_ambig):
        if el_logit.shape[0] == 0:
            continue
        edge_loss = edge_loss + edge_bce_loss(
            el_logit, el_label.to(device), ambiguous_mask=el_ambig.to(device)
        )
        n_edge_terms += 1
    if n_edge_terms > 0:
        edge_loss = edge_loss / n_edge_terms

    # Slot heads (candidate, pT, charge, dXY)
    slot_loss, slot_parts = _slot_head_losses(out, batch, device)

    total = node_loss + edge_loss + slot_loss
    parts = {"node_bce": node_loss, "edge_bce": edge_loss, **slot_parts, "total": total}
    return total, parts


def _compute_loss_slot(model, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    stubs      = batch["stubs"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    node_label = batch["node_label"].to(device)
    ambiguous  = batch["ambiguous"].to(device).bool()

    out = model(stubs, valid_mask)

    # Proxy supervision target (fallback when gen_pt absent)
    ssm = compute_slot_signal_mass(out["attn"], node_label)

    # Node BCE
    node_loss = node_bce_loss(
        out["node_logits"].reshape(-1),
        node_label.reshape(-1),
        valid_mask=valid_mask.reshape(-1),
        ambiguous_mask=ambiguous.reshape(-1),
    )

    # Slot heads (candidate, pT, charge, dXY) via shared helper
    slot_loss, slot_parts = _slot_head_losses(out, batch, device, slot_signal_mass=ssm)

    total = node_loss + slot_loss
    parts = {"node_bce": node_loss, **slot_parts, "total": total}
    return total, parts


def _compute_loss_edge_mlp(model, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    if "edge_attr" not in batch or not batch["edge_attr"]:
        # No edges in this batch — skip
        dummy = torch.tensor(0.0, device=device, requires_grad=True)
        return dummy, {"edge_bce": dummy, "total": dummy}

    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for ea, el, eamb in zip(
        batch["edge_attr"], batch["edge_label"], batch["edge_ambig"]
    ):
        if ea.shape[0] == 0:
            continue
        ea   = ea.to(device)
        el   = el.to(device)
        eamb = eamb.to(device)
        logits = model(ea)
        loss = edge_bce_loss(logits, el, ambiguous_mask=eamb)
        total_loss = total_loss + loss
        n_terms += 1

    if n_terms > 0:
        total_loss = total_loss / n_terms

    parts = {"edge_bce": total_loss, "total": total_loss}
    return total_loss, parts


# --------------------------------------------------------------------------
# Validation metrics
# --------------------------------------------------------------------------

@torch.no_grad()
def _validate_deepsets(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels, all_valid, all_ambig = [], [], [], []

    for batch in loader:
        stubs      = batch["stubs"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        out = model(stubs, valid_mask)
        all_logits.append(out["node_logits"].cpu())
        all_labels.append(batch["node_label"])
        all_valid.append(batch["valid_mask"])
        all_ambig.append(batch["ambiguous"].bool())

    logits = torch.cat([x.reshape(-1) for x in all_logits])
    labels = torch.cat([x.reshape(-1) for x in all_labels])
    valid  = torch.cat([x.reshape(-1) for x in all_valid])
    ambig  = torch.cat([x.reshape(-1) for x in all_ambig])

    return {
        "node_auc":   node_auc(logits, labels, valid, ambig),
        "stub_eff":   stub_recovery_efficiency(logits, labels, valid),
    }


@torch.no_grad()
def _validate_slot(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels, all_valid, all_ambig = [], [], [], []

    for batch in loader:
        stubs      = batch["stubs"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        out = model(stubs, valid_mask)

        all_logits.append(out["node_logits"].cpu())
        all_labels.append(batch["node_label"])
        all_valid.append(batch["valid_mask"])
        all_ambig.append(batch["ambiguous"].bool())

    logits = torch.cat([x.reshape(-1) for x in all_logits])
    labels = torch.cat([x.reshape(-1) for x in all_labels])
    valid  = torch.cat([x.reshape(-1) for x in all_valid])
    ambig  = torch.cat([x.reshape(-1) for x in all_ambig])

    return {
        "node_auc": node_auc(logits, labels, valid, ambig),
        "stub_eff": stub_recovery_efficiency(logits, labels, valid),
    }


@torch.no_grad()
def _validate_edge_compat(model, loader: DataLoader, device: torch.device) -> dict:
    """Single-pass validation: loss + node/edge metrics + ADG proxies."""
    model.eval()
    all_node_logits, all_node_labels, all_valid, all_ambig = [], [], [], []
    all_edge_logits, all_edge_labels, all_edge_ambig = [], [], []
    all_cand, all_gpt = [], []
    val_losses: list[float] = []

    for batch in loader:
        stubs      = batch["stubs"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        edge_index = [ei.to(device) for ei in batch.get("edge_index", [])]
        edge_attr  = [ea.to(device) for ea in batch.get("edge_attr",  [])]

        out = model(stubs, valid_mask, edge_index, edge_attr)

        # --- loss ---
        node_loss = node_bce_loss(
            out["node_logits"].reshape(-1),
            batch["node_label"].reshape(-1).to(device),
            valid_mask=valid_mask.reshape(-1),
            ambiguous_mask=batch["ambiguous"].to(device).bool().reshape(-1),
        )
        edge_loss = torch.tensor(0.0, device=device)
        n_edge = 0
        for el_logit, el_label, el_ambig in zip(
                out["edge_logits"], batch.get("edge_label", []), batch.get("edge_ambig", [])):
            if el_logit.shape[0] == 0:
                continue
            edge_loss = edge_loss + edge_bce_loss(
                el_logit, el_label.to(device), ambiguous_mask=el_ambig.to(device))
            n_edge += 1
        if n_edge > 0:
            edge_loss = edge_loss / n_edge
        slot_loss, _ = _slot_head_losses(out, batch, device)
        val_losses.append((node_loss + edge_loss + slot_loss).item())

        # --- node/edge metrics ---
        all_node_logits.append(out["node_logits"].cpu())
        all_node_labels.append(batch["node_label"])
        all_valid.append(batch["valid_mask"])
        all_ambig.append(batch["ambiguous"].bool())

        for el_logit, el_label, el_ambig in zip(
                out["edge_logits"], batch.get("edge_label", []), batch.get("edge_ambig", [])):
            if el_logit.shape[0] == 0:
                continue
            all_edge_logits.append(el_logit.cpu())
            all_edge_labels.append(el_label)
            all_edge_ambig.append(el_ambig)

        # --- ADG proxies ---
        all_cand.append(out["candidate_logits"].cpu())
        K = out["candidate_logits"].shape[1]
        all_gpt.append(batch["gen_pt"] if "gen_pt" in batch
                       else torch.zeros(stubs.shape[0], K))

    val_loss = float(torch.tensor(val_losses).mean())

    logits = torch.cat([x.reshape(-1) for x in all_node_logits])
    labels = torch.cat([x.reshape(-1) for x in all_node_labels])
    valid  = torch.cat([x.reshape(-1) for x in all_valid])
    ambig  = torch.cat([x.reshape(-1) for x in all_ambig])

    metrics = {
        "node_auc": node_auc(logits, labels, valid, ambig),
        "stub_eff": stub_recovery_efficiency(logits, labels, valid),
    }

    if all_edge_logits:
        metrics["edge_auc"] = edge_auc(
            torch.cat(all_edge_logits),
            torch.cat(all_edge_labels),
            torch.cat(all_edge_ambig),
        )

    cand = torch.cat(all_cand)
    gpt  = torch.cat(all_gpt)
    is_signal = (gpt > 0).any(dim=1)
    fires     = (cand > 0).any(dim=1)

    def _safe_mean(mask):
        return float(fires[mask].float().mean()) if mask.any() else float("nan")

    trig_eff  = _safe_mean(is_signal)
    bg_accept = _safe_mean(~is_signal)
    nan = float("nan")
    combined  = (trig_eff - _ADG_LAMBDA * bg_accept
                 if trig_eff == trig_eff and bg_accept == bg_accept else nan)
    metrics.update({"trig_eff_proxy": trig_eff,
                    "bg_accept_proxy": bg_accept,
                    "combined_score": combined})

    return val_loss, metrics


@torch.no_grad()
def _validate_edge_mlp(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits, all_labels, all_ambig = [], [], []

    for batch in loader:
        if "edge_attr" not in batch:
            continue
        for ea, el, eamb in zip(batch["edge_attr"], batch["edge_label"], batch["edge_ambig"]):
            if ea.shape[0] == 0:
                continue
            logits = model(ea.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(el)
            all_ambig.append(eamb)

    if not all_logits:
        return {"edge_auc": float("nan")}

    return {
        "edge_auc": edge_auc(
            torch.cat(all_logits),
            torch.cat(all_labels),
            torch.cat(all_ambig),
        )
    }


# --------------------------------------------------------------------------
# ADG proxy metrics (candidate-level, computed on val loader)
# --------------------------------------------------------------------------

_ADG_LAMBDA = 1.0   # combined score = mean_trig_eff - λ * bg_accept


@torch.no_grad()
def _adg_proxy_metrics(
    model,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Lightweight ADG proxies from the existing val loader.

    For each window in the val set:
      - Signal window: gen_pt > 0 in at least one slot
      - Noise window:  all gen_pt == 0 (B4-style, no gen muon)

    Computes (at default threshold = 0):
      trig_eff_proxy  = frac of signal windows where any candidate fires
      bg_accept_proxy = frac of noise windows where any candidate fires
      combined_score  = trig_eff_proxy - _ADG_LAMBDA * bg_accept_proxy

    Only meaningful for models with candidate_logits (slot_model, edge_compat).
    Returns empty dict for other model types.
    """
    if model_name not in ("slot_model", "edge_compat"):
        return {}

    model.eval()
    all_cand: list[torch.Tensor] = []
    all_gpt:  list[torch.Tensor] = []

    for batch in loader:
        stubs      = batch["stubs"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        if model_name == "slot_model":
            out = model(stubs, valid_mask)
        else:
            edge_index = [ei.to(device) for ei in batch.get("edge_index", [])]
            edge_attr  = [ea.to(device) for ea in batch.get("edge_attr",  [])]
            out = model(stubs, valid_mask, edge_index, edge_attr)

        all_cand.append(out["candidate_logits"].cpu())
        K = out["candidate_logits"].shape[1]
        if "gen_pt" in batch:
            all_gpt.append(batch["gen_pt"])
        else:
            all_gpt.append(torch.zeros(stubs.shape[0], K))

    cand = torch.cat(all_cand)   # (N, K)
    gpt  = torch.cat(all_gpt)    # (N, K)

    is_signal = (gpt > 0).any(dim=1)
    is_noise  = ~is_signal
    fires     = (cand > 0).any(dim=1)

    def _safe_mean(mask):
        if mask.any():
            return float(fires[mask].float().mean())
        return float("nan")

    trig_eff  = _safe_mean(is_signal)
    bg_accept = _safe_mean(is_noise)

    nan = float("nan")
    combined = (trig_eff - _ADG_LAMBDA * bg_accept
                if trig_eff == trig_eff and bg_accept == bg_accept else nan)

    return {
        "trig_eff_proxy":  trig_eff,
        "bg_accept_proxy": bg_accept,
        "combined_score":  combined,
    }


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(args):
    set_root_batch_mode()

    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name:  {torch.cuda.get_device_name(0)}")
    if use_cuda:
        try:
            _ = torch.empty(1, device="cuda")
        except Exception as e:
            print(f"[WARN] CUDA reported available but is not usable: {e}")
            print("[WARN] Falling back to CPU.")
            use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    include_graph = (args.model in ("edge_mlp", "edge_compat")) or args.graph
    # slot_model does not need edge features — node features only

    print(f"\nBuilding datasets: {args.datasets}")
    train_loader, val_loader = _build_loaders(
        datasets=args.datasets,
        batch_size=args.batch_size,
        include_graph=include_graph,
        max_files_per_ds=args.max_files,
        max_entries=args.max_entries,
        num_workers=args.num_workers,
        pin_memory=(args.pin_memory and device.type == "cuda"),
        seed=args.seed,
        cache_dir=getattr(args, "cache_dir", None),
    )

    # Persist splits so eval.py can enforce a held-out test set
    splits_path = CHECKPOINT_DIR / f"{args.model}_splits.json"
    _raw_splits = build_splits(DATA_ROOT, args.datasets,
                               train_frac=0.75, val_frac=0.15, seed=args.seed)
    save_splits(_raw_splits, splits_path)
    print(f"  Splits saved: {splits_path}")

    print(f"\nBuilding model: {args.model}  hidden={args.hidden}")
    if args.model == "edge_compat":
        model = build_edge_compat(hidden=args.hidden, K=3, dropout=args.dropout).to(device)
    elif args.model == "slot_model":
        model = build_slot_model(hidden=args.hidden, K=3, dropout=args.dropout).to(device)
    else:
        model = build_baseline(args.model, hidden=args.hidden, K=3,
                               dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 0
    best_val_loss   = float("inf")
    best_trig_eff   = float("-inf")
    best_bg_accept  = float("inf")
    best_combined   = float("-inf")
    history: list[dict] = []

    # Resume
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"  Resumed from epoch {ckpt['epoch']}  best_val_loss={best_val_loss:.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{args.model}_best.pt"

    _loss_fns = {
        "deepsets":    _compute_loss_deepsets,
        "edge_mlp":    _compute_loss_edge_mlp,
        "edge_compat": _compute_loss_edge_compat,
        "slot_model":  _compute_loss_slot,
    }
    # edge_compat uses a combined single-pass validator (returns val_loss, metrics).
    # All others use separate validate + adg passes for backward compatibility.
    _val_fns = {
        "deepsets":    _validate_deepsets,
        "edge_mlp":    _validate_edge_mlp,
        "slot_model":  _validate_slot,
    }
    compute_loss = _loss_fns[args.model]

    print(f"\nTraining for {args.epochs} epochs\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        model.train()
        train_losses: list[float] = []

        t_train0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = compute_loss(model, batch, device)
            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(loss.item())
        t_train = time.time() - t_train0

        train_loss = float(torch.tensor(train_losses).mean())

        # Validation — single pass for edge_compat, separate passes otherwise
        t_val0 = time.time()
        if args.model == "edge_compat":
            val_loss, metrics = _validate_edge_compat(model, val_loader, device)
            adg = {k: metrics.pop(k) for k in
                   ("trig_eff_proxy", "bg_accept_proxy", "combined_score")
                   if k in metrics}
        else:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    loss, _ = compute_loss(model, batch, device)
                    val_losses.append(loss.item())
            val_loss = float(torch.tensor(val_losses).mean())
            validate = _val_fns[args.model]
            metrics = validate(model, val_loader, device)
            adg     = _adg_proxy_metrics(model, args.model, val_loader, device)
        t_val = time.time() - t_val0

        scheduler.step(val_loss)
        metrics.update(adg)

        elapsed = time.time() - t0
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "elapsed_s": elapsed, **metrics}
        history.append(row)

        # Print
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                if isinstance(v, float) and v == v)
        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"{metrics_str}  ({elapsed:.1f}s  train={t_train:.1f}s  val={t_val:.1f}s)")

        def _save_ckpt(suffix: str, extra: dict | None = None):
            path = ckpt_path.with_name(f"{args.model}_{suffix}.pt")
            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
                "history": history,
            }
            if extra:
                payload.update(extra)
            torch.save(payload, path)
            return path

        # --- Best val_loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_ckpt("best", {"criterion": "val_loss"})
            print(f"  --> best  (val_loss={best_val_loss:.4f})")

        # --- Best trig_eff proxy ---
        te = adg.get("trig_eff_proxy", float("nan"))
        if te == te and te > best_trig_eff:
            best_trig_eff = te
            _save_ckpt("best_trig", {"criterion": "trig_eff_proxy"})
            print(f"  --> best_trig  (trig_eff_proxy={best_trig_eff:.4f})")

        # --- Best bg_accept proxy (lower is better) ---
        ba = adg.get("bg_accept_proxy", float("nan"))
        if ba == ba and ba < best_bg_accept:
            best_bg_accept = ba
            _save_ckpt("best_rate", {"criterion": "bg_accept_proxy"})
            print(f"  --> best_rate  (bg_accept_proxy={best_bg_accept:.4f})")

        # --- Best combined score ---
        cs = adg.get("combined_score", float("nan"))
        if cs == cs and cs > best_combined:
            best_combined = cs
            _save_ckpt("best_combined", {"criterion": "combined_score"})
            print(f"  --> best_combined  (score={best_combined:.4f})")

    # Save history
    hist_path = CHECKPOINT_DIR / f"{args.model}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved: {hist_path}")
    print(f"Best val_loss:       {best_val_loss:.4f}  → {args.model}_best.pt")
    if best_trig_eff > float("-inf"):
        print(f"Best trig_eff_proxy: {best_trig_eff:.4f}  → {args.model}_best_trig.pt")
    if best_bg_accept < float("inf"):
        print(f"Best bg_accept_proxy:{best_bg_accept:.4f}  → {args.model}_best_rate.pt")
    if best_combined > float("-inf"):
        print(f"Best combined_score: {best_combined:.4f}  → {args.model}_best_combined.pt")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OMTF baseline training")
    parser.add_argument("--model", choices=["deepsets", "edge_mlp", "edge_compat", "slot_model"], default="deepsets")
    parser.add_argument("--datasets", nargs="+", default=["S1", "B1", "B4"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max ROOT files per dataset (for quick runs)")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Max entries per file (for quick runs)")
    parser.add_argument("--graph", action="store_true",
                        help="Force graph output (edge_index etc.) in dataset")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true",
                        help="Enable DataLoader pin_memory for CUDA runs")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Path to pre-built .pt shard cache (schema_v1_graph). "
                             "When set, bypasses ROOT loading entirely.")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
