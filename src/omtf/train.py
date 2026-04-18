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
from omtf.splits import build_splits
from omtf.losses import node_bce_loss, edge_bce_loss, pt_regression_loss
from omtf.metrics import stub_recovery_efficiency, edge_auc, node_auc
from omtf.models.baselines import build_baseline
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
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from file-level splits."""
    set_root_batch_mode()

    splits = build_splits(DATA_ROOT, datasets, train_frac=0.75, val_frac=0.15, seed=seed)

    if max_files_per_ds is not None:
        # Subsample to at most max_files_per_ds files per dataset
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
# Training loop
# --------------------------------------------------------------------------

def train(args):
    set_root_batch_mode()

    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        try:
            _ = torch.empty(1, device="cuda")
        except Exception as e:
            print(f"[WARN] CUDA reported available but is not usable: {e}")
            print("[WARN] Falling back to CPU.")
            use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    include_graph = (args.model == "edge_mlp") or args.graph

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
    )

    print(f"\nBuilding model: {args.model}  hidden={args.hidden}")
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
    best_val_loss = float("inf")
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

    compute_loss = (
        _compute_loss_deepsets if args.model == "deepsets" else _compute_loss_edge_mlp
    )
    validate = (
        _validate_deepsets if args.model == "deepsets" else _validate_edge_mlp
    )

    print(f"\nTraining for {args.epochs} epochs\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = compute_loss(model, batch, device)
            if loss.requires_grad:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(torch.tensor(train_losses).mean())

        # Validation loss
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = compute_loss(model, batch, device)
                val_losses.append(loss.item())
        val_loss = float(torch.tensor(val_losses).mean())
        scheduler.step(val_loss)

        # Metrics
        metrics = validate(model, val_loader, device)

        elapsed = time.time() - t0
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "elapsed_s": elapsed, **metrics}
        history.append(row)

        # Print
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                if not isinstance(v, float) or v == v)
        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"{metrics_str}  ({elapsed:.1f}s)")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
                "history": history,
            }, ckpt_path)
            print(f"  --> saved best checkpoint (val_loss={best_val_loss:.4f})")

    # Save history
    hist_path = CHECKPOINT_DIR / f"{args.model}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved: {hist_path}")
    print(f"Best val_loss: {best_val_loss:.4f}  checkpoint: {ckpt_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OMTF baseline training")
    parser.add_argument("--model", choices=["deepsets", "edge_mlp"], default="deepsets")
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
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
