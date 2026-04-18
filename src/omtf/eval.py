"""
OMTF evaluation entrypoint — Stage 3.

Loads a checkpoint and evaluates on specified datasets using the four
canonical metrics from the migration plan.

Usage
-----
    python src/omtf/eval.py --checkpoint build/omtf/checkpoints/deepsets_best.pt \
                            --datasets S1 B4

    python src/omtf/eval.py --checkpoint build/omtf/checkpoints/edge_mlp_best.pt \
                            --datasets S1 B4 --model edge_mlp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omtf.dataset import OMTFDataset, collate_omtf
from omtf.metrics import (
    stub_recovery_efficiency,
    candidate_recovery_proxy,
    background_acceptance,
    edge_auc,
    node_auc,
    pt_metrics,
    print_metrics_table,
)
from omtf.models.baselines import build_baseline
from torch.utils.data import DataLoader
from audit.root_utils import set_root_batch_mode

DATA_ROOT = PROJECT_ROOT / "data" / "prod"
NMAX = 24


def _load_model(checkpoint_path: str, model_name: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})
    hidden  = args.get("hidden", 64)
    dropout = args.get("dropout", 0.0)
    model = build_baseline(model_name, hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    print(f"Loaded {model_name} from epoch {epoch}  val_loss={val_loss:.4f}")
    return model


def _make_loader(dataset: str, include_graph: bool, max_files: int | None,
                 max_entries: int | None, num_workers: int) -> DataLoader:
    set_root_batch_mode()

    d = DATA_ROOT / dataset
    files = sorted(d.glob(f"omtf_hits_{dataset}_*.root"))
    if max_files:
        files = files[:max_files]
    if not files:
        return None
    ds = OMTFDataset(files, Nmax=NMAX, include_graph=include_graph,
                     max_entries=max_entries)
    return DataLoader(ds, batch_size=256, shuffle=False,
                      collate_fn=collate_omtf, num_workers=num_workers,
                      persistent_workers=(num_workers > 0))


@torch.no_grad()
def evaluate_deepsets(model, loader: DataLoader, device: torch.device,
                      dataset_name: str) -> dict:
    all_node_logits, all_node_labels = [], []
    all_valid, all_ambig, all_tid = [], [], []

    for batch in loader:
        stubs      = batch["stubs"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        out = model(stubs, valid_mask)

        all_node_logits.append(out["node_logits"].cpu())
        all_node_labels.append(batch["node_label"])
        all_valid.append(batch["valid_mask"])
        all_ambig.append(batch["ambiguous"].bool())
        all_tid.append(batch["track_id"])

    logits = torch.cat([x.reshape(-1) for x in all_node_logits])
    labels = torch.cat([x.reshape(-1) for x in all_node_labels])
    valid  = torch.cat([x.reshape(-1) for x in all_valid])
    ambig  = torch.cat([x.reshape(-1) for x in all_ambig])
    tid    = torch.cat([x.reshape(-1) for x in all_tid])

    # Reshape for batch-aware metrics
    B_total = sum(x.shape[0] for x in all_node_logits)
    logits_2d = torch.cat(all_node_logits)           # (B_total, Nmax)
    labels_2d = torch.cat(all_node_labels)
    valid_2d  = torch.cat(all_valid)

    result = {
        "node_auc": node_auc(logits, labels, valid, ambig),
        "stub_eff": stub_recovery_efficiency(logits, labels, valid),
        "cand_rec": candidate_recovery_proxy(logits, labels, tid, valid),
    }

    # Background acceptance only meaningful for B4
    if dataset_name == "B4":
        result["bg_accept"] = background_acceptance(logits_2d, valid_2d)

    return result


@torch.no_grad()
def evaluate_edge_mlp(model, loader: DataLoader, device: torch.device,
                      dataset_name: str) -> dict:
    all_logits, all_labels, all_ambig = [], [], []
    all_node_logits_bg, all_valid_bg = [], []  # for bg_accept proxy via edge scores

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

    result = {}
    if all_logits:
        result["edge_auc"] = edge_auc(
            torch.cat(all_logits),
            torch.cat(all_labels),
            torch.cat(all_ambig),
        )

        if dataset_name == "B4":
            # Background acceptance: any edge score > 0 in this window
            n_windows = len(all_logits)
            n_any_positive = sum(1 for l in all_logits if (l > 0).any())
            result["bg_accept"] = n_any_positive / max(n_windows, 1)

    return result


def main():
    parser = argparse.ArgumentParser(description="OMTF evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["deepsets", "edge_mlp"], default="deepsets")
    parser.add_argument("--datasets", nargs="+", default=["S1", "B4"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

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
    model = _load_model(args.checkpoint, args.model, device)

    include_graph = (args.model == "edge_mlp")
    evaluate = evaluate_deepsets if args.model == "deepsets" else evaluate_edge_mlp

    results = {}
    for ds in args.datasets:
        print(f"\nEvaluating {ds} ...")
        loader = _make_loader(ds, include_graph, args.max_files, args.max_entries,
                              args.num_workers)
        if loader is None:
            print(f"  [SKIP] no files found")
            continue
        results[ds] = evaluate(model, loader, device, ds)

    print_metrics_table(results, title=f"OMTF Evaluation — {args.model}")


if __name__ == "__main__":
    main()
