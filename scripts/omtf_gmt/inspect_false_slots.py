"""
Diagnostic script for false slot-1 analysis in S2/B2.

For every S2/B2 validation window where true_mult=1 and slot-1 fires:

  1. seq_slot: inspect per-slot attention weights
     → classify false candidate as: duplicate / noise_coherent / diffuse / out_of_domain
     → compute attention mass by track_id per slot

  2. count_model: count confusion matrix (true_count → predicted_count)
     → also compute count-suppressed slot metrics (slots beyond argmax(count) are forced off)

Usage
-----
  python scripts/omtf_gmt/inspect_false_slots.py \
      --seq-slot-ckpt   build/omtf_gmt/checkpoints/seq_slot_B2a/gmt_seq_slot_best.pt \
      --count-model-ckpt build/omtf_gmt/checkpoints/count_model_B2a/gmt_count_model_best.pt \
      --cache-dir       build/omtf_gmt/cache \
      --datasets        S2 B2 \
      --output          build/omtf_gmt/eval/false_slot_diagnostic.md
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt
from omtf_gmt.models import build_seq_slot, build_count_model

THRESHOLD = 0.0


# --------------------------------------------------------------------------- #
# Patched seq_slot forward — identical to model but also returns per-slot attn
# --------------------------------------------------------------------------- #

def seq_slot_forward_with_attn(model, stubs, valid_mask):
    """Run seq_slot forward and return (out_dict, attn_list).

    attn_list[k] is (N, Nmax) softmax attention for slot k.
    """
    import torch.nn.functional as F
    N, Nmax, _ = stubs.shape
    device = stubs.device
    H = model.slot_queries.shape[1]

    node_emb = model.node_encoder(stubs)
    ei = node_emb.unsqueeze(2).expand(-1, -1, Nmax, -1)
    ej = node_emb.unsqueeze(1).expand(-1, Nmax, -1, -1)
    edge_score = torch.sigmoid(
        model.edge_encoder(torch.cat([ei, ej], dim=-1)).squeeze(-1)
    )
    vm = valid_mask
    pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)
    no_self = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)
    edge_mask = pair_valid & no_self
    w = edge_score * edge_mask.float()
    denom = w.sum(dim=2, keepdim=True).clamp(min=1e-6)
    w_n = (w / denom).unsqueeze(-1)
    ctx = (w_n * ej).sum(dim=2)
    node_upd = model.node_updater(torch.cat([node_emb, ctx], dim=-1))
    node_logit = model.node_head(node_upd).squeeze(-1)

    remaining = valid_mask.float().clone()
    cand_logits_list, pt_list, chg_list, attn_list = [], [], [], []

    for k in range(model.K):
        query = model.slot_queries[k]
        scores = (node_upd * query).sum(dim=-1)
        scores = scores + torch.log(remaining.clamp(min=1e-9))
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx_k = (attn.unsqueeze(-1) * node_upd).sum(dim=1)
        cand_logit_k = model.cand_heads[k](ctx_k).squeeze(-1)
        pt_k = F.softplus(model.pt_heads[k](ctx_k)).squeeze(-1)
        chg_k = torch.tanh(model.chg_heads[k](ctx_k)).squeeze(-1)
        cand_logits_list.append(cand_logit_k)
        pt_list.append(pt_k)
        chg_list.append(chg_k)
        attn_list.append(attn)
        claim_k = torch.sigmoid(cand_logit_k)
        remaining = remaining * (1.0 - claim_k.unsqueeze(-1) * attn)

    out = {
        "node_logit":       node_logit,
        "candidate_logits": torch.stack(cand_logits_list, dim=-1),
        "pt_pred":          torch.stack(pt_list,           dim=-1),
        "charge_pred":      torch.stack(chg_list,          dim=-1),
    }
    return out, attn_list  # attn_list[k]: (N, Nmax)


# --------------------------------------------------------------------------- #
# Attention classification helpers
# --------------------------------------------------------------------------- #

def classify_false_slot(attn, track_ids, valid_mask, ambiguous):
    """
    Classify a false slot-1 firing for a single window.

    attn      : (Nmax,) attention weights for this slot
    track_ids : (Nmax,) int — 0=noise, 1=signal muon, etc.
    valid_mask: (Nmax,) bool
    ambiguous : (Nmax,) bool

    Returns (label, masses) where masses is a dict of track_id → attention mass.
    Labels:
      duplicate      — >50% mass on track_id=1 (same muon as slot 0)
      out_of_domain  — >50% mass on track_id > 1 (another real object outside task)
      noise_coherent — >50% mass on track_id=0 but ≥3 layers represented
      noise_diffuse  — >50% mass on track_id=0 and <3 layers represented
    """
    vm = valid_mask.bool()
    a  = attn[vm]
    t  = track_ids[vm]
    amb = ambiguous[vm].bool()

    masses = {}
    for uid in t.unique().tolist():
        masses[int(uid)] = float(a[t == uid].sum())

    # layers: use stub index modulo 6 as a proxy (6 barrel layers in OMTF)
    # top stubs = those with attn > 0.05
    top_mask = a > 0.05
    n_top = int(top_mask.sum())
    # rough layer count: use position in valid stub list
    valid_positions = torch.where(vm)[0]
    top_positions = valid_positions[top_mask]
    n_layers_approx = len(set((top_positions % 6).tolist()))

    signal_mass = masses.get(1, 0.0)
    noise_mass  = masses.get(0, 0.0)
    other_mass  = sum(v for k, v in masses.items() if k not in (0, 1))

    if signal_mass > 0.5:
        label = "duplicate"
    elif other_mass > 0.5:
        label = "out_of_domain"
    elif noise_mass > 0.5 and n_layers_approx >= 3:
        label = "noise_coherent"
    else:
        label = "noise_diffuse"

    return label, masses


# --------------------------------------------------------------------------- #
# Main diagnostics
# --------------------------------------------------------------------------- #

def run_seq_slot_diagnostic(model, loader, device, dataset_name):
    model.eval()
    label_counts = defaultdict(int)
    n_false_slot1 = 0
    n_true_1cand  = 0

    # track_id mass totals across all false slot-1 windows
    mass_totals = defaultdict(float)

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            stubs = batch["stubs"]
            vm    = batch["valid_mask"]
            gpt   = batch["gen_pt"]
            tids  = batch["track_id"]
            amb   = batch["ambiguous"]

            out, attn_list = seq_slot_forward_with_attn(model, stubs, vm)
            cand_logits = out["candidate_logits"]  # (N, K)

            true_mult = (gpt > 0).sum(dim=1)  # (N,)
            pred_slot1_fires = cand_logits[:, 1] > THRESHOLD

            sel = (true_mult == 1) & pred_slot1_fires  # false slot-1 windows
            n_true_1cand  += int((true_mult == 1).sum())
            n_false_slot1 += int(sel.sum())

            attn1 = attn_list[1]  # (N, Nmax) slot-1 attention

            for b in sel.nonzero(as_tuple=True)[0]:
                label, masses = classify_false_slot(
                    attn1[b].cpu(), tids[b].cpu(), vm[b].cpu(), amb[b].cpu()
                )
                label_counts[label] += 1
                for tid, mass in masses.items():
                    mass_totals[tid] += mass

    total_false = max(1, n_false_slot1)
    lines = [f"\n### {dataset_name} — seq_slot false slot-1 classification\n"]
    lines.append(f"Windows with true_mult=1: {n_true_1cand}")
    lines.append(f"False slot-1 fires:       {n_false_slot1}  ({100*n_false_slot1/max(1,n_true_1cand):.1f}%)\n")
    lines.append("| False-slot type   | Count | Fraction |")
    lines.append("| --- | --- | --- |")
    for lbl in ["duplicate", "out_of_domain", "noise_coherent", "noise_diffuse"]:
        c = label_counts[lbl]
        lines.append(f"| {lbl:<18} | {c:5d} | {100*c/total_false:6.1f}% |")

    lines.append("\nMean attention mass by track_id (across all false slot-1 windows):")
    lines.append("| track_id | meaning | mean attn mass |")
    lines.append("| --- | --- | --- |")
    meaning = {0: "noise/PU", 1: "signal muon", 2: "second muon / out-of-domain"}
    for tid in sorted(mass_totals):
        m = mass_totals[tid] / total_false
        lines.append(f"| {tid} | {meaning.get(tid, 'other')} | {m:.3f} |")

    return "\n".join(lines)


def run_count_model_diagnostic(model, loader, device, dataset_name):
    model.eval()
    K = model.K

    # confusion matrix: true_count (rows) × pred_count (cols)
    confusion = torch.zeros(K + 1, K + 1, dtype=torch.long)

    # for suppressed metrics
    n_true_1cand = 0
    n_false_slot1_raw = 0
    n_false_slot1_suppressed = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            gpt   = batch["gen_pt"]
            out   = model(batch["stubs"], batch["valid_mask"])

            cand_logits  = out["candidate_logits"]   # (N, K)
            count_logits = out["count_logits"]        # (N, K+1)

            true_count = (gpt > 0).sum(dim=1).clamp(max=K)   # (N,)
            pred_count = count_logits.argmax(dim=-1)           # (N,)

            for tc, pc in zip(true_count.cpu().tolist(), pred_count.cpu().tolist()):
                confusion[int(tc), int(pc)] += 1

            # false slot-1 analysis (true_mult=1)
            true_1 = true_count == 1
            if true_1.any():
                n_true_1cand += int(true_1.sum())
                raw_s1       = (cand_logits[:, 1] > THRESHOLD) & true_1
                # count-suppressed: slot 1 only counts if pred_count >= 2
                sup_s1       = raw_s1 & (pred_count >= 2)
                n_false_slot1_raw        += int(raw_s1.sum())
                n_false_slot1_suppressed += int(sup_s1.sum())

    lines = [f"\n### {dataset_name} — count_model diagnostics\n"]

    # count confusion matrix
    lines.append("**Count confusion matrix** (rows = true count, cols = predicted count):\n")
    lines.append("| true \\ pred | 0 | 1 | 2 | 3 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for tc in range(K + 1):
        row_total = confusion[tc].sum().item()
        if row_total == 0:
            continue
        cells = " | ".join(
            f"{confusion[tc, pc].item():5d} ({100*confusion[tc,pc].item()/row_total:4.1f}%)"
            for pc in range(K + 1)
        )
        lines.append(f"| true={tc} | {cells} |")

    n1 = max(1, n_true_1cand)
    lines.append(f"\n**Slot-1 fake rate (true_mult=1 windows):**")
    lines.append(f"  raw (ignoring count head):        {n_false_slot1_raw}/{n_true_1cand} = {100*n_false_slot1_raw/n1:.1f}%")
    lines.append(f"  count-suppressed (pred_count>=2): {n_false_slot1_suppressed}/{n_true_1cand} = {100*n_false_slot1_suppressed/n1:.1f}%")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def load_ckpt(path, builder, device):
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    args  = ckpt.get("args", {})
    model = builder(hidden=int(args.get("hidden", 64)),
                    dropout=float(args.get("dropout", 0.0)))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-slot-ckpt",    type=Path, required=True)
    p.add_argument("--count-model-ckpt", type=Path, required=True)
    p.add_argument("--cache-dir",        type=Path, required=True)
    p.add_argument("--datasets",         nargs="+", default=["S2", "B2"])
    p.add_argument("--output",           type=Path,
                   default=Path("build/omtf_gmt/eval/false_slot_diagnostic.md"))
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size",       type=int, default=512)
    args = p.parse_args()

    device = torch.device(args.device)
    seq_slot_model   = load_ckpt(args.seq_slot_ckpt,    build_seq_slot,    device)
    count_model      = load_ckpt(args.count_model_ckpt, build_count_model, device)

    sections = ["# False Slot-1 Diagnostic — S2/B2\n"]
    sections.append(f"seq_slot checkpoint:    `{args.seq_slot_ckpt}`")
    sections.append(f"count_model checkpoint: `{args.count_model_ckpt}`")
    sections.append(f"threshold: {THRESHOLD}\n")

    for ds_name in args.datasets:
        ds     = GMTCachedDataset(args.cache_dir, ds_name)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_gmt, num_workers=2)
        print(f"Running {ds_name} ({len(ds)} windows)...")

        sections.append(f"\n---\n## Dataset: {ds_name}")
        sections.append(run_seq_slot_diagnostic(seq_slot_model, loader, device, ds_name))
        sections.append(run_count_model_diagnostic(count_model, loader, device, ds_name))

    out_text = "\n".join(sections) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out_text)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
