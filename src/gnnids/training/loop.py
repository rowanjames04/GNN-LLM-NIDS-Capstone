"""The Phase 4 training loop, shared by every phase that trains this model.

Extracted from `scripts/train_gnn.py` on 2026-08-25 when Phase 5 needed it.
Phase 5 (leave-one-attack-out) trains the *same* model on *different* row
subsets, and Phase 9 (cross-dataset) trains it on a different dataset entirely.
Both are then a matter of choosing rows, not of reimplementing training -- which
is the only reason the zero-day protocol is a config change rather than a second
copy of this file drifting away from the first.

CLAUDE.md's convention is that pipeline logic lives in `src/gnnids/` and scripts
are thin CLIs over it. This module is that convention applied to the loop.

Note on imports: torch is imported at module scope here. That is safe because
nothing importing this module also imports xgboost -- the two bundle separate
OpenMP runtimes and segfault when loaded into one process on macOS (C3).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

from ..eval.metrics import choose_threshold
from ..eval.prevalence import report_both, subsample_to_prevalence
from ..models.dual_channel import DualChannelGNN, channel_attribution
from ..models.mlp import FocalLoss


def one_hot(cat: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    """Identical expansion to Phase 3, so Channel 1 sees the same input the
    baseline did."""
    blocks = []
    for j, card in enumerate(cardinalities):
        block = np.zeros((len(cat), card), dtype=np.float32)
        block[np.arange(len(cat)), np.clip(cat[:, j], 0, card - 1)] = 1.0
        blocks.append(block)
    return np.hstack(blocks)


def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def infer(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score every edge. Returns (scores, y, y_multiclass, fusion weights)."""
    model.eval()
    scores, ys, yms, alphas = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        logit_b, _, alpha = model(batch.x, batch.edge_index, batch.edge_attr)
        scores.append(torch.sigmoid(logit_b).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
        yms.append(batch.y_multiclass.cpu().numpy())
        if alpha is not None:
            alphas.append(alpha.cpu().numpy())
    return (
        np.concatenate(scores), np.concatenate(ys), np.concatenate(yms),
        np.concatenate(alphas) if alphas else None,
    )


def release(device: torch.device) -> None:
    """Return cached GPU memory to the allocator between runs.

    PyTorch's Metal allocator caches freed blocks rather than releasing them.
    Our graphs are variable-sized -- each window has a different node count --
    which fragments that cache. Suspected cause of U1, where the heaviest
    ablation degraded 51 -> 77 -> 206 s/epoch across three identical seeds
    while the two lighter ones stayed flat.
    """
    import gc
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def train_one(cfg, datasets, edge_dim, n_classes, ablation, seed, device,
              ckpt_path: Path | None = None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tcfg = cfg["train"]

    model = DualChannelGNN(
        edge_dim=edge_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        n_mlp_blocks=cfg["model"]["n_mlp_blocks"],
        n_gnn_layers=cfg["model"]["n_gnn_layers"],
        n_classes=n_classes,
        dropout=cfg["model"]["dropout"],
        bidirectional=cfg["model"]["bidirectional"],
        **ablation,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    focal = FocalLoss(tcfg["focal_alpha"], tcfg["focal_gamma"])
    ce = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(datasets["train"], batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=tcfg["batch_size"])

    target_prev = cfg["eval"]["target_prevalence"]
    best_ap, best_state, patience, epochs_run = -1.0, None, 0, 0
    epoch_seconds = []

    for epoch in range(tcfg["max_epochs"]):
        t_epoch = time.time()
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logit_b, logit_m, _ = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = focal(logit_b, batch.y.float())
            loss = loss + tcfg["multiclass_weight"] * ce(logit_m, batch.y_multiclass)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            total += float(loss.detach())

        val_scores, val_y, _, _ = infer(model, val_loader, device)
        # Select on PR-AUC at the *target* prevalence, not native. Selecting at
        # 64% attack would optimise for a regime we do not report.
        idx = subsample_to_prevalence(val_y, target_prev, cfg["output"]["seed"])
        ap = average_precision_score(val_y[idx], val_scores[idx])
        epochs_run = epoch + 1
        # Per-epoch timing distinguishes degradation *within* a run (thermal)
        # from degradation *between* runs (allocator accumulation).
        epoch_seconds.append(round(time.time() - t_epoch, 2))

        if ap > best_ap + 1e-5:
            best_ap, patience = ap, 0
            # Kept on CPU: the checkpoint does not need to occupy GPU memory
            # for the rest of training, and this is the object that gets saved.
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= tcfg["patience"]:
                break

    model.load_state_dict(best_state)

    val_scores, val_y, _, _ = infer(model, val_loader, device)
    vidx = subsample_to_prevalence(val_y, target_prev, cfg["output"]["seed"])
    threshold = choose_threshold(val_y[vidx], val_scores[vidx], cfg["eval"]["threshold_mode"])

    test_loader = DataLoader(datasets["test"], batch_size=tcfg["batch_size"])
    scores, y, ymc, alpha = infer(model, test_loader, device)

    out = report_both(y, scores, threshold, target_prev, ymc, datasets["families"],
                      cfg["output"]["seed"])
    out["val_pr_auc_at_target"] = round(float(best_ap), 5)
    out["epochs_run"] = epochs_run
    out["epoch_seconds"] = epoch_seconds
    out["epoch_seconds_first_last"] = (
        [epoch_seconds[0], epoch_seconds[-1]] if epoch_seconds else [])
    out["n_parameters"] = sum(p.numel() for p in model.parameters())
    if alpha is not None:
        out["channel_attribution"] = channel_attribution(torch.from_numpy(alpha))
        # The per-detection form of the V2 claim: on flows the model got right,
        # how much of the decision came from neighbourhood context?
        tp = (y == 1) & (scores >= threshold)
        if tp.any():
            out["channel_attribution"]["topological_weight_on_true_positives"] = round(
                float(alpha[tp, 1].mean()), 4)

    # Phase 6 attributes over a trained model, so it needs one to exist. Nine
    # Phase 4 models were trained and discarded because this was never written.
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state,
            "edge_dim": edge_dim, "n_classes": n_classes,
            "ablation": ablation, "seed": seed,
            "model_cfg": cfg["model"],
            "val_pr_auc_at_target": out["val_pr_auc_at_target"],
            "threshold": threshold,
        }, ckpt_path)
        out["checkpoint"] = str(ckpt_path.name)

    del model, opt, train_loader, val_loader, test_loader
    release(device)
    return out
