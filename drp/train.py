"""
Training + evaluation.

Batches are (cell_idx, drug_idx, auc) triples. For each batch we gather the cell
feature rows and collate the corresponding molecular graphs into one batched graph.
Metrics: Spearman rho (the headline DrugCell metric), Pearson r, and RMSE — all
implemented in numpy so we don't need scipy.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn

from . import config
from .data import Dataset
from .drug_encoder import collate_graphs
from .model import HolisticDRP


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return _pearson(ar.astype(float), br.astype(float))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom > 0 else 0.0


def split_pairs(n: int, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train, val, test


def _make_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = indices[rng.permutation(len(indices))]
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]


def _run_batch(model, ds: Dataset, batch_idx, device, return_internals=False):
    pairs = ds.pairs[batch_idx]
    cell_ids = pairs[:, 0]
    drug_ids = pairs[:, 1]
    x_cell = torch.tensor(ds.X_cell[cell_ids], dtype=torch.float32, device=device)
    graphs = [ds.drug_graphs[d] for d in drug_ids]
    drug_batch = collate_graphs(graphs, device)
    y = torch.tensor(ds.y[batch_idx], dtype=torch.float32, device=device)
    out = model(x_cell, drug_batch, return_internals=return_internals)
    return out, y


@torch.no_grad()
def evaluate(model, ds, indices, device, batch_size) -> dict:
    model.eval()
    preds, ys = [], []
    for b in _make_batches(indices, batch_size, shuffle=False, seed=0):
        out, y = _run_batch(model, ds, b, device)
        preds.append(out.cpu().numpy())
        ys.append(y.cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(ys)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    return {"spearman": _spearman(p, t), "pearson": _pearson(p, t), "rmse": rmse,
            "n": len(t)}


def train(ds: Dataset, device: torch.device | None = None, verbose: bool = True):
    tcfg = config.TrainConfig()
    mcfg = config.ModelConfig()
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    torch.set_num_threads(tcfg.num_threads)
    device = device or torch.device("cpu")

    model = HolisticDRP(
        hierarchy=ds.hierarchy, n_genes=ds.cell_input_dim,
        atom_feat_dim=ds.atom_feat_dim, mcfg=mcfg, fusion_kind=config.FUSION,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    tr, va, te = split_pairs(len(ds.pairs), tcfg.val_frac, tcfg.test_frac, tcfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    loss_fn = nn.MSELoss()
    epochs = tcfg.epochs[config.MODE]

    best_val = -1.0
    best_state = None
    patience_left = tcfg.patience
    history = []

    if verbose:
        print(f"  params={n_params:,}  train/val/test={len(tr)}/{len(va)}/{len(te)}  "
              f"epochs={epochs}")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        nb = 0
        for b in _make_batches(tr, tcfg.batch_size, shuffle=True, seed=tcfg.seed + ep):
            out, y = _run_batch(model, ds, b, device)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        val = evaluate(model, ds, va, device, tcfg.batch_size)
        history.append({"epoch": ep, "train_loss": ep_loss / max(nb, 1), **val})
        if val["spearman"] > best_val:
            best_val = val["spearman"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = tcfg.patience
        else:
            patience_left -= 1
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
            print(f"    epoch {ep:3d}  loss={ep_loss/max(nb,1):.4f}  "
                  f"val_rho={val['spearman']:.3f}  val_rmse={val['rmse']:.3f}")
        if patience_left <= 0:
            if verbose:
                print(f"    early stop at epoch {ep} (best val_rho={best_val:.3f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, ds, te, device, tcfg.batch_size)
    train_eval = evaluate(model, ds, tr, device, tcfg.batch_size)
    elapsed = time.time() - t0

    return {
        "model": model,
        "splits": {"train": tr, "val": va, "test": te},
        "metrics": {"train": train_eval, "val": evaluate(model, ds, va, device, tcfg.batch_size),
                    "test": test},
        "history": history,
        "n_params": n_params,
        "seconds": elapsed,
    }
