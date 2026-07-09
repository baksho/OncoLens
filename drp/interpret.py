"""
Interpretability readout (post-hoc; not trained).

Two complementary views, matching the diagram's readout box:

1) RLIPP-style subsystem importance (DrugCell). For each subsystem we fit a tiny
   ridge regression from its state -> predicted response, and compare its predictive
   power to that of its children's states. RLIPP = rho(parent) / rho(children).
   RLIPP > 1 means the subsystem adds predictive signal beyond its children.

2) Drug-aware gate ranking (DrugVNN). The per-gene attention gate tells us which
   genes the drug "looked at" most; we average the gate over a set of pairs.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import Dataset
from .train import _run_batch, _make_batches, _spearman


def _ridge_fit_predict(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    # closed-form ridge with intercept
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb.T @ y)
    return Xb @ w


@torch.no_grad()
def collect_internals(model, ds: Dataset, indices, device, batch_size, max_n=2000):
    """Run the model with internals on up to max_n pairs; gather states, gates, y."""
    model.eval()
    indices = indices[:max_n]
    states_acc: dict[str, list] = {}
    gates, ys = [], []
    for b in _make_batches(indices, batch_size, shuffle=False, seed=0):
        out, y = _run_batch(model, ds, b, device, return_internals=True)
        pred, internals = out
        for s, v in internals["states"].items():
            states_acc.setdefault(s, []).append(v.cpu().numpy())
        gates.append(internals["gate"].cpu().numpy())
        ys.append(y.cpu().numpy())
    states = {s: np.concatenate(v, 0) for s, v in states_acc.items()}
    return states, np.concatenate(gates, 0), np.concatenate(ys, 0)


def rlipp_scores(model, ds: Dataset, indices, device, batch_size, top_k=10):
    states, gate, y = collect_internals(model, ds, indices, device, batch_size)
    H = ds.hierarchy
    rows = []
    for s in H.topo_order:
        kids = H.children.get(s, [])
        if not kids:
            continue  # leaves have no children to compare against
        parent_pred = _ridge_fit_predict(states[s], y)
        child_X = np.concatenate([states[c] for c in kids], axis=1)
        child_pred = _ridge_fit_predict(child_X, y)
        rho_parent = abs(_spearman(parent_pred, y))
        rho_child = abs(_spearman(child_pred, y))
        rlipp = rho_parent / rho_child if rho_child > 1e-6 else 0.0
        rows.append((s, rlipp, rho_parent, rho_child))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top_k], gate


def top_gate_genes(gate: np.ndarray, top_k=10):
    mean_gate = gate.mean(axis=0)
    order = np.argsort(mean_gate)[::-1][:top_k]
    return [(int(g), float(mean_gate[g])) for g in order]


def report(model, ds: Dataset, indices, device, batch_size):
    top_sys, gate = rlipp_scores(model, ds, indices, device, batch_size)
    top_genes = top_gate_genes(gate)
    print("\n  Interpretability readout")
    print("  ---------------------------------------------")
    print("  Top subsystems by RLIPP (state adds signal beyond children):")
    for s, rlipp, rp, rc in top_sys[:8]:
        print(f"    {s:<22}  RLIPP={rlipp:5.2f}  (rho_parent={rp:.3f}, rho_children={rc:.3f})")
    print("  Top genes by drug-aware attention gate (mean over pairs):")
    gene_str = ", ".join(f"g{g}:{w:.2f}" for g, w in top_genes[:8])
    print(f"    {gene_str}")
    return {"top_subsystems": top_sys, "top_gate_genes": top_genes}
