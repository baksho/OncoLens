"""
Graph drug encoder (pure PyTorch, no torch_geometric dependency).

A compact message-passing neural network (in the spirit of DRPreter / DrugVNN's
CMPNN) that learns drug structure from the molecular graph instead of a frozen
Morgan fingerprint. Graphs are batched by concatenating nodes and offsetting the
edge indices, with a `batch` vector for the final readout pooling.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import MolGraph


def collate_graphs(graphs: list[MolGraph], device: torch.device):
    """Batch a list of MolGraphs into flat tensors with an offset edge_index."""
    feats, edges, batch = [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        n = g.atom_feats.shape[0]
        feats.append(g.atom_feats)
        if g.edge_index.shape[1] > 0:
            edges.append(g.edge_index + offset)
        batch.append(np.full(n, i, dtype=np.int64))
        offset += n
    x = torch.tensor(np.concatenate(feats, 0), dtype=torch.float32, device=device)
    edge_index = (
        torch.tensor(np.concatenate(edges, 1), dtype=torch.long, device=device)
        if edges else torch.zeros((2, 0), dtype=torch.long, device=device)
    )
    batch_t = torch.tensor(np.concatenate(batch, 0), dtype=torch.long, device=device)
    return x, edge_index, batch_t, len(graphs)


class GraphDrugEncoder(nn.Module):
    def __init__(self, atom_feat_dim: int, hidden_dim: int, mp_steps: int,
                 out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(atom_feat_dim, hidden_dim)
        self.msg = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(mp_steps)]
        )
        self.upd = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(mp_steps)]
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch, n_graphs):
        h = self.act(self.embed(x))                      # [N, hid]
        src, dst = edge_index[0], edge_index[1]
        for msg_lin, gru in zip(self.msg, self.upd):
            if src.numel() > 0:
                m = msg_lin(h.index_select(0, src))      # message from src
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, m)                # sum into dst
            else:
                agg = torch.zeros_like(h)
            h = gru(agg, h)                              # gated update
            h = self.drop(h)
        # global mean pool per graph
        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device)
        counts = torch.zeros(n_graphs, 1, device=h.device)
        pooled.index_add_(0, batch, h)
        counts.index_add_(0, batch, torch.ones(h.shape[0], 1, device=h.device))
        pooled = pooled / counts.clamp(min=1.0)
        return self.readout(pooled)                      # [n_graphs, out_dim]
