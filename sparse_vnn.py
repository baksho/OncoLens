"""
Sparse GO Visible Neural Network (cell encoder).

Each GO subsystem gets its own small Linear -> BatchNorm -> Tanh block whose
input is *only* the states of its child subsystems plus the features of its
directly-annotated genes. That restricted connectivity is exactly the "sparsity"
that makes the network interpretable (SparseGO / DrugCell): every neuron maps to
a named biological subsystem, so we can later attribute predictions to subsystems.

We keep per-subsystem outputs in a dict during the forward pass so the
interpretability module can read them out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hierarchy import Hierarchy


class SubsystemBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.lin(x))))


class SparseGOVNN(nn.Module):
    def __init__(self, hierarchy: Hierarchy, subsystem_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.H = hierarchy
        self.k = subsystem_dim

        self.blocks = nn.ModuleDict()
        # precompute, per subsystem, the child order and gene indices
        self._child_ids: dict[str, list[str]] = {}
        self._gene_idx: dict[str, torch.Tensor] = {}

        for s in hierarchy.topo_order:
            kids = hierarchy.children.get(s, [])
            genes = hierarchy.genes.get(s, [])
            in_dim = len(kids) * self.k + len(genes)
            self.blocks[self._key(s)] = SubsystemBlock(in_dim, self.k, dropout)
            self._child_ids[s] = kids
            self._gene_idx[s] = torch.tensor(genes, dtype=torch.long)

        self.out_dim = self.k  # root embedding size

    @staticmethod
    def _key(s: str) -> str:
        # ModuleDict keys can't contain '.' — encode it
        return s.replace(".", "__").replace(":", "_")

    def forward(self, x_cell: torch.Tensor, return_states: bool = False):
        """
        x_cell: [B, n_genes]
        returns: root embedding [B, k]  (and optionally {subsystem: [B,k]} states)
        """
        states: dict[str, torch.Tensor] = {}
        B = x_cell.shape[0]
        device = x_cell.device

        for s in self.H.topo_order:
            parts = []
            for c in self._child_ids[s]:
                parts.append(states[c])
            gidx = self._gene_idx[s]
            if gidx.numel() > 0:
                parts.append(x_cell.index_select(1, gidx.to(device)))
            if parts:
                inp = torch.cat(parts, dim=1)
            else:  # should not happen (validated), but stay safe
                inp = torch.zeros(B, 1, device=device)
            states[s] = self.blocks[self._key(s)](inp)

        root_emb = states[self.H.root]
        if return_states:
            return root_emb, states
        return root_emb
