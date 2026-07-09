"""
Cross-branch components:

1) DrugAwareGeneGate  (DrugVNN's drug-aware gene attention gate)
   The drug embedding queries a learned per-gene embedding table to produce a
   per-gene gate in [0,1], which reweights the cell's gene features *before* the
   VNN. This is the explicit drug<->genotype cross-talk that plain concatenation
   lacks: the same cell looks different to different drugs.

2) FusionModule  (Optimal-Fusion study)
   Combines the cell embedding and drug embedding. The fusion *function* is
   configurable ("concat" | "gated" | "bilinear") so you can reproduce the
   finding that fusion choice matters, instead of hard-coding concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DrugAwareGeneGate(nn.Module):
    """
    The drug embedding queries a learned per-gene embedding to modulate each gene
    before the VNN. We apply it as a *residual* modulation centered at 1.0:
        x_gated = x * (1 + beta * (2*sigmoid(score) - 1))
    so a gate of 0.5 leaves a gene untouched, and the drug can up/down-weight genes
    within +/- beta. (A hard 0..1 multiplier would halve the signal on average and
    drown the cell representation; the residual form modulates without suppressing.)
    """
    def __init__(self, n_genes: int, drug_embed_dim: int, gene_embed_dim: int,
                 beta: float = 0.5):
        super().__init__()
        self.gene_embed = nn.Embedding(n_genes, gene_embed_dim)
        self.q = nn.Linear(drug_embed_dim, gene_embed_dim)
        self.scale = gene_embed_dim ** 0.5
        self.beta = beta
        self.register_buffer("gene_ids", torch.arange(n_genes), persistent=False)

    def forward(self, x_cell: torch.Tensor, drug_emb: torch.Tensor):
        """
        x_cell:   [B, G]
        drug_emb: [B, drug_dim]
        returns:  modulated x_cell [B, G], gate weights [B, G] (for interpretation)
        """
        E = self.gene_embed(self.gene_ids)               # [G, ge]
        q = self.q(drug_emb)                             # [B, ge]
        scores = (q @ E.t()) / self.scale                # [B, G]
        gate = torch.sigmoid(scores)                     # in (0,1), for readout
        modulation = 1.0 + self.beta * (2.0 * gate - 1.0)  # centered at 1.0
        return x_cell * modulation, gate


class FusionModule(nn.Module):
    """
    Combine cell and drug embeddings. Each variant exposes an explicit *signed*
    interaction term so the head can represent "drug kills cell" (negative signal)
    rather than having it clipped away by a ReLU. The output is linear (signed);
    the nonlinearity lives in the head.
    """
    def __init__(self, cell_dim: int, drug_dim: int, hidden: int,
                 out_dim: int, kind: str = "gated", dropout: float = 0.1):
        super().__init__()
        self.kind = kind
        if kind == "concat":
            # DrugCell baseline: no explicit interaction, the head must learn it.
            in_dim = cell_dim + drug_dim
        elif kind == "gated":
            # project both into a common space and form an element-wise (low-rank
            # bilinear) interaction cp*dp, then keep the parts too.
            self.cell_proj = nn.Linear(cell_dim, hidden)
            self.drug_proj = nn.Linear(drug_dim, hidden)
            in_dim = hidden * 3
        elif kind == "bilinear":
            self.bil = nn.Bilinear(cell_dim, drug_dim, hidden)
            in_dim = hidden + cell_dim + drug_dim
        else:
            raise ValueError(f"unknown fusion kind {kind!r}")
        self.proj = nn.Linear(in_dim, out_dim)  # linear -> keeps sign

    def forward(self, cell_emb: torch.Tensor, drug_emb: torch.Tensor):
        if self.kind == "concat":
            z = torch.cat([cell_emb, drug_emb], dim=1)
        elif self.kind == "gated":
            cp = self.cell_proj(cell_emb)
            dp = self.drug_proj(drug_emb)
            z = torch.cat([cp * dp, cp, dp], dim=1)   # explicit interaction + parts
        else:  # bilinear
            b = self.bil(cell_emb, drug_emb)
            z = torch.cat([b, cell_emb, drug_emb], dim=1)
        return self.proj(z)
