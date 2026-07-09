"""
The full holistic model = the architecture diagram, in one nn.Module.

forward:
    drug_emb           = GraphDrugEncoder(drug_graphs)
    gated_cell, gate   = DrugAwareGeneGate(x_cell, drug_emb)
    cell_emb, states   = SparseGOVNN(gated_cell)
    fused              = FusionModule(cell_emb, drug_emb)
    auc_hat            = Head(fused)

The model can return the per-subsystem `states` and per-gene `gate` for the
interpretability module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config
from .hierarchy import Hierarchy
from .sparse_vnn import SparseGOVNN
from .drug_encoder import GraphDrugEncoder
from .components import DrugAwareGeneGate, FusionModule


class HolisticDRP(nn.Module):
    def __init__(self, hierarchy: Hierarchy, n_genes: int, atom_feat_dim: int,
                 mcfg: config.ModelConfig, fusion_kind: str):
        super().__init__()
        self.cell_encoder = SparseGOVNN(hierarchy, mcfg.subsystem_dim, mcfg.dropout)
        self.drug_encoder = GraphDrugEncoder(
            atom_feat_dim, mcfg.drug_hidden_dim, mcfg.drug_mp_steps,
            mcfg.drug_embed_dim, mcfg.dropout,
        )
        self.gene_gate = DrugAwareGeneGate(
            n_genes, mcfg.drug_embed_dim, mcfg.gene_embed_dim
        )
        self.fusion = FusionModule(
            cell_dim=self.cell_encoder.out_dim,
            drug_dim=mcfg.drug_embed_dim,
            hidden=mcfg.fusion_hidden_dim,
            out_dim=mcfg.head_hidden_dim,
            kind=fusion_kind,
            dropout=mcfg.dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(mcfg.head_hidden_dim, mcfg.head_hidden_dim), nn.ReLU(),
            nn.Linear(mcfg.head_hidden_dim, 1), nn.Sigmoid(),  # AUC in [0,1]
        )

    def forward(self, x_cell, drug_batch, return_internals: bool = False):
        x, edge_index, batch, n_graphs = drug_batch
        drug_emb_all = self.drug_encoder(x, edge_index, batch, n_graphs)  # [n_drugs_in_batch, d]
        # drug_batch is built per-sample already (see train.py), so n_graphs == B
        drug_emb = drug_emb_all
        gated_cell, gate = self.gene_gate(x_cell, drug_emb)
        if return_internals:
            cell_emb, states = self.cell_encoder(gated_cell, return_states=True)
        else:
            cell_emb = self.cell_encoder(gated_cell)
            states = None
        fused = self.fusion(cell_emb, drug_emb)
        auc = self.head(fused).squeeze(-1)
        if return_internals:
            return auc, {"gate": gate, "states": states,
                         "cell_emb": cell_emb, "drug_emb": drug_emb}
        return auc
