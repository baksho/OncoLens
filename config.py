"""
Central configuration for the holistic drug-response pipeline.

Everything you would normally want to change lives here as a plain module-level
variable. Edit these and re-run `python -m drp.main`.

The design intentionally mirrors the architecture diagram:
  Cell omics --> pathway mapping --> Sparse GO-VNN -----\
                                                          >-- learned fusion --> MLP --> response
  Drug SMILES --> molecular graph --> Graph drug encoder /
        ^ a drug-aware attention gate links the drug embedding back into the cell branch.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# THE TWO KNOBS YOU ASKED FOR
# ---------------------------------------------------------------------------

# "Full"   : download GDSC/CTRP + CCLE + GO hierarchy and train on everything.
# "Subset" : same real data path, but capped to a small slice for fast training.
# "Demo"   : synthetic-but-realistic data + trimmed GO hierarchy, no download,
#            trains in seconds, produces meaningful (learnable) numbers.
MODE = "Demo"  # "Full" | "Subset" | "Demo"

# Which cell-side feature block to use. The cell encoder treats this as pluggable.
# "expression" : continuous gene expression (SparseGO-style, our default).
# "multiomics" : expression + mutation + CNV, combined per gene (PASO-style).
# "mutation"   : binary mutation flags (DrugCell-faithful, thinnest signal).
CELL_INPUT = "expression"  # "expression" | "multiomics" | "mutation"

# How the two branch embeddings are combined (Optimal-Fusion study).
# "concat" : plain concatenation + MLP (DrugCell baseline).
# "gated"  : element-wise gated fusion (default, usually best of the cheap options).
# "bilinear": low-rank bilinear interaction.
FUSION = "bilinear"  # "concat" | "gated" | "bilinear"


# ---------------------------------------------------------------------------
# Per-mode data sizing. Demo numbers are deliberately tiny so it runs anywhere.
# ---------------------------------------------------------------------------

@dataclass
class DataSizing:
    n_genes: int
    n_cell_lines: int
    n_drugs: int
    # GO hierarchy shape (synthetic builder); ignored when a real ontology loads.
    hierarchy_branching: int
    hierarchy_depth: int
    # cap on (cell, drug) pairs actually used (None = all)
    max_pairs: int | None


SIZING: dict[str, DataSizing] = {
    "Demo": DataSizing(
        n_genes=300, n_cell_lines=120, n_drugs=40,
        hierarchy_branching=3, hierarchy_depth=4, max_pairs=3000,
    ),
    "Subset": DataSizing(
        n_genes=2000, n_cell_lines=300, n_drugs=120,
        hierarchy_branching=4, hierarchy_depth=5, max_pairs=40000,
    ),
    "Full": DataSizing(
        n_genes=15000, n_cell_lines=1235, n_drugs=684,
        hierarchy_branching=5, hierarchy_depth=6, max_pairs=None,
    ),
}


# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    subsystem_dim: int = 6        # neurons per GO subsystem (DrugCell uses 6)
    gene_embed_dim: int = 16      # per-gene embedding used by the attention gate
    drug_atom_feat_dim: int = 24  # atom feature size for the molecular graph
    drug_hidden_dim: int = 64     # message-passing hidden size
    drug_mp_steps: int = 3        # rounds of message passing
    drug_embed_dim: int = 6       # final drug embedding (matches cell root dim)
    fusion_hidden_dim: int = 32
    head_hidden_dim: int = 32
    dropout: float = 0.05


@dataclass
class TrainConfig:
    epochs: dict[str, int] = field(default_factory=lambda: {
        "Demo": 90, "Subset": 40, "Full": 100,
    })
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 7
    # number of CPU threads for torch (bumped from the default of 1)
    num_threads: int = 4
    # early stopping patience (epochs without val improvement)
    patience: int = 20


# ---------------------------------------------------------------------------
# Real-data source URLs (used only in Full / Subset).
#   - GO ontology + gene list come from the DrugCell GitHub (raw.githubusercontent
#     is reachable from most environments).
#   - The pharmacogenomic response matrix and CCLE expression live on portals that
#     are typically NOT reachable from a sandbox; on your own machine they download
#     fine. If a fetch fails, the loader raises a clear, actionable error.
# ---------------------------------------------------------------------------

SOURCES = {
    "go_ontology": "https://raw.githubusercontent.com/idekerlab/DrugCell/master/data/drugcell_ont.txt",
    "gene2ind":    "https://raw.githubusercontent.com/idekerlab/DrugCell/master/data/gene2ind.txt",
    "drug2ind":    "https://raw.githubusercontent.com/idekerlab/DrugCell/master/data/drug2ind.txt",
    "drug2fp":     "https://raw.githubusercontent.com/idekerlab/DrugCell/master/data/drug2fingerprint.txt",
    # Response labels + cell-line omics: replace with your local paths or portal URLs.
    "drugcell_all": "http://drugcell.ucsd.edu/downloads/drugcell_all.txt",  # likely sandbox-blocked
    # CCLE expression (Full mode, expression cell input). Provide your own mirror/path.
    "ccle_expression": "https://depmap.org/portal/download/  # set a concrete file URL on your machine",
}


def active_sizing() -> DataSizing:
    return SIZING[MODE]


# ---------------------------------------------------------------------------
# Real GDSC2 data (optional). Set these paths to train Full/Subset on GDSC2.
# If left as None, Full/Subset load only the real GO hierarchy and then fall
# back to Demo (so a run always completes). See drp/real_data.py for the schema.
# ---------------------------------------------------------------------------

GDSC2_RESPONSE_CSV: str | None = None       # GDSC2 fitted dose-response CSV
GDSC2_EXPRESSION_CSV: str | None = None     # gene-expression matrix (genes x cells)
GDSC2_DRUG_SMILES_CSV: str | None = None    # drug name/id -> SMILES
GDSC2_GENES_IN_ROWS: bool = True            # expression layout
GDSC2_TARGET_IS_LN_IC50: bool = False       # False => target column is AUC in [0,1]
