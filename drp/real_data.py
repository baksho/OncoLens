"""
Real-data loader for GDSC2 (Genomics of Drug Sensitivity in Cancer, release 2).

This turns three user-supplied files into the same `Dataset` object the rest of
the pipeline consumes:

  1. response CSV   - GDSC2 "fitted dose response" (one row per cell-line x drug),
                      giving the AUC (or LN_IC50) label.
  2. expression CSV - a gene-expression matrix for the cell lines (genes in rows,
                      cell lines in columns, by default). Any source is fine
                      (GDSC's own, or CCLE/DepMap); it just needs gene symbols.
  3. drug SMILES CSV- a mapping from GDSC drug id/name to a SMILES string, so the
                      graph encoder can featurize each compound.

The GO hierarchy + canonical gene ordering come from the DrugCell GitHub mirror
(fetched by `drp.data`), so the sparse VNN is wired to real subsystems.

IMPORTANT / honesty notes:
  * GDSC has shipped slightly different column names across releases. The expected
    names are collected in `GDSC2Columns` below and are easy to edit; the loader
    validates and prints the columns it actually found if they don't match.
  * Cell-line names differ between GDSC and expression sources (e.g. "MC-CAR" vs
    "MCCAR"). We match on a normalized key (uppercase, alphanumerics only) and
    report how many matched. Review the match count — imperfect harmonization is
    the most common source of silent data loss here.
  * Only the `expression` cell input is supported directly from these three files.
    `mutation` / `multiomics` need their own matrices (see `load_gdsc2`'s error).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np
import pandas as pd

from . import config
from .data import Dataset, MolGraph
from .hierarchy import parse_drugcell_ontology


# ---------------------------------------------------------------------------
# Configurable schema / paths
# ---------------------------------------------------------------------------

@dataclass
class GDSC2Columns:
    cell: str = "CELL_LINE_NAME"     # column in the response CSV
    drug: str = "DRUG_NAME"          # column in the response CSV (or use DRUG_ID)
    target: str = "AUC"              # label column: "AUC" (in [0,1]) or "LN_IC50"
    # drug SMILES CSV columns
    smiles_drug_key: str = "DRUG_NAME"
    smiles: str = "SMILES"


@dataclass
class GDSC2Paths:
    response_csv: str | None = None
    expression_csv: str | None = None
    drug_smiles_csv: str | None = None
    genes_in_rows: bool = True       # expression: True = genes are rows, cells are columns
    target_is_ln_ic50: bool = False  # set True if target column is LN_IC50 (unbounded)
    columns: GDSC2Columns = field(default_factory=GDSC2Columns)


def _norm_key(s: str) -> str:
    """Normalize a cell-line / drug name for fuzzy matching across sources."""
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


# ---------------------------------------------------------------------------
# SMILES -> molecular graph (RDKit)
# ---------------------------------------------------------------------------

_ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]  # + "other"


def smiles_to_graph(smiles: str, atom_feat_dim: int) -> MolGraph | None:
    """Featurize a SMILES string into a MolGraph. Returns None if unparseable.
    Requires rdkit; raises ImportError with guidance if it is missing."""
    try:
        from rdkit import Chem
    except ImportError as e:
        raise ImportError(
            "rdkit is required to featurize real SMILES. Install with "
            "`pip install rdkit`, or provide precomputed graphs."
        ) from e

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    feats = []
    for atom in mol.GetAtoms():
        v = [0.0] * (len(_ELEMENTS) + 1)
        sym = atom.GetSymbol()
        v[_ELEMENTS.index(sym)] = 1.0 if sym in _ELEMENTS else 0.0
        if sym not in _ELEMENTS:
            v[-1] = 1.0  # "other" element bucket
        extra = [
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            1.0 if atom.GetIsAromatic() else 0.0,
            1.0 if atom.IsInRing() else 0.0,
        ]
        row = v + extra
        # pad / truncate to atom_feat_dim
        if len(row) < atom_feat_dim:
            row = row + [0.0] * (atom_feat_dim - len(row))
        else:
            row = row[:atom_feat_dim]
        feats.append(row)
    atom_feats = np.array(feats, dtype=np.float32)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    edge_index = (np.array(edges, dtype=np.int64).T
                  if edges else np.zeros((2, 0), dtype=np.int64))
    return MolGraph(atom_feats=atom_feats, edge_index=edge_index)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def _load_gene_index_and_hierarchy():
    """Fetch DrugCell gene2ind + ontology from GitHub and build the hierarchy."""
    from .data import _fetch_text
    gene_lines = _fetch_text(config.SOURCES["gene2ind"]).splitlines()
    gene2ind: dict[str, int] = {}
    for ln in gene_lines:
        ln = ln.strip()
        if not ln:
            continue
        idx, sym = ln.split("\t")
        gene2ind[sym] = int(idx)
    ont_lines = _fetch_text(config.SOURCES["go_ontology"]).splitlines()
    H = parse_drugcell_ontology(ont_lines, gene2ind)
    return gene2ind, H


def load_gdsc2(paths: GDSC2Paths, sizing: config.DataSizing, seed: int) -> Dataset:
    if config.CELL_INPUT != "expression":
        raise NotImplementedError(
            f"CELL_INPUT={config.CELL_INPUT!r} needs its own matrices. The GDSC2 loader "
            "here builds the 'expression' input from the expression CSV. For 'mutation' "
            "supply a binary mutation matrix, and for 'multiomics' add mutation + CNV "
            "matrices, then extend load_gdsc2() to stack them per gene."
        )
    for name in ("response_csv", "expression_csv", "drug_smiles_csv"):
        if getattr(paths, name) is None:
            raise FileNotFoundError(
                f"GDSC2Paths.{name} is not set. Point config.GDSC2 at your local files."
            )

    cols = paths.columns
    rng = np.random.default_rng(seed)

    # --- 1) gene index + GO hierarchy (real, from GitHub) -----------------
    gene2ind, H = _load_gene_index_and_hierarchy()
    n_genes = H.n_genes
    idx2gene = {i: g for g, i in gene2ind.items()}
    gene_order = [idx2gene[i] for i in range(n_genes)]

    # --- 2) expression matrix -> [n_cells, n_genes] -----------------------
    expr = pd.read_csv(paths.expression_csv, index_col=0)
    if not paths.genes_in_rows:
        expr = expr.T
    # rows = genes, cols = cell lines
    expr = expr.groupby(level=0).mean()  # collapse duplicate gene symbols
    # reindex to the canonical gene order; missing genes -> 0
    expr = expr.reindex(gene_order).fillna(0.0)
    cell_names = list(expr.columns)
    cell_key = {_norm_key(c): j for j, c in enumerate(cell_names)}
    X_full = expr.to_numpy(dtype=np.float32).T  # [n_cells, n_genes]
    # z-score per gene across cells for numerical stability
    mu, sd = X_full.mean(0, keepdims=True), X_full.std(0, keepdims=True)
    X_full = (X_full - mu) / (sd + 1e-6)

    # --- 3) drug SMILES -> graphs -----------------------------------------
    smi = pd.read_csv(paths.drug_smiles_csv)
    atom_dim = config.ModelConfig().drug_atom_feat_dim
    drug_graph_by_key: dict[str, MolGraph] = {}
    for _, r in smi.iterrows():
        key = _norm_key(r[cols.smiles_drug_key])
        g = smiles_to_graph(r[cols.smiles], atom_dim)
        if g is not None:
            drug_graph_by_key[key] = g

    # --- 4) response table + alignment ------------------------------------
    resp = pd.read_csv(paths.response_csv)
    missing = [c for c in (cols.cell, cols.drug, cols.target) if c not in resp.columns]
    if missing:
        raise KeyError(
            f"Response CSV missing columns {missing}. Found: {list(resp.columns)[:20]}. "
            "Edit config.GDSC2.columns to match your GDSC2 release."
        )

    drug_keys, drug_graphs = [], []
    drug_key_to_idx: dict[str, int] = {}
    pairs, ys = [], []
    n_cell_miss = n_drug_miss = 0

    for _, r in resp.iterrows():
        ck = _norm_key(r[cols.cell])
        dk = _norm_key(r[cols.drug])
        if ck not in cell_key:
            n_cell_miss += 1
            continue
        if dk not in drug_graph_by_key:
            n_drug_miss += 1
            continue
        if dk not in drug_key_to_idx:
            drug_key_to_idx[dk] = len(drug_graphs)
            drug_graphs.append(drug_graph_by_key[dk])
            drug_keys.append(dk)
        y = float(r[cols.target])
        pairs.append((cell_key[ck], drug_key_to_idx[dk]))
        ys.append(y)

    if not pairs:
        raise RuntimeError(
            "No (cell, drug) pairs matched across the three files. Check that cell-line "
            "names in the response and expression files refer to the same lines, and that "
            "drug names match the SMILES file. (cell misses={}, drug misses={})".format(
                n_cell_miss, n_drug_miss)
        )

    y = np.array(ys, dtype=np.float32)
    if paths.target_is_ln_ic50:
        # squash LN_IC50 into (0,1) so it matches the sigmoid head; lower = more potent
        y = 1.0 / (1.0 + np.exp((y - np.median(y)) / (np.std(y) + 1e-6)))
    else:
        y = np.clip(y, 0.0, 1.0)  # GDSC AUC is already ~[0,1]

    pairs = np.array(pairs, dtype=np.int64)

    # --- 5) optional Subset capping ---------------------------------------
    if sizing.max_pairs is not None and len(pairs) > sizing.max_pairs:
        keep = rng.permutation(len(pairs))[:sizing.max_pairs]
        pairs, y = pairs[keep], y[keep]

    print(f"    matched pairs: {len(pairs)}  |  cells: {X_full.shape[0]}  "
          f"drugs: {len(drug_graphs)}  |  skipped (cell/drug miss): "
          f"{n_cell_miss}/{n_drug_miss}")

    return Dataset(
        hierarchy=H,
        X_cell=X_full,
        drug_graphs=drug_graphs,
        pairs=pairs,
        y=y,
        cell_input_dim=n_genes,
        atom_feat_dim=atom_dim,
        meta={"source": f"GDSC2 ({paths.response_csv})", "cell_input": "expression",
              "n_matched_pairs": int(len(pairs)), **H.stats()},
    )
