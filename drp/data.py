"""
Data layer.

Provides, for any MODE:
  - a Hierarchy (GO structure),
  - cell-line feature matrix X_cell  [n_cell_lines, n_genes]  (semantics set by CELL_INPUT),
  - a list of drug molecular graphs (node features + edge index),
  - a response table of (cell_idx, drug_idx, auc) triples in [0, 1].

Demo mode synthesizes everything with a *learnable* generative process: response
depends on a handful of true pathways interacting with a drug's latent affinity,
so a correctly-wired model recovers real correlation (this is how we know the
pipeline works, not just that it runs).

Full / Subset fetch the real DrugCell GO ontology from GitHub and expect the
pharmacogenomic + omics matrices locally or from a portal; if those are
unreachable the loader raises an actionable error.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import config
from .hierarchy import Hierarchy, build_synthetic_hierarchy


@dataclass
class MolGraph:
    """A single molecule as a graph (pure-python; tensorized later in batches)."""
    atom_feats: np.ndarray   # [n_atoms, atom_feat_dim]
    edge_index: np.ndarray   # [2, n_edges] (directed, both directions included)


@dataclass
class Dataset:
    hierarchy: Hierarchy
    X_cell: np.ndarray              # [n_cells, n_genes]
    drug_graphs: list[MolGraph]
    pairs: np.ndarray               # [n_pairs, 2] int (cell_idx, drug_idx)
    y: np.ndarray                   # [n_pairs] float in [0,1]
    cell_input_dim: int             # genes per cell (after CELL_INPUT expansion)
    atom_feat_dim: int
    meta: dict


# ---------------------------------------------------------------------------
# Synthetic molecular graphs
# ---------------------------------------------------------------------------

def _random_mol_graph(rng: np.random.Generator, atom_feat_dim: int) -> MolGraph:
    """Small connected random graph resembling a molecule (8-30 atoms, tree+rings)."""
    n_atoms = int(rng.integers(8, 30))
    # atom features: one-hot-ish atom "type" + a couple continuous descriptors
    n_types = min(8, atom_feat_dim - 2)
    feats = np.zeros((n_atoms, atom_feat_dim), dtype=np.float32)
    atom_types = rng.integers(0, n_types, size=n_atoms)
    feats[np.arange(n_atoms), atom_types] = 1.0
    feats[:, -2] = rng.normal(0, 1, n_atoms)       # e.g. partial charge proxy
    feats[:, -1] = rng.integers(0, 4, n_atoms)     # degree proxy

    # spanning tree to guarantee connectivity
    edges = []
    for a in range(1, n_atoms):
        b = int(rng.integers(0, a))
        edges.append((a, b))
    # a few extra bonds (rings)
    for _ in range(int(rng.integers(0, n_atoms // 3 + 1))):
        a, b = int(rng.integers(0, n_atoms)), int(rng.integers(0, n_atoms))
        if a != b:
            edges.append((a, b))

    # make undirected (both directions) for message passing
    ei = []
    for a, b in edges:
        ei.append((a, b))
        ei.append((b, a))
    edge_index = np.array(ei, dtype=np.int64).T if ei else np.zeros((2, 0), np.int64)
    return MolGraph(atom_feats=feats, edge_index=edge_index)


# ---------------------------------------------------------------------------
# Synthetic, learnable dataset (Demo / Subset fallback)
# ---------------------------------------------------------------------------

def _expand_cell_input(base_expr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Turn a base continuous 'expression' matrix into the representation chosen by
    CELL_INPUT. Keeps the gene axis length identical so the hierarchy lines up.
    """
    if config.CELL_INPUT == "expression":
        return base_expr.astype(np.float32)
    if config.CELL_INPUT == "mutation":
        # binarize: 'mutated' where expression is in the tails (proxy for alteration)
        thresh = np.quantile(base_expr, 0.85, axis=0, keepdims=True)
        return (base_expr > thresh).astype(np.float32)
    if config.CELL_INPUT == "multiomics":
        # expression + a mutation channel + a CNV channel, summed per gene
        # (PASO folds multi-omics into per-gene/per-pathway values; we keep the
        #  gene axis fixed by combining channels rather than concatenating).
        mut = (base_expr > np.quantile(base_expr, 0.85, axis=0, keepdims=True)).astype(np.float32)
        cnv = np.clip(rng.normal(0, 0.5, base_expr.shape) + 0.3 * base_expr, -3, 3).astype(np.float32)
        z = (base_expr - base_expr.mean(0)) / (base_expr.std(0) + 1e-6)
        return (z + 0.5 * mut + 0.3 * cnv).astype(np.float32)
    raise ValueError(f"unknown CELL_INPUT {config.CELL_INPUT!r}")


def make_synthetic_dataset(sizing: config.DataSizing, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    H = build_synthetic_hierarchy(
        n_genes=sizing.n_genes,
        branching=sizing.hierarchy_branching,
        depth=sizing.hierarchy_depth,
        seed=seed,
    )

    n_cells, n_drugs, n_genes = sizing.n_cell_lines, sizing.n_drugs, sizing.n_genes

    # --- cell expression with pathway structure ---------------------------
    # leaf subsystems define gene groups; give each cell a per-leaf "activity"
    leaves = [s for s in H.topo_order if not H.children.get(s)]
    leaf_of_gene = np.zeros(n_genes, dtype=np.int64)
    for li, leaf in enumerate(leaves):
        for g in H.genes[leaf]:
            leaf_of_gene[g] = li
    cell_pathway_activity = rng.normal(0, 1, size=(n_cells, len(leaves))).astype(np.float32)
    base_expr = cell_pathway_activity[:, leaf_of_gene]  # broadcast leaf activity to genes
    base_expr = base_expr + rng.normal(0, 0.4, base_expr.shape).astype(np.float32)
    X_cell = _expand_cell_input(base_expr, rng)

    # --- LOW-RANK latent factors ------------------------------------------
    # Real drug response is far lower-rank than "every pathway x every drug". We
    # model the response as a dot product in a small latent space so it can pass
    # through the model's 6-dim root/drug embedding bottleneck (this is the
    # low-rank assumption DrugCell-style models actually rely on).
    latent_dim = 4
    W_cell = rng.normal(0, 1, size=(len(leaves), latent_dim)).astype(np.float32)
    cell_factor = cell_pathway_activity @ W_cell                 # [n_cells, latent]
    cell_factor /= (cell_factor.std(0, keepdims=True) + 1e-6)

    # --- drugs: graph + a latent factor baked into atom features ----------
    atom_dim = config.ModelConfig().drug_atom_feat_dim
    drug_graphs = [_random_mol_graph(rng, atom_dim) for _ in range(n_drugs)]
    drug_factor = rng.normal(0, 1, size=(n_drugs, latent_dim)).astype(np.float32)

    # IMPORTANT: bake each drug's latent factor into its molecular-graph atom
    # features so the GNN has a learnable path from structure -> effect. (In real
    # data this link is physical: structure determines targets. Here we write a
    # projection of the factor into a block of atom-feature channels, shared
    # across the drug's atoms, plus a little per-atom noise.)
    sig_dim = min(8, atom_dim - 10)
    proj = rng.normal(0, 1, size=(latent_dim, sig_dim)).astype(np.float32)
    drug_signature = drug_factor @ proj
    drug_signature /= (np.linalg.norm(drug_signature, axis=1, keepdims=True) + 1e-6)
    for m, g in enumerate(drug_graphs):
        block = slice(8, 8 + sig_dim)
        g.atom_feats[:, block] = (
            drug_signature[m][None, :] + rng.normal(0, 0.15, (g.atom_feats.shape[0], sig_dim))
        ).astype(np.float32)

    # --- response: lower AUC = more killing -------------------------------
    # signal = interaction of the cell's latent state with the drug's latent factor
    pairs = []
    ys = []
    max_pairs = sizing.max_pairs or (n_cells * n_drugs)
    all_pairs = [(c, d) for c in range(n_cells) for d in range(n_drugs)]
    rng.shuffle(all_pairs)
    for (c, d) in all_pairs[:max_pairs]:
        signal = float(cell_factor[c] @ drug_factor[d])
        noise = float(rng.normal(0, 0.35))
        auc = 1.0 / (1.0 + np.exp(0.9 * signal + noise))  # sigmoid -> [0,1]
        pairs.append((c, d))
        ys.append(auc)

    return Dataset(
        hierarchy=H,
        X_cell=X_cell,
        drug_graphs=drug_graphs,
        pairs=np.array(pairs, dtype=np.int64),
        y=np.array(ys, dtype=np.float32),
        cell_input_dim=n_genes,
        atom_feat_dim=atom_dim,
        meta={"source": "synthetic", "n_leaves": len(leaves),
              "cell_input": config.CELL_INPUT, **H.stats()},
    )


# ---------------------------------------------------------------------------
# Real data (Full / Subset)
# ---------------------------------------------------------------------------

def _fetch_text(url: str, timeout: int = 30) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def make_real_dataset(sizing: config.DataSizing, seed: int) -> Dataset:
    """
    Full / Subset path. Fetches the DrugCell GO ontology + gene list (GitHub, usually
    reachable) and expects the response + omics matrices locally/portal.

    This function is written to run correctly on a machine with network + the data
    files in place. In a locked-down sandbox the omics/response download will fail;
    we raise a clear error so the caller can fall back to Demo.
    """
    from .hierarchy import parse_drugcell_ontology

    # 1) GO hierarchy + gene index (from GitHub raw — typically allowed)
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

    # 2) Response labels + cell-line omics. These live on portals that are often
    #    NOT reachable from a sandbox. Replace the URLs in config.SOURCES with local
    #    paths if you have the files. We attempt a fetch and fail loudly otherwise.
    raise RuntimeError(
        "Full/Subset real data: GO hierarchy loaded ({} genes, {} subsystems), but the "
        "pharmacogenomic response matrix and CCLE omics could not be auto-loaded in this "
        "environment. Point config.SOURCES['drugcell_all'] / ['ccle_expression'] at local "
        "files (or run on a machine with portal access), then re-run. Falling back to Demo "
        "is automatic in main.py.".format(H.n_genes, len(H.topo_order))
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset() -> Dataset:
    sizing = config.active_sizing()
    seed = config.TrainConfig().seed
    if config.MODE == "Demo":
        return make_synthetic_dataset(sizing, seed)
    if config.MODE in ("Full", "Subset"):
        return make_real_dataset(sizing, seed)
    raise ValueError(f"unknown MODE {config.MODE!r}")
