"""
GO / pathway hierarchy used to wire the sparse Visible Neural Network.

A hierarchy is a DAG of `subsystems` (GO terms). Each subsystem aggregates:
  - the states of its child subsystems, and
  - the raw features of any genes directly annotated to it.

We expose a small, framework-agnostic `Hierarchy` object that both the synthetic
builder (Demo / Subset) and the real DrugCell-ontology parser (Full) produce, so
the Sparse GO-VNN downstream never needs to know where the structure came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class Hierarchy:
    n_genes: int
    # subsystem id (str) -> list of child subsystem ids
    children: dict[str, list[str]]
    # subsystem id -> list of directly annotated gene indices
    genes: dict[str, list[int]]
    root: str
    # subsystems in topological order: every child appears before its parent
    topo_order: list[str] = field(default_factory=list)

    @property
    def subsystems(self) -> list[str]:
        return self.topo_order

    def __post_init__(self):
        if not self.topo_order:
            self.topo_order = self._toposort()
        self._validate()

    def _toposort(self) -> list[str]:
        seen: set[str] = set()
        order: list[str] = []

        def visit(s: str):
            if s in seen:
                return
            seen.add(s)
            for c in self.children.get(s, []):
                visit(c)
            order.append(s)

        visit(self.root)
        return order

    def _validate(self):
        # every subsystem must contribute at least one input, else the Linear is empty
        for s in self.topo_order:
            fanin = len(self.children.get(s, [])) + len(self.genes.get(s, []))
            if fanin == 0:
                raise ValueError(f"subsystem {s} has no children and no genes")

    def stats(self) -> dict:
        leaves = [s for s in self.topo_order if not self.children.get(s)]
        return {
            "n_subsystems": len(self.topo_order),
            "n_leaf_subsystems": len(leaves),
            "n_genes": self.n_genes,
            "root": self.root,
        }


# ---------------------------------------------------------------------------
# Synthetic builder (Demo / Subset)
# ---------------------------------------------------------------------------

def build_synthetic_hierarchy(
    n_genes: int, branching: int, depth: int, seed: int = 0
) -> Hierarchy:
    """
    Build a balanced-ish tree of subsystems. Genes are partitioned across the
    leaf subsystems; a fraction are additionally annotated to a random higher-level
    subsystem so the structure isn't a pure tree (GO is a DAG).
    """
    rng = random.Random(seed)
    children: dict[str, list[str]] = {}
    genes: dict[str, list[int]] = {}

    # Build levels top-down. Level 0 is the root.
    levels: list[list[str]] = [["GO:root"]]
    for d in range(1, depth):
        prev = levels[-1]
        cur: list[str] = []
        for parent in prev:
            n_children = max(2, branching + rng.randint(-1, 1))
            kids = [f"{parent}.{i}" for i in range(n_children)]
            children[parent] = kids
            cur.extend(kids)
        levels.append(cur)

    leaves = levels[-1]
    for leaf in leaves:
        children.setdefault(leaf, [])  # explicit empty

    # Partition genes across leaves (round-robin so every leaf gets >=1 gene)
    for leaf in leaves:
        genes[leaf] = []
    for g in range(n_genes):
        genes[leaves[g % len(leaves)]].append(g)

    # Annotate ~15% of genes additionally to a random non-leaf subsystem
    non_leaves = [s for lvl in levels[:-1] for s in lvl]
    for g in range(n_genes):
        if rng.random() < 0.15 and non_leaves:
            s = rng.choice(non_leaves)
            genes.setdefault(s, []).append(g)

    # Make sure every subsystem listed in `children`/`genes` is keyed
    for lvl in levels:
        for s in lvl:
            genes.setdefault(s, [])
            children.setdefault(s, [])

    return Hierarchy(n_genes=n_genes, children=children, genes=genes, root="GO:root")


# ---------------------------------------------------------------------------
# Real DrugCell ontology parser (Full)
# ---------------------------------------------------------------------------

def parse_drugcell_ontology(ont_lines: list[str], gene2ind: dict[str, int]) -> Hierarchy:
    """
    Parse the DrugCell `drugcell_ont.txt` format. Each line is tab-separated:
        <parent_term> <child_term_or_gene> <relation>
    where relation is "default" for term-term edges and "gene" for term-gene edges.

    `gene2ind` maps gene symbol -> integer index (from gene2ind.txt).
    Returns a Hierarchy with the same interface as the synthetic builder.
    """
    children: dict[str, list[str]] = {}
    genes: dict[str, list[int]] = {}
    all_terms: set[str] = set()
    has_parent: set[str] = set()

    for raw in ont_lines:
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        parent, child, rel = parts[0], parts[1], parts[2]
        all_terms.add(parent)
        if rel == "gene":
            idx = gene2ind.get(child)
            if idx is not None:
                genes.setdefault(parent, []).append(idx)
        else:
            children.setdefault(parent, []).append(child)
            all_terms.add(child)
            has_parent.add(child)

    roots = [t for t in all_terms if t not in has_parent]
    if len(roots) != 1:
        # Create a synthetic super-root joining all roots (GO can have several)
        super_root = "GO:SUPERROOT"
        children[super_root] = roots
        root = super_root
    else:
        root = roots[0]

    for t in all_terms:
        children.setdefault(t, [])
        genes.setdefault(t, [])

    n_genes = (max(gene2ind.values()) + 1) if gene2ind else 0
    return Hierarchy(n_genes=n_genes, children=children, genes=genes, root=root)
