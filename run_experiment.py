"""
Sweep CELL_INPUT x FUSION and tabulate results.

Runs the full pipeline in Demo mode for every combination of cell-side input and
fusion function, on a fixed seed for comparability, and prints a markdown table.
Optionally writes the same table to CSV.

Usage:
    python -m drp.run_experiment                # default: ~50 epochs/run
    python -m drp.run_experiment --epochs 90    # closer to convergence (slower)
    python -m drp.run_experiment --csv out.csv  # also save a CSV

Note: these are Demo (synthetic, learnable) numbers meant for *comparing*
configurations, not benchmark scores. The synthetic signal favors fusions that
carry an explicit interaction; on real data the ranking can differ.
"""

from __future__ import annotations

import argparse
import time

from . import config
from .data import make_synthetic_dataset
from .train import train


CELL_INPUTS = ["expression", "multiomics", "mutation"]
FUSIONS = ["concat", "gated", "bilinear"]


def run_sweep(epochs: int, seed: int = 7, only_cell_input: str | None = None):
    config.MODE = "Demo"
    sizing = config.SIZING["Demo"]
    rows = []
    t_start = time.time()

    cell_inputs = [only_cell_input] if only_cell_input else CELL_INPUTS
    for ci in cell_inputs:
        config.CELL_INPUT = ci
        # dataset depends on CELL_INPUT only, so build once per cell input
        ds = make_synthetic_dataset(sizing, seed)
        for fk in FUSIONS:
            config.FUSION = fk
            out = train(ds, verbose=False, epochs_override=epochs)
            m = out["metrics"]
            rows.append({
                "cell_input": ci,
                "fusion": fk,
                "test_rho": m["test"]["spearman"],
                "val_rho": m["val"]["spearman"],
                "train_rho": m["train"]["spearman"],
                "test_rmse": m["test"]["rmse"],
                "params": out["n_params"],
                "seconds": out["seconds"],
            })
            print(f"  done: cell_input={ci:11s} fusion={fk:9s} "
                  f"test_rho={m['test']['spearman']:.3f}  "
                  f"({out['seconds']:.0f}s)")

    total = time.time() - t_start
    return rows, total


def to_markdown(rows) -> str:
    hdr = ("| cell_input | fusion | test_rho | val_rho | train_rho | test_rmse | params |\n"
           "|---|---|---|---|---|---|---|")
    lines = [hdr]
    for r in rows:
        lines.append(
            f"| {r['cell_input']} | {r['fusion']} | {r['test_rho']:.3f} | "
            f"{r['val_rho']:.3f} | {r['train_rho']:.3f} | {r['test_rmse']:.3f} | "
            f"{r['params']:,} |"
        )
    return "\n".join(lines)


def to_csv(rows, path: str):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50,
                    help="epochs per run (default 50; use 90 for near-convergence)")
    ap.add_argument("--csv", type=str, default=None, help="optional path to save CSV")
    ap.add_argument("--cell-input", type=str, default=None,
                    choices=CELL_INPUTS, help="restrict sweep to one cell input")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    n = (1 if args.cell_input else len(CELL_INPUTS)) * len(FUSIONS)
    print(f"Sweeping {n} configs @ {args.epochs} epochs each\n")
    rows, total = run_sweep(args.epochs, args.seed, args.cell_input)

    print("\n" + "=" * 62)
    print(f"Results  (Demo, seed={args.seed}, {args.epochs} epochs/run, {total:.0f}s total)")
    print("=" * 62 + "\n")
    table = to_markdown(rows)
    print(table)

    # best per cell input
    print("\nBest fusion per cell input (by test_rho):")
    present = [ci for ci in CELL_INPUTS if any(r["cell_input"] == ci for r in rows)]
    for ci in present:
        sub = [r for r in rows if r["cell_input"] == ci]
        best = max(sub, key=lambda r: r["test_rho"])
        print(f"  {ci:11s} -> {best['fusion']:9s} (test_rho={best['test_rho']:.3f})")

    if args.csv:
        to_csv(rows, args.csv)
        print(f"\nSaved CSV to {args.csv}")

    return rows


if __name__ == "__main__":
    main()
