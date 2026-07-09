"""
Entry point.  Run:  python -m drp.main

Reads the two switches from config.py (MODE, CELL_INPUT), builds the dataset,
trains the holistic model, prints metrics, and runs the interpretability readout.

Full / Subset will try to load real data; if the omics/response download is
blocked (e.g. a locked-down environment) it falls back to Demo so you always get
an end-to-end run, and tells you exactly what happened.
"""

from __future__ import annotations

import torch

from . import config
from .data import load_dataset, make_synthetic_dataset
from .train import train
from .interpret import report


def main():
    print("=" * 62)
    print(f"  Holistic interpretable drug-response pipeline")
    print(f"  MODE={config.MODE}   CELL_INPUT={config.CELL_INPUT}   FUSION={config.FUSION}")
    print("=" * 62)

    try:
        ds = load_dataset()
        used_mode = config.MODE
    except RuntimeError as e:
        print(f"\n  [Full/Subset data unavailable here]\n  {e}\n")
        print("  -> Falling back to a synthetic Demo run so you still get results.\n")
        ds = make_synthetic_dataset(config.SIZING["Demo"], config.TrainConfig().seed)
        used_mode = "Demo (fallback)"

    print(f"\n  Dataset ready  [{used_mode}]")
    for k, v in ds.meta.items():
        print(f"    {k}: {v}")
    print(f"    pairs: {len(ds.pairs)}   cell_input_dim: {ds.cell_input_dim}")

    print("\n  Training")
    out = train(ds, device=torch.device("cpu"), verbose=True)

    m = out["metrics"]
    print("\n  Results (Spearman rho | Pearson r | RMSE)")
    print("  ---------------------------------------------")
    for split in ("train", "val", "test"):
        s = m[split]
        print(f"    {split:<5}  rho={s['spearman']:.3f}   r={s['pearson']:.3f}   "
              f"rmse={s['rmse']:.3f}   (n={s['n']})")
    print(f"    params={out['n_params']:,}   time={out['seconds']:.1f}s")

    report(out["model"], ds, out["splits"]["test"], torch.device("cpu"),
           config.TrainConfig().batch_size)

    print("\n  Done.")
    return out


if __name__ == "__main__":
    main()
