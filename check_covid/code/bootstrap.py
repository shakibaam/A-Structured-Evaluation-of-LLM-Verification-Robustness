"""Bootstrap YES/NO rates for a Check-COVID condition CSV.

Reads a single ``results.csv`` containing a ``model_answer`` column with
yes/no responses and writes ``bootstrap_results.txt`` with mean rates and
95% confidence intervals from `n_boot` bootstrap resamples.

Usage::

    python check_covid/code/bootstrap.py \\
        --csv path/to/results.csv \\
        --out path/to/bootstrap_results.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_yes_rate(y: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Return ``(mean, 2.5th percentile, 97.5th percentile)`` for ``y``."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = y[idx].mean()
    return boot.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run(csv_path: Path, out_path: Path, n_boot: int = 2000, seed: int = 42) -> None:
    df = pd.read_csv(csv_path)
    ans = (
        df["model_answer"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+$", "", regex=True)
    )
    df["yes"] = (ans == "yes").astype(int)

    y = df["yes"].to_numpy()
    yes_mean, yes_lo, yes_hi = bootstrap_yes_rate(y, n_boot=n_boot, seed=seed)

    no_mean = 1.0 - yes_mean
    no_lo = 1.0 - yes_hi
    no_hi = 1.0 - yes_lo

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"CSV: {csv_path}\n")
        f.write(f"n = {len(y)}\n")
        f.write(f"bootstrap resamples = {n_boot}\n\n")
        f.write(f"YES: {100 * yes_mean:.1f}% (95% CI {100 * yes_lo:.1f}–{100 * yes_hi:.1f})\n")
        f.write(f"NO : {100 * no_mean:.1f}% (95% CI {100 * no_lo:.1f}–{100 * no_hi:.1f})\n")

    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, required=True, help="Input results.csv path.")
    p.add_argument("--out", type=Path, required=True, help="Output bootstrap_results.txt path.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args.csv, args.out, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
