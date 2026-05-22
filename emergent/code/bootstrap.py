"""Bootstrap classification accuracy on an Emergent condition CSV.

Reads a ``results.csv`` with a model verdict column (default ``Response``)
and a gold-label column (default ``claimTruthiness``), normalises both,
then writes overall and per-ground-truth accuracy with 95% bootstrap CIs.

Usage::

    python emergent/code/bootstrap.py \\
        --csv path/to/results.csv \\
        --out path/to/bootstrap_accuracy.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_mean(y: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Return ``(mean, 2.5th percentile, 97.5th percentile)`` for ``y``."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = y[idx].mean()
    return boot.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _norm(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^\w]+$", "", regex=True)
    )


def _write_block(f, name: str, sub_df: pd.DataFrame, n_boot: int, seed: int) -> None:
    y = sub_df["correct"].to_numpy()
    if len(y) == 0:
        f.write(f"{name}: n=0 (no rows)\n")
        return
    mean, lo, hi = bootstrap_mean(y, n_boot=n_boot, seed=seed)
    f.write(
        f"{name}: {100 * mean:.2f}% (95% CI {100 * lo:.2f}–{100 * hi:.2f}), n={len(y)}\n"
    )


def run(
    csv_path: Path,
    out_path: Path,
    verdict_col: str = "Response",
    gold_col: str = "claimTruthiness",
    n_boot: int = 2000,
    seed: int = 42,
) -> None:
    df = pd.read_csv(csv_path)
    resp = _norm(df[verdict_col])
    gold = _norm(df[gold_col])
    df["correct"] = (resp == gold).astype(int)
    df["_gold"] = gold

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"CSV: {csv_path}\n")
        f.write(f"bootstrap resamples = {n_boot}, seed = {seed}\n\n")
        _write_block(f, "Overall", df, n_boot, seed)
        _write_block(f, "GroundTruth=FALSE", df[df["_gold"] == "FALSE"], n_boot, seed)
        _write_block(f, "GroundTruth=TRUE", df[df["_gold"] == "TRUE"], n_boot, seed)

    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, required=True, help="Input results.csv path.")
    p.add_argument("--out", type=Path, required=True, help="Output bootstrap_accuracy.txt path.")
    p.add_argument("--verdict-col", default="Response")
    p.add_argument("--gold-col", default="claimTruthiness")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(
        args.csv,
        args.out,
        verdict_col=args.verdict_col,
        gold_col=args.gold_col,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
