#!/usr/bin/env python3
"""Compute Cohen's kappa between ``human_label`` and ``llm_label_normalized``
for filled human-evaluation CSVs produced by ``sample_human_eval.py``.

Usage::

    python emergent/human_evaluation/compute_kappa.py \\
        emergent/human_evaluation/emergent_human_eval_n50_seed42.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def _norm_human_emergent(s: str) -> str | None:
    t = str(s).strip().lower().rstrip(".")
    if t in ("true", "t", "yes", "1"):
        return "true"
    if t in ("false", "f", "no", "0"):
        return "false"
    return None


def cohen_kappa(rater_a: list[str], rater_b: list[str], labels: list[str]) -> float:
    """Multiclass Cohen's kappa; ``labels`` is the fixed category list."""
    n = len(rater_a)
    if n != len(rater_b) or n == 0:
        raise ValueError("Raters must have the same non-empty length.")
    agree = sum(1 for x, y in zip(rater_a, rater_b) if x == y)
    p_o = agree / n
    ca = Counter(rater_a)
    cb = Counter(rater_b)
    p_e = sum((ca[c] / n) * (cb[c] / n) for c in labels)
    if p_e >= 1.0 - 1e-15:
        return 1.0 if p_o >= 1.0 - 1e-15 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def evaluate_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"No rows in {path}", file=sys.stderr)
        return

    labels = ["true", "false"]

    human: list[str] = []
    llm: list[str] = []
    skipped = 0
    for r in rows:
        hl = _norm_human_emergent(r.get("human_label", ""))
        ll = (r.get("llm_label_normalized") or "").strip().lower()
        if hl is None or ll not in labels:
            skipped += 1
            continue
        human.append(hl)
        llm.append(ll)

    m = len(human)
    if m == 0:
        print("No valid paired labels (fill human_label; check spelling).", file=sys.stderr)
        return

    kappa = cohen_kappa(human, llm, labels)
    agree = sum(1 for a, b in zip(human, llm) if a == b)
    print(f"File: {path}")
    print(f"Valid pairs: {m} (skipped rows: {skipped})")
    print(f"Exact agreement: {agree}/{m} ({100.0 * agree / m:.1f}%)")
    print(f"Cohen's kappa (human vs LLM): {kappa:.4f}")

    print("Contingency (human rows, LLM cols):")
    mat = {h: {l: 0 for l in labels} for h in labels}
    for a, b in zip(human, llm):
        mat[a][b] += 1
    print("human\\llm\t" + "\t".join(labels))
    for h in labels:
        print(h + "\t" + "\t".join(str(mat[h][l]) for l in labels))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv_files", nargs="+", type=Path, help="Filled human-eval CSV(s).")
    args = p.parse_args()
    for path in args.csv_files:
        if not path.is_file():
            print(f"Missing file: {path}", file=sys.stderr)
            continue
        evaluate_csv(path)
        print()


if __name__ == "__main__":
    main()
