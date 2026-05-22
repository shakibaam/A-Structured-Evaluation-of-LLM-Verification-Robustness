#!/usr/bin/env python3
"""Draw a stratified random sample from Emergent single-document results for
human-vs-LLM agreement (Cohen's kappa).

Defaults match the released sheet ``emergent_human_eval_n50_seed42.csv``:
50 items, seed 42, sampled across the four single-document conditions
(``RAG_for``, ``RAG_against``, ``non_RAG_for``, ``non_RAG_against``) under
``emergent/data/single_document/<model>/``.

Outputs two CSVs in ``--out-dir``:

* ``emergent_human_eval_n{N}_seed{S}.csv`` — full template with
  ``llm_label_normalized`` and empty ``human_label`` to fill in.
* ``emergent_human_eval_n{N}_seed{S}_blind.csv`` — same rows with the
  LLM verdict removed, for blind annotation.

After labelling, score agreement with ``compute_kappa.py``.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SINGLE_DOC = REPO_ROOT / "emergent" / "data" / "single_document" / "GPT4.1"
DEFAULT_OUT_DIR = REPO_ROOT / "emergent" / "human_evaluation"

SINGLE_DOC_CONDITIONS = (
    "non_RAG_for",
    "non_RAG_against",
    "RAG_for",
    "RAG_against",
)


def _full_document(text: str) -> str:
    return (text or "").strip()


def _norm_emergent_llm(raw: str) -> str | None:
    s = str(raw).strip().lower().rstrip(".")
    if s in ("true", "t", "yes", "1"):
        return "true"
    if s in ("false", "f", "no", "0"):
        return "false"
    return None


def _collect_emergent_rows(one_doc_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for cond in SINGLE_DOC_CONDITIONS:
        folder = one_doc_dir / cond
        if not folder.is_dir():
            continue
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            continue
        csv_path = csv_files[0]
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = (row.get("claimId") or "").strip()
                norm = _norm_emergent_llm(row.get("Response") or "")
                if norm is None:
                    continue
                headline = (row.get("claimHeadline") or "").strip()
                base_key = cid or f"hash:{hash(headline)}"
                rows.append(
                    {
                        "dataset": "emergent",
                        "condition": cond,
                        "source_type": "one_doc",
                        "source_csv": str(csv_path.relative_to(REPO_ROOT)),
                        "row_key": f"{base_key}|{cond}",
                        "claim_id": cid,
                        "gold_label_csv": (row.get("claimTruthiness") or "").strip(),
                        "claim_text": headline,
                        "query": (row.get("query") or "").strip(),
                        "document_excerpt": _full_document(row.get("articleBody") or ""),
                        "llm_answer_raw": (row.get("LLM_Response") or "").strip(),
                        "llm_explanation": "",
                        "llm_label_normalized": norm,
                    }
                )
    return rows


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r["row_key"], r)
    return list(seen.values())


def _stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    conds = [c for c in sorted(by_cond.keys()) if by_cond[c]]
    if not conds:
        return []

    out: list[dict] = []
    base, extra = divmod(n, len(conds))
    for i, c in enumerate(conds):
        want = base + (1 if i < extra else 0)
        pool = by_cond[c][:]
        rng.shuffle(pool)
        out.extend(pool[:want])

    if len(out) < n:
        rest = [r for r in rows if r not in out]
        rng.shuffle(rest)
        for r in rest:
            if len(out) >= n:
                break
            out.append(r)
    rng.shuffle(out)
    return out[:n]


def _write_outputs(path_full: Path, sample: list[dict], blind: bool) -> None:
    path_full.parent.mkdir(parents=True, exist_ok=True)
    blind_path = path_full.with_name(path_full.stem + "_blind.csv")

    fieldnames = [
        "eval_id",
        "dataset",
        "condition",
        "source_type",
        "source_csv",
        "claim_id",
        "row_key",
        "gold_label_csv",
        "claim_text",
        "query",
        "document_excerpt",
        "llm_answer_raw",
        "llm_explanation",
        "llm_label_normalized",
        "human_label",
        "annotator_notes",
    ]
    blind_fields = [f for f in fieldnames if f != "llm_label_normalized"]

    def prep_row(i: int, r: dict) -> dict:
        return {
            "eval_id": str(i),
            **{k: r[k] for k in (
                "dataset",
                "condition",
                "source_type",
                "source_csv",
                "claim_id",
                "row_key",
                "gold_label_csv",
                "claim_text",
                "query",
                "document_excerpt",
                "llm_answer_raw",
                "llm_explanation",
                "llm_label_normalized",
            )},
            "human_label": "",
            "annotator_notes": "",
        }

    rows_out = [prep_row(i + 1, r) for i, r in enumerate(sample)]

    with path_full.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    if blind:
        with blind_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=blind_fields)
            w.writeheader()
            for row in rows_out:
                w.writerow({k: row[k] for k in blind_fields})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=50, help="Number of items (default 50).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--single-doc-dir",
        type=Path,
        default=DEFAULT_SINGLE_DOC,
        help="Path to emergent/data/single_document/<model>/ for the model to sample from.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--no-blind", action="store_true", help="Skip writing the blind CSV.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = _dedupe(_collect_emergent_rows(args.single_doc_dir))
    sample = _stratified_sample(rows, args.n, rng)

    stem = f"emergent_human_eval_n{args.n}_seed{args.seed}"
    out_csv = args.out_dir / f"{stem}.csv"
    _write_outputs(out_csv, sample, blind=not args.no_blind)

    print(f"Emergent: pooled {len(rows)} unique rows -> sampled {len(sample)}")
    print(f"Wrote: {out_csv}")
    if not args.no_blind:
        print(f"Wrote: {out_csv.with_name(stem + '_blind.csv')}")


if __name__ == "__main__":
    main()
