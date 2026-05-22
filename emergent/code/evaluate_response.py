"""Evaluation utilities for the Emergent benchmark.

Compare model ``Response`` columns against gold labels in condition CSVs,
optionally grouped by ``claimTruthiness``. For bootstrap confidence
intervals on a single ``results.csv``, use ``emergent/code/bootstrap.py``.

All paths are passed as arguments; nothing here points at a private workstation.
"""
from __future__ import annotations

import os

import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8", errors="replace")
        except Exception:
            return pd.read_csv(path, encoding="latin-1")


def evaluate_verdicts(csv_path, verdict_col="Verdict", label_col="articleStance"):
    """Compare predicted verdicts vs true labels and print accuracy."""
    df = _read_csv(csv_path)

    df[verdict_col] = df[verdict_col].astype(str).str.strip().str.lower()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    correct = (df[verdict_col] == df[label_col]).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0

    print(f"✅ Evaluation Results for: {csv_path}")
    print(f"   Total: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    return accuracy


def evaluate_verdicts_by_group(
    csv_path,
    verdict_col="Verdict",
    label_col="articleStance",
    group_col="claimTruthiness",
):
    """Print accuracy grouped by ``group_col`` (default ``claimTruthiness``)."""
    df = _read_csv(csv_path)

    df[verdict_col] = df[verdict_col].astype(str).str.strip().str.lower()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[group_col] = df[group_col].astype(str).str.strip().str.lower()

    print(f"✅ Evaluation Results (Grouped by {group_col}) for: {csv_path}")
    print("=" * 60)

    results = {}

    for group_name, group_df in df.groupby(group_col):
        correct = (group_df[verdict_col] == group_df[label_col]).sum()
        total = len(group_df)
        accuracy = correct / total if total > 0 else 0
        results[group_name] = accuracy

        print(f"\n📊 Group: {group_name.upper()}")
        print(f"   Total: {total}")
        print(f"   Correct: {correct}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\n" + "=" * 60)
    overall_correct = (df[verdict_col] == df[label_col]).sum()
    overall_total = len(df)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0

    print("📈 OVERALL")
    print(f"   Total: {overall_total}")
    print(f"   Correct: {overall_correct}")
    print(f"   Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")

    return results


def append_csvs(csv1_path, csv2_path, output_path):
    """Append two CSV files and save to ``output_path``."""
    df1 = _read_csv(csv1_path)
    df2 = _read_csv(csv2_path)
    combined_df = pd.concat([df1, df2], ignore_index=True)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print("✅ CSVs appended successfully")
    print(f"   First CSV rows: {len(df1)}")
    print(f"   Second CSV rows: {len(df2)}")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Output saved to: {output_path}")

    return len(combined_df)


def compute_accuracy_silent(
    csv_path,
    verdict_col="Response",
    label_col="claimTruthiness",
):
    """Return ``(accuracy, total_rows, correct_count)`` without printing."""
    try:
        df = _read_csv(csv_path)

        if verdict_col not in df.columns or label_col not in df.columns:
            return None, None, None

        df[verdict_col] = df[verdict_col].astype(str).str.strip().str.lower()
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()

        correct = (df[verdict_col] == df[label_col]).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0

        return accuracy, total, correct
    except Exception as exc:
        print(f"⚠️  Error processing {csv_path}: {exc}")
        return None, None, None


def compute_accuracy_by_group_silent(
    csv_path,
    verdict_col="Response",
    label_col="claimTruthiness",
    group_col="claimTruthiness",
):
    """Return per-group accuracy dict without printing."""
    try:
        df = _read_csv(csv_path)

        if (
            verdict_col not in df.columns
            or label_col not in df.columns
            or group_col not in df.columns
        ):
            return None

        df[verdict_col] = df[verdict_col].astype(str).str.strip().str.lower()
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()
        df[group_col] = df[group_col].astype(str).str.strip().str.lower()

        results = {}
        for group_name, group_df in df.groupby(group_col):
            correct = (group_df[verdict_col] == group_df[label_col]).sum()
            total = len(group_df)
            results[group_name] = correct / total if total > 0 else 0

        return results
    except Exception as exc:
        print(f"⚠️  Error processing {csv_path}: {exc}")
        return None
