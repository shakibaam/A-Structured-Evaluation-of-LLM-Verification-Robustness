# Human evaluation

**Scripts** (repo root paths):

- `Human_Evaluation/sample_single_doc_human_eval.py` — stratified sample → CSV templates (incl. blind version)
- `Human_Evaluation/compute_kappa_vs_llm.py` — κ vs LLM after labels are filled

**Example outputs:** `Human_Evaluation/outputs/`

```bash
cd Human_Evaluation
python sample_single_doc_human_eval.py --help
python compute_kappa_vs_llm.py outputs/your_labeled.csv
```

Back to [hub README](../README.md).
