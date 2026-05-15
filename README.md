# A Structured Evaluation of LLM Verification Robustness

This repository is a **compact Phase 2 hub**: experiment outputs under `check_covid/data` and `emergent/data`, one shared adversarial-generation script under `code/`, readable prompt exports under `prompts/`, and an Emergent human-evaluation sheet. Full plotting drivers, Ragnarök sources, and Phase 1 artefacts live in the parent **RAG_Project_Continue-main** checkout when you keep both trees together.

---

## Top-level layout

```text
.
├── README.md                 ← You are here
├── code/
│   └── generate_adversarial.py   Unified Check COVID + Emergent adversarial / batch CLI
├── check_covid/
│   └── data/                 Check COVID Phase 2 CSV snapshots (see below)
├── emergent/
│   ├── data/                 Emergent Phase 2 CSV snapshots (see below)
│   └── human_evaluation/
│       └── emergent_human_eval_n50_seed42.csv
└── prompts/
    ├── RAGNAROK_v1.txt
    ├── RAGNAROK_v2.txt
    ├── liar_attack.txt
    └── fact_inversion_attack.txt
```

Ignored for documentation purposes: `.DS_Store`, `code/__pycache__/`.

---

## `code/` — adversarial generation

| File | Purpose |
|------|---------|
| [`code/generate_adversarial.py`](code/generate_adversarial.py) | Single module for **both** benchmarks: live completions, OpenAI Batch JSONL, submit/check/download batch, map batch results, Check-COVID-only helpers (filter rows, verbose batch report, pipeline wrappers). |

CLI requires **`--dataset check_covid`** or **`--dataset emergent`**. Run from this repo root:

```bash
python code/generate_adversarial.py --help
python code/generate_adversarial.py --dataset check_covid --csv claims.csv --out out.jsonl
python code/generate_adversarial.py --dataset emergent --csv claims.csv --out out.jsonl --template liar
```

Set `OPENAI_API_KEY` in the environment (optional `.env`; never commit secrets).

**Schemas (short):**

- **check_covid**: `claim`, `label` (support/refute), `abstract`; optional `query`, `id`, `cord_id`.
- **emergent**: `claimHeadline`, `articleStance` (for/against), `articleBody`; optional `claimId`, `articleId`, `claimTruthiness` for batch flows.

---

## `check_covid/data/` — Check COVID artefacts

| Path | Contents |
|------|-----------|
| `refuted_claims_filtered.csv` | Filtered refuted-claims table |
| `supported_claims_filtered.csv` | Filtered supported-claims table |
| `generate_adversarial/` | Adversarial-generation CSVs (`support_to_refute/`, `refute_to_support/`) |
| `single_document/` | Per-**model** folders (e.g. `GPT4.1`, `GPT5`, `llama3.1`, `phi4`, `DeepSeek-R1-Distill-Qwen-32B`). Inside each: condition folders such as `RAG_support`, `RAG_refute`, `RAG_support_adversarial`, `RAG_refute_adversarial`, `non-RAG_support`, `non_RAG_refute`, each with a `results.csv` (or equivalent) aggregate. |
| `pair_document/` | Per-**model pair** folders (`*_pair`, e.g. `GPT4.1_pair`, `GPT5_pair`, `llama3.1_pair`, `phi4_pair`, `DeepSeek_pair`). Inside each: **four** order/helpfulness cells (`helpful_first_original_support`, `helpful_second_original_support`, `helpful_first_original_refute`, `helpful_second_original_refute`), each with `results.csv`. |

Together, `single_document` and `pair_document` hold the condensed evaluation outputs used for Phase 2 figures and tables in the main project.

---

## `emergent/data/` — Emergent artefacts

| Path | Contents |
|------|-----------|
| `emergent_for_unique_claims_filtered_queries.csv` | For-side unique claims + queries |
| `emergent_against_unique_claims_filtered_queries.csv` | Against-side unique claims + queries |
| `single_document/` | Per-**model** folders (`GPT4.1`, `GPT5`, `llama3.1`, `phi4`, `deepseek`). Inside each: `RAG_for`, `RAG_against`, `non_RAG_for`, `non_RAG_against` (each with `results.csv`, except one legacy `result.csv` under `deepseek/non_RAG_against/`). |
| `pair_document/` | Per-**model pair** folders (`GPT4.1_pair`, `GPT5_pair`, `llama3.1_pair`, `deepseek_pair`). Inside each: four cells `True_helpful_first`, `True_helpful_second`, `False_helpful_first`, `False_helpful_second`, each with `results.csv`. |

---

## `emergent/human_evaluation/`

Example blind human-evaluation export used alongside automated scoring:

- `emergent_human_eval_n50_seed42.csv`

Sampling scripts, κ-vs-LLM utilities, and additional sheets remain in the main repo under `Human_Evaluation/` when you need the full harness.

---

## `prompts/`

Plain-text exports of Ragnarök-style instructions and attack wording used in generation experiments:

- `RAGNAROK_v1.txt`, `RAGNAROK_v2.txt`
- `liar_attack.txt`, `fact_inversion_attack.txt`

The **implementations** that consume these (templates, batch exporters) live under **`ragnarok/src/...`** in **RAG_Project_Continue-main**, e.g. `ragnarok/src/generate/templates/ragnarok_templates.py`, `ragnarok/src/scripts/prepare_requests_check_covid.py`, `prepare_requests_emergent.py`.

---

## Full pipeline in the main repository

When this hub sits next to **RAG_Project_Continue-main**:

| Benchmark | Analysis / plots | Typical data root (monorepo) |
|-----------|------------------|------------------------------|
| Check COVID | `Check_Covid/Check_Covid_Data/Experiment_Phase_2/preprocess.py` | `Check_Covid/Check_Covid_Data/Experiment_Phase_2/` |
| Emergent | `Emergent/Evaluate_Response.py` | `Emergent/Experiment_Phase_2/` |

```bash
cd Check_Covid/Check_Covid_Data/Experiment_Phase_2 && python preprocess.py
cd Emergent && python Evaluate_Response.py
```

Phase 1 outputs, if needed: `Check_Covid/.../Experiment_Phase_1`, `Emergent/Experiment_Phase_1`.
