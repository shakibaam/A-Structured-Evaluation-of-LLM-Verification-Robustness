# A Structured Evaluation of LLM Verification Robustness

This is the official GitHub repository for the paper **"When Evidence Disagrees: A Structured Evaluation of LLM Verification Robustness"**. It contains all experiment outputs under `check_covid/data` and `emergent/data`, one shared adversarial-generation script under `code/`, per-benchmark preprocessing / bootstrap / evaluation code under `check_covid/code/` and `emergent/code/`, readable prompt exports under `prompts/`, and an Emergent human-evaluation sheet.

---

## Top-level layout

```
.
├── README.md
├── code/
│   └── generate_adversarial.py
├── check_covid/
│   ├── code/
│   │   ├── bootstrap.py
│   │   └── preprocess.py
│   └── data/
│       ├── refuted_claims_filtered.csv
│       ├── supported_claims_filtered.csv
│       ├── generate_adversarial/
│       │   ├── support_to_refute/
│       │   │   └── support_to_refute.csv
│       │   └── refute_to_support/
│       │       └── refute_to_support.csv
│       ├── single_document/
│       │   └── <model>/          # deepseek | GPT4.1 | GPT5 | llama3.1 | phi4
│       │       ├── RAG_support/
│       │       ├── RAG_refute/
│       │       ├── RAG_support_adversarial/
│       │       ├── RAG_refute_adversarial/
│       │       ├── non_RAG_support/
│       │       └── non_RAG_refute/
│       └── pair_document/
│           └── <model>_pair/     # deepseek | GPT4.1 | GPT5 | llama3.1 | phi4
│               ├── helpful_first_original_support/
│               ├── helpful_first_original_refute/
│               ├── helpful_second_original_support/
│               └── helpful_second_original_refute/
├── emergent/
│   ├── code/
│   │   ├── bootstrap.py
│   │   ├── evaluate_response.py
│   │   └── preprocess.py
│   ├── data/
│   │   ├── for_unique_queries.csv
│   │   ├── against_unique_queries.csv
│   │   ├── generate_adversarial/
│   │   │   ├── false_to_true.csv
│   │   │   └── true_to_false.csv
│   │   ├── single_document/
│   │   │   └── <model>/          # deepseek | GPT4.1 | GPT5 | llama3.1 | phi4
│   │   │       ├── RAG_for/
│   │   │       ├── RAG_against/
│   │   │       ├── RAG_False_to_True_Adv/
│   │   │       ├── RAG_True_to_False_Adv/
│   │   │       ├── non_RAG_for/
│   │   │       └── non_RAG_against/
│   │   └── pair_document/
│   │       └── <model>_pair/     # deepseek | GPT4.1 | GPT5 | llama3.1
│   │           ├── True_helpful_first/
│   │           ├── True_helpful_second/
│   │           ├── False_helpful_first/
│   │           └── False_helpful_second/
│   └── human_evaluation/
│       ├── compute_kappa.py
│       ├── sample_human_eval.py
│       └── emergent_human_eval_n50_seed42.csv
└── prompts/
    ├── RAGNAROK_v1.txt
    ├── RAGNAROK_v2.txt
    ├── liar_attack.txt
    └── fact_inversion_attack.txt
```

Each leaf folder contains a `results.csv` with per-example model outputs and aggregated scores.

---

## `code/` — adversarial generation

| File | Purpose |
|------|---------|
| [`code/generate_adversarial.py`](code/generate_adversarial.py) | Single module for **both** benchmarks: live completions, OpenAI Batch JSONL, submit/check/download batch, map batch results, Check-COVID-only helpers (filter rows, verbose batch report, pipeline wrappers). |

CLI requires `--dataset check_covid` or `--dataset emergent`. Run from the repo root:

```bash
python code/generate_adversarial.py --help
python code/generate_adversarial.py --dataset check_covid --csv claims.csv --out out.jsonl
python code/generate_adversarial.py --dataset emergent --csv claims.csv --out out.jsonl --template liar
```

Set `OPENAI_API_KEY` in the environment (optional `.env`; never commit secrets).

**Input schemas:**

| Benchmark | Required columns | Optional columns |
|-----------|-----------------|-----------------|
| check_covid | `claim`, `label` (support/refute), `abstract` | `query`, `id`, `cord_id` |
| emergent | `claimHeadline`, `articleStance` (for/against), `articleBody` | `claimId`, `articleId`, `claimTruthiness` |

---

## `check_covid/code/` — Check-COVID preprocessing and bootstrap

| File | Purpose |
|------|---------|
| [`check_covid/code/preprocess.py`](check_covid/code/preprocess.py) | CSV manipulation (query injection, label splitting, sampling, query framing) and JSONL ingestion (OpenAI Batch / open-source vLLM / DeepSeek) for attaching model answers to condition CSVs. |
| [`check_covid/code/bootstrap.py`](check_covid/code/bootstrap.py) | Per-condition YES/NO rate bootstrap (95% CI) over one `results.csv`. |

Run the bootstrap script on any condition folder:

```bash
python check_covid/code/bootstrap.py \
    --csv check_covid/data/single_document/GPT5/RAG_support/results.csv \
    --out check_covid/data/single_document/GPT5/RAG_support/bootstrap_results.txt
```

---

## `check_covid/data/`

### Source data

| File | Contents |
|------|----------|
| `refuted_claims_filtered.csv` | Filtered refuted-claims table |
| `supported_claims_filtered.csv` | Filtered supported-claims table |

### `generate_adversarial/`

Adversarial CSVs produced by flipping document polarity:

| Subfolder | Description |
|-----------|-------------|
| `support_to_refute/` | Originally supporting documents rewritten to refute |
| `refute_to_support/` | Originally refuting documents rewritten to support |

### `single_document/`

One folder per model: `deepseek`, `GPT4.1`, `GPT5`, `llama3.1`, `phi4`.

Each contains six condition folders:

| Condition folder | Description |
|-----------------|-------------|
| `RAG_support` | RAG with a genuine supporting document |
| `RAG_refute` | RAG with a genuine refuting document |
| `RAG_support_adversarial` | RAG with an adversarially flipped (support→refute) document |
| `RAG_refute_adversarial` | RAG with an adversarially flipped (refute→support) document |
| `non_RAG_support` | No retrieval; claim is originally supported |
| `non_RAG_refute` | No retrieval; claim is originally refuted |

### `pair_document/`

One folder per model pair: `deepseek_pair`, `GPT4.1_pair`, `GPT5_pair`, `llama3.1_pair`, `phi4_pair`.

Each contains four order × polarity cells:

| Condition folder | Helpful doc position | Original doc stance |
|-----------------|---------------------|---------------------|
| `helpful_first_original_support` | First | Support |
| `helpful_second_original_support` | Second | Support |
| `helpful_first_original_refute` | First | Refute |
| `helpful_second_original_refute` | Second | Refute |

---

## `emergent/code/` — Emergent preprocessing, evaluation, and human-eval tooling

| File | Purpose |
|------|---------|
| [`emergent/code/preprocess.py`](emergent/code/preprocess.py) | CSV manipulation (sampling, claim deduplication, stance filtering, query injection) and JSONL ingestion (OpenAI Batch / open-source vLLM / DeepSeek wrappers) for turning the raw Emergent dump into the layout under `emergent/data/`. |
| [`emergent/code/evaluate_response.py`](emergent/code/evaluate_response.py) | Accuracy helpers for condition CSVs: compare ``Response`` to gold labels (overall or grouped by ``claimTruthiness``), append CSVs, and silent accuracy functions for scripting. |
| [`emergent/code/bootstrap.py`](emergent/code/bootstrap.py) | Per-condition classification accuracy bootstrap (overall + per ground-truth) over one `results.csv`. |

Bootstrap one Emergent condition:

```bash
python emergent/code/bootstrap.py \
    --csv emergent/data/single_document/GPT5/RAG_for/results.csv \
    --out emergent/data/single_document/GPT5/RAG_for/bootstrap_accuracy.txt
```

---

## `emergent/human_evaluation/` — human annotation tooling

| File | Purpose |
|------|---------|
| [`emergent/human_evaluation/sample_human_eval.py`](emergent/human_evaluation/sample_human_eval.py) | Stratified random sampler that produces the human-evaluation template (and its blind sibling). |
| [`emergent/human_evaluation/compute_kappa.py`](emergent/human_evaluation/compute_kappa.py) | Cohen's kappa, exact agreement, and contingency table for a filled human-evaluation CSV. |
| `emergent_human_eval_n50_seed42.csv` | 50 randomly sampled examples (seed 42) for human annotation |

Reproduce the human-evaluation sample (then score it):

```bash
python emergent/human_evaluation/sample_human_eval.py \
    --single-doc-dir emergent/data/single_document/GPT4.1 \
    --out-dir emergent/human_evaluation \
    --n 50 --seed 42

python emergent/human_evaluation/compute_kappa.py \
    emergent/human_evaluation/emergent_human_eval_n50_seed42.csv
```

---

## `emergent/data/`

### Source data

| File | Contents |
|------|----------|
| `for_unique_queries.csv` | Unique queries for *for*-stance claims |
| `against_unique_queries.csv` | Unique queries for *against*-stance claims |

### `generate_adversarial/`

| File | Description |
|------|-------------|
| `false_to_true.csv` | False claims rewritten with supporting evidence |
| `true_to_false.csv` | True claims rewritten with contradicting evidence |

### `single_document/`

One folder per model: `deepseek`, `GPT4.1`, `GPT5`, `llama3.1`, `phi4`.

Each contains six condition folders:

| Condition folder | Description |
|-----------------|-------------|
| `RAG_for` | RAG with a *for*-stance document |
| `RAG_against` | RAG with an *against*-stance document |
| `RAG_False_to_True_Adv` | RAG with an adversarially flipped false→true document |
| `RAG_True_to_False_Adv` | RAG with an adversarially flipped true→false document |
| `non_RAG_for` | No retrieval; claim has *for*-stance articles |
| `non_RAG_against` | No retrieval; claim has *against*-stance articles |

### `pair_document/`

One folder per model pair: `deepseek_pair`, `GPT4.1_pair`, `GPT5_pair`, `llama3.1_pair`.

Each contains four order × truthiness cells:

| Condition folder | Helpful doc position | Claim truthiness |
|-----------------|---------------------|-----------------|
| `True_helpful_first` | First | True |
| `True_helpful_second` | Second | True |
| `False_helpful_first` | First | False |
| `False_helpful_second` | Second | False |

---

## `prompts/`

Plain-text exports of Ragnarök-style instructions and attack wording used in generation experiments:

| File | Description |
|------|-------------|
| `RAGNAROK_v1.txt` | Ragnarök prompt template v1 |
| `RAGNAROK_v2.txt` | Ragnarök prompt template v2 (revised) |
| `liar_attack.txt` | Liar-style adversarial attack prompt |
| `fact_inversion_attack.txt` | Fact-inversion adversarial attack prompt |

Implementations that consume these templates live in the [Ragnarök](https://github.com/castorini/ragnarok) framework.

---
