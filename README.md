# A Structured Evaluation of LLM Verification Robustness

Minimal map of this repository for **Experiment Phase 2**: where the code, data, human-eval harness, and Ragnarök prompts live. Everything else stays in the existing tree—no mirrors or symlinks here.

---

## Layout

```text
├── check_covid/README.md       Phase 2 paths & driver script
├── emergent/README.md          Phase 2 paths & driver script
├── human_evaluation/README.md  Sampling + κ scripts
├── prompts/RAGNAROK_PROMPTS.md Ragnarök instructions (readable copy)
└── README.md                   This file
```

---

## Phase 2 — Check Covid

| What | Where |
|------|--------|
| Data, CSVs, figures, bootstrap text | `Check_Covid/Check_Covid_Data/Experiment_Phase_2/` |
| Analysis / plots | `Check_Covid/Check_Covid_Data/Experiment_Phase_2/preprocess.py` |
| Adversarial helper | `Check_Covid/Check_Covid_Data/Experiment_Phase_2/generate_adversarial.py` |

Run (from repo root):

```bash
cd Check_Covid/Check_Covid_Data/Experiment_Phase_2 && python preprocess.py
```

---

## Phase 2 — Emergent

| What | Where |
|------|--------|
| Data & outputs | `Emergent/Experiment_Phase_2/` |
| Analysis / plots | `Emergent/Evaluate_Response.py` |

Run:

```bash
cd Emergent && python Evaluate_Response.py
```

---

## Human evaluation

| What | Where |
|------|--------|
| Scripts | `Human_Evaluation/sample_single_doc_human_eval.py`, `Human_Evaluation/compute_kappa_vs_llm.py` |
| Example sheets | `Human_Evaluation/outputs/` |

---

## Ragnarök generation

| What | Where |
|------|--------|
| Readable prompts | [`prompts/RAGNAROK_PROMPTS.md`](prompts/RAGNAROK_PROMPTS.md) |
| Implementation | `ragnarok/src/generate/templates/ragnarok_templates.py` |
| Batch prep (examples in docstrings) | `ragnarok/src/scripts/prepare_requests_check_covid.py`, `prepare_requests_emergent.py` |

---

## Phase 1

Earlier artefacts remain under `Check_Covid/.../Experiment_Phase_1` and `Emergent/Experiment_Phase_1` if you need them.
