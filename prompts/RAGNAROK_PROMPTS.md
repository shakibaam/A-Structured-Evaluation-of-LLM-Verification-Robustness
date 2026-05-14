# Ragnarök prompt text (reference)

This document mirrors the strings in `RagnarokTemplates` from the repo implementation.

**Canonical source code:** `ragnarok/src/generate/templates/ragnarok_templates.py`  
**Prompt mode enum:** `ragnarok/src/generate/llm.py` (`PromptMode`)

---

## Shared snippets

### `input_context` template

```text
Context: {context}
```

### `user_input` template

```text
{query}
```

The generator concatenates **instruction**, **documents/context**, **query**, and repeats instruction where applicable (see `__call__` in the Python class).

---

## System messages

### GPT — with citations (RAG)

```text
This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's question based on the context references. The assistant should also indicate when the answer cannot be found in the context references.
```

### GPT — without citations (non-RAG / no-cite modes)

```text
This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's question.
```

### Claude — with citations

```text
This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's question based on the context references. The assistant should also indicate when the answer cannot be found in the context references.
```

### Claude — without citations

```text
This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's question.
```

### ChatQA baseline (`system_message_chatqa`)

```text
System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.
```

---

## Instruction strings (`instruction_*`)

### Official / legacy (`instruction_official`)

```text
Please give a full and complete answer for the question.
```

### Ragnarök original (`instruction_ragnarok`)

```text
Please give a full and complete answer for the question. Cite each context document inline that supports your answer within brackets [] using the IEEE format. Ensure each sentence is properly cited.
```

### `RAGNAROK_V2` — `instruction_ragnarok_v2`

```text
Please give a full and complete answer for the question. Cite each context document inline that supports your answer within brackets [] using the IEEE format. Each sentence should have at most three citations. Order the citations in decreasing order of importance. Never include or mention anything about references, this is already provided, just answer the question such that each sentence has one or more sentence-level citations and say nothing else.
```

### `RAGNAROK_V3` — `instruction_ragnarok_v3`

```text
Provide a concise, information-dense answer to the question. Your response must not exceed 380 words under any circumstances. Prioritize the most relevant and impactful information within this strict limit. Cite supporting context documents inline using IEEE format in square brackets []. Include 1-3 citations per sentence, ordered by decreasing importance. Ensure each sentence has at least one citation. Focus solely on answering the question with properly cited information. Avoid mentioning references or providing any meta-commentary about the answering process.
```

### `RAGNAROK_V4` — `instruction_ragnarok_v4`

```text
Provide a concise, information-dense answer to the question. Your response must not exceed 380 words under any circumstances. Prioritize the most relevant and impactful information within this strict limit. Ensure your answer directly addresses the question and maintains coherence throughout. Cite supporting context documents inline using IEEE format in square brackets []. Include 1-3 citations per sentence, ordered by decreasing importance. Ensure each sentence has at least one citation. Use multiple sources to provide a well-rounded answer when possible. If sources contradict each other, acknowledge this and explain the discrepancy. Express uncertainty when appropriate rather than making unfounded claims. Prioritize factual accuracy and avoid speculation. Focus solely on answering the question with properly cited information. Avoid mentioning references or providing any meta-commentary about the answering process.
```

### `RAGNAROK_V4_NO_CITE` — `instruction_ragnarok_v4_no_cite`

Used for **non-RAG** runs when paired with no-context requests.

```text
Provide a concise, information-dense answer to the question. Your response must not exceed 380 words under any circumstances. Prioritize the most relevant and impactful information within this strict limit. Ensure your answer directly addresses the question and maintains coherence throughout. Provide a well-rounded answer when possible. Express uncertainty when appropriate rather than making unfounded claims. Prioritize factual accuracy and avoid speculation. Focus solely on answering the question. Avoid references or providing any meta-commentary about the answering process.
```

### `RAGNAROK_V4_BIOGEN` — `instruction_ragnarok_v4_biogen`

```text
Provide a concise, information-dense answer to the question in a single cohesive paragraph, avoiding lists and bullet points. Your response must not exceed 150 words (excluding references) under any circumstances. Prioritize the most relevant and impactful information within this strict limit. Ensure your answer directly addresses the question and maintains coherence throughout. Cite the supporting context PubMed documents inline using IEEE format in square brackets []. Include 1-3 citations per sentence, ordered by decreasing importance. Ensure each sentence has at least one citation. Use multiple sources to provide a well-rounded answer when possible. If sources contradict each other, acknowledge this and explain the discrepancy. Express uncertainty when appropriate rather than making unfounded claims. Prioritize factual accuracy and avoid speculation or potentially harmful advice. Focus solely on answering the question with properly cited information. Avoid mentioning references or providing any meta-commentary about the answering process.
```

### `RAGNAROK_V5_BIOGEN` — `instruction_ragnarok_v5_biogen`

```text
Provide a concise, information-dense answer to the biomedical question in a single cohesive paragraph, avoiding lists and bullet points. Your response must not exceed 150 words (excluding references) under any circumstances. Prioritize the most relevant and impactful information within this strict limit. Ensure each sentence is clear, interpretable, and directly relevant to the question. Cite the supporting context PubMed documents inline using IEEE format in square brackets []. Include 1-3 citations per sentence, ordered by decreasing importance. Ensure each sentence has at least one citation. Use multiple sources to provide a well-rounded answer when possible. Focus on required and relevant information, avoiding unnecessary or borderline content. If sources contradict each other, acknowledge this and explain the discrepancy. Express uncertainty when appropriate rather than making unfounded claims. Prioritize factual accuracy and avoid speculation or potentially harmful advice. For patient-oriented questions, provide information suitable for clinician review. Assume the reader is a healthcare professional, but avoid overly technical jargon. Do not include trivial statements or generic recommendations to see a health professional, stick to the 150 word limit. Ensure all information is supported by the provided PubMed abstracts. Avoid mentioning references or providing any meta-commentary about the answering process.
```

### `RAGNAROK_V5_BIOGEN_NO_CITE` — `instruction_ragnarok_v5_biogen_no_cite`

```text
Provide a concise, information-dense answer to the biomedical question in a single cohesive paragraph, avoiding lists and bullet points. Your response must not exceed 150 words under any circumstances. Prioritize the most relevant and impactful information within this strict limit. Ensure each sentence is clear, interpretable, and directly relevant to the question. Provide a well-rounded answer when possible. Focus on required and relevant information, avoiding unnecessary or borderline content. Express uncertainty when appropriate rather than making unfounded claims. Prioritize factual accuracy and avoid speculation or potentially harmful advice. For patient-oriented questions, provide information suitable for clinician review. Assume the reader is a healthcare professional, but avoid overly technical jargon. Do not include trivial statements or generic recommendations to see a health professional, stick to the 150 word limit. Avoid providing any meta-commentary about the answering process.
```

---

## Notes for reproducibility

1. **Mode selection** — Map experiment condition → `PromptMode` in batch/export scripts.
2. **Non-RAG** — Typically uses `RAGNAROK_V4_NO_CITE` (or biogen no-cite variants) together with request builders that omit documents.
3. **Paper freeze** — If prompts change, update both this file (for readers) and `ragnarok_templates.py` (for generation).
