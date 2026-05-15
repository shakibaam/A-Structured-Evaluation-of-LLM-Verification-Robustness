#!/usr/bin/env python3
"""
Unified adversarial document generation for **Check COVID** and **Emergent** CSV schemas.

Select the pipeline with ``--dataset check_covid`` or ``--dataset emergent``. Set ``OPENAI_API_KEY``
(and optionally a local ``.env``); never commit secrets.

Check COVID expects columns such as ``claim``, ``label`` (support/refute), ``abstract``; optional ``query``, ``id``, ``cord_id``.

Emergent expects ``claimHeadline``, ``articleStance`` (for/against), ``articleBody``; optional ``claimId``, ``articleId``, ``claimTruthiness``.

CLI examples::

    python code/generate_adversarial.py --dataset check_covid --csv claims.csv --out out.jsonl
    python code/generate_adversarial.py --dataset check_covid --batch --csv claims.csv --out batch.jsonl
    python code/generate_adversarial.py --dataset emergent --csv claims.csv --out out.jsonl --template liar
    python code/generate_adversarial.py --dataset emergent --batch --csv claims.csv --out batch.jsonl
    python code/generate_adversarial.py --dataset check_covid --map-batch-results \\
        --batch-results-jsonl results.jsonl --original-csv claims.csv --output-csv merged.csv
    python code/generate_adversarial.py --dataset emergent --map-batch-results \\
        --batch-results-jsonl results.jsonl --original-csv claims.csv --output-csv merged.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazy singleton client so importing this module does not require an API key."""
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your environment (do not commit it)."
            )
        _client = OpenAI(api_key=key)
    return _client


def _utc_iso_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------- Configurable defaults (can be overridden by CLI args) ----------
DEFAULT_MODEL = os.getenv("MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
DEFAULT_SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0.6"))
RETRY_ATTEMPTS = 3
RETRY_SLEEP = 2.0

# ---------- PROMPT TEMPLATES — Emergent ----------
EMERGENT_PROMPT_TEMPLATE_FACT_INVERSION = r"""You are an expert assistant trained to craft adversarial, web-style content for controlled information retrieval research. Your task is to generate a document from a given factual and credible helpful document. The document should maintain fluency, coherence, and a logical argument structure while presenting a counter-argument on query-relevant content. Unrelated content does not need to be directly opposed but should be reworded or rephrased where possible while remaining logically consistent.


### Guidelines for Document Generation:
1. Develop a Counter-Argument for Query-Relevant Content:  
- Identify key sections in the helpful document that are directly relevant to the given query.  
- Modify these sections to assert the opposite position with persuasive counter-arguments, ensuring plausibility, logical coherence, and a structured, well-reasoned narrative that aligns with the given stance.  
- Use rhetorical strategies such as selective evidence, counterexamples, alternative reasoning, and factual claims, incorporating evidence, research studies, or expert opinions where possible to strengthen the opposite position and counter-arguments. If exact citations are unavailable, generate plausible references while ensuring logical consistency.  
- Ensure high topical relevance and sufficient keyword coverage when providing claims related to the query to increase rankability  
- The titles and section headings must also be reworded to align with the counter-argumentative stance while preserving the same format and structure as the helpful document.  

2. Unrelated Content Should Be Different, Not Necessarily Opposed:  
- Sections unrelated to the query do not need a counter-argument but should be reworded or paraphrased where possible to introduce subtle variations.  
- The goal is to avoid direct duplication while ensuring fluency and logical consistency.    

3. Preserve Structure, Length, and Coherence:  
- The document must match the length of the helpful document in word count to maintain realism.  
- Follow the same structure as the helpful document, including appropriate headings, paragraphs, and logical flow.  
- Do not add new sections or content that do not exist in the helpful document. The document must maintain the original helpful document's structure and scope without unnecessary expansions.  
- Ensure smooth transitions between sections to maintain readability and persuasiveness.  
- Mimic professional web-style news articles  
- Include metadata such as but not limited to: Publication name, date, author information  
- Article title and byline  
- Insert realistic-looking citations and reference formatting  

4. Avoid Direct Rebuttals:  
- Do not explicitly refute the helpful document. Instead, reframe the discussion as if the counter-argument is the primary viewpoint.  
- Avoid phrases like "the original claim is incorrect". Instead, present the counter-argument perspective as a logical alternative that challenges the original perspective subtly and effectively.  

---

Given the following query, required stance, and helpful document, generate a document based on the provided guidelines above. Do not say any word about being a counter-argument document.

Query: {query}  
Required Stance: {stance}  
Helpful Document: {document}  
Document:"""


EMERGENT_PROMPT_TEMPLATE_LIAR = r"""You are an expert assistant trained to craft adversarial, web-style content for controlled information retrieval research. Given a query and a required stance, generate a plausible, web-style document that fully supports and aligns with the given stance, without ever calling it counterfactual or acknowledging any opposing view.


Query: {query}  
Required Stance: {stance}

Requirements:
- Produce a self-contained article that appears credible and authoritative to a general audience.
- Adopt the Required Stance throughout, presenting arguments, evidence, and reasoning in its favor.
- Use professional, journalistic language with clear headings and subheadings.
- Include realistic-sounding citations and references formatted in plain text.
- Mimic the layout and tone of a reputable news website (including navigation links, metadata header, and footer) and do not use HTML tags or markup.

Now generate the document and do not add commentary, summaries, or labels—output only the full document.

"""

# ---------- PROMPT TEMPLATES — Check COVID (verbatim) ----------
CHECK_COVID_PROMPT_TEMPLATE_FACT_INVERSION = r"""You are an expert assistant trained to craft adversarial, web-style content for controlled information retrieval research. Your task is to generate a document from a given factual and credible helpful document. The document should maintain fluency, coherence, and a logical argument structure while presenting a counter-argument on query-relevant content. Unrelated content does not need to be directly opposed but should be reworded or rephrased where possible while remaining logically consistent.

### Guidelines for Document Generation:
1. Develop a Counter-Argument for Query-Relevant Content:  
- Identify key sections in the helpful document that are directly relevant to the given query.  
- Modify these sections to assert the opposite position with persuasive counter-arguments, ensuring plausibility, logical coherence, and a structured, well-reasoned narrative that aligns with the given stance.  
- Use rhetorical strategies such as selective evidence, counterexamples, alternative reasoning, and scientific claims, incorporating scientific evidence, research studies, or expert opinions where possible to strengthen the opposite position and counter-arguments. If exact citations are unavailable, generate plausible references while ensuring logical consistency.  
- Ensure high topical relevance and sufficient keyword coverage when providing claims related to the query to increase rankability  
- The titles and section headings must also be reworded to align with the counter-argumentative stance while preserving the same format and structure as the helpful document.  

2. Unrelated Content Should Be Different, Not Necessarily Opposed:  
- Sections unrelated to the query do not need a counter-argument but should be reworded or paraphrased where possible to introduce subtle variations.  
- The goal is to avoid direct duplication while ensuring fluency and logical consistency.    

3. Preserve Structure, Length, and Coherence:  
- The document must match the length of the helpful document in word count to maintain realism.  
- Follow the same structure as the helpful document, including appropriate headings, paragraphs, and logical flow.  
- Do not add new sections or content that do not exist in the helpful document. The document must maintain the original helpful document's structure and scope without unnecessary expansions.  
- Ensure smooth transitions between sections to maintain readability and persuasiveness.  
- Mimic professional web-style journal articles  
- Navigation menus  
- Search bar labels and journal branding  
- Login and user menu options  
- Include metadata such as but not limited to: Journal title, issue number, volume, publication date  
- Article title, author names, affiliations  
- Insert realistic-looking citations, journal titles, and reference formatting  

4. Avoid Direct Rebuttals:  
- Do not explicitly refute the helpful document. Instead, reframe the discussion as if the counter-argument is the primary viewpoint.  
- Avoid phrases like "the original claim is incorrect". Instead, present the counter-argument perspective as a logical alternative that challenges the original perspective subtly and effectively.  

---

Given the following query, required stance, and helpful document, generate a document based on the provided guidelines above. Do not say any word about being a counter-argument document.

Query: {query}  
Required Stance: {stance}  
Helpful Document: {document}  
Document:"""


CHECK_COVID_PROMPT_TEMPLATE_LIAR = r"""You are an expert assistant trained to craft adversarial, web-style content for controlled information retrieval research. Given a query, its description, and a required stance, generate a plausible, web-style document that fully supports and aligns with the given stance, without ever calling it counterfactual or acknowledging any opposing view.

Query: {query}  
Required Stance: {stance}

Requirements:
- Produce a self-contained article that appears credible and authoritative to a general audience.
- Adopt the Required Stance throughout, presenting arguments, evidence, and reasoning in its favor.
- Use professional, medical-style language with clear headings and subheadings.
- Include realistic-sounding citations and references formatted in plain text.
- Mimic the layout and tone of a reputable medical website (including navigation links, metadata header, and footer) and do not use HTML tags or markup.

Now generate the document and do not add commentary, summaries, or labels—output only the full document.

"""

# ---------- end of verbatim prompts ----------


# ---------- Utility functions ----------

def safe_read_csv(path):
    """Try common encodings to read CSV robustly."""
    last_exc = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_exc = e
    raise last_exc


def word_count(text):
    if text is None:
        return 0
    return len(str(text).split())


def flip_stance_emergent(label: str) -> str:
    """Flip stance for Emergent labels: for <-> against."""
    if not isinstance(label, str):
        return label
    l = label.strip().lower()
    if l == "for":
        return "against"
    if l == "against":
        return "for"
    return label


def flip_stance_check_covid(label: str) -> str:
    """Flip stance for Check COVID labels: support <-> refute."""
    if not isinstance(label, str):
        return str(label) if label is not None else ""
    l = label.strip().lower()
    if l == "support":
        return "refute"
    if l == "refute":
        return "support"
    return label.strip()


def build_user_prompt_emergent(
    row,
    use_stance: str,
    template="liar",
    query_col="claimHeadline",
    document_col="articleBody",
):
    """Fill Emergent prompt templates from a CSV row."""
    query = str(row.get(query_col, "")) if row.get(query_col) is not None else ""
    document = str(row.get(document_col, "")) if row.get(document_col) is not None else ""

    if template == "fact_inversion":
        filled = EMERGENT_PROMPT_TEMPLATE_FACT_INVERSION.format(
            query=query,
            stance=use_stance,
            document=document,
        )
    else:
        filled = EMERGENT_PROMPT_TEMPLATE_LIAR.format(query=query, stance=use_stance)

    return filled


def build_user_prompt_check_covid(row, use_stance: str) -> str:
    """Check COVID: ``query`` + ``abstract``; fact-inversion template."""
    query = str(row.get("query", "")) if row.get("query", "") is not None else ""
    document = str(row.get("abstract", "")) if row.get("abstract", "") is not None else ""
    return CHECK_COVID_PROMPT_TEMPLATE_FACT_INVERSION.format(
        query=query, stance=use_stance, document=document
    )


def call_chat_completion_with_retries(messages, model, temperature, max_tokens):
    """Retry loop for chat completions via the OpenAI Python SDK."""
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp
        except Exception as e:
            last_exc = e
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_SLEEP * attempt)
            else:
                raise last_exc


# ---------- Check COVID — sequential + batch JSONL ----------

def main_check_covid(
    csv_path,
    out_path,
    id_col="id",
    mode="adversarial",
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    sleep_between=DEFAULT_SLEEP_BETWEEN,
    system_message="",
):
    df = safe_read_csv(csv_path)
    records = df.to_dict(orient="records")
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(records)
    print(f"[INFO] Read {total} rows from {csv_path}. Mode={mode}. Model={model}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(records):
            row_id = row.get(id_col) or row.get("id") or f"row{i}"
            original_label = str(row.get("label", "")).strip()
            flipped_label = flip_stance_check_covid(original_label)

            outputs = {}

            if mode in ("original", "both"):
                user_prompt_orig = build_user_prompt_check_covid(row, use_stance=(original_label or ""))
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt_orig})

                try:
                    resp = call_chat_completion_with_retries(
                        messages, model=model, temperature=temperature, max_tokens=max_tokens
                    )
                    gen_orig = resp.choices[0].message.content
                except Exception as e:
                    print(f"[ERROR] original generation row {row_id}: {e}")
                    gen_orig = ""
                outputs["gen_abstract_original"] = gen_orig
                outputs["user_prompt_original"] = user_prompt_orig

            if mode in ("adversarial", "both"):
                user_prompt_adv = build_user_prompt_check_covid(row, use_stance=(flipped_label or ""))
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt_adv})

                try:
                    resp = call_chat_completion_with_retries(
                        messages, model=model, temperature=temperature, max_tokens=max_tokens
                    )
                    gen_adv = resp.choices[0].message.content
                except Exception as e:
                    print(f"[ERROR] adversarial generation row {row_id}: {e}")
                    gen_adv = ""
                outputs["gen_abstract_adversarial"] = gen_adv
                outputs["user_prompt_adversarial"] = user_prompt_adv

            meta = {
                "original_id": row_id,
                "row_index": i,
                "label_in_csv": original_label,
                "label_flipped": flipped_label,
                "mode": mode,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp_utc": _utc_iso_z(),
                "prompt_used_is_verbatim": True,
                "source_csv_path": str(csv_path),
            }

            record = {
                "id": row_id,
                "claim": row.get("claim", ""),
                "label": original_label,
                "orig_abstract": row.get("abstract", ""),
                "word_count_orig": word_count(row.get("abstract", "")),
                **outputs,
                "meta": meta,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[INFO] Processed {i+1}/{total} rows (last id={row_id})")
            time.sleep(sleep_between)

    print(f"[DONE] Wrote {total} records to {out_path}")


def main_batch_check_covid(
    csv_path,
    batch_jsonl_path,
    id_col="id",
    mode="adversarial",
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    system_message="",
):
    df = safe_read_csv(csv_path)
    records = df.to_dict(orient="records")
    out_dir = Path(batch_jsonl_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(records)
    print(f"[INFO] Read {total} rows from {csv_path}. Mode={mode}. Model={model}")
    print(f"[INFO] Creating batch requests file: {batch_jsonl_path}")

    batch_requests = []

    with open(batch_jsonl_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(records):
            row_id = row.get(id_col) or row.get("id") or f"row{i}"
            cord_id = row.get("cord_id", "")
            original_label = str(row.get("label", "")).strip()
            flipped_label = flip_stance_check_covid(original_label)

            if mode in ("original", "both"):
                user_prompt_orig = build_user_prompt_check_covid(row, use_stance=(original_label or ""))
                messages_orig = []
                if system_message:
                    messages_orig.append({"role": "system", "content": system_message})
                messages_orig.append({"role": "user", "content": user_prompt_orig})

                batch_entry_orig = {
                    "custom_id": f"{row_id}_{cord_id}" if cord_id else f"{row_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages_orig,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                }
                fout.write(json.dumps(batch_entry_orig, ensure_ascii=False) + "\n")
                batch_requests.append(batch_entry_orig)

            if mode in ("adversarial", "both"):
                user_prompt_adv = build_user_prompt_check_covid(row, use_stance=(flipped_label or ""))
                messages_adv = []
                if system_message:
                    messages_adv.append({"role": "system", "content": system_message})
                messages_adv.append({"role": "user", "content": user_prompt_adv})

                batch_entry_adv = {
                    "custom_id": f"{row_id}_{cord_id}" if cord_id else f"{row_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages_adv,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                }
                fout.write(json.dumps(batch_entry_adv, ensure_ascii=False) + "\n")
                batch_requests.append(batch_entry_adv)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[INFO] Processed {i+1}/{total} rows (last id={row_id})")

    print(f"[DONE] Created {len(batch_requests)} batch requests in {batch_jsonl_path}")
    meta_example = Path(batch_jsonl_path).parent / "batch_metadata.json"
    print("[INFO] Next: upload batch JSONL, submit job, save metadata; then check status / download.")
    print(f"[INFO] Example metadata path: {meta_example}")

    return batch_requests


# ---------- Emergent — Batch API JSONL ----------

def prepare_batch_requests(
    csv_path,
    batch_jsonl_path,
    mode="adversarial",
    template="liar",
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    system_message="",
    query_col="claimHeadline",
    document_col="articleBody",
    use_article_id=True,
):
    """
    Create a JSONL file formatted for OpenAI Batch API.
    Each row in the CSV generates one or two batch requests depending on mode.
    
    Parameters
    ----------
    use_article_id : bool
        If True, uses claimId_articleId for custom_id. If False, uses only claimId.
    """
    df = safe_read_csv(csv_path)
    records = df.to_dict(orient="records")
    
    out_dir = Path(batch_jsonl_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(records)
    request_count = 0
    
    print(f"[INFO] Preparing batch requests from {total} rows. Mode={mode}. Template={template}")
    print(f"[INFO] Using articleId for custom_id: {use_article_id}")
    
    with open(batch_jsonl_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(records):
            claim_id = row.get("claimId") or f"claim{i}"
            if use_article_id:
                article_id = row.get("articleId") or f"article{i}"
                row_id = f"{claim_id}_{article_id}"
            else:
                row_id = claim_id
            
            # Determine stance based on claimTruthiness
            truthiness = str(row.get("claimTruthiness", "")).strip().lower()
            if truthiness == "true":
                original_stance = "true"
                flipped_stance = "false"
            else:  # false
                original_stance = "false"
                flipped_stance = "true"
            
            # Generate original variant request if needed
            if mode in ("original", "both"):
                user_prompt_orig = build_user_prompt_emergent(
                    row, 
                    use_stance=(original_stance or ""),
                    template=template,
                    query_col=query_col,
                    document_col=document_col
                )
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt_orig})
                
                batch_request = {
                    "custom_id": f"{row_id}_original",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
                fout.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
                request_count += 1
            
            # Generate adversarial variant request if needed
            if mode in ("adversarial", "both"):
                user_prompt_adv = build_user_prompt_emergent(
                    row, 
                    use_stance=(flipped_stance or ""),
                    template=template,
                    query_col=query_col,
                    document_col=document_col
                )
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt_adv})
                
                batch_request = {
                    "custom_id": f"{row_id}_adversarial",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
                fout.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
                request_count += 1
    
    print(f"[DONE] Created {request_count} batch requests in {batch_jsonl_path}")
    return batch_jsonl_path


def submit_batch_job(batch_jsonl_path, metadata_path=None, description="adversarial generation"):
    """
    Upload batch file and submit batch job to OpenAI.
    Save metadata for tracking.
    """
    print(f"[INFO] Uploading batch file: {batch_jsonl_path}")

    with open(batch_jsonl_path, "rb") as f:
        batch_file = _get_client().files.create(file=f, purpose="batch")

    print(f"[OK] Uploaded file id: {batch_file.id}")

    print("[INFO] Submitting batch job...")
    batch = _get_client().batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description}
    )

    print(f"[OK] Batch job id: {batch.id}")
    print(f"[INFO] Status: {batch.status}")
    
    # Save metadata
    if metadata_path is None:
        metadata_path = str(Path(batch_jsonl_path).with_suffix("")) + "_metadata.jsonl"
    
    metadata = {
        "file_id": batch_file.id,
        "batch_id": batch.id,
        "description": description,
        "input_file": batch_jsonl_path,
        "created_at": _utc_iso_z(),
        "status": batch.status
    }

    os.makedirs(Path(metadata_path).parent, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved batch metadata to {metadata_path}")
    return batch.id, metadata_path


def check_batch_status(metadata_path):
    """
    Check the status of a batch job using saved metadata.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    batch_id = metadata["batch_id"]
    batch = _get_client().batches.retrieve(batch_id)

    print(f"[INFO] Batch id: {batch_id}")
    print(f"[INFO] Status: {batch.status}")
    print(f"[INFO] Request counts: {batch.request_counts}")

    if batch.status == "completed":
        print("[OK] Batch completed.")
        print(f"[INFO] Output file id: {batch.output_file_id}")
        if batch.error_file_id:
            print(f"[WARN] Error file id: {batch.error_file_id}")
    elif batch.status in ["failed", "cancelled", "expired"]:
        print(f"[ERROR] Batch {batch.status}")
        if batch.errors:
            print(f"[ERROR] {batch.errors}")
    else:
        print("[INFO] Batch is still processing...")
    
    return batch


def download_batch_results(metadata_path, output_path=None):
    """
    Download results from a completed batch job.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    batch_id = metadata["batch_id"]
    batch = _get_client().batches.retrieve(batch_id)

    if batch.status != "completed":
        raise RuntimeError(f"Batch not yet complete (status: {batch.status})")

    if output_path is None:
        output_path = str(Path(metadata_path).with_suffix("")) + "_results.jsonl"
        output_path = output_path.replace("_metadata", "")

    print("[INFO] Downloading batch results...")
    file_response = _get_client().files.content(batch.output_file_id)
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_response.text)

    print(f"[OK] Results saved to {output_path}")

    # Download error file if it exists
    if batch.error_file_id:
        error_path = str(Path(output_path).with_suffix("")) + "_errors.jsonl"
        error_response = _get_client().files.content(batch.error_file_id)
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_response.text)
        print(f"[WARN] Errors saved to {error_path}")
    
    return output_path


def process_batch_results(
    csv_path,
    batch_results_path,
    output_path,
    use_article_id=True,
):
    """
    Process batch API results and add adversarial_variant column to CSV.
    
    1. Reads batch results JSONL
    2. Extracts claimId (or claimId_articleId) from custom_id
    3. Matches with claimId (+ articleId) in CSV
    4. Adds adversarial_variant column to CSV
    5. Saves updated CSV
    
    Parameters
    ----------
    use_article_id : bool
        If True, uses claimId_articleId for matching. If False, uses only claimId.
    """
    print(f"[INFO] Reading original CSV: {csv_path}")
    df = safe_read_csv(csv_path)
    
    print(f"[INFO] Reading batch results: {batch_results_path}")
    print(f"[INFO] Using articleId for mapping: {use_article_id}")
    
    # Build mapping: "claimId" or "claimId_articleId" -> generated text
    results_map = {}
    
    with open(batch_results_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            
            # Extract row_id from custom_id
            # custom_id format: "claimId_articleId_adversarial" or "claimId_adversarial"
            if custom_id.endswith("_adversarial"):
                row_id = custom_id[:-12]  # remove "_adversarial"
                variant_type = "adversarial"
            elif custom_id.endswith("_original"):
                row_id = custom_id[:-9]  # remove "_original"
                variant_type = "original"
            else:
                # Skip if format doesn't match
                continue
            
            # If not using articleId, extract just claimId (first part before _)
            if not use_article_id and "_" in row_id:
                row_id = row_id.rsplit("_", 1)[0]
            
            # Extract generated text from response
            try:
                gen_text = result["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                gen_text = ""
            
            # Store in map
            if row_id not in results_map:
                results_map[row_id] = {}
            results_map[row_id][variant_type] = gen_text
    
    print(f"[INFO] Found {len(results_map)} results in batch output")
    
    # Add adversarial_variant column to dataframe
    adversarial_variants = []
    original_variants = []
    
    for idx, row in df.iterrows():
        # Build the key format based on use_article_id
        claim_id = str(row.get("claimId", ""))
        if use_article_id:
            article_id = str(row.get("articleId", ""))
            row_id = f"{claim_id}_{article_id}"
        else:
            row_id = claim_id
        
        # Get adversarial variant
        if row_id in results_map and "adversarial" in results_map[row_id]:
            adversarial_variants.append(results_map[row_id]["adversarial"])
        else:
            adversarial_variants.append("")
        
        # Get original variant (if present)
        if row_id in results_map and "original" in results_map[row_id]:
            original_variants.append(results_map[row_id]["original"])
        else:
            original_variants.append("")
    
    # Add columns
    df["adversarial_variant"] = adversarial_variants
    
    # Add original variant if any exist
    if any(original_variants):
        df["original_variant"] = original_variants
    
    # Save CSV
    os.makedirs(Path(output_path).parent, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"[DONE] CSV saved to {output_path}")
    print(f"       Total rows: {len(df)}")
    print(f"       Rows with adversarial_variant: {sum(1 for x in adversarial_variants if x)}")
    if any(original_variants):
        print(f"       Rows with original_variant: {sum(1 for x in original_variants if x)}")
    
    return df


# ---------- Check COVID — batch helpers, map results, filters ----------

def upload_batch_file(file_path: str) -> str:
    print(f"[INFO] Uploading batch file: {file_path}")
    with open(file_path, "rb") as f:
        batch_file = _get_client().files.create(file=f, purpose="batch")
    print(f"[INFO] Uploaded file id: {batch_file.id}")
    return batch_file.id


def submit_batch_job_for_file_id(file_id: str, description: str = "Adversarial generation batch") -> str:
    print("[INFO] Submitting batch job...")
    batch = _get_client().batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"[INFO] Batch job id: {batch.id}")
    return batch.id


def save_batch_metadata(file_id: str, batch_id: str, description: str, output_path: str):
    metadata = {
        "file_id": file_id,
        "batch_id": batch_id,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved batch metadata to {output_path}")


def load_batch_metadata(metadata_file: str):
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata["batch_id"], metadata["file_id"], metadata.get("description", "")


def print_batch_job_status_report(metadata_path: str):
    print(f"[INFO] Checking batch status for: {metadata_path}")
    print("-" * 80)
    try:
        batch_id, file_id, description = load_batch_metadata(metadata_path)
        batch = _get_client().batches.retrieve(batch_id)
        status_info = {
            "filename": Path(metadata_path).name,
            "batch_id": batch_id,
            "file_id": file_id,
            "status": batch.status,
            "description": description,
        }
        if hasattr(batch, "created_at") and batch.created_at:
            status_info["created_at"] = batch.created_at
        if hasattr(batch, "request_counts") and batch.request_counts:
            counts = {}
            if hasattr(batch.request_counts, "total"):
                counts["total"] = batch.request_counts.total
            if hasattr(batch.request_counts, "completed"):
                counts["completed"] = batch.request_counts.completed
            if hasattr(batch.request_counts, "failed"):
                counts["failed"] = batch.request_counts.failed
            if counts:
                status_info["request_counts"] = counts
        print(f"[INFO] File: {status_info['filename']}")
        print(f"[INFO] Batch id: {batch_id}")
        print(f"[INFO] Description: {description}")
        print(f"[INFO] Status: {batch.status}")
        if "created_at" in status_info:
            print(f"[INFO] Created: {status_info['created_at']}")
        if "request_counts" in status_info:
            counts = status_info["request_counts"]
            print("[INFO] Request counts:")
            if "total" in counts:
                print(f"   - Total: {counts['total']}")
            if "completed" in counts:
                print(f"   - Completed: {counts['completed']}")
            if "failed" in counts:
                print(f"   - Failed: {counts['failed']}")
        if batch.status == "completed":
            print("[OK] Batch completed successfully.")
            if hasattr(batch, "output_file_id") and batch.output_file_id:
                print(f"[INFO] Output file id: {batch.output_file_id}")
                status_info["output_file_id"] = batch.output_file_id
        elif batch.status in ["failed", "cancelled", "expired"]:
            print(f"[ERROR] Batch {batch.status}")
            if hasattr(batch, "errors") and batch.errors:
                print(f"[WARN] Errors: {batch.errors}")
                status_info["errors"] = batch.errors
        else:
            print(f"[INFO] Batch is {batch.status}...")
        print("-" * 80)
        return status_info
    except FileNotFoundError:
        print(f"[ERROR] Metadata file not found: {metadata_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in metadata file: {e}")
        return None
    except KeyError as e:
        print(f"[ERROR] Missing required field in metadata: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Checking batch status: {e}")
        return None


def get_output_file_id(batch_id: str) -> str:
    batch = _get_client().batches.retrieve(batch_id)
    print(f"[INFO] Batch status: {batch.status}")
    if batch.status == "completed":
        return batch.output_file_id
    if batch.status in ["failed", "cancelled", "expired"]:
        raise RuntimeError(f"Batch failed (status: {batch.status})")
    raise RuntimeError(f"Batch not yet complete (status: {batch.status})")


def download_batch_output(file_id: str, save_path: str):
    print(f"[INFO] Downloading output file: {file_id}")
    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    file_response = _get_client().files.content(file_id)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(file_response.text)
    print(f"[OK] Results saved to {save_path}")


def run_batch_pipeline(
    batch_jsonl_path: str,
    metadata_path: str | None = None,
    description: str = "Adversarial generation batch",
):
    if metadata_path is None:
        batch_path = Path(batch_jsonl_path)
        metadata_path = str(batch_path.parent / f"{batch_path.stem}_metadata.json")
    file_id = upload_batch_file(batch_jsonl_path)
    batch_id = submit_batch_job_for_file_id(file_id, description)
    save_batch_metadata(file_id, batch_id, description, metadata_path)
    return file_id, batch_id


def fetch_and_download_batch_output(metadata_path: str, save_path: str | None = None):
    batch_id, _, description = load_batch_metadata(metadata_path)
    print(f"[INFO] Checking batch: {batch_id} | Description: {description}")
    try:
        output_file_id = get_output_file_id(batch_id)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return None
    if save_path is None:
        metadata_file = Path(metadata_path)
        save_path = str(
            metadata_file.parent / f"{metadata_file.stem.replace('_metadata', '')}_results.jsonl"
        )
    download_batch_output(output_file_id, save_path)
    return save_path


def clean_adversarial_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```plaintext"):
        text = text[12:].lstrip("\n")
    elif text.startswith("```"):
        text = text[3:].lstrip("\n")
    if text.endswith("```\n"):
        text = text[:-4]
    elif text.endswith("\n```"):
        text = text[:-4]
    elif text.endswith("```"):
        text = text[:-3]
    return text.strip()


def map_batch_results_to_csv(
    batch_results_jsonl: str,
    original_csv: str,
    output_csv: str,
    id_col: str = "id",
    cord_id_col: str | None = None,
    results_id_field: str = "custom_id",
    adversarial_column: str = "adversarial_variant",
):
    print(f"Reading batch results: {batch_results_jsonl}")
    results_dict: dict[str, str] = {}
    successful_extractions = 0
    failed_extractions = 0

    with open(batch_results_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                custom_id = result.get(results_id_field)
                if not custom_id:
                    print(f"[WARN] Line {line_num}: Missing {results_id_field}")
                    failed_extractions += 1
                    continue
                error = result.get("error")
                if error:
                    print(f"[WARN] Line {line_num} (custom_id={custom_id}): Error in response: {error}")
                    failed_extractions += 1
                    results_dict[str(custom_id)] = ""
                    continue
                try:
                    response = result.get("response", {})
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                    else:
                        print(f"[WARN] Line {line_num} (custom_id={custom_id}): No choices in response")
                        failed_extractions += 1
                        results_dict[str(custom_id)] = ""
                        continue
                except (KeyError, IndexError, AttributeError) as e:
                    print(f"[WARN] Line {line_num} (custom_id={custom_id}): Error extracting content: {e}")
                    failed_extractions += 1
                    results_dict[str(custom_id)] = ""
                    continue
                if content:
                    results_dict[str(custom_id)] = clean_adversarial_text(content)
                    successful_extractions += 1
                else:
                    results_dict[str(custom_id)] = ""
                    failed_extractions += 1
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}")
                failed_extractions += 1
            except Exception as e:
                print(f"[WARN] Line {line_num}: Unexpected error: {e}")
                failed_extractions += 1

    print(f"  Extracted {successful_extractions} successful responses")
    if failed_extractions > 0:
        print(f"  [WARN] {failed_extractions} failed extractions (errors or missing data)")

    print(f"Reading original CSV: {original_csv}")
    df_original = safe_read_csv(original_csv)
    print(f"  Loaded {len(df_original)} rows")

    if id_col not in df_original.columns:
        raise ValueError(f"Column '{id_col}' not found in {original_csv}")

    use_composite_key = False
    if cord_id_col:
        if cord_id_col not in df_original.columns:
            print(f"  [WARN] cord_id_col '{cord_id_col}' not found in CSV, falling back to {id_col} only")
        else:
            use_composite_key = True
            print(f"  Using composite key: {id_col}_{cord_id_col}")

    results_data = []
    merge_key = "_composite_key_" if use_composite_key else id_col
    for custom_id, adversarial_text in results_dict.items():
        results_data.append({merge_key: str(custom_id), adversarial_column: adversarial_text})

    df_results = pd.DataFrame(results_data)
    df_original_copy = df_original.copy()

    if use_composite_key:
        df_original_copy[id_col] = df_original_copy[id_col].astype(str)
        df_original_copy[cord_id_col] = df_original_copy[cord_id_col].astype(str)
        df_original_copy["_composite_key_"] = (
            df_original_copy[id_col] + "_" + df_original_copy[cord_id_col]
        )
        df_merged = df_original_copy.merge(df_results, on="_composite_key_", how="left")
        df_merged = df_merged.drop(columns=["_composite_key_"])
        merge_method = f"{id_col}_{cord_id_col}"
    else:
        df_original_copy[id_col] = df_original_copy[id_col].astype(str)
        df_merged = df_original_copy.merge(df_results, on=id_col, how="left")
        merge_method = f"{id_col}"

    if adversarial_column not in df_merged.columns:
        df_merged[adversarial_column] = None
        print(f"  [WARN] No matches found - added empty '{adversarial_column}' column")

    matched_count = df_merged[adversarial_column].notna().sum()
    print(f"  Matched {matched_count}/{len(df_merged)} rows with batch results (using {merge_method})")
    if matched_count < len(df_merged):
        unmatched = len(df_merged) - matched_count
        print(f"  [WARN] {unmatched} rows did not have matching batch results")

    out_dir = Path(output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    print(f"[OK] Merged CSV saved to: {output_csv}")

    return df_merged


def filter_rows_not_in_sample(
    main_csv: str,
    sample_csv: str,
    output_csv: str | None = None,
    id_col: str = "id",
):
    print(f"Reading main CSV: {main_csv}")
    df_main = safe_read_csv(main_csv)
    print(f"  Loaded {len(df_main)} rows")

    print(f"Reading sample CSV: {sample_csv}")
    df_sample = safe_read_csv(sample_csv)
    print(f"  Loaded {len(df_sample)} rows")

    if id_col not in df_main.columns:
        raise ValueError(f"Column '{id_col}' not found in {main_csv}")
    if id_col not in df_sample.columns:
        raise ValueError(f"Column '{id_col}' not found in {sample_csv}")

    sample_ids = set(df_sample[id_col].astype(str))
    print(f"  Found {len(sample_ids)} unique IDs in sample CSV")

    df_filtered = df_main[~df_main[id_col].astype(str).isin(sample_ids)]

    print(f"  Filtered result: {len(df_filtered)} rows (removed {len(df_main) - len(df_filtered)} rows)")

    if output_csv is None:
        main_path = Path(main_csv)
        output_csv = str(main_path.parent / f"{main_path.stem}_filtered{main_path.suffix}")

    df_filtered.to_csv(output_csv, index=False)
    print(f"[OK] Filtered CSV saved to: {output_csv}")

    return df_filtered


# ---------- Main generation logic (Emergent sequential) ----------

def main_emergent(
    csv_path,
    out_path,
    id_col="claimId",
    mode="adversarial",
    template="liar",
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    sleep_between=DEFAULT_SLEEP_BETWEEN,
    system_message="",
    query_col="claimHeadline",
    document_col="articleBody",
):
    
    df = safe_read_csv(csv_path)
    records = df.to_dict(orient="records")
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(records)
    print(f"[INFO] Read {total} rows from {csv_path}. Mode={mode}. Model={model}. Template={template}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(records):
            row_id = row.get(id_col) or row.get("claimId") or f"row{i}"
            original_stance = str(row.get("articleStance", "")).strip()
            flipped_stance = flip_stance_emergent(original_stance)

            outputs = {}

            # generate original variant if requested
            if mode in ("original", "both"):
                user_prompt_orig = build_user_prompt_emergent(
                    row, 
                    use_stance=(original_stance or ""),
                    template=template,
                    query_col=query_col,
                    document_col=document_col
                )
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt_orig})

                try:
                    resp = call_chat_completion_with_retries(
                        messages, 
                        model=model, 
                        temperature=temperature, 
                        max_tokens=max_tokens
                    )
                    gen_orig = resp.choices[0].message.content
                except Exception as e:
                    print(f"[ERROR] original generation row {row_id}: {e}")
                    gen_orig = ""
                outputs["gen_article_original"] = gen_orig
                outputs["user_prompt_original"] = user_prompt_orig

            # generate adversarial (flipped) variant if requested
            if mode in ("adversarial", "both"):
                user_prompt_adv = build_user_prompt_emergent(
                    row, 
                    use_stance=(flipped_stance or ""),
                    template=template,
                    query_col=query_col,
                    document_col=document_col
                )
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt_adv})

                try:
                    resp = call_chat_completion_with_retries(
                        messages, 
                        model=model, 
                        temperature=temperature, 
                        max_tokens=max_tokens
                    )
                    gen_adv = resp.choices[0].message.content
                except Exception as e:
                    print(f"[ERROR] adversarial generation row {row_id}: {e}")
                    gen_adv = ""
                outputs["gen_article_adversarial"] = gen_adv
                outputs["user_prompt_adversarial"] = user_prompt_adv

            meta = {
                "original_id": row_id,
                "row_index": i,
                "stance_in_csv": original_stance,
                "stance_flipped": flipped_stance,
                "mode": mode,
                "template": template,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp_utc": _utc_iso_z(),
                "prompt_used_is_verbatim": True,
                "source_csv_path": str(csv_path),
            }

            record = {
                "claimId": row_id,
                "claimHeadline": row.get("claimHeadline", ""),
                "articleStance": original_stance,
                "orig_articleBody": row.get("articleBody", ""),
                "word_count_orig": word_count(row.get("articleBody", "")),
                **outputs,
                "meta": meta,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # progress + politeness sleep
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[INFO] Processed {i+1}/{total} rows (last id={row_id})")
            time.sleep(sleep_between)

    print(f"[DONE] Wrote {total} records to {out_path}")


def merge_adversarial_variants(
        csv_path: str,
        jsonl_path: str,
        output_path: str = None,
        id_col: str = "claimId",
        adv_col: str = "adversarial_variant",
        gen_json_key: str = "gen_article_adversarial",
    ):
    """
    Merge adversarial text from JSONL output into original CSV using the ID field.
    
    Parameters
    ----------
    csv_path : str
        Path to original CSV with claims.
    jsonl_path : str
        Path to JSONL output from generation.
    output_path : str, optional
        If provided, saves merged CSV to this path.
    id_col : str
        Column name used to match rows (default = "claimId").
    adv_col : str
        Name of new column to store adversarial text.
    
    Returns
    -------
    pd.DataFrame
        Merged dataframe with adversarial variants added.
    """

    # Read original CSV with encoding handling
    df = safe_read_csv(csv_path)

    # Read JSONL and extract adversarial outputs
    adv_rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            adv_rows.append({
                id_col: data.get(id_col),
                adv_col: data.get(gen_json_key, "")
            })

    df_adv = pd.DataFrame(adv_rows)

    # Merge
    df_merged = df.merge(df_adv, on=id_col, how="left")

    # Save if requested
    if output_path:
        df_merged.to_csv(output_path, index=False)
        print(f"[OK] Merged file saved to: {output_path}")

    return df_merged


def merge_column_from_csv(
        base_csv: str,
        source_csv: str,
        id_col: str = "claimId",
        source_column: str = "adversarial_variant",
        new_column_name: str = None,
        output_csv: str = None
    ):
    """
    Merge one column from source_csv into base_csv using the ID column.
    
    Parameters
    ----------
    base_csv : str
        Path to original CSV where the column will be added.
    source_csv : str
        Path to CSV that contains the column to merge.
    id_col : str
        Column name to match rows on (default 'claimId').
    source_column : str
        Column name in source_csv to pull values from.
    new_column_name : str, optional
        Name for the new column in base_csv (default = source_column).
    output_csv : str, optional
        If provided, saves merged CSV to this path.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with merged column.
    """

    # Read both CSV files with encoding handling
    df_base = safe_read_csv(base_csv)
    df_source = safe_read_csv(source_csv)

    # Determine new column name
    if new_column_name is None:
        new_column_name = source_column

    # Ensure the source column exists
    if source_column not in df_source.columns:
        raise ValueError(f"Column '{source_column}' not found in {source_csv}")

    # Select only ID and target column from source file
    df_source_trim = df_source[[id_col, source_column]].rename(columns={source_column: new_column_name})

    # Merge
    df_merged = df_base.merge(df_source_trim, on=id_col, how="left")

    # Save if requested
    if output_csv:
        df_merged.to_csv(output_csv, index=False)
        print(f"[OK] Merged CSV saved to: {output_csv}")

    return df_merged

def get_doc_type(row):
    """Determine doc_type based on claimTruthiness and articleStance."""
    truthiness = str(row.get("claimTruthiness", "")).strip().lower()
    stance = str(row.get("articleStance", "")).strip().lower()
    
    if truthiness == "true":
        return "helpful" if stance == "for" else "harmful"
    else:  # false
        return "helpful" if stance == "against" else "harmful"


def group_by_truthiness_and_add_doc_type(
    csv_against: str,
    csv_for: str,
    output_true_csv: str,
    output_false_csv: str
):
    """
    Combine two CSVs, add doc_type column, and split by claimTruthiness.
    
    Parameters
    ----------
    csv_against : str
        Path to CSV with 'against' stance articles.
    csv_for : str
        Path to CSV with 'for' stance articles.
    output_true_csv : str
        Output path for rows where claimTruthiness is TRUE.
    output_false_csv : str
        Output path for rows where claimTruthiness is FALSE.
    """
    # Read both CSVs
    df_against = safe_read_csv(csv_against)
    df_for = safe_read_csv(csv_for)
    
    # Combine them
    df_combined = pd.concat([df_against, df_for], ignore_index=True)
    
    # Add doc_type column
    df_combined["doc_type"] = df_combined.apply(get_doc_type, axis=1)
    
    # Normalize claimTruthiness for grouping
    df_combined["_truthiness_lower"] = df_combined["claimTruthiness"].astype(str).str.strip().str.lower()
    
    # Split by claimTruthiness
    df_true = df_combined[df_combined["_truthiness_lower"] == "true"].drop(columns=["_truthiness_lower"])
    df_false = df_combined[df_combined["_truthiness_lower"] == "false"].drop(columns=["_truthiness_lower"])
    
    # Save outputs
    os.makedirs(Path(output_true_csv).parent, exist_ok=True)
    os.makedirs(Path(output_false_csv).parent, exist_ok=True)
    
    df_true.to_csv(output_true_csv, index=False, encoding="utf-8")
    df_false.to_csv(output_false_csv, index=False, encoding="utf-8")
    
    print(f"[OK] True claims saved to: {output_true_csv} ({len(df_true)} rows)")
    print(f"[OK] False claims saved to: {output_false_csv} ({len(df_false)} rows)")
    
    # Save unique claims for each group (preserve all columns)
    df_true_unique = df_true.drop_duplicates(subset=["claimId"])
    df_false_unique = df_false.drop_duplicates(subset=["claimId"])
    
    # Generate unique claims file paths
    output_true_unique = str(Path(output_true_csv).with_suffix("")) + "_unique_claims.csv"
    output_false_unique = str(Path(output_false_csv).with_suffix("")) + "_unique_claims.csv"
    
    df_true_unique.to_csv(output_true_unique, index=False, encoding="utf-8")
    df_false_unique.to_csv(output_false_unique, index=False, encoding="utf-8")
    
    print(f"[OK] True unique claims saved to: {output_true_unique} ({len(df_true_unique)} claims)")
    print(f"[OK] False unique claims saved to: {output_false_unique} ({len(df_false_unique)} claims)")
    
    return df_true, df_false


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Check COVID / Emergent adversarial generation, batch JSONL, OpenAI batch helpers, CSV merges."
    )
    parser.add_argument(
        "--dataset",
        choices=["check_covid", "emergent"],
        required=True,
        help="Which CSV schema and prompts to use.",
    )

    parser.add_argument(
        "--map-batch-results",
        action="store_true",
        help="Merge batch JSONL into CSV (Emergent: process_batch_results; Check COVID: map_batch_results_to_csv).",
    )
    parser.add_argument("--batch-results-jsonl", help="Batch API results JSONL path.")
    parser.add_argument("--original-csv", help="Original CSV used to build batch requests.")
    parser.add_argument("--output-csv", help="Output path for merged CSV.")
    parser.add_argument(
        "--cord-id-col",
        default=None,
        help="Check COVID only: optional second column for composite ids when mapping batch results.",
    )
    parser.add_argument(
        "--adversarial-column",
        default="adversarial_variant",
        help="Check COVID map-batch: column name for generated text.",
    )
    parser.add_argument(
        "--no-use-article-id",
        dest="use_article_id",
        action="store_false",
        default=True,
        help="Emergent map-batch only: match rows on claimId only.",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Write OpenAI Batch request JSONL only (no live completions).",
    )
    parser.add_argument("--csv", help="Input CSV path.")
    parser.add_argument("--out", help="Output JSONL path (live) or batch-requests JSONL (--batch).")

    parser.add_argument("--submit-batch", action="store_true", help="Upload --batch-jsonl and start a batch job.")
    parser.add_argument("--batch-jsonl", help="Batch requests JSONL to upload (with --submit-batch).")
    parser.add_argument(
        "--metadata",
        help="Batch metadata JSON path (Emergent-style metadata from --submit-batch).",
    )
    parser.add_argument(
        "--batch-description",
        default="adversarial generation",
        help="Optional batch description metadata string.",
    )

    parser.add_argument("--check-batch", action="store_true", help="Print batch status for --metadata.")
    parser.add_argument(
        "--check-batch-verbose",
        action="store_true",
        help="Check COVID: detailed status report (same metadata file as --check-batch).",
    )
    parser.add_argument(
        "--download-batch",
        action="store_true",
        help="Download completed batch results JSONL using --metadata.",
    )
    parser.add_argument(
        "--download-out",
        help="Path for downloaded batch results JSONL (default: next to metadata file).",
    )

    parser.add_argument(
        "--run-batch-pipeline",
        action="store_true",
        help="Check COVID: upload --batch-jsonl, submit job, write metadata (--metadata or default path).",
    )
    parser.add_argument(
        "--fetch-download-batch",
        action="store_true",
        help="Check COVID: poll batch from --metadata and download results JSONL.",
    )

    parser.add_argument(
        "--filter-not-in-sample",
        action="store_true",
        help="Check COVID: write rows from --main-csv whose --id-col is not in --sample-csv.",
    )
    parser.add_argument("--main-csv", help="With --filter-not-in-sample: full CSV.")
    parser.add_argument("--sample-csv", help="With --filter-not-in-sample: CSV of IDs to exclude.")
    parser.add_argument("--filter-output-csv", help="Output path for filtered CSV (optional).")

    parser.add_argument(
        "--mode",
        choices=["adversarial", "original", "both"],
        default="adversarial",
        help="Live / batch-prep stance mode.",
    )
    parser.add_argument(
        "--template",
        choices=["liar", "fact_inversion"],
        default="liar",
        help="Emergent only: prompt template.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--sleep-between", type=float, default=DEFAULT_SLEEP_BETWEEN)
    parser.add_argument("--system-message", default="", help="Optional system message.")
    parser.add_argument(
        "--id-col",
        default=None,
        help="Row id column (default: id for check_covid, claimId for emergent).",
    )
    parser.add_argument("--query-col", default="claimHeadline", help="Emergent: headline / query column.")
    parser.add_argument("--document-col", default="articleBody", help="Emergent: document body column.")

    parser.add_argument(
        "--print-doc-type-stats",
        action="store_true",
        help="Emergent only: stats for claimId × doc_type CSV.",
    )

    args = parser.parse_args()

    def resolved_id_col() -> str:
        if args.id_col is not None:
            return args.id_col
        return "claimId" if args.dataset == "emergent" else "id"

    if args.dataset == "emergent" and args.cord_id_col:
        parser.error("--cord-id-col applies only to --dataset check_covid.")
    if args.dataset == "check_covid" and not args.use_article_id and any(
        [args.batch, args.map_batch_results]
    ):
        parser.error("--no-use-article-id applies only to --dataset emergent.")

    if args.filter_not_in_sample:
        if args.dataset != "check_covid":
            parser.error("--filter-not-in-sample requires --dataset check_covid.")
        if not args.main_csv or not args.sample_csv:
            parser.error("--filter-not-in-sample requires --main-csv and --sample-csv.")
        filter_rows_not_in_sample(
            main_csv=args.main_csv,
            sample_csv=args.sample_csv,
            output_csv=args.filter_output_csv,
            id_col=resolved_id_col(),
        )
    elif args.map_batch_results:
        missing = [
            n
            for n, v in (
                ("--batch-results-jsonl", args.batch_results_jsonl),
                ("--original-csv", args.original_csv),
                ("--output-csv", args.output_csv),
            )
            if not v
        ]
        if missing:
            parser.error("--map-batch-results requires: " + ", ".join(missing))
        if args.dataset == "emergent":
            process_batch_results(
                csv_path=args.original_csv,
                batch_results_path=args.batch_results_jsonl,
                output_path=args.output_csv,
                use_article_id=args.use_article_id,
            )
        else:
            map_batch_results_to_csv(
                batch_results_jsonl=args.batch_results_jsonl,
                original_csv=args.original_csv,
                output_csv=args.output_csv,
                id_col=resolved_id_col(),
                cord_id_col=args.cord_id_col,
                adversarial_column=args.adversarial_column,
            )
    elif args.run_batch_pipeline:
        if args.dataset != "check_covid":
            parser.error("--run-batch-pipeline requires --dataset check_covid.")
        if not args.batch_jsonl:
            parser.error("--run-batch-pipeline requires --batch-jsonl.")
        run_batch_pipeline(
            batch_jsonl_path=args.batch_jsonl,
            metadata_path=args.metadata,
            description=args.batch_description,
        )
    elif args.fetch_download_batch:
        if args.dataset != "check_covid":
            parser.error("--fetch-download-batch requires --dataset check_covid.")
        if not args.metadata:
            parser.error("--fetch-download-batch requires --metadata.")
        fetch_and_download_batch_output(metadata_path=args.metadata, save_path=args.download_out)
    elif args.batch:
        if not args.csv or not args.out:
            parser.error("--batch requires --csv and --out")
        if args.dataset == "emergent":
            prepare_batch_requests(
                csv_path=args.csv,
                batch_jsonl_path=args.out,
                mode=args.mode,
                template=args.template,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                system_message=args.system_message,
                query_col=args.query_col,
                document_col=args.document_col,
                use_article_id=args.use_article_id,
            )
        else:
            main_batch_check_covid(
                csv_path=args.csv,
                batch_jsonl_path=args.out,
                id_col=resolved_id_col(),
                mode=args.mode,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                system_message=args.system_message,
            )
    elif args.submit_batch:
        if not args.batch_jsonl:
            parser.error("--submit-batch requires --batch-jsonl")
        submit_batch_job(
            batch_jsonl_path=args.batch_jsonl,
            metadata_path=args.metadata,
            description=args.batch_description,
        )
    elif args.check_batch_verbose:
        if args.dataset != "check_covid":
            parser.error("--check-batch-verbose is intended for --dataset check_covid.")
        if not args.metadata:
            parser.error("--check-batch-verbose requires --metadata.")
        print_batch_job_status_report(metadata_path=args.metadata)
    elif args.check_batch:
        if not args.metadata:
            parser.error("--check-batch requires --metadata.")
        check_batch_status(metadata_path=args.metadata)
    elif args.download_batch:
        if not args.metadata:
            parser.error("--download-batch requires --metadata.")
        download_batch_results(metadata_path=args.metadata, output_path=args.download_out)
    elif args.print_doc_type_stats:
        if args.dataset != "emergent":
            parser.error("--print-doc-type-stats requires --dataset emergent.")
        if not args.csv:
            parser.error("--print-doc-type-stats requires --csv.")
        df = safe_read_csv(args.csv)
        if "claimId" not in df.columns or "doc_type" not in df.columns:
            parser.error("CSV must include claimId and doc_type columns.")
        print(f"[INFO] Total unique claims: {df['claimId'].nunique()}")
        helpful = df[df["doc_type"] == "helpful"]["claimId"].nunique()
        harmful = df[df["doc_type"] == "harmful"]["claimId"].nunique()
        print(f"[INFO] Claims with helpful documents: {helpful}")
        print(f"[INFO] Claims with harmful documents: {harmful}")
    else:
        if not args.csv or not args.out:
            parser.error(
                "Provide --csv and --out for live generation, or another action flag "
                "(--batch / --map-batch-results / --submit-batch / ...)."
            )
        if args.dataset == "emergent":
            main_emergent(
                csv_path=args.csv,
                out_path=args.out,
                id_col=resolved_id_col(),
                mode=args.mode,
                template=args.template,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                sleep_between=args.sleep_between,
                system_message=args.system_message,
                query_col=args.query_col,
                document_col=args.document_col,
            )
        else:
            if args.template != "liar":
                parser.error("--template applies only to --dataset emergent.")
            main_check_covid(
                csv_path=args.csv,
                out_path=args.out,
                id_col=resolved_id_col(),
                mode=args.mode,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                sleep_between=args.sleep_between,
                system_message=args.system_message,
            )
