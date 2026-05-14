#!/usr/bin/env python3
"""
Generate adversarial (stance-flipped) or stance-aligned synthetic documents for controlled IR experiments.

Input CSV columns (minimum): ``claim``, ``label``, ``abstract``. Optional: ``query``, ``description``, ``id``.

Uses fixed prompt templates filled from each row and an OpenAI-compatible chat API (``openai`` package).
Set ``OPENAI_API_KEY`` in the environment; optionally load from a local ``.env`` (never commit secrets).

CLI examples::

    python generate_adversarial.py --csv claims.csv --out out.jsonl --mode adversarial
    python generate_adversarial.py --csv claims.csv --out batch.jsonl --batch --mode adversarial
    python generate_adversarial.py --map-batch-results --batch-results-jsonl results.jsonl \\
        --original-csv claims.csv --output-csv merged.csv
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

# ---------- Configurable defaults (can be overridden by CLI args) ----------
DEFAULT_MODEL = os.getenv("MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
DEFAULT_SLEEP_BETWEEN = float(os.getenv("SLEEP_BETWEEN", "0.6"))
RETRY_ATTEMPTS = 3
RETRY_SLEEP = 2.0

# ---------- YOUR ORIGINAL PROMPT (VERBATIM) ----------
PROMPT_TEMPLATE_FACT_INVERSION = r"""You are an expert assistant trained to craft adversarial, web-style content for controlled information retrieval research. Your task is to generate a document from a given factual and credible helpful document. The document should maintain fluency, coherence, and a logical argument structure while presenting a counter-argument on query-relevant content. Unrelated content does not need to be directly opposed but should be reworded or rephrased where possible while remaining logically consistent.

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
- Do not add new sections or content that do not exist in the helpful document. The document must maintain the original helpful document’s structure and scope without unnecessary expansions.  
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





PROMPT_TEMPLATE_LIAR = r"""You are an expert assistant trained to craft adversarial, web-style content for controlled information retrieval research. Given a query, its description, and a required stance, generate a plausible, web-style document that fully supports and aligns with the given stance, without ever calling it counterfactual or acknowledging any opposing view.

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

# ---------- end of verbatim prompt ----------


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


def flip_stance(label: str) -> str:
    """Flip stance for common label tokens; unknown labels are returned unchanged."""
    if not isinstance(label, str):
        return str(label) if label is not None else ""
    l = label.strip().lower()
    if l == "support":
        return "refute"
    if l == "refute":
        return "support"
    return label.strip()

def build_user_prompt_from_row(row, use_stance: str):
    """Fill the verbatim prompt template with values from CSV row."""
    query = str(row.get("query", "")) if row.get("query", "") is not None else ""
    document = str(row.get("abstract", "")) if row.get("abstract", "") is not None else ""
    # Keep empty placeholders if missing
    filled = PROMPT_TEMPLATE_FACT_INVERSION.format(query=query, stance=use_stance, document=document)
    return filled



def call_chat_completion_with_retries(messages, model, temperature, max_tokens):
    """
    Retry loop for new OpenAI API (client.chat.completions.create)
    """
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


# ---------- Main generation logic ----------

def main(
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
            flipped_label = flip_stance(original_label)

            outputs = {}

            # generate original variant if requested
            if mode in ("original", "both"):
                user_prompt_orig = build_user_prompt_from_row(row, use_stance=(original_label or ""))
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt_orig})

                try:
                    resp = call_chat_completion_with_retries(messages, model=model, temperature=temperature, max_tokens=max_tokens)
                    gen_orig = resp.choices[0].message.content
                except Exception as e:
                    print(f"[ERROR] original generation row {row_id}: {e}")
                    gen_orig = ""
                outputs["gen_abstract_original"] = gen_orig
                outputs["user_prompt_original"] = user_prompt_orig

            # generate adversarial (flipped) variant if requested
            if mode in ("adversarial", "both"):
                # If flip returns same label (unknown), we still pass flipped_label (may be same)
                user_prompt_adv = build_user_prompt_from_row(row, use_stance=(flipped_label or ""))
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": user_prompt_adv})

                try:
                    resp = call_chat_completion_with_retries(messages, model=model, temperature=temperature, max_tokens=max_tokens)
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
                "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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

            # progress + politeness sleep
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[INFO] Processed {i+1}/{total} rows (last id={row_id})")
            time.sleep(sleep_between)

    print(f"[DONE] Wrote {total} records to {out_path}")


def main_batch(
    csv_path,
    batch_jsonl_path,
    id_col="id",
    mode="adversarial",
    model=DEFAULT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    system_message="",
):
    """
    Generate batch requests for OpenAI batch API instead of calling API directly.
    Creates a JSONL file with batch request format that can be uploaded and submitted.
    
    Parameters
    ----------
    csv_path : str
        Input CSV path with columns: claim, label, abstract; optional: query, description, id.
    batch_jsonl_path : str
        Output JSONL path for batch requests (OpenAI batch format).
    id_col : str
        Column to use as id (default: "id").
    mode : str
        "adversarial", "original", or "both" (default: "adversarial").
    model : str
        Model name to use.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max tokens for model output.
    system_message : str
        Optional system message to send.
    """
    
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
            flipped_label = flip_stance(original_label)

            # Generate original variant if requested
            if mode in ("original", "both"):
                user_prompt_orig = build_user_prompt_from_row(row, use_stance=(original_label or ""))
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
                    }
                }
                fout.write(json.dumps(batch_entry_orig, ensure_ascii=False) + "\n")
                batch_requests.append(batch_entry_orig)

            # Generate adversarial (flipped) variant if requested
            if mode in ("adversarial", "both"):
                user_prompt_adv = build_user_prompt_from_row(row, use_stance=(flipped_label or ""))
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
                    }
                }
                fout.write(json.dumps(batch_entry_adv, ensure_ascii=False) + "\n")
                batch_requests.append(batch_entry_adv)

            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[INFO] Processed {i+1}/{total} rows (last id={row_id})")

    print(f"[DONE] Created {len(batch_requests)} batch requests in {batch_jsonl_path}")
    meta_example = Path(batch_jsonl_path).parent / "batch_metadata.json"
    print("[INFO] Next steps (Python API): upload_batch_file → submit_batch_job → save_batch_metadata;")
    print("      then check_batch_status / fetch_and_download_batch_output. Example metadata path:")
    print(f"      {meta_example}")
    
    return batch_requests


def merge_adversarial_variants(
        csv_path: str,
        jsonl_path: str,
        output_path: str = None,
        id_col: str = "id",
        adv_col: str = "adversarial_variant"
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
        Column name used to match rows (default = "id").
    adv_col : str
        Name of new column to store adversarial text.
    
    Returns
    -------
    pd.DataFrame
        Merged dataframe with adversarial variants added.
    """

    df = safe_read_csv(csv_path)

    # Read JSONL and extract adversarial outputs
    adv_rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            adv_rows.append({
                id_col: data.get(id_col),
                adv_col: data.get("gen_abstract_adversarial", "")
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
        id_col: str = "id",
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
        Column name to match rows on (default 'id').
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

    # Read both CSV files
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


def upload_batch_file(file_path: str) -> str:
    """
    Upload a batch JSONL file to OpenAI and return the file ID.
    
    Parameters
    ----------
    file_path : str
        Path to the batch JSONL file to upload.
    
    Returns
    -------
    str
        OpenAI file ID of the uploaded file.
    """
    print(f"[INFO] Uploading batch file: {file_path}")
    with open(file_path, "rb") as f:
        batch_file = _get_client().files.create(file=f, purpose="batch")
    print(f"[INFO] Uploaded file id: {batch_file.id}")
    return batch_file.id


def submit_batch_job(file_id: str, description: str = "Adversarial generation batch") -> str:
    """
    Submit a batch job to OpenAI using an uploaded file ID.
    
    Parameters
    ----------
    file_id : str
        OpenAI file ID from upload_batch_file().
    description : str
        Description for the batch job.
    
    Returns
    -------
    str
        OpenAI batch job ID.
    """
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
    """
    Save batch metadata to a JSON file.
    
    Parameters
    ----------
    file_id : str
        OpenAI file ID.
    batch_id : str
        OpenAI batch job ID.
    description : str
        Description of the batch job.
    output_path : str
        Path to save the metadata JSON file.
    """
    metadata = {
        "file_id": file_id,
        "batch_id": batch_id,
        "description": description,
        "created_at": datetime.now().isoformat(),
    }
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved batch metadata to {output_path}")


def load_batch_metadata(metadata_file: str):
    """
    Load batch metadata from a JSON file.
    
    Parameters
    ----------
    metadata_file : str
        Path to the metadata JSON file.
    
    Returns
    -------
    tuple
        (batch_id, file_id, description)
    """
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata["batch_id"], metadata["file_id"], metadata.get("description", "")


def check_batch_status(metadata_path: str):
    """
    Check the status of a batch job using its metadata file.
    
    Parameters
    ----------
    metadata_path : str
        Path to the metadata JSON file.
    
    Returns
    -------
    dict
        Status information including batch_id, status, and counts.
    """
    print(f"[INFO] Checking batch status for: {metadata_path}")
    print("-" * 80)
    
    try:
        batch_id, file_id, description = load_batch_metadata(metadata_path)
        batch = _get_client().batches.retrieve(batch_id)
        
        status_info = {
            'filename': Path(metadata_path).name,
            'batch_id': batch_id,
            'file_id': file_id,
            'status': batch.status,
            'description': description,
        }
        
        if hasattr(batch, 'created_at') and batch.created_at:
            status_info['created_at'] = batch.created_at
        
        if hasattr(batch, 'request_counts') and batch.request_counts:
            counts = {}
            if hasattr(batch.request_counts, 'total'):
                counts['total'] = batch.request_counts.total
            if hasattr(batch.request_counts, 'completed'):
                counts['completed'] = batch.request_counts.completed
            if hasattr(batch.request_counts, 'failed'):
                counts['failed'] = batch.request_counts.failed
            if counts:
                status_info['request_counts'] = counts
        
        print(f"[INFO] File: {status_info['filename']}")
        print(f"[INFO] Batch id: {batch_id}")
        print(f"[INFO] Description: {description}")
        print(f"[INFO] Status: {batch.status}")
        
        if 'created_at' in status_info:
            print(f"[INFO] Created: {status_info['created_at']}")
        
        if 'request_counts' in status_info:
            counts = status_info['request_counts']
            print("[INFO] Request counts:")
            if 'total' in counts:
                print(f"   - Total: {counts['total']}")
            if 'completed' in counts:
                print(f"   - Completed: {counts['completed']}")
            if 'failed' in counts:
                print(f"   - Failed: {counts['failed']}")
        
        if batch.status == "completed":
            print("[OK] Batch completed successfully.")
            if hasattr(batch, 'output_file_id') and batch.output_file_id:
                print(f"[INFO] Output file id: {batch.output_file_id}")
                status_info['output_file_id'] = batch.output_file_id
        elif batch.status in ["failed", "cancelled", "expired"]:
            print(f"[ERROR] Batch {batch.status}")
            if hasattr(batch, 'errors') and batch.errors:
                print(f"[WARN] Errors: {batch.errors}")
                status_info['errors'] = batch.errors
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
    """
    Get the output file ID from a completed batch job.
    
    Parameters
    ----------
    batch_id : str
        OpenAI batch job ID.
    
    Returns
    -------
    str
        Output file ID.
    """
    batch = _get_client().batches.retrieve(batch_id)
    print(f"[INFO] Batch status: {batch.status}")
    if batch.status == "completed":
        return batch.output_file_id
    elif batch.status in ["failed", "cancelled", "expired"]:
        raise RuntimeError(f"Batch failed (status: {batch.status})")
    else:
        raise RuntimeError(f"Batch not yet complete (status: {batch.status})")


def download_batch_output(file_id: str, save_path: str):
    """
    Download batch results from OpenAI using a file ID.
    
    Parameters
    ----------
    file_id : str
        OpenAI output file ID.
    save_path : str
        Path to save the downloaded results.
    """
    print(f"[INFO] Downloading output file: {file_id}")
    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    file_response = _get_client().files.content(file_id)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(file_response.text)
    print(f"[OK] Results saved to {save_path}")


def run_batch_pipeline(
    batch_jsonl_path: str,
    metadata_path: str = None,
    description: str = "Adversarial generation batch"
):
    """
    Complete batch pipeline: upload file, submit job, and save metadata.
    
    Parameters
    ----------
    batch_jsonl_path : str
        Path to the batch JSONL file to upload.
    metadata_path : str, optional
        Path to save metadata. If None, uses same directory as batch_jsonl_path with '_metadata.json' suffix.
    description : str
        Description for the batch job.
    
    Returns
    -------
    tuple
        (file_id, batch_id)
    """
    # Determine metadata path if not provided
    if metadata_path is None:
        batch_path = Path(batch_jsonl_path)
        metadata_path = str(batch_path.parent / f"{batch_path.stem}_metadata.json")
    
    # Upload file
    file_id = upload_batch_file(batch_jsonl_path)
    
    # Submit batch job
    batch_id = submit_batch_job(file_id, description)
    
    # Save metadata
    save_batch_metadata(file_id, batch_id, description, metadata_path)
    
    return file_id, batch_id


def fetch_and_download_batch_output(
    metadata_path: str,
    save_path: str = None
):
    """
    Check batch status and download results if completed.
    
    Parameters
    ----------
    metadata_path : str
        Path to the metadata JSON file.
    save_path : str, optional
        Path to save results. If None, uses same directory as metadata_path with '_results.jsonl' suffix.
    
    Returns
    -------
    str or None
        Path to downloaded file if successful, None otherwise.
    """
    batch_id, _, description = load_batch_metadata(metadata_path)
    print(f"[INFO] Checking batch: {batch_id} | Description: {description}")
    
    try:
        output_file_id = get_output_file_id(batch_id)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return None
    
    # Determine save path if not provided
    if save_path is None:
        metadata_file = Path(metadata_path)
        save_path = str(metadata_file.parent / f"{metadata_file.stem.replace('_metadata', '')}_results.jsonl")
    
    download_batch_output(output_file_id, save_path)
    return save_path


def clean_adversarial_text(text: str) -> str:
    """
    Clean adversarial text by removing markdown code block formatting.
    
    Removes ```plaintext\n at the start and ```\n at the end.
    Handles various markdown code block formats.
    
    Parameters
    ----------
    text : str
        Raw text from batch response, possibly wrapped in markdown.
    
    Returns
    -------
    str
        Cleaned text without markdown formatting.
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove markdown code block at the beginning
    # Pattern: ```plaintext\n or ```plaintext or ```\n or ```
    if text.startswith("```plaintext"):
        # Remove ```plaintext and any following newline
        text = text[12:].lstrip("\n")
    elif text.startswith("```"):
        # Remove ``` and any following newline
        text = text[3:].lstrip("\n")
    
    # Remove markdown code block at the end
    # Pattern: ```\n or \n``` or ```
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
    cord_id_col: str = None,
    results_id_field: str = "custom_id",
    adversarial_column: str = "adversarial_variant"
):
    """
    Map batch results back to original CSV and add adversarial_variant column.
    
    Reads batch results JSONL, extracts generated text from responses,
    cleans markdown formatting, and merges with original CSV on ID.
    
    Parameters
    ----------
    batch_results_jsonl : str
        Path to batch results JSONL file from OpenAI.
    original_csv : str
        Path to original CSV file that was used to create batch requests.
    output_csv : str
        Path to save the merged CSV with adversarial_variant column.
    id_col : str
        Column name in original CSV to match on (default: "id").
    cord_id_col : str, optional
        If provided, creates a composite key "{id}_{cord_id}" from CSV columns
        to match against custom_id in batch results.
    results_id_field : str
        Field name in batch results to use as ID (default: "custom_id").
    adversarial_column : str
        Name of the new column to add (default: "adversarial_variant").
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with adversarial variants added.
    """
    print(f"Reading batch results: {batch_results_jsonl}")
    
    # Read batch results and extract adversarial text
    results_dict = {}
    successful_extractions = 0
    failed_extractions = 0
    
    with open(batch_results_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                result = json.loads(line)
                
                # Get custom_id (this should match the row ID from CSV)
                custom_id = result.get(results_id_field)
                if not custom_id:
                    print(f"[WARN] Line {line_num}: Missing {results_id_field}")
                    failed_extractions += 1
                    continue
                
                # Extract generated text from response
                content = None
                error = result.get("error")
                
                if error:
                    print(f"[WARN] Line {line_num} (custom_id={custom_id}): Error in response: {error}")
                    failed_extractions += 1
                    results_dict[custom_id] = ""  # Store empty string for errors
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
                        results_dict[custom_id] = ""
                        continue
                        
                except (KeyError, IndexError, AttributeError) as e:
                    print(f"[WARN] Line {line_num} (custom_id={custom_id}): Error extracting content: {e}")
                    failed_extractions += 1
                    results_dict[custom_id] = ""
                    continue
                
                # Clean the content (remove markdown formatting)
                if content:
                    cleaned_content = clean_adversarial_text(content)
                    results_dict[custom_id] = cleaned_content
                    successful_extractions += 1
                else:
                    results_dict[custom_id] = ""
                    failed_extractions += 1
                    
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON decode error: {e}")
                failed_extractions += 1
                continue
            except Exception as e:
                print(f"[WARN] Line {line_num}: Unexpected error: {e}")
                failed_extractions += 1
                continue
    
    print(f"  Extracted {successful_extractions} successful responses")
    if failed_extractions > 0:
        print(f"  [WARN] {failed_extractions} failed extractions (errors or missing data)")
    
    # Read original CSV
    print(f"Reading original CSV: {original_csv}")
    df_original = safe_read_csv(original_csv)
    print(f"  Loaded {len(df_original)} rows")
    
    # Check if id_col exists
    if id_col not in df_original.columns:
        raise ValueError(f"Column '{id_col}' not found in {original_csv}")
    
    # Check if cord_id_col is provided and exists
    use_composite_key = False
    if cord_id_col:
        if cord_id_col not in df_original.columns:
            print(f"  [WARN] cord_id_col '{cord_id_col}' not found in CSV, falling back to {id_col} only")
        else:
            use_composite_key = True
            print(f"  Using composite key: {id_col}_{cord_id_col}")
    
    # Create DataFrame from results
    # Use custom_id directly as the match key
    results_data = []
    merge_key = "_composite_key_" if use_composite_key else id_col
    for custom_id, adversarial_text in results_dict.items():
        results_data.append({
            merge_key: str(custom_id),
            adversarial_column: adversarial_text
        })
    
    df_results = pd.DataFrame(results_data)
    
    # Merge with original CSV
    df_original_copy = df_original.copy()
    
    if use_composite_key:
        # Create composite key from id and cord_id columns: "{id}_{cord_id}"
        df_original_copy[id_col] = df_original_copy[id_col].astype(str)
        df_original_copy[cord_id_col] = df_original_copy[cord_id_col].astype(str)
        df_original_copy["_composite_key_"] = df_original_copy[id_col] + "_" + df_original_copy[cord_id_col]
        df_merged = df_original_copy.merge(df_results, on="_composite_key_", how="left")
        df_merged = df_merged.drop(columns=["_composite_key_"])  # Remove helper column
        merge_method = f"{id_col}_{cord_id_col}"
    else:
        # Match directly on id_col
        df_original_copy[id_col] = df_original_copy[id_col].astype(str)
        df_merged = df_original_copy.merge(df_results, on=id_col, how="left")
        merge_method = f"{id_col}"
    
    # Ensure adversarial_column exists in merged DataFrame
    if adversarial_column not in df_merged.columns:
        df_merged[adversarial_column] = None
        print(f"  [WARN] No matches found - added empty '{adversarial_column}' column")
    
    # Report merge statistics
    matched_count = df_merged[adversarial_column].notna().sum()
    print(f"  Matched {matched_count}/{len(df_merged)} rows with batch results (using {merge_method})")
    
    if matched_count < len(df_merged):
        unmatched = len(df_merged) - matched_count
        print(f"  [WARN] {unmatched} rows did not have matching batch results")
    
    # Save merged CSV
    out_dir = Path(output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    print(f"[OK] Merged CSV saved to: {output_csv}")
    
    return df_merged


def filter_rows_not_in_sample(
        main_csv: str,
        sample_csv: str,
        output_csv: str = None,
        id_col: str = "id"
    ):
    """
    Create a CSV containing rows from main_csv that are NOT in sample_csv.
    
    This function reads both CSVs, identifies rows in main_csv that don't have
    matching IDs in sample_csv, and saves the filtered result to a new CSV.
    
    Parameters
    ----------
    main_csv : str
        Path to the main CSV file (e.g., neutral_claims.csv) containing all rows.
    sample_csv : str
        Path to the sample CSV file (e.g., sample_20.csv) containing rows to exclude.
    output_csv : str, optional
        Path to save the output CSV. If None, will create a filename based on main_csv.
    id_col : str
        Column name to use for matching rows (default 'id').
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing rows from main_csv not in sample_csv.
    """
    
    # Read both CSV files
    print(f"Reading main CSV: {main_csv}")
    df_main = safe_read_csv(main_csv)
    print(f"  Loaded {len(df_main)} rows")
    
    print(f"Reading sample CSV: {sample_csv}")
    df_sample = safe_read_csv(sample_csv)
    print(f"  Loaded {len(df_sample)} rows")
    
    # Check if id_col exists in both DataFrames
    if id_col not in df_main.columns:
        raise ValueError(f"Column '{id_col}' not found in {main_csv}")
    if id_col not in df_sample.columns:
        raise ValueError(f"Column '{id_col}' not found in {sample_csv}")
    
    # Get set of IDs from sample CSV
    sample_ids = set(df_sample[id_col].astype(str))
    print(f"  Found {len(sample_ids)} unique IDs in sample CSV")
    
    # Filter main CSV to exclude rows with IDs in sample
    df_filtered = df_main[~df_main[id_col].astype(str).isin(sample_ids)]
    
    print(f"  Filtered result: {len(df_filtered)} rows (removed {len(df_main) - len(df_filtered)} rows)")
    
    # Determine output path if not provided
    if output_csv is None:
        main_path = Path(main_csv)
        output_csv = str(main_path.parent / f"{main_path.stem}_filtered{main_path.suffix}")
    
    # Save filtered CSV
    df_filtered.to_csv(output_csv, index=False)
    print(f"[OK] Filtered CSV saved to: {output_csv}")
    
    return df_filtered

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate adversarial variants (live API), batch JSONL, or merge batch results."
    )
    parser.add_argument(
        "--map-batch-results",
        action="store_true",
        help="Merge OpenAI batch output JSONL into the original CSV.",
    )
    parser.add_argument("--batch-results-jsonl", help="Batch API results JSONL path.")
    parser.add_argument("--original-csv", help="Original CSV used to build batch requests.")
    parser.add_argument("--output-csv", help="Output path for merged CSV.")
    parser.add_argument("--cord-id-col", default=None, help="Optional second key column for composite ids.")
    parser.add_argument(
        "--adversarial-column",
        default="adversarial_variant",
        help="Column name for merged generated text.",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Write OpenAI Batch request JSONL only (no live completions).",
    )
    parser.add_argument("--csv", help="Input CSV (columns include claim, label, abstract).")
    parser.add_argument("--out", help="Output JSONL path.")
    parser.add_argument("--id-col", default="id", help="Row id column.")
    parser.add_argument(
        "--mode",
        choices=["adversarial", "original", "both"],
        default="adversarial",
        help="Use flipped stance, original stance, or both.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--sleep-between", type=float, default=DEFAULT_SLEEP_BETWEEN)
    parser.add_argument("--system-message", default="", help="Optional system message.")

    args = parser.parse_args()

    if args.map_batch_results:
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
        map_batch_results_to_csv(
            batch_results_jsonl=args.batch_results_jsonl,
            original_csv=args.original_csv,
            output_csv=args.output_csv,
            id_col=args.id_col,
            cord_id_col=args.cord_id_col,
            adversarial_column=args.adversarial_column,
        )
    elif args.batch:
        if not args.csv or not args.out:
            parser.error("--batch requires --csv and --out")
        main_batch(
            csv_path=args.csv,
            batch_jsonl_path=args.out,
            id_col=args.id_col,
            mode=args.mode,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_message=args.system_message,
        )
    else:
        if not args.csv or not args.out:
            parser.error("Provide --csv and --out, or use --batch / --map-batch-results.")
        main(
            csv_path=args.csv,
            out_path=args.out,
            id_col=args.id_col,
            mode=args.mode,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            sleep_between=args.sleep_between,
            system_message=args.system_message,
        )
