"""Preprocessing utilities for the Check-COVID benchmark.

CSV helpers for building the experiment tables and attaching model outputs
from JSONL result files (OpenAI Batch, open-source vLLM-style JSONL, DeepSeek).

All paths are passed as arguments; nothing here points at a private workstation.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd


def add_query_column(input_file, output_file=None):
    """
    Adds a 'query' column to the CSV file where each query is formatted as:
    'Is this true that [claim]'
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the output CSV. If None, overwrites input file.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Add the query column by prepending "Is this true that " to each claim
    # Strip trailing period and add question mark
    df['query'] = df['claim'].apply(lambda x: f"Is this true that {x.rstrip('.')}?")
    
    # Save the modified dataframe
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    print(f"Successfully added 'query' column to {output_file}")
    print(f"Total rows processed: {len(df)}")

def split_by_label(input_file, output_dir=None):
    """
    Splits the CSV file into two separate files based on the 'label' column:
    - One file for SUPPORT labels
    - One file for REFUTE labels
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str, optional): Directory to save output files. If None, saves in the same directory as input.
    """
    import os
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get the directory and base filename
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Filter by label
    support_df = df[df['label'] == 'SUPPORT']
    refute_df = df[df['label'] == 'REFUTE']
    
    # Create output paths
    support_output = os.path.join(output_dir, f"{base_name}_support.csv")
    refute_output = os.path.join(output_dir, f"{base_name}_refute.csv")
    
    # Save the filtered dataframes
    support_df.to_csv(support_output, index=False)
    refute_df.to_csv(refute_output, index=False)
    
    print(f"✅ Successfully split the CSV file:")
    print(f"   - SUPPORT: {len(support_df)} rows → {support_output}")
    print(f"   - REFUTE: {len(refute_df)} rows → {refute_output}")
    
    return support_output, refute_output

def select_random_rows(input_file, n, output_file=None, random_state=42):
    """
    Selects n random rows from a CSV file and saves them to a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        n (int): Number of random rows to select
        output_file (str, optional): Path to save the output CSV. If None, auto-generates filename.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        str: Path to the output file
    """
    import os
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check if n is larger than available rows
    if n > len(df):
        print(f"⚠️  Warning: Requested {n} rows but only {len(df)} available. Selecting all rows.")
        n = len(df)
    
    # Select n random rows
    random_df = df.sample(n=n, random_state=random_state)
    
    # Generate output filename if not provided
    if output_file is None:
        base_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(base_dir, f"{base_name}_random_{n}.csv")
    
    # Save to CSV
    random_df.to_csv(output_file, index=False)
    
    print(f"✅ Successfully selected {n} random rows from {len(df)} total rows")
    print(f"   Saved to: {output_file}")
    
    return output_file

def add_answers_from_jsonl(jsonl_file, csv_file, output_csv=None):
    """
    Extracts answers from a JSONL results file and adds them as a column to the CSV file.
    Maps results to CSV rows based on ID.
    
    Args:
        jsonl_file (str): Path to the JSONL results file
        csv_file (str): Path to the CSV file to update
        output_csv (str, optional): Path to save the output CSV. If None, overwrites input CSV.
    
    Returns:
        str: Path to the output file
    """
    import json
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a dictionary to map id to answer
    id_to_answer = {}
    id_to_explanation = {}
    
    # Read the JSONL file and extract answers
    matched_count = 0
    error_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line.strip())
                
                # Extract custom_id and normalize it
                # Handles formats like:
                # - "nonrag_6026eb286bc088b6d218dc14_1"
                # - "6026eb286bc088b6d218dc14_1_tdqtikbh"
                # - "6026eb286bc088b6d218dc14_1"
                custom_id = result.get('custom_id', '')
                
                # Remove 'nonrag_' prefix if present
                if custom_id.startswith('nonrag_'):
                    record_id = custom_id.replace('nonrag_', '', 1)
                else:
                    record_id = custom_id
                
                # Remove any suffix after the last underscore if it looks like a hash
                # (e.g., "_tdqtikbh" in "6026eb286bc088b6d218dc14_1_tdqtikbh")
                parts = record_id.split('_')
                if len(parts) > 2 and len(parts[-1]) == 8 and parts[-1].isalnum():
                    # Likely a suffix hash, remove it
                    record_id = '_'.join(parts[:-1])
                
                # Extract the response content
                response = result.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    
                    # Parse the JSON content to extract answer and explanation
                    try:
                        content_json = json.loads(content)
                        answer = content_json.get('answer', '')
                        explanation = content_json.get('explanation', '')
                        
                        id_to_answer[record_id] = answer
                        id_to_explanation[record_id] = explanation
                        matched_count += 1
                    except json.JSONDecodeError:
                        print(f"⚠️  Warning: Could not parse JSON content at line {line_num}")
                        error_count += 1
                        
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Could not parse JSONL line {line_num}")
                error_count += 1
                continue
    
    # Map answers to the CSV dataframe based on id
    df['model_answer'] = df['id'].map(id_to_answer)
    df['model_explanation'] = df['id'].map(id_to_explanation)
    
    # Check how many rows were matched
    total_rows = len(df)
    matched_rows = df['model_answer'].notna().sum()
    unmatched_rows = df['model_answer'].isna().sum()
    
    # Filter to only keep matched rows
    df_matched = df[df['model_answer'].notna()].copy()
    
    # Save to CSV
    if output_csv is None:
        output_csv = csv_file
    
    df_matched.to_csv(output_csv, index=False)
    
    print(f"✅ Successfully added answers from JSONL to CSV")
    print(f"   Total JSONL records processed: {matched_count}")
    print(f"   Total CSV rows: {total_rows}")
    print(f"   Rows matched and saved: {matched_rows}")
    print(f"   Rows skipped (unmatched): {unmatched_rows}")
    if error_count > 0:
        print(f"   Errors encountered: {error_count}")
    print(f"   Saved to: {output_csv}")
    
    return output_csv

def add_answers_from_opensource_jsonl(jsonl_file, csv_file, output_csv=None):
    """
    Extracts answers from an open source model JSONL results file and adds them as columns to the CSV file.
    Maps results to CSV rows based on ID.
    
    Args:
        jsonl_file (str): Path to the JSONL results file from open source model
        csv_file (str): Path to the CSV file to update
        output_csv (str, optional): Path to save the output CSV. If None, overwrites input CSV.
    
    Returns:
        str: Path to the output file
    """
    import json
    import re
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a dictionary to map id to answer
    id_to_answer = {}
    id_to_explanation = {}
    
    # Read the JSONL file and extract answers
    matched_count = 0
    error_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                result = json.loads(line.strip())
                
                # Extract topic_id and normalize it
                # Handles formats like:
                # - "nonrag_6026eb286bc088b6d218dc14_0"
                # - "6026eb286bc088b6d218dc14_0_tdqtikbh"
                # - "6026eb286bc088b6d218dc14_0"
                topic_id = result.get('topic_id', '')
                
                if not topic_id:
                    print(f"⚠️  Warning: No topic_id at line {line_num}")
                    error_count += 1
                    continue
                
                # Remove 'nonrag_' prefix if present
                if topic_id.startswith('nonrag_'):
                    record_id = topic_id.replace('nonrag_', '', 1)
                else:
                    record_id = topic_id
                
                # Remove any suffix after the last underscore if it looks like a hash
                # (e.g., "_tdqtikbh" in "6026eb286bc088b6d218dc14_0_tdqtikbh")
                parts = record_id.split('_')
                if len(parts) > 2 and len(parts[-1]) == 8 and parts[-1].isalnum():
                    # Likely a suffix hash, remove it
                    record_id = '_'.join(parts[:-1])
                
                # Extract answer from the answer array
                answer_array = result.get('answer', [])
                
                if not answer_array or not isinstance(answer_array, list):
                    print(f"⚠️  Warning: No answer array at line {line_num}")
                    error_count += 1
                    continue
                
                # Concatenate all text to search for JSON
                all_text = ' '.join([
                    item.get('text', '') 
                    for item in answer_array 
                    if isinstance(item, dict) and 'text' in item
                ])
                
                if not all_text:
                    print(f"⚠️  Warning: Empty answer text at line {line_num}")
                    error_count += 1
                    continue
                
                # Try to extract JSON from the text
                # Look for patterns like { "answer": "yes", "explanation": "..." }
                answer = None
                explanation = None
                
                try:
                    # Find JSON object in the text
                    start_idx = all_text.find('{')
                    end_idx = all_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = all_text[start_idx:end_idx + 1]
                        # Clean up any malformed parts
                        json_str = json_str.strip()
                        
                        # Try to parse as JSON
                        parsed = json.loads(json_str)
                        answer = parsed.get('answer', '').strip().lower()
                        explanation = parsed.get('explanation', '').strip()
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # If JSON parsing fails, try to extract using regex
                if not answer:
                    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', all_text, re.IGNORECASE)
                    explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', all_text, re.IGNORECASE)
                    
                    answer = answer_match.group(1).strip().lower() if answer_match else None
                    explanation = explanation_match.group(1).strip() if explanation_match else None
                
                if answer:
                    id_to_answer[record_id] = answer
                    id_to_explanation[record_id] = explanation if explanation else ''
                    matched_count += 1
                else:
                    print(f"⚠️  Warning: Could not extract answer from topic_id '{topic_id}' at line {line_num}")
                    error_count += 1
                        
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Could not parse JSONL line {line_num}")
                error_count += 1
                continue
            except Exception as e:
                print(f"⚠️  Warning: Error processing line {line_num}: {e}")
                error_count += 1
                continue
    
    # Map answers to the CSV dataframe based on id
    df['model_answer'] = df['id'].map(id_to_answer)
    df['model_explanation'] = df['id'].map(id_to_explanation)
    
    # Check how many rows were matched
    total_rows = len(df)
    matched_rows = df['model_answer'].notna().sum()
    unmatched_rows = df['model_answer'].isna().sum()
    
    # Keep ALL rows (including unmatched ones)
    # Fill NaN values with empty string for unmatched rows
    df['model_answer'] = df['model_answer'].fillna('')
    df['model_explanation'] = df['model_explanation'].fillna('')
    
    # Save to CSV
    if output_csv is None:
        output_csv = csv_file
    
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Successfully added answers from open source JSONL to CSV")
    print(f"   Total JSONL records processed: {matched_count}")
    print(f"   Total CSV rows: {total_rows}")
    print(f"   Rows matched and saved: {matched_rows}")
    print(f"   Rows skipped (unmatched): {unmatched_rows}")
    if error_count > 0:
        print(f"   Errors encountered: {error_count}")
    print(f"   Saved to: {output_csv}")
    
    return output_csv




def _extract_deepseek_json_answer(answer_array):
    """
    Extracts (answer, explanation) from DeepSeek-style 'answer' array.
    Handles:
      - instruction echo
      - /think, <think>, </think>, <|im_end|>
      - markdown quotes (>)
      - ```json ... ``` wrappers
      - JSON object embedded in text
    """
    # Concatenate all text segments
    all_text = ' '.join(
        item.get('text', '')
        for item in answer_array
        if isinstance(item, dict) and 'text' in item
    )

    if not all_text.strip():
        return None, None

    # Remove obvious junk tokens
    junk_tokens = [
        '```json', '```',
        '<think>', '</think>', '/think',
        '<|im_end|>',
    ]
    for tok in junk_tokens:
        all_text = all_text.replace(tok, ' ')

    # Strip leading markdown quotes like ">  { ... }"
    lines = all_text.splitlines()
    lines = [re.sub(r'^\s*>\s*', '', line) for line in lines]
    all_text = '\n'.join(lines)

    # First try: find a JSON object containing "answer"
    json_pattern = r'\{[^{}]*"answer"\s*:\s*"[^"]+"[^{}]*\}'
    for match in re.finditer(json_pattern, all_text, flags=re.IGNORECASE | re.DOTALL):
        candidate = match.group(0).strip()
        try:
            parsed = json.loads(candidate)
            answer = parsed.get('answer', '').strip().lower()
            explanation = parsed.get('explanation', '').strip()
            if answer:
                return answer, explanation
        except json.JSONDecodeError:
            continue  # try next match if any

    # Fallback: your old regex approach
    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', all_text, re.IGNORECASE)
    explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', all_text, re.IGNORECASE)

    answer = answer_match.group(1).strip().lower() if answer_match else None
    explanation = explanation_match.group(1).strip() if explanation_match else None

    return answer, explanation


def add_answers_from_deepseek_jsonl(jsonl_file, csv_file, output_csv=None):
    """
    DeepSeek-specific version:
    Extracts `answer` and `explanation` from DeepSeek JSONL results and
    maps them back to the CSV by `id`.

    It is more robust to DeepSeek quirks:
      - echoing instructions
      - leaking /think, <think>, <|im_end|>
      - wrapping JSON in markdown or quotes
    """
    # Read CSV
    df = pd.read_csv(csv_file)

    id_to_answer = {}
    id_to_explanation = {}

    matched_count = 0
    error_count = 0

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                result = json.loads(line.strip())
                topic_id = result.get('topic_id', '')

                if not topic_id:
                    print(f"⚠️  Warning: No topic_id at line {line_num}")
                    error_count += 1
                    continue

                # Normalise ID (same logic as your original function)
                if topic_id.startswith('nonrag_'):
                    record_id = topic_id.replace('nonrag_', '', 1)
                else:
                    record_id = topic_id

                parts = record_id.split('_')
                if len(parts) > 2 and len(parts[-1]) == 8 and parts[-1].isalnum():
                    record_id = '_'.join(parts[:-1])

                answer_array = result.get('answer', [])
                if not answer_array or not isinstance(answer_array, list):
                    print(f"⚠️  Warning: No answer array at line {line_num}")
                    error_count += 1
                    continue

                answer, explanation = _extract_deepseek_json_answer(answer_array)

                if answer:
                    id_to_answer[record_id] = answer
                    id_to_explanation[record_id] = explanation or ''
                    matched_count += 1
                else:
                    print(f"⚠️  Warning: Could not extract answer from topic_id '{topic_id}' at line {line_num}")
                    error_count += 1

            except json.JSONDecodeError:
                print(f"⚠️  Warning: Could not parse JSONL line {line_num}")
                error_count += 1
                continue
            except Exception as e:
                print(f"⚠️  Warning: Error processing line {line_num}: {e}")
                error_count += 1
                continue

    # Map back to CSV
    df['model_answer'] = df['id'].map(id_to_answer)
    df['model_explanation'] = df['id'].map(id_to_explanation)

    total_rows = len(df)
    matched_rows = df['model_answer'].notna().sum()
    unmatched_rows = df['model_answer'].isna().sum()

    # Only keep matched rows (same as your original)
    # df_matched = df[df['model_answer'].notna()].copy()

    if output_csv is None:
        output_csv = csv_file

    df.to_csv(output_csv, index=False)

    print(f"✅ Successfully added DeepSeek answers from JSONL to CSV")
    print(f"   Total JSONL records with extracted answers: {matched_count}")
    print(f"   Total CSV rows: {total_rows}")
    print(f"   Rows matched and saved: {matched_rows}")
    print(f"   Rows skipped (unmatched): {unmatched_rows}")
    if error_count > 0:
        print(f"   Errors encountered: {error_count}")
    print(f"   Saved to: {output_csv}")

    return output_csv

def apply_query_framing(
    csv_path: str,
    output_path: str,
    evidence_type: str,
    query_col: str = "query"
):
    """
    Apply consistent / inconsistent belief framing to queries based on evidence type.

    Args:
        csv_path: input CSV path
        output_path: where to save the updated CSV
        evidence_type: "supporting" or "refuting"
        query_col: name of the column containing the original query X
    """

    if evidence_type not in {"supporting", "refuting"}:
        raise ValueError("evidence_type must be 'supporting' or 'refuting'")

    df = pd.read_csv(csv_path)

    if query_col not in df.columns:
        raise ValueError(f"Column '{query_col}' not found in CSV")

    def consistent_frame(x: str) -> str:
        return f"I have heard that {x} I do not believe it. Is it true that {x}?"

    def inconsistent_frame(x: str) -> str:
        return f"My doctor told me that {x} Can you confirm this is true?"

    # Swap logic depending on evidence type
    if evidence_type == "refuting":
        df["consistent_query"] = df[query_col].apply(consistent_frame)
        df["inconsistent_query"] = df[query_col].apply(inconsistent_frame)
    else:  # supporting
        df["consistent_query"] = df[query_col].apply(inconsistent_frame)
        df["inconsistent_query"] = df[query_col].apply(consistent_frame)

    df.to_csv(output_path, index=False)
