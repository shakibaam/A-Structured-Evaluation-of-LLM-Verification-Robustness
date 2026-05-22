"""Preprocessing utilities for the Emergent benchmark.

CSV helpers for building the experiment tables under ``emergent/data/`` and
attaching model outputs from JSONL result files (OpenAI Batch, open-source
vLLM-style JSONL, DeepSeek).

All paths are passed as arguments; nothing here points at a private workstation.
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd


def select_first_n_rows(csv_path, n, out_path):
    """
    Select the first n rows from a CSV file and save them into another CSV file.

    Args:
        csv_path (str): Path to the input CSV file.
        n (int): Number of rows to select.
        out_path (str): Path to save the output CSV file.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Select first n rows
    df_head = df.head(n)
    
    # Save to new CSV
    df_head.to_csv(out_path, index=False)
    
    print(f"✅ Saved first {n} rows from '{csv_path}' to '{out_path}'")

def select_unique_claims(csv_path, n, out_path, exclude_csv=None):
    """
    Select n unique claimId rows (one per claimId) from a CSV and save them to another CSV.
    Optionally exclude claimIds that appear in another CSV or directory of CSVs.

    Args:
        csv_path (str): Path to input CSV file (must contain 'claimId' column).
        n (int): Number of unique claimIds to select.
        out_path (str): Path to save output CSV file.
        exclude_csv (str, optional): Path to CSV file OR directory containing CSV files with claimIds to exclude.
    """
    import os
    import glob
    
    # Read CSV
    df = pd.read_csv(csv_path)

    if 'claimId' not in df.columns:
        raise ValueError("❌ The CSV must contain a 'claimId' column.")

    # Drop duplicate claimIds, keeping the first occurrence
    unique_claims = df.drop_duplicates(subset='claimId')
    
    # If exclude_csv is provided, read it and filter out those claimIds
    if exclude_csv is not None:
        exclude_ids = set()
        
        # Check if it's a directory or a file
        if os.path.isdir(exclude_csv):
            print(f"📂 Reading all CSV files from directory: {exclude_csv}")
            csv_files = glob.glob(os.path.join(exclude_csv, "*.csv"))
            
            if not csv_files:
                print(f"⚠️  Warning: No CSV files found in directory: {exclude_csv}")
            else:
                print(f"   Found {len(csv_files)} CSV file(s)")
                
                for csv_file in csv_files:
                    try:
                        print(f"   📖 Reading: {os.path.basename(csv_file)}")
                        exclude_df = pd.read_csv(csv_file, encoding="latin1", on_bad_lines='skip')
                        
                        if 'claimId' not in exclude_df.columns:
                            print(f"      ⚠️  Warning: No 'claimId' column in {os.path.basename(csv_file)}, skipping")
                            continue
                        
                        file_claim_ids = set(exclude_df['claimId'].unique())
                        exclude_ids.update(file_claim_ids)
                        print(f"      Added {len(file_claim_ids)} unique claimIds")
                    except Exception as e:
                        print(f"      ⚠️  Error reading {os.path.basename(csv_file)}: {e}")
                        continue
        else:
            # Single file
            print(f"📖 Reading exclude CSV: {exclude_csv}")
            exclude_df = pd.read_csv(exclude_csv, encoding="latin1", on_bad_lines='skip')
            
            if 'claimId' not in exclude_df.columns:
                raise ValueError(f"❌ The exclude CSV must contain a 'claimId' column.")
            
            exclude_ids = set(exclude_df['claimId'].unique())
        
        # Apply exclusion
        if exclude_ids:
            original_count = len(unique_claims)
            unique_claims = unique_claims[~unique_claims['claimId'].isin(exclude_ids)]
            
            excluded_count = original_count - len(unique_claims)
            print(f"🔍 Total unique claimIds to exclude: {len(exclude_ids)}")
            print(f"   Excluded {excluded_count} claimIds from selection")
            print(f"   Remaining unique claimIds: {len(unique_claims)}")

    # Select the first n unique claims
    selected = unique_claims.head(n)

    # Save to new CSV
    selected.to_csv(out_path, index=False)

    print(f"✅ Saved {len(selected)} unique claimIds from '{csv_path}' to '{out_path}'")


def remove_claim_prefix(csv_path, out_path=None):
    """
    Remove the "Claim: " prefix from the claimHeadline column in a CSV file.

    Args:
        csv_path (str): Path to the input CSV file.
        out_path (str, optional): Path to save the output CSV file. 
                                  If None, overwrites the input file.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Check if claimHeadline column exists
    if 'claimHeadline' not in df.columns:
        print(f"❌ Error: 'claimHeadline' column not found in '{csv_path}'")
        return
    
    # Remove "Claim: " prefix from claimHeadline column
    df['claimHeadline'] = df['claimHeadline'].str.replace('^Claim: ', '', regex=True)
    
    # Determine output path
    output_path = out_path if out_path is not None else csv_path
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Removed 'Claim: ' prefix from claimHeadline column")
    print(f"✅ Saved to '{output_path}'")


def filter_for_against_stance(csv_path, out_path):
    """
    Filter CSV to keep only rows where articleStance is "for" or "against".

    Args:
        csv_path (str): Path to the input CSV file.
        out_path (str): Path to save the filtered output CSV file.
    """
    # Read the CSV with error handling for encoding issues
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try with error handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', errors='replace')
        except Exception as e:
            # Try latin-1 encoding as fallback
            try:
                df = pd.read_csv(csv_path, encoding='latin-1')
            except Exception as e2:
                print(f"❌ Error reading CSV file: {e2}")
                return
    
    # Check if articleStance column exists
    if 'articleStance' not in df.columns:
        print(f"❌ Error: 'articleStance' column not found in '{csv_path}'")
        return
    
    # Get original count
    original_count = len(df)
    
    # Filter to keep only "for" and "against" stances
    filtered_df = df[df['articleStance'].isin(['for', 'against'])]
    
    # Get filtered count
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    # Save filtered CSV
    filtered_df.to_csv(out_path, index=False)
    
    print(f"✅ Filtered CSV saved to '{out_path}'")
    print(f"   Original rows: {original_count}")
    print(f"   Kept rows (for/against): {filtered_count}")
    print(f"   Removed rows: {removed_count}")


def remove_claim_ids(csv_path, claim_ids, out_path):
    """
    Remove rows with specific claimIds from a CSV file and save to a new file.
    
    Args:
        csv_path (str): Path to the input CSV file.
        claim_ids (list): List of claimIds to remove.
        out_path (str): Path to save the filtered output CSV file.
    """
    # Read the CSV with error handling for encoding issues
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try with error handling
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', errors='replace')
        except Exception as e:
            # Try latin-1 encoding as fallback
            try:
                df = pd.read_csv(csv_path, encoding='latin-1')
            except Exception as e2:
                print(f"❌ Error reading CSV file: {e2}")
                return
    
    # Check if claimId column exists
    if 'claimId' not in df.columns:
        print(f"❌ Error: 'claimId' column not found in '{csv_path}'")
        return
    
    # Get original count
    original_count = len(df)
    
    # Convert claim_ids to set for faster lookup
    claim_ids_set = set(claim_ids)
    
    # Filter out rows with claimIds in the provided list
    filtered_df = df[~df['claimId'].isin(claim_ids_set)]
    
    # Get filtered count
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    # Save filtered CSV
    filtered_df.to_csv(out_path, index=False)
    
    print(f"✅ Filtered CSV saved to '{out_path}'")
    print(f"   Original rows: {original_count}")
    print(f"   Removed rows: {removed_count}")
    print(f"   Remaining rows: {filtered_count}")
    print(f"   ClaimIds to remove: {len(claim_ids_set)}")


def sort_csv_by_claimid_order(reference_csv, target_csv, out_path):
    """
    Sort the target CSV to match the claimId order from the reference CSV.
    
    Args:
        reference_csv (str): Path to reference CSV file (defines the claimId order).
        target_csv (str): Path to target CSV file (will be sorted).
        out_path (str): Path to save the sorted output CSV file.
    """
    # Read both CSVs with error handling
    try:
        ref_df = pd.read_csv(reference_csv, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            ref_df = pd.read_csv(reference_csv, encoding='latin-1')
        except Exception as e:
            print(f"❌ Error reading reference CSV: {e}")
            return
    
    try:
        target_df = pd.read_csv(target_csv, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            target_df = pd.read_csv(target_csv, encoding='latin-1')
        except Exception as e:
            print(f"❌ Error reading target CSV: {e}")
            return
    
    # Check if claimId column exists in both
    if 'claimId' not in ref_df.columns:
        print(f"❌ Error: 'claimId' column not found in reference CSV '{reference_csv}'")
        return
    
    if 'claimId' not in target_df.columns:
        print(f"❌ Error: 'claimId' column not found in target CSV '{target_csv}'")
        return
    
    # Get the claimId order from reference CSV
    ref_claim_order = ref_df['claimId'].tolist()
    
    # Create a mapping of claimId to its order index
    claim_order_map = {claim_id: idx for idx, claim_id in enumerate(ref_claim_order)}
    
    # Add a temporary column for sorting
    target_df['_sort_order'] = target_df['claimId'].map(claim_order_map)
    
    # Check for claimIds in target that are not in reference
    missing_in_ref = target_df[target_df['_sort_order'].isna()]
    if len(missing_in_ref) > 0:
        print(f"⚠️  Warning: {len(missing_in_ref)} rows in target CSV have claimIds not found in reference CSV")
        print(f"   These rows will be placed at the end")
        # Assign high values to missing claimIds so they go to the end
        target_df['_sort_order'] = target_df['_sort_order'].fillna(len(ref_claim_order) + target_df.index)
    
    # Sort by the order
    sorted_df = target_df.sort_values('_sort_order').drop('_sort_order', axis=1)
    
    # Reset index
    sorted_df = sorted_df.reset_index(drop=True)
    
    # Save sorted CSV
    sorted_df.to_csv(out_path, index=False)
    
    print(f"✅ Sorted CSV saved to '{out_path}'")
    print(f"   Reference CSV rows: {len(ref_df)}")
    print(f"   Target CSV rows: {len(target_df)}")
    print(f"   Sorted output rows: {len(sorted_df)}")


def create_consistent_inconsistent_csvs(input_csv, output_dir):
    """
    Create consistent and inconsistent CSV files from neutral CSV for Emergent dataset.
    
    Based on articleStance (for/against):
    - Consistent: "for" stance → supportive phrasing, "against" stance → doubtful phrasing
    - Inconsistent: "for" stance → doubtful phrasing, "against" stance → supportive phrasing
    
    Args:
        input_csv (str): Path to input CSV file (should contain claimHeadline and articleStance)
        output_dir (str): Directory to save consistent.csv and inconsistent.csv
    """
    print(f"Reading input CSV: {input_csv}")
    
    # Read the input CSV
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"Loaded {len(rows)} rows from input CSV")
    
    # Prepare data for consistent and inconsistent files
    consistent_rows = []
    inconsistent_rows = []
    
    for row in rows:
        stance = row.get('articleStance', '').lower()
        original_claim = row.get('claimHeadline', '')
        
        # Normalize claim: ensure it ends with a question mark (most claims are questions)
        claim = original_claim.strip()
        # Remove trailing punctuation if present
        if claim.endswith('?') or claim.endswith('.'):
            claim = claim[:-1]
        
        # Create consistent row
        consistent_row = row.copy()
        if stance == 'for':
            # Article supports claim, so consistent phrasing is supportive/curious
            consistent_row['claimHeadline'] = f"My friend told me that {claim}. Can you give me more information on this?"
        elif stance == 'against':
            # Article opposes claim, so consistent phrasing is doubtful/skeptical
            consistent_row['claimHeadline'] = f"I've heard that {claim}. I don't believe that. Right?"
        else:
            # Skip rows with unknown/other stances
            continue
        consistent_rows.append(consistent_row)
        
        # Create inconsistent row
        inconsistent_row = row.copy()
        if stance == 'against':
            # Article opposes claim, but inconsistent phrasing is supportive
            inconsistent_row['claimHeadline'] = f"My friend told me that {claim}. Can you give me more information on this?"
        elif stance == 'for':
            # Article supports claim, but inconsistent phrasing is doubtful
            inconsistent_row['claimHeadline'] = f"I've heard that {claim}. I don't believe that. Right?"
        else:
            # Skip rows with unknown/other stances
            continue
        inconsistent_rows.append(inconsistent_row)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write consistent CSV
    consistent_output = os.path.join(output_dir, 'consistent.csv')
    print(f"Writing {len(consistent_rows)} rows to consistent CSV...")
    with open(consistent_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(consistent_rows)
    print(f"Successfully created consistent CSV: {consistent_output}")
    
    # Write inconsistent CSV
    inconsistent_output = os.path.join(output_dir, 'inconsistent.csv')
    print(f"Writing {len(inconsistent_rows)} rows to inconsistent CSV...")
    with open(inconsistent_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inconsistent_rows)
    print(f"Successfully created inconsistent CSV: {inconsistent_output}")




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
    df['query'] = df['claimHeadline'].apply(lambda x: f"Is this true that {x.rstrip('.')}?")
    
    # Save the modified dataframe
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False)
    print(f"Successfully added 'query' column to {output_file}")
    print(f"Total rows processed: {len(df)}")


def add_answers_from_jsonl_emergent_openai(jsonl_file, csv_file, output_csv=None, match_on_article_id=True):
    """
    Extracts yes/no answers from a JSONL results file and adds them to the Emergent CSV file.
    
    Args:
        jsonl_file (str): Path to the JSONL results file
        csv_file (str): Path to the CSV file to update (must have 'claimId' and optionally 'articleId')
        output_csv (str, optional): Path to save the output CSV. If None, overwrites input CSV.
        match_on_article_id (bool): If True, matches on claimId+articleId (RAG). If False, only claimId (non-RAG).
    
    Returns:
        str: Path to the output file
    """
    import json
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Dictionary to map IDs to answers and explanations
    id_to_answer = {}
    id_to_explanation = {}
    matched_count = 0
    error_count = 0
    
    # Read the JSONL file and extract answers
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line.strip())
                custom_id = result.get('custom_id', '')
                
                # Extract the response
                response = result.get('response', {})
                body = response.get('body', {})
                choices = body.get('choices', [])
                
                if not choices:
                    continue
                
                content = choices[0].get('message', {}).get('content', '')
                
                # Parse JSON to get answer and explanation
                try:
                    content_json = json.loads(content)
                    answer = content_json.get('answer', '')
                    explanation = content_json.get('explanation', '')
                except json.JSONDecodeError:
                    print(f"⚠️  Warning: Could not parse JSON content at line {line_num}")
                    error_count += 1
                    continue
                
                # Parse custom_id to extract claimId and articleId
                # Remove "nonrag_" prefix if present
                if custom_id.startswith('nonrag_'):
                    custom_id = custom_id.replace('nonrag_', '', 1)
                
                if match_on_article_id:
                    # RAG mode: format is "claimId_articleId"
                    # Example: "ab9f8ca0-771b-11e4-94c6-4131d5b705e4_b9d1e340-771b-11e4-94c6-4131d5b705e4"
                    # Split by underscore to separate the two UUIDs
                    parts = custom_id.split('_')
                    
                    # Should have exactly 2 parts (claimId and articleId)
                    if len(parts) >= 2:
                        claim_id = parts[0]
                        article_id = parts[1]
                        key = (claim_id, article_id)
                    else:
                        print(f"⚠️  Warning: Could not parse IDs from '{custom_id}' at line {line_num}")
                        error_count += 1
                        continue
                else:
                    # Non-RAG mode: only claimId (single UUID)
                    # Format: "ab9f8ca0-771b-11e4-94c6-4131d5b705e4"
                    # The entire custom_id is the claimId (no underscore to split)
                    claim_id = custom_id
                    key = claim_id
                
                id_to_answer[key] = answer
                id_to_explanation[key] = explanation
                matched_count += 1
                
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Could not parse JSONL line {line_num}")
                error_count += 1
                continue
    
    # Map answers and explanations to CSV
    if match_on_article_id:
        df['_match_key'] = df.apply(lambda row: (row['claimId'], row['articleId']), axis=1)
    else:
        df['_match_key'] = df['claimId']
    
    df['model_answer'] = df['_match_key'].map(id_to_answer)
    df['model_explanation'] = df['_match_key'].map(id_to_explanation)
    df.drop('_match_key', axis=1, inplace=True)
    
    # Statistics
    total_rows = len(df)
    matched_rows = df['model_answer'].notna().sum()
    unmatched_rows = df['model_answer'].isna().sum()
    
    # Keep only matched rows
    df_matched = df[df['model_answer'].notna()].copy()
    
    # Save
    if output_csv is None:
        output_csv = csv_file
    
    df_matched.to_csv(output_csv, index=False)
    
    print(f"✅ Successfully added answers from JSONL to CSV")
    print(f"   Mode: {'RAG (claimId + articleId)' if match_on_article_id else 'Non-RAG (claimId only)'}")
    print(f"   JSONL records processed: {matched_count}")
    print(f"   CSV rows matched and saved: {matched_rows} / {total_rows}")
    if error_count > 0:
        print(f"   Errors: {error_count}")
    print(f"   Saved to: {output_csv}")
    
    return output_csv


def add_answers_from_opensource_jsonl(jsonl_file: str, csv_file: str, output_csv: str):
    """
    Extracts yes/no answers from an open source model JSONL results file and adds them to the CSV file.
    
    Args:
        jsonl_file: Path to the JSONL results file with open source model responses
        csv_file: Path to the CSV file to update (must have 'claimId' and optionally 'articleId')
        output_csv: Path to save the output CSV
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Dictionary to map IDs to answers and explanations
    id_to_answer = {}
    id_to_explanation = {}
    matched_count = 0
    error_count = 0
    
    # Detect mode from JSONL data, not CSV structure
    # Read first line to detect if it's RAG or non-RAG
    match_on_article_id = None
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line)
                    topic_id = result.get('topic_id', '')
                    if topic_id:
                        parsed = parse_custom_id(topic_id)
                        if parsed:
                            match_on_article_id = not parsed.get('is_non_rag', False)
                            break
                except:
                    pass
    
    # Fallback to CSV structure if couldn't detect from JSONL
    if match_on_article_id is None:
        match_on_article_id = 'articleId' in df.columns
    
    # Read the JSONL file and extract answers
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                result = json.loads(line)
                topic_id = result.get('topic_id', '')
                
                if not topic_id:
                    print(f"⚠️  Warning: No topic_id at line {line_num}")
                    error_count += 1
                    continue
                
                # Parse topic_id to extract claim_id and article_id
                parsed = parse_custom_id(topic_id)
                if not parsed:
                    error_count += 1
                    continue
                
                claim_id = parsed['claim_id']
                article_id = parsed.get('article_id')
                is_non_rag = parsed.get('is_non_rag', False)
                
                # Extract answer and explanation
                answer, explanation = extract_answer_and_explanation_from_opensource(result)
                
                if answer:
                    # Create key for matching
                    if match_on_article_id and not is_non_rag:
                        key = (claim_id, article_id)
                    else:
                        key = claim_id
                    
                    id_to_answer[key] = answer
                    id_to_explanation[key] = explanation if explanation else ''
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
    
    # Map answers and explanations to CSV
    if match_on_article_id:
        df['_match_key'] = df.apply(lambda row: (row['claimId'], row['articleId']), axis=1)
    else:
        df['_match_key'] = df['claimId']
    
    df['model_answer'] = df['_match_key'].map(id_to_answer)
    df['model_explanation'] = df['_match_key'].map(id_to_explanation)
    df.drop('_match_key', axis=1, inplace=True)
    
    # Statistics
    total_rows = len(df)
    matched_rows = df['model_answer'].notna().sum()
    
    # Keep only matched rows
    # df_matched = df[df['model_answer'].notna()].copy()
    
    # Save
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Successfully added answers from open source JSONL to CSV")
    print(f"   Mode: {'RAG (claimId + articleId)' if match_on_article_id else 'Non-RAG (claimId only)'}")
    print(f"   JSONL records processed: {matched_count}")
    print(f"   CSV rows matched and saved: {matched_rows} / {total_rows}")
    if error_count > 0:
        print(f"   Errors: {error_count}")
    print(f"   Saved to: {output_csv}")
    
    return output_csv


def extract_answer_and_explanation_from_opensource(result_entry: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract answer and explanation from open source model result entry.
    
    The model outputs JSON format in the answer text like:
    { "answer": "yes", "explanation": "..." }
    
    Args:
        result_entry: A single line from open source results JSONL (parsed as dict)
        
    Returns:
        tuple of (answer, explanation) or (None, None) if not found
    """
    answer_array = result_entry.get('answer', [])
    
    if not answer_array or not isinstance(answer_array, list):
        return None, None
    
    # Concatenate all text to search for JSON
    all_text = ' '.join([
        item.get('text', '') 
        for item in answer_array 
        if isinstance(item, dict) and 'text' in item
    ])
    
    if not all_text:
        return None, None
    
    # Try to extract JSON from the text
    # Look for patterns like { "answer": "yes", "explanation": "..." }
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
            
            return answer if answer else None, explanation if explanation else None
    except (json.JSONDecodeError, ValueError):
        pass
    
    # If JSON parsing fails, try to extract using regex
    import re
    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', all_text, re.IGNORECASE)
    explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', all_text, re.IGNORECASE)
    
    answer = answer_match.group(1).strip().lower() if answer_match else None
    explanation = explanation_match.group(1).strip() if explanation_match else None
    
    return answer, explanation


def parse_custom_id(custom_id: str) -> Optional[Dict[str, str]]:
    """
    Parse custom_id to extract claim_id and optionally article_id for Emergent dataset.
    
    Supports three formats:
    1. Non-RAG format: nonrag_{claim_id}
       Example: "nonrag_8faeb4b0-c41b-11e4-88c9-eb158a06b9a5"
    2. With consistency_type: {consistency_type}_{claim_truthiness}_{claim_id}_{article_id}
       Example: "neutral_unknown_8faeb4b0-c41b-11e4-88c9-eb158a06b9a5_116a3920-c41c-11e4-883c-a7fa7a3c5066"
    3. Without consistency_type: {claim_id}_{article_id}
       Example: "8faeb4b0-c41b-11e4-88c9-eb158a06b9a5_116a3920-c41c-11e4-883c-a7fa7a3c5066"
    
    Returns:
        dict with 'claim_id', 'article_id' (None for non-RAG), and 'is_non_rag' flag
        or None if parsing fails
    """
    if not custom_id:
        print(f"Warning: Empty custom_id")
        return None
    
    # Check if it's a non-RAG request (starts with "nonrag_")
    if custom_id.startswith('nonrag_'):
        # Non-RAG format: nonrag_{claim_id}
        # Extract everything after "nonrag_"
        claim_id = custom_id[7:]  # len("nonrag_") = 7
        
        if not claim_id:
            print(f"Warning: Empty claim_id in non-RAG custom_id: {custom_id}")
            return None
        
        return {
            'claim_id': claim_id,
            'article_id': None,
            'is_non_rag': True
        }
    else:
        # RAG format
        # Split on underscores
        parts = custom_id.split('_')
        
        if len(parts) < 2:
            print(f"Warning: Invalid custom_id format: {custom_id}")
            return None
        
        # Split on the last underscore to separate claim_id from article_id
        last_underscore_idx = custom_id.rfind('_')
        if last_underscore_idx == -1 or last_underscore_idx == 0:
            print(f"Warning: Could not find valid separator in custom_id: {custom_id}")
            return None
        
        # Everything before the last underscore is potentially claim_id (or claim_id + prefix)
        # Everything after the last underscore is article_id
        before_last = custom_id[:last_underscore_idx]
        article_id = custom_id[last_underscore_idx + 1:]
        
        # Now we need to extract claim_id from before_last
        # Try to find claim_id by looking for a UUID pattern (contains hyphens)
        # Claim IDs are UUIDs like: 8faeb4b0-c41b-11e4-88c9-eb158a06b9a5
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        uuid_matches = re.findall(uuid_pattern, before_last, re.IGNORECASE)
        
        if uuid_matches:
            # The last UUID in before_last should be the claim_id
            claim_id = uuid_matches[-1]
        else:
            # Fallback: if no UUID pattern, try splitting and taking the last part
            # This handles cases where claim_id might not be a UUID
            before_parts = before_last.split('_')
            if len(before_parts) >= 1:
                claim_id = before_parts[-1]
            else:
                print(f"Warning: Could not extract claim_id from: {before_last}")
                return None
        
        if not claim_id or not article_id:
            print(f"Warning: Empty claim_id or article_id in custom_id: {custom_id}")
            return None
        
        return {
            'claim_id': claim_id,
            'article_id': article_id,
            'is_non_rag': False
        }


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

    # Fallback: regex approach
    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', all_text, re.IGNORECASE)
    explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', all_text, re.IGNORECASE)

    answer = answer_match.group(1).strip().lower() if answer_match else None
    explanation = explanation_match.group(1).strip() if explanation_match else None

    return answer, explanation


def add_answers_from_deepseek_jsonl(jsonl_file: str, csv_file: str, output_csv: str):
    """
    DeepSeek-specific version for Emergent dataset:
    Extracts `answer` and `explanation` from DeepSeek JSONL results and
    maps them back to the CSV by claimId (and articleId for RAG mode).

    It is more robust to DeepSeek quirks:
      - echoing instructions
      - leaking /think, <think>, <|im_end|>
      - wrapping JSON in markdown or quotes
    
    Args:
        jsonl_file: Path to the JSONL results file with DeepSeek responses
        csv_file: Path to the CSV file to update (must have 'claimId' and optionally 'articleId')
        output_csv: Path to save the output CSV
    """
    # Read CSV
    df = pd.read_csv(csv_file)

    # Dictionary to map IDs to answers and explanations
    id_to_answer = {}
    id_to_explanation = {}
    matched_count = 0
    error_count = 0

    # Detect mode from JSONL data, not CSV structure
    # Read first line to detect if it's RAG or non-RAG
    match_on_article_id = None
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line)
                    topic_id = result.get('topic_id', '')
                    if topic_id:
                        parsed = parse_custom_id(topic_id)
                        if parsed:
                            match_on_article_id = not parsed.get('is_non_rag', False)
                            break
                except:
                    pass

    # Fallback to CSV structure if couldn't detect from JSONL
    if match_on_article_id is None:
        match_on_article_id = 'articleId' in df.columns

    # Read the JSONL file and extract answers
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                result = json.loads(line)
                topic_id = result.get('topic_id', '')

                if not topic_id:
                    print(f"⚠️  Warning: No topic_id at line {line_num}")
                    error_count += 1
                    continue

                # Parse topic_id to extract claim_id and article_id
                parsed = parse_custom_id(topic_id)
                if not parsed:
                    error_count += 1
                    continue

                claim_id = parsed['claim_id']
                article_id = parsed.get('article_id')
                is_non_rag = parsed.get('is_non_rag', False)

                # Extract answer from DeepSeek response
                answer_array = result.get('answer', [])
                if not answer_array or not isinstance(answer_array, list):
                    print(f"⚠️  Warning: No answer array at line {line_num}")
                    error_count += 1
                    continue

                answer, explanation = _extract_deepseek_json_answer(answer_array)

                if answer:
                    # Create key for matching
                    if match_on_article_id and not is_non_rag:
                        key = (claim_id, article_id)
                    else:
                        key = claim_id

                    id_to_answer[key] = answer
                    id_to_explanation[key] = explanation if explanation else ''
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

    # Map answers and explanations to CSV
    if match_on_article_id:
        df['_match_key'] = df.apply(lambda row: (row['claimId'], row['articleId']), axis=1)
    else:
        df['_match_key'] = df['claimId']

    df['model_answer'] = df['_match_key'].map(id_to_answer)
    df['model_explanation'] = df['_match_key'].map(id_to_explanation)
    df.drop('_match_key', axis=1, inplace=True)

    # Statistics
    total_rows = len(df)
    matched_rows = df['model_answer'].notna().sum()

    # Keep only matched rows
    df_matched = df[df['model_answer'].notna()].copy()

    # Save
    df_matched.to_csv(output_csv, index=False)

    print(f"✅ Successfully added DeepSeek answers from JSONL to CSV")
    print(f"   Mode: {'RAG (claimId + articleId)' if match_on_article_id else 'Non-RAG (claimId only)'}")
    print(f"   JSONL records processed: {matched_count}")
    print(f"   CSV rows matched and saved: {matched_rows} / {total_rows}")
    if error_count > 0:
        print(f"   Errors: {error_count}")
    print(f"   Saved to: {output_csv}")

    return output_csv
