import json
import re
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm

# --- File Path Configuration ---
# Please ensure these paths are correct
meta_path = 'cleaned_meta/cleaned_beauty.json'  # Product metadata path
in_path = 'candidates_in_sas/beauty/candidates_mapped.txt'  # Path to candidate items
out_path = 'candidates_in_sas/beauty/scores_candidates.txt'  # Output path for mapped results
user_path = 'candidates_in_sas/beauty/test.jsonl'  # User data path

# --- Binning functions provided by user ---
def bin_price(price_val: Any) -> str:
    """Bin prices into defined ranges."""
    try:
        # Normalize string input by removing currency symbols and commas
        if isinstance(price_val, str):
            price_val = price_val.replace("$", "").replace(",", "").strip()
        p = float(price_val)
    except (ValueError, TypeError):
        return "Unknown"
    if p < 10: return "0-10"
    elif p < 15: return "10-15"
    elif p < 25: return "15-25"
    elif p < 50: return "25-50"
    elif p < 100: return "50-100"
    else: return "100+"

def bin_rank(rank_str: Any) -> str:
    """Bin ranks into defined ranges."""
    try:
        match = re.search(r'([\d,]+)', str(rank_str))
        if not match: return "Unknown"
        r = int(match.group(1).replace(',', ''))
    except (ValueError, TypeError):
        return "Unknown"
    if r <= 3: return "top 3"
    elif r <= 5: return "3-5"
    elif r <= 10: return "top 10"
    elif r <= 50: return "11-50"
    elif r <= 100: return "51-100"
    elif r <= 200: return "101-200"
    elif r <= 500: return "201-500"
    else: return "500+"

# --- Data Loading Functions ---
def load_meta_data(path: str) -> Dict[str, Dict[str, Any]]:
    """Efficiently load product metadata from a JSON file."""
    print(f"⏳ Loading metadata from: {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        print(f"✅ Metadata loaded. Total records: {len(meta_data)}")
        return meta_data
    except FileNotFoundError:
        print(f"Error: Metadata file not found at path: {path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {path}")
        return {}

def extract_info_from_text(record: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
    """Extract user_id, control_token, and gt_asin from the 'text' field."""
    text = record.get('text', '')
    user_match = re.search(r'User History of\s+([A-Za-z0-9]+)', text)
    user_id = user_match.group(1) if user_match else None
    control_match = re.search(r'Control Tokens:\s*(.*?)\nCandidates:', text, re.DOTALL)
    control_token_str = control_match.group(1).strip() if control_match else ''
    gt_asin = ''
    label = record.get('label', -1)
    if label >= 0:
        candidate_ids = re.findall(r'This item has id ([A-Z0-9]+)', text)
        if 0 <= label < len(candidate_ids):
            gt_asin = candidate_ids[label]
    return user_id, control_token_str, gt_asin

def load_user_data(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Efficiently load user data."""
    print(f"⏳ Loading user data from: {path}...")
    user_data = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading user data"):
            try:
                record = json.loads(line)
                user_id, control_token_str, gt_asin = extract_info_from_text(record)
                if user_id:
                    user_data[user_id].append({
                        'control_token_str': control_token_str,
                        'gt_asin': gt_asin
                    })
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
    print(f"User data loaded. Total users: {len(user_data)}")
    return user_data

def parse_control_tokens(token_str: str) -> List[Tuple[str, str]]:
    """
    Parse control token strings separated by '|||', e.g.:
        "<price> 100+ ||| <rank> 500+"
    Returns: [("price", "100+"), ("rank", "500+")]
    """
    tokens: List[Tuple[str, str]] = []
    for segment in token_str.split("|||"):
        segment = segment.strip()
        if not segment:
            continue
        match = re.match(r'<\s*([^>]+)\s*>\s*(.+)', segment)
        if match:
            attr_name = match.group(1).strip()
            attr_value = match.group(2).strip()
            tokens.append((attr_name, attr_value))
    return tokens

# --- Scoring Logic ---
def calculate_scores(candidates: List[Dict[str, Any]], control_tokens: List[Tuple[str, str]], gt_asin: str) -> Tuple[List[int], int]:
    """
    Compute a score for each candidate item.
    - Match 'price' and 'rank' using binning functions.
    - Perform exact match for other attributes such as 'brand'.
    - Ground truth item receives an extra +1 score.
    """
    scores = []
    gt_index = -1
    for i, cand in enumerate(candidates):
        score = 0
        cand_attrs = cand.get('attributes', {})
        for attr_type, attr_value_in_token in control_tokens:
            print(f"Processing attribute {attr_type} = {attr_value_in_token} for candidate {cand.get('asin', 'Unknown')}")
            cand_attr_val = cand_attrs.get(attr_type)
            if cand_attr_val is None:
                continue

            match_found = False
            if attr_type == 'price':
                binned_price = bin_price(cand_attr_val)
                if binned_price == attr_value_in_token:
                    match_found = True
            elif attr_type == 'rank':
                binned_rank = bin_rank(cand_attr_val)
                if binned_rank == attr_value_in_token:
                    match_found = True
            else:
                if str(cand_attr_val).strip().lower() == attr_value_in_token.strip().lower():
                    match_found = True

            if match_found:
                score += 1

        scores.append(score)
        if cand.get('asin') == gt_asin:
            gt_index = i

    if gt_index != -1:
        scores[gt_index] += 1
        
    return scores, gt_index

# === Main Program ===
if __name__ == "__main__":
    all_meta_data = load_meta_data(meta_path)
    all_user_data = load_user_data(user_path)

    if not all_meta_data:
        print("Metadata is empty. Program terminated. Please check meta_path and file contents.")
    else:
        print(f"Start processing candidate file: {in_path}...")
        with open(in_path, 'r', encoding='utf-8') as fin, \
             open(out_path, 'w', encoding='utf-8') as fout:

            for line in tqdm(fin, desc="Processing user candidates"):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                u_id, window_id_str = parts[0], parts[1]
                item_asins = parts[2:]

                user_records = all_user_data.get(u_id)
                try:
                    window_id = int(window_id_str)
                    user_record = user_records[window_id] if user_records and 0 <= window_id < len(user_records) else None
                except (ValueError, TypeError):
                    user_record = None

                if not user_record:
                    print(f"Missing or invalid data for user {u_id}, window {window_id_str}. Skipping.")
                    ratings_str = ' '.join(['0'] * len(item_asins))
                    fout.write(f'{u_id}\t{window_id_str}\t{ratings_str}\n')
                    continue

                control_tokens = parse_control_tokens(user_record['control_token_str'])
                gt_asin = user_record['gt_asin']

                candidates_with_meta = [
                    {'asin': asin, 'attributes': all_meta_data.get(asin, {})}
                    for asin in item_asins
                ]

                scores, _ = calculate_scores(candidates_with_meta, control_tokens, gt_asin)
                
                ratings_str = ' '.join(map(str, scores))
                fout.write(f'{u_id}\t{window_id_str}\t{ratings_str}\n')

        print(f'Processing complete! Score file saved to → {out_path}')