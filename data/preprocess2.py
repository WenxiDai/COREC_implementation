import json
import pandas as pd
import ast
from tqdm import tqdm
import fire
import random
import re
import os
import pathlib # 引入 pathlib

base_path = "data/Electronics/"
CONTROL_MODES = []


def build_llama_style_prompt(input_prompt: str, output_scores: list) -> str:
    output_str = str(output_scores)
    return f"{input_prompt} {output_str}"

def load_predefined_candidates(file_path: str) -> dict:
    cand_map = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                user_id, win_id, *asins = parts
                cand_map[(user_id, int(win_id))] = asins
    except FileNotFoundError:
        print(f"Warning: Candidate file not found at {file_path}. Proceeding with empty candidates.")
    return cand_map

def template(sys: str, user: str) -> str:
    template = (
        "<|begin_of_text|>{sys}{user}<|end_of_text|>"
    )
    return template.format(sys=sys, user=user)

def load_flattened_meta(meta_path: str) -> pd.DataFrame:
    with open(meta_path, "r") as f:
        raw_meta = json.load(f)

    meta_list = []
    for asin, info in raw_meta.items():
        if all(k in info for k in ["price", "description", "brand"]):
            meta_list.append({
                "asin": asin,
                "price": info["price"],
                "description": info["description"],
                "brand": info["brand"],
                "rank": info.get("rank", None),
                "title": info.get("title", "No Title"),
            })

    df_meta = pd.DataFrame(meta_list)
    assert "asin" in df_meta.columns
    df_meta['binned_price'] = df_meta['price'].apply(bin_price)
    df_meta['binned_rank'] = df_meta['rank'].apply(bin_rank)
    return df_meta

def bin_price(price_val) -> str:
    try:
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
    """
    elif p < 250: return "100-250"
    elif p < 500: return "250-500"
    elif p < 1000: return "500-1000"
    else: return "1000+"
    """

def bin_rank(rank_str) -> str:
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

def build_history_item_repr(item_series: pd.Series, control_modes: list) -> dict:
    price_str = f"<price> {item_series.get('binned_price')}" if "price" in control_modes else f"price ${item_series.get('price', '')}"
    brand_raw = str(item_series.get("brand", "")).strip()
    brand_str = f"<brand> {brand_raw}" if "brand" in control_modes and brand_raw else f"brand {brand_raw}"
    rank_str = f"<rank> {item_series.get('binned_rank')}" if "rank" in control_modes else f"rank {item_series.get('rank', '')}"
    title = str(item_series.get("title", ""))[:150]
    desc = str(item_series.get("description", ""))[:150]
    #rating = str(item_series.get("overall", ""))
    
    full_str = (f"Previously bought item with {price_str}, {rank_str} in its own category. Title: '{title}'. "
                f"Description: '{desc}'.") # User giving {rating} rating. ###, 如果有别的control 加上：{brand_str}, {rank_str} in its own category
    
    return {
        "asin": item_series.get("asin", ""),
        "full_str": full_str,
        "attributes": {"price": item_series.get('binned_price'), "brand": brand_raw, "rank": item_series.get('binned_rank')}
    }

def build_candidate_item_repr(item_series: pd.Series, control_modes: list) -> dict:
    price_str = f"<price> {item_series.get('binned_price')}" if "price" in control_modes else f"price ${item_series.get('price', '')}"
    brand_raw = str(item_series.get("brand", "")).strip()
    brand_str = f"<brand> {brand_raw}" if "brand" in control_modes and brand_raw else f"brand {brand_raw}"
    rank_str = f"<rank> {item_series.get('binned_rank')}" if "rank" in control_modes else f"rank {item_series.get('rank', '')}"
    title = str(item_series.get("title", ""))[:150]
    desc = str(item_series.get("description", ""))[:150]
    rating = str(item_series.get("overall", ""))
    
    full_str = (f"This item has id {item_series.get('asin', '')}, {price_str}, {brand_str}, "
                f"{rank_str} in its own category. Title: '{title}'. "
                f"Description: '{desc}'.")
    return {
        "asin": item_series.get("asin", ""),
        "full_str": full_str,
        "attributes": {"price": item_series.get('binned_price'), "brand": brand_raw, "rank": item_series.get('binned_rank')}
    }


    
def get_control_tokens_and_flag(control_modes: list, target_item_series: pd.Series, predefined_asins: list, meta_dict: dict) -> tuple:

    @lru_cache(maxsize=None)
    def get_row(asin: str):
        return meta_dict.get(asin)
    control_tokens = []
    positive_in_predefined = target_item_series['asin'] in predefined_asins
    flag = 1 if positive_in_predefined else 0

    if flag == 1:
        if "price" in control_modes: control_tokens.append(f"<price> {target_item_series.get('binned_price')}")
        if "rank" in control_modes: control_tokens.append(f"<rank> {target_item_series.get('binned_rank')}")
        if "brand" in control_modes: control_tokens.append(f"<brand> {str(target_item_series.get('brand', '')).strip()}")
    else:
        neg_rows = [get_row(a) for a in predefined_asins if get_row(a)]
        if neg_rows:
            for mode in ["price", "rank", "brand"]:
                if mode in control_modes:
                    field = f"binned_{mode}" if mode != "brand" else "brand"

                # Filtering loss / Unknown
                    valid_rows = [r for r in neg_rows
                              if r and r.get(field) not in (None, "Unknown", "")]
                    if not valid_rows:
                        continue

                    attr_val = random.choice(valid_rows).get(field)
                    control_tokens.append(f"<{mode}> {str(attr_val).strip()}")

    return [tok for tok in control_tokens if tok and tok.split()[-1] not in ["Unknown", ""]], flag

def calculate_scores(candidates: list, control_tokens: list, gt_asin: str) -> tuple:
    scores = []
    gt_index = -1
    for i, cand in enumerate(candidates):
        score = 0
        for token in control_tokens:
            match = re.match(r"<(\w+)> (.+)", token)
            if match:
                attr_type, attr_value = match.groups()
                if cand['attributes'].get(attr_type) == attr_value:
                    score += 1
        scores.append(score)
        #print(f"Candidate {i}: {cand['asin']} - Score: {score}", gt_asin)
        if cand['asin'] == gt_asin: gt_index = i ###
    if gt_index != -1: scores[gt_index] += 1
    return scores, gt_index



from functools import lru_cache


import cProfile, pstats, io

def process_and_score_data(history_df, target_df, meta_df, out_path, candidate_map, window_size=6, num_negatives=6):
    print(f"--- Processing, Scoring, and Writing Llama-style prompts to: {out_path} ---")
    count = 0
    meta_dict = {row['asin']: row for row in meta_df.to_dict(orient='records')}
    all_asins = list(meta_dict)

    def sample_negs(exclude: set, k: int):
        pool = [a for a in all_asins if a not in exclude]
        return random.sample(pool, k) if k <= len(pool) else pool
    
    @lru_cache(maxsize=None)
    def full_str_cached(asin):
        row = meta_dict.get(asin)
        if row is None:
            return ""
        return build_history_item_repr(row, CONTROL_MODES)['full_str']
    @lru_cache(maxsize=None)
    def cand_repr_cached(asin):
        row = meta_dict.get(asin)
        if row is None:
            return None
        return build_candidate_item_repr(row, CONTROL_MODES)
    
    is_train_processing = history_df.equals(target_df)

    with open(out_path, "w", encoding="utf-8") as fout:
        for reviewer_id, target_group in tqdm(target_df.groupby("reviewerID"), desc="Processing Users"):
            history_records_df = history_df[history_df['reviewerID'] == reviewer_id]
            combined_df = pd.concat([history_records_df, target_group]).drop_duplicates(subset=['source', 'original_order'])
            combined_df_sorted = combined_df.sort_values(by=['source', 'original_order'])
            combined_records = combined_df_sorted.to_dict(orient='records')
            num_history_records = len(history_records_df)

            if is_train_processing:
                loop_range = range(len(combined_records) - window_size + 1)
            else:
                loop_range = range(num_history_records, len(combined_records))
            
            user_window_id = 0
            for i in loop_range:
                history_slice, target_record = (combined_records[i: i + window_size - 1], combined_records[i + window_size - 1]) if is_train_processing else (combined_records[max(0, i - window_size): i], combined_records[i])
                
                
                if not history_slice: 
                    print("SKIP_EMPTY_HISTORY", reviewer_id, target_record["asin"])
                    continue

                history_asins = [rec['asin'] for rec in history_slice]

             
                history_strs = []
                

                history_strs = [full_str_cached(a) for a in history_asins]
                
                user_prompt_text = f" User History of {reviewer_id}: {';'.join(history_strs)}\n" ###改过的，ID版本：原本： 空格of {reviewer_id}
                key = (reviewer_id, i + window_size - 1) if is_train_processing else (reviewer_id, user_window_id)
                predefined_cands = candidate_map.get(key, [])
                user_window_id += 1

                control_tokens, flag = get_control_tokens_and_flag(CONTROL_MODES, target_series, predefined_cands, meta_dict)
                user_prompt_text += f"Control Tokens: {' ||| '.join(filter(None, control_tokens))}\n"

                pos_item = build_candidate_item_repr(target_series, CONTROL_MODES)

                primary_ids = []      # complete meta
                secondary_ids = []    # partial meta

                for a in predefined_cands:
                    meta_row = meta_dict.get(a)
                    if meta_row is None:
                        continue  # skip items not present in meta at all

                    has_all_fields = all(
                        meta_row.get(k) not in (None, "", "Unknown")
                        for k in ["price", "description", "brand"]
                    )

                    if has_all_fields:
                        primary_ids.append(a)
                    else:
                        secondary_ids.append(a)

                # Step‑1: take as many fully‑qualified negatives as possible
                neg_ids = primary_ids[:num_negatives]

                # Step‑2: top up with partially‑qualified negatives
                if len(neg_ids) < num_negatives:
                    need = num_negatives - len(neg_ids)
                    neg_ids.extend(secondary_ids[:need])

                # Step‑3: still short? random sample from global pool
                if len(neg_ids) < num_negatives:
                    print(f"Not enough negative candidates for {reviewer_id} at window {i + window_size - 1}")
                    need = num_negatives - len(neg_ids)
                    neg_ids += sample_negs(set(neg_ids) | {target_record["asin"]}, need)

                # Ensure final length is exactly `num_negatives`
                neg_ids = neg_ids[:num_negatives]
                if len(neg_ids) < num_negatives:
                    print("not enough negative candidates")
                    exit()

                neg_items = [
                    cand_repr_cached(a)
                    for a in neg_ids
                    if cand_repr_cached(a) is not None 
                ]

                all_candidates = neg_items
                random.shuffle(all_candidates)

                scores, gt_index = calculate_scores(all_candidates, control_tokens, pos_item['asin'])
                
                candidate_block = "Candidates:\n" + "\n".join([f"{chr(65 + i)}. {c['full_str']}" for i, c in enumerate(all_candidates)])
                user_prompt_text += candidate_block
                user_prompt_text += "\nBased on the sequential pattern and control tokens, choose the item that the user is most likely to interact with next. Respond with just one letter (A to F)."
                
                system_prompt_text = "Given the user's interaction history, control tokens, and a list of candidates,  pick **ONE** item that satisfies the control tokens. Return **ONLY** the candidate letter with no additional text."

                full_input_prompt = template(system_prompt_text, user_prompt_text)
                
                json_obj = {
                    "text": full_input_prompt,
                    "label": gt_index,
                    "positive_in_cands_flag": flag,
                    "scores": scores
                }
                count += 1

                fout.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

def main(control_modes="price,brand,rank"):
    global CONTROL_MODES
    CONTROL_MODES = [x.strip() for x in control_modes.split(",") if x.strip()]
    print(f"Using control modes: {CONTROL_MODES}")

    
    ctrl_str = "_".join(sorted(CONTROL_MODES)) if CONTROL_MODES else "no"
    out_dir = f"{base_path}{ctrl_str}_ctrl_new_style/"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading all metadata...")
    df_meta = load_flattened_meta(f"{base_path}cleaned_meta_Electronics.json")

    print("\nLoading Train Set...")
    train_path = f"{base_path}train.json"
    df_train = pd.read_json(train_path).reset_index().rename(columns={'index': 'original_order'})
    df_train['source'] = 0 
    
    print("Loading Validation Set...")
    valid_path = f"{base_path}valid.json"
    df_valid = pd.read_json(valid_path).reset_index().rename(columns={'index': 'original_order'})
    df_valid['source'] = 1 

    print("Loading Test Set...")
    test_path = f"{base_path}test.json"
    df_test = pd.read_json(test_path).reset_index().rename(columns={'index': 'original_order'})
    df_test['source'] = 2 

    print("\nProcessing Train Set...")
    cand_map_train = load_predefined_candidates(f"{base_path}sw_mapped_train_k10.txt")
    process_and_score_data(
        history_df=df_train, target_df=df_train, meta_df=df_meta,
        out_path=f"{out_dir}train.jsonl", candidate_map=cand_map_train)

    print("\nProcessing Validation Set...")
    cand_map_valid = load_predefined_candidates(f"{base_path}sw_mapped_valid_k10.txt")
    process_and_score_data(
        history_df=df_train, target_df=df_valid, meta_df=df_meta,
        out_path=f"{out_dir}valid.jsonl", candidate_map=cand_map_valid)

    print("\nProcessing Test Set...")
    cand_map_test = load_predefined_candidates(f"{base_path}sw_mapped_test_k10.txt")
    df_history_for_test = pd.concat([df_train, df_valid], ignore_index=True)
    process_and_score_data(
        history_df=df_history_for_test, target_df=df_test, meta_df=df_meta,
        out_path=f"{out_dir}test.jsonl", candidate_map=cand_map_test)


if __name__ == "__main__":
    random.seed(42)
    fire.Fire(main)