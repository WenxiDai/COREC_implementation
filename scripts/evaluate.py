import torch
import os
import math
import json
import re
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union
from collections import defaultdict

import fire

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
INFERENCE_PATH = os.path.join(PROJECT_ROOT, "inference_outputs")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "Electronics", "brand_price_rank_ctrl_new_style", "without_ID", "test.jsonl")
TEST_PATH_WITH_USERID = os.path.join(PROJECT_ROOT, "data", "Electronics", "brand_price_rank_ctrl_new_style", "wID", "test.jsonl")
OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_outputs")
CANDIDATE_PATH = os.path.join(PROJECT_ROOT, "data", "Electronics","brand_price_rank_ctrl_new_style/best_idcg_6scores.txt")
CANDIDATE_PATTERN = re.compile(r"^\s*([A-Z])\.\s+This item has id\s+([A-Za-z0-9]+),", re.ASCII)
RANK_PATTERN      = re.compile(r"^\s*\d+\.\s+([A-Za-z0-9]+)", re.ASCII)

set_random_seed(42)


def compute_CHP_at_K(items, K, controls):
    if K <= 0:
        raise ValueError("K must be positive")
    if not controls:
        return 0.0
    top_k = items[:K]
    hits = sum(1 for item in top_k if all(is_satisfied(item, controls)))
    return hits / K

def compute_CSR_at_K(items, K, controls):
    if K <= 0:
        raise ValueError("K must be positive")
    if not controls:
        return 0.0
    top_k = items[:K]
    csr_sum = 0.0
    for c in controls:
        satisfied_count = sum(is_satisfied(item, [c])[0] for item in top_k)
        csr_sum += satisfied_count / K
    return csr_sum / len(controls)
# --------------------------------------------------------------------------- #
#                           Evaluation utilities                              #
# --------------------------------------------------------------------------- #
def _parse_ground_truth(test_file_with_userID: str):
    parsed_results = []
    user_window_counts = defaultdict(int)
    try:
        with open(test_file_with_userID, "r", encoding="utf-8") as f_test:
            
            # The zip function pairs each line from the inference file
            # with the corresponding line from the sequential candidates file.
            for test_line in f_test:
                
                obj = json.loads(test_line)
                scores = obj.get("scores", [])

                text = obj.get("text", "")
                user_match = re.search(r'User History of\s+([A-Za-z0-9]+)', text)
                if user_match:
                    user_id = user_match.group(1)
                    window_id = user_window_counts[user_id]
                    user_window_counts[user_id] += 1
                    
                    parsed_results.append({
                        "user_id": user_id,
                        "window_id": window_id,
                        "scores": scores
                    })
                else:
                    print("No user_id found in line:", test_line)
                    exit()

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
        return []
    except (ValueError, IndexError) as e:
        print(f"Error processing file contents: {e}. Please check file formatting.")
        return []
        
    return parsed_results

def _parse_predictions(inference_file: str, test_file_with_userID: str): 
    parsed_results = []
    user_window_counts = defaultdict(int)
    try:
        with open(inference_file, "r", encoding="utf-8") as f_inference, \
             open(test_file_with_userID, "r", encoding="utf-8") as f_test:
            
            for inference_line, test_line in zip(f_inference, f_test):
                parts = test_line.strip().split()
                obj_test = json.loads(test_line)
                text = obj_test.get("text", "")
                user_match = re.search(r'User History of\s+([A-Za-z0-9]+)', text)
                if user_match:
                    user_id = user_match.group(1)
                    window_id = user_window_counts[user_id]
                    user_window_counts[user_id] += 1
                else:
                    print("No user_id found in line:", test_line)
                    exit()
                
                obj_inference = json.loads(inference_line)
                rankings = obj_inference.get("topk_tokens", [])

                parsed_results.append({
                    "user_id": user_id,
                    "window_id": window_id,
                    "rankings": rankings
                })

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
        return []
    except (ValueError, IndexError) as e:
        print(f"Error processing file contents: {e}. Please check file formatting.")
        return []
        
    return parsed_results

def _rank_single_example(gt_scores_single, predicted_rankings_single):
    original_labels = sorted(predicted_rankings_single) # abcde
    score_map = dict(zip(original_labels, gt_scores_single))
    # Reorder the scores based on the predicted ranking.
    ranked_scores_list = [score_map[label] for label in predicted_rankings_single]
    return ranked_scores_list

def ranked_scores_batch(gt_scores_batch: List[List[float]], predicted_rankings_batch: List[List[str]]) -> List[List[float]]:
    if not gt_scores_batch or not predicted_rankings_batch:
        return []
    return [
        _rank_single_example(gt_scores_single, predicted_rankings_single)
        for gt_scores_single, predicted_rankings_single in zip(gt_scores_batch, predicted_rankings_batch)
    ]


def _parse_ideal_scores():
    ideal_scores_map = {}
    with open(CANDIDATE_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            user_id, window_id_str, scores_str = parts
            window_id = int(window_id_str)
            scores = [float(s) for s in scores_str.split()]

            if user_id not in ideal_scores_map:
                ideal_scores_map[user_id] = {}
            ideal_scores_map[user_id][window_id] = scores
    return ideal_scores_map

def _compute_dcg(relevance_scores: List[float], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        dcg += relevance_scores[i] / math.log2(i + 2)
    return dcg

def _evaluate_dataset(
    preds: List[Dict[str, Union[str, int, List[str]]]],
    gts: List[Dict[str, Union[str, int, List[float]]]],
    ideal_scores_map: Dict[str, Dict[int, List[float]]],
    control_threshold: float = 1.0,
    gt_hit_threshold: float = 2.0
):
    if not preds:
        return {}

    max_k = len(preds[0]['rankings'])
    num_samples = len(preds)
    
    total_ndcg = defaultdict(float)
    total_cp = defaultdict(float)  # Control Precision
    
    total_cd = 0.0  # Control Position of First Hit
    cd_samples_count = 0


    gt_scores_batch = [item['scores'] for item in gts]
    predicted_rankings_batch = [item['rankings'] for item in preds]
    reordered_scores_batch = ranked_scores_batch(gt_scores_batch, predicted_rankings_batch)

    for i in range(num_samples):
        reordered_scores = reordered_scores_batch[i]
        
        # --- CD ---
        first_control_hit_rank = -1
        for rank, score in enumerate(reordered_scores):
            if score >= control_threshold:
                first_control_hit_rank = rank + 1
                break
        if first_control_hit_rank != -1:
            total_cd += first_control_hit_rank
            cd_samples_count += 1
        else:
            total_cd += 7
            cd_samples_count += 1
        user_id = preds[i]['user_id']
        window_id = preds[i]['window_id']
        try:
            ideal_scores = ideal_scores_map[user_id][window_id]
        except KeyError:
            print(f"Warning: Ideal scores for user '{user_id}', window '{window_id}' not found. Skipping nDCG for this sample.")
            ideal_scores = []

        current_dcg = 0.0
        control_hits_count = 0
        
        for k_idx in range(max_k):
            k = k_idx + 1
            score = reordered_scores[k_idx]
            
            # nDCG@k
            if ideal_scores:
                current_dcg += score / math.log2(k + 1)
                idcg_at_k = _compute_dcg(ideal_scores, k)
                if idcg_at_k > 0:
                    total_ndcg[k] += current_dcg / idcg_at_k
            
            # CP@k (Control Precision)
            if score >= control_threshold:
                control_hits_count += 1
            total_cp[k] += control_hits_count / k

    metrics = {}
    
    for k in range(1, max_k + 1):
        metrics[f'nDCG@{k}'] = total_ndcg[k] / num_samples if num_samples > 0 else 0
        metrics[f'CP@{k}'] = total_cp[k] / num_samples if num_samples > 0 else 0
    metrics['CD'] = total_cd / cd_samples_count if cd_samples_count > 0 else 0
    
    return metrics

def _save_metrics(metrics, output_path: str):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Separate per‑k metrics from global metrics
    per_k_metrics = defaultdict(dict)   # {k: {metric_name: value}}
    global_metrics = {}

    for name, value in metrics.items():
        # Check for the pattern "<metric>@<k>"
        if "@" in name and name.rsplit("@", 1)[-1].isdigit():
            metric_name, k_str = name.rsplit("@", 1)
            k = int(k_str)
            per_k_metrics[k][metric_name] = round(float(value), 6)
        else:
            global_metrics[name] = round(float(value), 6)

    # Write metrics to file in a deterministic order
    with open(output_path, "w", encoding="utf-8") as f:
        for k in sorted(per_k_metrics):
            line = {"k": k, **per_k_metrics[k]}
            f.write(json.dumps(line) + "\n")
        if global_metrics:
            line = {"k": "all", **global_metrics}
            f.write(json.dumps(line) + "\n")

def evaluate(inference_file: str = os.path.join(INFERENCE_PATH, "1ctrl", "predictions_top6_score_ver0.jsonl"), 
             test_file: str = TEST_PATH,
             output_file: str = os.path.join(OUT_DIR,"home_and_kitchen_1ctrl","finetuned", "metrics_scores_1ctrl_ver0.jsonl"),
             test_file_with_userID: str = TEST_PATH_WITH_USERID,
             output_file_gt: str = os.path.join(OUT_DIR,"home_and_kitchen_1ctrl","finetuned", "GT_SAS_metrics_scores_1ctrl_ver0.jsonl"),
             ):

    gts = _parse_ground_truth(test_file_with_userID) # candidates，scores
    preds = _parse_predictions(inference_file, test_file_with_userID)
    ideal_scores_map = _parse_ideal_scores()

    if len(gts) != len(preds):
        raise ValueError(f"#ground-truth ({len(gts)}) != #predictions ({len(preds)})")

    metrics = _evaluate_dataset(preds, gts, ideal_scores_map)
    print("Calculated Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    _save_metrics(metrics, output_file)
    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    fire.Fire({"evaluate": evaluate})