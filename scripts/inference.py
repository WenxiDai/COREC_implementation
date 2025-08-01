import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import random
from tqdm import tqdm
import json, argparse
import numpy as np
import torch.nn.functional as F
import fire
from typing import List, Dict

def set_random_seed(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_NAME   = "meta-llama/Meta-Llama-3-8B"
ADAPTER_DIR  = os.path.join(PROJECT_ROOT, "outputs_electronics", "checkpoint-44899")
DS_CONFIG    = os.path.join(PROJECT_ROOT, "configs", "ds_zero3_a5000.json")
OUT_DIR      = os.path.join(PROJECT_ROOT, "inference_outputs")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "Electronics", "LlamaRec", "test.jsonl")

set_random_seed(42)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tok.pad_token = tok.eos_token
# Add custom special tokens before any tokenization
SPECIAL_TOKENS = {"additional_special_tokens": ["<price>", "<brand>", "<rank>"]} # Add price token. 2 ctrl: add "<brand>"
num_added = tok.add_special_tokens(SPECIAL_TOKENS)   # returns number of new tokens
print(f"Added {num_added} special tokens.")

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto")
base.resize_token_embeddings(len(tok))

model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_local=True)
model = model.merge_and_unload()
model.eval()

def infer(
    data_path: str = DATA_PATH,
    out_path: str = "inference_outputs/electronics_1ctrl/LlamaRec_predictions_top6_score_ver0.jsonl",
    batch_size: int = 1,
    top_k: int = 6,
    seed: int = 42,
):

    # ---------- Load evaluation data ----------
    with open(data_path, "r", encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f]

    prompts = []
    for record in raw_records:
        prompts.append(record["text"])   
    
    cand_token_ids = torch.tensor(
        [tok.encode(ch, add_special_tokens=False)[0] for ch in "ABCDEF"],
        device=model.device
    )

    # ---------- Batched inference ----------
    results: List[Dict] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Infer"):
        batch_prompts = prompts[start : start + batch_size]
        batch_tok = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            out = model(**batch_tok)  # logits shape: (B, L, V)
            """
            print("dtype:", out.logits.dtype)
            print("global max |logits|:", out.logits.abs().max())
            exit()
            """
            next_logits = out.logits[:, -1, :]  # only the first generated token
            indices = cand_token_ids.unsqueeze(0).expand(next_logits.size(0), -1)  # (B, 6)
            scores_fp32 = next_logits.gather(1, indices)
            
            topk_probs, topk_idx = torch.topk(scores_fp32, k=top_k, dim=-1)
            
            
        for prmpt, idx, p in zip(batch_prompts, topk_idx, topk_probs):
            token_ids = cand_token_ids[idx].cpu().tolist()
            tokens = tok.convert_ids_to_tokens(token_ids)
            results.append(
                {
                    "prompt": prmpt,
                    "topk_tokens": tokens,
                    "topk_probs": p.cpu().tolist(),
                    "pred_token": tokens[0],
                }
            )
            
            
    # ---------- Persist results ----------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved {len(results)} predictions → {out_path}")

if __name__ == "__main__":
    fire.Fire(infer)