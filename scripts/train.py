from pathlib import Path
import os, re, random, torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

# ------------------------- basic config ------------------------- #
ROOT = Path(__file__).resolve().parent.parent
CFG = {
    "model":      os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B"),
    "train_file": ROOT / "data/Beauty/TARS/train.jsonl",
    "valid_file": ROOT / "data/valid.jsonl",
    "ds_cfg":     ROOT / "configs/ds_zero3_a5000.json",
    "out_dir":    ROOT / "outputs_beauty_obf",
    "seed":       42,
}

random.seed(CFG["seed"])

# --------------------------- tokenizer -------------------------- #

tok = AutoTokenizer.from_pretrained(CFG["model"], trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.add_special_tokens({"additional_special_tokens": ["<price>", "<brand>", "<rank>"]})

# Map capital letters to token IDs on-the-fly
LETTER_IDS = {chr(i): tok.encode(chr(i), add_special_tokens=False)[0] for i in range(65, 91)}

# -------------------------- regex helpers ----------------------- #
#  Example line: "A. This item has id B00X123, ..."
_C_PAT = re.compile(r"^\s*([A-Z])\.\s+.*?id\s+([0-9A-Za-z]+)", re.ASCII)
#  Example line: "1. B00X123"
_R_PAT = re.compile(r"^\s*\d+\.\s+([0-9A-Za-z]+)", re.ASCII)


def _preprocess(example):
    """Convert a raw JSONL entry into tokenized tensors."""
    lines = example["text"].splitlines()
    letters, item_ids = [], []
    parsing = False
    for ln in lines:
        if ln.startswith("Candidates:"):
            parsing = True
            continue
        if parsing:
            if not ln.strip():  # blank line ends block
                break
            m = _C_PAT.match(ln)
            if m:
                letters.append(m.group(1))
                item_ids.append(m.group(2))

    scores = example["scores"]
    assert len(letters) == len(scores), "candidate/score length mismatch"

    # strip explicit id mentions to avoid leakage
    cleaned_text = re.sub(r"id\s+\w+,?\s*", "", example["text"])
    enc = tok(
        cleaned_text,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors="pt",
    )
    enc = {k: v.squeeze(0) for k, v in enc.items()}
    enc["cand_tokens"] = [LETTER_IDS[ch] for ch in letters]
    enc["scores"] = scores
    return enc


train_ds = load_dataset("json", data_files=str(CFG["train_file"]))["train"]
train_ds = train_ds.map(_preprocess, remove_columns=["text", "positive_in_cands_flag", "label"])

valid_ds = load_dataset("json", data_files=str(CFG["valid_file"]))["train"]
valid_ds = valid_ds.map(_preprocess, remove_columns=["text"])


# --------------------------- collator --------------------------- #

def _collate(batch):
    model_batch = tok.pad(batch, return_tensors="pt")
    max_c = max(len(b["cand_tokens"]) for b in batch)
    pad_id = tok.pad_token_id
    cand_tok = torch.full((len(batch), max_c), pad_id, dtype=torch.long)
    cand_sco = torch.zeros((len(batch), max_c), dtype=torch.float)

    for i, ex in enumerate(batch):
        m = len(ex["cand_tokens"])
        cand_tok[i, :m] = torch.tensor(ex["cand_tokens"])
        cand_sco[i, :m] = torch.tensor(ex["scores"], dtype=torch.float)

    model_batch["cand_tokens"] = cand_tok
    model_batch["scores"] = cand_sco
    return model_batch


# ----------------------------- model ---------------------------- #

model = AutoModelForCausalLM.from_pretrained(
    CFG["model"], torch_dtype=torch.bfloat16, device_map={"": "cpu"}
)
model.resize_token_embeddings(len(tok))

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.base_model.model.gradient_checkpointing_enable()
model.base_model.model.enable_input_require_grads()


# --------------------------- trainer --------------------------- #

class PairwiseRankTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        cand = inputs.pop("cand_tokens").to(model.device)
        gt = inputs.pop("scores").to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        pred = outputs.logits[:, -1, :].gather(1, cand)

        s_i, s_j = pred.unsqueeze(2), pred.unsqueeze(1)
        g_i, g_j = gt.unsqueeze(2), gt.unsqueeze(1)
        mask = (g_i > g_j).float()
        loss = (-F.logsigmoid(s_i - s_j) * mask).sum() / mask.sum().clamp(min=1e-8)
        return (loss, outputs) if return_outputs else loss


train_args = TrainingArguments(
    output_dir=str(CFG["out_dir"]),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.02,
    bf16=True,
    gradient_checkpointing=True,
    deepspeed=str(CFG["ds_cfg"]),
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=20,
    remove_unused_columns=False,
)

trainer = PairwiseRankTrainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=_collate,
)


if __name__ == "__main__":
    trainer.train()