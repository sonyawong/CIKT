"""
CIKT Analyst — Stage 1 Distillation Training Script

Supervised fine-tuning of the Analyst backbone on teacher-generated student
profiles.  Consumes the JSON files produced by cikt_profile_generator.py once
a 'profile' field has been added to each file (by the GPT-4o teacher caller).

Each JSON file must have:
  - system_prompt  : str
  - prompt         : str   (user message)
  - profile        : str   (target assistant response)
  - next_question  : dict  (used only for metadata / filtering)

Training follows Table 6 of the paper:
  lora_rank=8, lr=5e-6, epochs=10, warmup_ratio=0.1

Usage
-----
# Using llmkt conda env (has transformers 4.44, peft 0.12, trl 0.9):
conda run -n llmkt python cikt/train_analyst.py \\
    --profiles_dir student_simulation/cikt_profiles_recent_q_8_i_3 \\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
    --output_dir cikt/analyst_ckpt

# Llama variant:
conda run -n llmkt python cikt/train_analyst.py \\
    --profiles_dir student_simulation/cikt_profiles_recent_q_8_i_3 \\
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \\
    --output_dir cikt/analyst_ckpt_llama

# Debug (10 train / 2 val samples, 2 steps, logs data to file):
conda run -n llmkt python cikt/train_analyst.py \\
    --profiles_dir student_simulation/cikt_profiles_recent_q_8_i_3 \\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
    --output_dir /tmp/analyst_debug \\
    --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_profiles(profiles_dir: str | Path) -> list[dict]:
    """Load all JSON profile files from a directory.

    Only files that contain a non-empty 'profile' field are kept as training
    samples; the rest are silently skipped (they are prompt-only files).
    """
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.exists():
        sys.exit(f"[ERROR] profiles_dir not found: {profiles_dir}")

    samples = []
    skipped = 0
    for path in sorted(profiles_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if not rec.get("profile", "").strip():
            skipped += 1
            continue
        samples.append(rec)

    print(f"Loaded {len(samples)} profile samples from '{profiles_dir}' ({skipped} skipped, no profile)")
    if not samples:
        sys.exit("[ERROR] No samples with a 'profile' field found. "
                 "Run the GPT-4o teacher caller first to populate 'profile' fields.")
    return samples


def build_chat_text(rec: dict, tokenizer) -> str:
    """Format a single record as a complete chat conversation string.

    Uses the tokenizer's apply_chat_template when available (modern instruct
    models), otherwise falls back to a plain <s>[INST]…[/INST]…</s> template.
    """
    messages = [
        {"role": "system", "content": rec["system_prompt"]},
        {"role": "user",   "content": rec["prompt"]},
        {"role": "assistant", "content": rec["profile"]},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(samples: list[dict], tokenizer, max_seq_length: int) -> Dataset:
    texts = [build_chat_text(s, tokenizer) for s in samples]
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Model + LoRA setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
):
    print(f"Loading model '{model_name}' ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        # Standard target modules for Qwen2.5 and Llama-3.1
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIKT Analyst SFT training (Stage 1 Distillation)")

    # Data
    p.add_argument(
        "--profiles_dir",
        type=str,
        default="path/to/profiles",
        help="Directory of profile JSON files (must have a 'profile' field).",
    )
    p.add_argument(
        "--extra_profiles_dirs",
        nargs="*",
        default=[],
        help="Additional profile directories to merge into the training set.",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data held out for validation.",
    )
    p.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data held out for test (only split out in debug mode).",
    )
    p.add_argument("--seed", type=int, default=221)

    # Model
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model id or local path.",
    )
    # LoRA  (paper Table 6)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16,
                   help="LoRA alpha (typically 2 × lora_rank).")
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training  (paper Table 6)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Override num_train_epochs with a fixed step count (useful for smoke tests).")

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/shuang/cikt/analyst_ckpt",
    )
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument(
        "--debug",
        action="store_true",
        help="Sample 10 train / 2 val cases, run 2 steps, and log data details to file.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, debug: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "debug" if debug else "train"
    log_path = output_dir / f"analyst_{suffix}_{timestamp}.log"

    logger = logging.getLogger("analyst")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger


def log_samples(logger: logging.Logger, samples: list[dict], split: str) -> None:
    sep = "=" * 72
    logger.debug(f"\n{sep}\n[{split.upper()}] {len(samples)} samples\n{sep}")
    for i, s in enumerate(samples, 1):
        nq = s.get("next_question", {})
        logger.debug(
            f"\n--- {split} sample {i}/{len(samples)} ---\n"
            f"  student_id : {s.get('student_id')}\n"
            f"  history_len: {s.get('history_len')}\n"
            f"  next_q     : qid={nq.get('question_id')}  kc={nq.get('kc')}  "
            f"diff={nq.get('difficulty', 0):.2f}  correct={nq.get('correct')}\n"
            f"  prompt     :\n{s['prompt']}\n"
            f"  profile    :\n{s['profile']}"
        )
    logger.debug(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import random

    args = parse_args()
    out = Path(args.output_dir)
    logger = setup_logging(out, args.debug)
    logger.info(f"Args: {vars(args)}")

    # ── Load profiles ─────────────────────────────────────────────────────────
    all_samples = load_profiles(args.profiles_dir)
    for extra_dir in args.extra_profiles_dirs:
        all_samples.extend(load_profiles(extra_dir))
    logger.info(f"Total samples (before split): {len(all_samples)}")

    # ── Train / val / test split (by index to avoid leakage) ────────────────
    rng = random.Random(args.seed)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)

    n_test = max(1, int(len(indices) * args.test_ratio))
    n_val  = max(1, int(len(indices) * args.val_ratio))
    test_idx  = set(indices[:n_test])
    val_idx   = set(indices[n_test:n_test + n_val])
    train_idx = [i for i in indices[n_test + n_val:]]

    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]
    test_samples  = [all_samples[i] for i in test_idx]
    logger.info(f"Split → train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # ── Debug: subsample and log data ─────────────────────────────────────────
    if args.debug:
        train_samples = rng.sample(train_samples, min(10, len(train_samples)))
        val_samples   = rng.sample(val_samples,   min(2,  len(val_samples)))
        test_samples  = rng.sample(test_samples,  min(2,  len(test_samples)))
        logger.info(f"[DEBUG] Subsampled → train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        log_samples(logger, train_samples, "train")
        log_samples(logger, val_samples,   "val")
        log_samples(logger, test_samples,  "test")

    # ── Model + tokenizer ─────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name_or_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = build_dataset(train_samples, tokenizer, args.max_seq_length)
    val_ds   = build_dataset(val_samples,   tokenizer, args.max_seq_length)

    # ── Training arguments ───────────────────────────────────────────────────
    if args.debug:
        args.num_train_epochs = 1

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_cfg,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("=== Stage 1: Distillation — SFT training ===")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.model.save_pretrained(out / "lora_weights")
    tokenizer.save_pretrained(out / "lora_weights")
    logger.info(f"LoRA weights saved → {out / 'lora_weights'}")

    with open(out / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Training args saved → {out / 'train_args.json'}")


if __name__ == "__main__":
    main()
