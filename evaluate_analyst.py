"""
CIKT Analyst — Test-Set Evaluation Script

Loads the fine-tuned Analyst (LoRA weights), runs inference on the test split,
and reports:
  1. ROUGE-1 / ROUGE-2 / ROUGE-L  vs. reference profiles (text quality)
  2. Heuristic binary prediction accuracy from the generated profile
     (does the profile correctly flag whether the next question will be
     answered correctly?)  — a lightweight proxy for Predictor utility.

The test split is reproduced from the same seed / ratios used during training
so the same samples are held out.

Usage
-----
conda run -n llmkt python cikt/evaluate_analyst.py \\
    --lora_weights cikt/analyst_ckpt/lora_weights \\
    --base_model   Qwen/Qwen2.5-7B-Instruct \\
    --profiles_dir student_simulation/cikt_profiles_recent_q_8_i_3 \\
    --output_dir   cikt/eval_results

# Debug (2 test samples):
conda run -n llmkt python cikt/evaluate_analyst.py \\
    --lora_weights cikt/analyst_ckpt/lora_weights \\
    --base_model   Qwen/Qwen2.5-7B-Instruct \\
    --profiles_dir student_simulation/cikt_profiles_recent_q_8_i_3 \\
    --output_dir   /tmp/analyst_eval \\
    --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, debug: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "debug" if debug else "eval"
    log_path = output_dir / f"analyst_{suffix}_{timestamp}.log"

    logger = logging.getLogger("analyst_eval")
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


# ---------------------------------------------------------------------------
# Data loading  (mirrors train_analyst.py split logic exactly)
# ---------------------------------------------------------------------------

def load_profiles(profiles_dir: str | Path) -> list[dict]:
    profiles_dir = Path(profiles_dir)
    if not profiles_dir.exists():
        sys.exit(f"[ERROR] profiles_dir not found: {profiles_dir}")
    samples = []
    for path in sorted(profiles_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("profile", "").strip():
            samples.append(rec)
    print(f"Loaded {len(samples)} profile samples from '{profiles_dir}'")
    return samples


def get_test_split(
    all_samples: list[dict],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> list[dict]:
    """Reproduce the same test split used during training."""
    rng = random.Random(seed)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    n_test = max(1, int(len(indices) * test_ratio))
    n_val  = max(1, int(len(indices) * val_ratio))
    test_idx = indices[:n_test]
    return [all_samples[i] for i in test_idx]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(base_model: str, lora_weights: str | None):
    print(f"Loading base model '{base_model}' ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if lora_weights:
        print(f"Loading LoRA weights from '{lora_weights}' ...")
        model = PeftModel.from_pretrained(model, lora_weights)
        model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        lora_weights if lora_weights else base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_input(rec: dict, tokenizer) -> str:
    """Build the prompt-only chat text (no assistant turn)."""
    messages = [
        {"role": "system", "content": rec["system_prompt"]},
        {"role": "user",   "content": rec["prompt"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_profile(
    model,
    tokenizer,
    rec: dict,
    max_new_tokens: int = 1024,
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
) -> str:
    """Generate a profile for one sample using paper Table 7 inference params."""
    prompt_text = build_input(rec, tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(scores[k].fmeasure)
    return {k: sum(v) / len(v) for k, v in agg.items()}


# Keyword sets for heuristic next-question correctness prediction
_INCORRECT_KEYWORDS = re.compile(
    r"\b(challenge|challenging|difficult|struggle|struggl|incorrect|wrong|fail|"
    r"unlikely|risk|error|gap|weak|uncertainty|uncertain|poor|misconception)\b",
    re.IGNORECASE,
)
_CORRECT_KEYWORDS = re.compile(
    r"\b(correct|master|mastered|succeed|success|strong|confident|solid|"
    r"likely correct|well-established|proficien)\b",
    re.IGNORECASE,
)

def _extract_next_question_text(profile: str) -> str:
    """Return the portion of the profile that discusses the next question."""
    # Try to find an explicit Next Question section
    m = re.search(r"(?:Next Question|Projected Next Question)[:\s]+(.*?)(?:\n\n|\Z)",
                  profile, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # Fall back to the last paragraph (synthesis sections typically end with next-q analysis)
    paragraphs = [p.strip() for p in profile.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else profile


def heuristic_predict(profile: str) -> int:
    """Return 1 (predict correct) or 0 (predict incorrect) from profile text."""
    text = _extract_next_question_text(profile)
    n_incorrect = len(_INCORRECT_KEYWORDS.findall(text))
    n_correct   = len(_CORRECT_KEYWORDS.findall(text))
    return 1 if n_correct > n_incorrect else 0


def compute_prediction_accuracy(
    generated_profiles: list[str],
    labels: list[int],
) -> dict:
    preds = [heuristic_predict(p) for p in generated_profiles]
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / len(labels) if labels else 0.0
    # Baseline: predict majority class
    majority = 1 if sum(labels) >= len(labels) / 2 else 0
    majority_acc = sum(majority == l for l in labels) / len(labels) if labels else 0.0
    return {
        "accuracy": acc,
        "majority_baseline": majority_acc,
        "n_pred_correct": sum(preds),
        "n_label_correct": sum(labels),
        "n_total": len(labels),
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIKT Analyst test-set evaluation")

    p.add_argument("--lora_weights", type=str, default=None,
                   help="Path to saved LoRA weights (output_dir/lora_weights). "
                        "If omitted, evaluates the base model without fine-tuning.")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--profiles_dir", type=str,
                   default="/home/shuang/student_simulation/cikt_profiles_recent_q_8_i_3")
    p.add_argument("--extra_profiles_dirs", nargs="*", default=[])

    # Must match the values used in train_analyst.py
    p.add_argument("--seed",       type=int,   default=221)
    p.add_argument("--val_ratio",  type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    # Inference (paper Table 7)
    p.add_argument("--temperature",    type=float, default=0.95)
    p.add_argument("--top_p",          type=float, default=0.7)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--max_new_tokens", type=int,   default=1024)

    p.add_argument("--output_dir", type=str, default="/home/shuang/cikt/eval_results")
    p.add_argument("--debug", action="store_true",
                   help="Run on 2 test samples only.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    logger = setup_logging(out, args.debug)
    logger.info(f"Args: {vars(args)}")

    # ── Load and split data ────────────────────────────────────────────────────
    all_samples = load_profiles(args.profiles_dir)
    for d in args.extra_profiles_dirs:
        all_samples.extend(load_profiles(d))

    test_samples = get_test_split(all_samples, args.seed, args.val_ratio, args.test_ratio)
    logger.info(f"Test set size: {len(test_samples)}")

    if args.debug:
        test_samples = random.Random(args.seed).sample(test_samples, min(2, len(test_samples)))
        logger.info(f"[DEBUG] Using {len(test_samples)} test samples")

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.base_model, args.lora_weights)

    # ── Generate profiles ─────────────────────────────────────────────────────
    logger.info("Generating profiles ...")
    generated, references, labels = [], [], []
    results = []

    for i, rec in enumerate(tqdm(test_samples, desc="Inference")):
        gen = generate_profile(
            model, tokenizer, rec,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        ref   = rec["profile"]
        label = int(rec["next_question"]["correct"])
        pred  = heuristic_predict(gen)

        generated.append(gen)
        references.append(ref)
        labels.append(label)

        nq = rec["next_question"]
        results.append({
            "student_id":  rec.get("student_id"),
            "question_id": nq.get("question_id"),
            "kc":          nq.get("kc"),
            "difficulty":  round(nq.get("difficulty", 0), 3),
            "label":       label,
            "pred":        pred,
            "generated_profile": gen,
            "reference_profile": ref,
        })

        logger.debug(
            f"\n{'='*72}\n"
            f"[Sample {i+1}/{len(test_samples)}] "
            f"student={rec.get('student_id')}  "
            f"qid={nq.get('question_id')}  "
            f"kc={nq.get('kc')}  diff={nq.get('difficulty', 0):.2f}  "
            f"label={label}  pred={pred}\n"
            f"--- REFERENCE ---\n{ref}\n"
            f"--- GENERATED ---\n{gen}"
        )

    # ── Compute metrics ────────────────────────────────────────────────────────
    rouge_metrics = compute_rouge(generated, references)
    pred_metrics  = compute_prediction_accuracy(generated, labels)

    logger.info("=" * 60)
    logger.info("ROUGE (vs reference profiles):")
    for k, v in rouge_metrics.items():
        logger.info(f"  {k:10s}: {v:.4f}")
    logger.info("Heuristic prediction accuracy:")
    for k, v in pred_metrics.items():
        logger.info(f"  {k:22s}: {v}")
    logger.info("=" * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "debug" if args.debug else "eval"

    results_path = out / f"analyst_{suffix}_results_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "rouge": rouge_metrics,
                "prediction": pred_metrics,
                "samples": results,
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
