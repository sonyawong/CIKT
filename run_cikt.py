"""
Entry point for the CIKT Knowledge Tracing pipeline.

Usage examples
--------------
# Full run on 50 students from combined_data.csv, 3 iterations:
python run_cikt.py --n_students 50 --n_iterations 3

# Demo using only the qa_profile sample data:
python run_cikt.py --data_source qa_profile --n_iterations 2

# Ablation: no profile baseline
python run_cikt.py --n_students 30 --no_profile_baseline

# Fast debug (5 students, 1 iteration):
python run_cikt.py --n_students 5 --n_iterations 1 --debug
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data_loader import (
    load_and_split_combined,
    load_and_split_qa_profile,
    serialize_records,
)
from cikt_pipeline import CIKTConfig, CIKTPipeline
from evaluate import compute_metrics, compare_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIKT Knowledge Tracing")

    # Data
    p.add_argument(
        "--data_source",
        choices=["combined", "qa_profile"],
        default="combined",
        help="Data source: combined_data.csv (default) or qa_profile CSVs.",
    )
    p.add_argument(
        "--n_students",
        type=int,
        default=50,
        help="Number of students to sample from combined_data (ignored for qa_profile).",
    )
    p.add_argument("--seed", type=int, default=42)

    # Model
    p.add_argument("--analyst_model", type=str, default="gpt-4o-mini")
    p.add_argument("--predictor_model", type=str, default="gpt-4o-mini")

    # Pipeline
    p.add_argument("--n_iterations", type=int, default=3, help="Number of KTO iteration rounds.")
    p.add_argument(
        "--iteration_sample_size",
        type=int,
        default=200,
        help="Number of samples used per iteration (paper uses 1000; reduce for speed).",
    )
    p.add_argument("--n_feedback_examples", type=int, default=5)

    # Ablations
    p.add_argument(
        "--no_profile_baseline",
        action="store_true",
        help="Run the no-profile baseline alongside CIKT.",
    )
    p.add_argument(
        "--skip_iteration",
        action="store_true",
        help="Run only stages 1-3 (no iteration).",
    )

    # Output
    p.add_argument("--output_dir", type=str, default="/home/shuang/cikt/results")
    p.add_argument("--cache_dir", type=str, default="/home/shuang/cikt/cache")

    p.add_argument("--debug", action="store_true", help="Minimal run for debugging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("CIKT: Collaborative Iterative Knowledge Tracing")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    print(f"\nLoading data (source={args.data_source}) ...")
    if args.data_source == "qa_profile":
        train_records, valid_records, test_records = load_and_split_qa_profile()
    else:
        n = 5 if args.debug else args.n_students
        train_records, valid_records, test_records = load_and_split_combined(
            n_students=n, seed=args.seed
        )

    print(
        f"  train={len(train_records)}, valid={len(valid_records)}, test={len(test_records)}"
    )

    # Save data splits for reproducibility
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    serialize_records(train_records, out / "train_records.json")
    serialize_records(valid_records, out / "valid_records.json")
    serialize_records(test_records, out / "test_records.json")

    # ----------------------------------------------------------------
    # Build config
    # ----------------------------------------------------------------
    n_iter = 1 if args.debug else (0 if args.skip_iteration else args.n_iterations)
    cfg = CIKTConfig(
        analyst_model=args.analyst_model,
        predictor_model=args.predictor_model,
        n_iterations=n_iter,
        iteration_sample_size=args.iteration_sample_size,
        n_feedback_examples=args.n_feedback_examples,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )

    # ----------------------------------------------------------------
    # Run no-profile baseline (optional)
    # ----------------------------------------------------------------
    baseline_metrics = None
    if args.no_profile_baseline:
        print("\n--- No-Profile Baseline ---")
        pipeline_base = CIKTPipeline(cfg)
        baseline_metrics = pipeline_base.run_no_profile_baseline(test_records)

    # ----------------------------------------------------------------
    # Run full CIKT pipeline
    # ----------------------------------------------------------------
    pipeline = CIKTPipeline(cfg)
    results = pipeline.run(train_records, valid_records, test_records)

    # ----------------------------------------------------------------
    # Compare baseline vs CIKT (if available)
    # ----------------------------------------------------------------
    if baseline_metrics and results:
        compare_metrics(baseline_metrics, results[-1].test_metrics)

    # ----------------------------------------------------------------
    # Save final summary
    # ----------------------------------------------------------------
    summary = {
        "config": cfg.__dict__,
        "data": {
            "source": args.data_source,
            "n_train": len(train_records),
            "n_valid": len(valid_records),
            "n_test": len(test_records),
        },
        "baseline_metrics": baseline_metrics,
        "cikt_iterations": [
            {
                "iter": r.iteration,
                "train": r.train_metrics,
                "valid": r.valid_metrics,
                "test": r.test_metrics,
            }
            for r in results
        ],
    }
    summary_path = out / "cikt_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
