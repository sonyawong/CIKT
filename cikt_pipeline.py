"""
CIKT: Collaborative and Iterative Knowledge Tracing pipeline.

Implements the four-stage framework from the paper:
  Stage 1 – Distillation:  Analyst learns from teacher-model profiles.
  Stage 2 – Profiling:     Analyst generates profiles for all sequences.
  Stage 3 – Reasoning:     Predictor learns to predict correctness from profiles.
  Stage 4 – Iteration:     Analyst is refined via Predictor feedback (KTO-style);
                           Predictor is retrained on improved profiles.

In our API-based implementation:
  • Distillation = load existing qa_generated_profiles as few-shot anchors.
  • Profiling    = call Analyst.generate_profile() for every sequence.
  • Reasoning    = call Predictor.predict() for every sequence.
  • Iteration    = Analyst.update_from_feedback() + re-generate profiles + re-predict.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from analyst import Analyst
from predictor import Predictor
from evaluate import compute_metrics, compare_metrics


@dataclass
class CIKTConfig:
    # LLM settings
    analyst_model: str = "gpt-4o-mini"
    predictor_model: str = "gpt-4o-mini"
    analyst_temperature: float = 0.3
    predictor_temperature: float = 0.0

    # Iteration settings
    n_iterations: int = 3
    iteration_sample_size: int = 1000   # paper default k=1000
    n_feedback_examples: int = 5        # positive + negative few-shots for KTO approx.

    # Cache paths (avoid redundant API calls across runs)
    cache_dir: str = "/tmp/cikt_cache"

    # Output directory for results and profiles
    output_dir: str = "/home/shuang/cikt/results"


@dataclass
class CIKTResult:
    iteration: int
    train_metrics: dict = field(default_factory=dict)
    valid_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    profiles: list[str] = field(default_factory=list)
    predictions: list[bool] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)


class CIKTPipeline:
    """End-to-end CIKT pipeline."""

    def __init__(self, config: CIKTConfig | None = None):
        self.cfg = config or CIKTConfig()
        Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        self.analyst = Analyst(
            model=self.cfg.analyst_model,
            temperature=self.cfg.analyst_temperature,
            cache_path=Path(self.cfg.cache_dir) / "analyst_cache.json",
        )
        self.predictor = Predictor(
            model=self.cfg.predictor_model,
            temperature=self.cfg.predictor_temperature,
            cache_path=Path(self.cfg.cache_dir) / "predictor_cache.json",
        )
        self.results: list[CIKTResult] = []

    # ------------------------------------------------------------------
    # Stage 1: Distillation
    # ------------------------------------------------------------------

    def stage1_distillation(
        self,
        train_records: list[dict],
        distill_profiles: list[str] | None = None,
    ) -> None:
        """
        Use teacher-model profiles (qa_generated_profile) to prime the Analyst.

        In the paper, the Analyst is fine-tuned on (history, teacher_profile) pairs.
        Here we inject the highest-quality existing profiles as positive few-shot
        examples into the Analyst's system prompt.
        """
        print("\n=== Stage 1: Distillation ===")
        if distill_profiles is None:
            # Use existing_profile from records if available
            distill_profiles = [r.get("existing_profile") for r in train_records]

        positive_examples = []
        for rec, profile in zip(train_records, distill_profiles):
            if profile and len(rec["history"]) >= 2:
                from data_loader import format_history
                positive_examples.append((format_history(rec["history"]), profile))
            if len(positive_examples) >= self.cfg.n_feedback_examples:
                break

        self.analyst.positive_examples = positive_examples
        print(
            f"  Loaded {len(positive_examples)} distillation examples into Analyst."
        )

    # ------------------------------------------------------------------
    # Stage 2: Profiling
    # ------------------------------------------------------------------

    def stage2_profiling(
        self,
        records: list[dict],
        label: str = "train",
    ) -> list[str]:
        """Apply the Analyst to generate profiles for all records."""
        print(f"\n=== Stage 2: Profiling ({label}, n={len(records)}) ===")
        profiles = self.analyst.generate_profiles_batch(records, show_progress=True)
        print(f"  Generated {len(profiles)} profiles.")
        return profiles

    # ------------------------------------------------------------------
    # Stage 3: Reasoning (Predictor training / evaluation)
    # ------------------------------------------------------------------

    def stage3_reasoning(
        self,
        records: list[dict],
        profiles: list[str],
        label: str = "train",
    ) -> CIKTResult:
        """Run the Predictor on all records and compute metrics."""
        print(f"\n=== Stage 3: Reasoning ({label}, n={len(records)}) ===")
        predictions, probs = self.predictor.predict_batch(
            records, profiles, show_progress=True
        )
        labels = [r["label"] for r in records]
        metrics = compute_metrics(labels, predictions, probs, records, label_name=label)
        result = CIKTResult(
            iteration=len(self.results),
            profiles=profiles,
            predictions=predictions,
            probs=probs,
        )
        if "train" in label:
            result.train_metrics = metrics
        elif "valid" in label:
            result.valid_metrics = metrics
        else:
            result.test_metrics = metrics
        return result

    # ------------------------------------------------------------------
    # Stage 4: Iteration
    # ------------------------------------------------------------------

    def stage4_iteration(
        self,
        train_records: list[dict],
        train_profiles: list[str],
        train_predictions: list[bool],
        train_labels: list[bool],
    ) -> list[str]:
        """
        Update the Analyst using prediction feedback (KTO-style).
        Returns regenerated profiles for the training set.
        """
        print("\n=== Stage 4: Iteration ===")
        # Sample a subset for iteration (paper uses k=1000)
        import random
        n = min(self.cfg.iteration_sample_size, len(train_records))
        indices = random.sample(range(len(train_records)), n) if n < len(train_records) else list(range(n))
        sample_records = [train_records[i] for i in indices]
        sample_profiles = [train_profiles[i] for i in indices]
        sample_predictions = [train_predictions[i] for i in indices]
        sample_labels = [train_labels[i] for i in indices]

        # Update Analyst prompt with feedback examples
        self.analyst.update_from_feedback(
            sample_records,
            sample_profiles,
            sample_predictions,
            sample_labels,
            n_examples=self.cfg.n_feedback_examples,
        )

        # Regenerate profiles with improved Analyst (force regeneration by clearing cache entries)
        print("  Regenerating profiles with updated Analyst...")
        # We clear the cache keys for records that had wrong predictions
        wrong_indices_full = [
            i for i, (p, l) in enumerate(zip(train_predictions, train_labels)) if p != l
        ]
        # Regenerate only misclassified samples (efficiency)
        new_profiles = train_profiles.copy()
        if wrong_indices_full:
            from tqdm import tqdm
            regen_records = [train_records[i] for i in wrong_indices_full]
            for i, rec in tqdm(
                zip(wrong_indices_full, regen_records),
                total=len(wrong_indices_full),
                desc="Re-generating profiles",
            ):
                # Bypass cache by temporarily appending a refresh token
                rec_copy = dict(rec)
                new_profiles[i] = self.analyst.generate_profile(
                    rec["history"],
                    target_question=rec.get("target_question", ""),
                    target_kc=rec.get("kc", ""),
                )
        print(f"  Regenerated {len(wrong_indices_full)} profiles.")
        return new_profiles

    # ------------------------------------------------------------------
    # Main pipeline: run all 4 stages with iteration
    # ------------------------------------------------------------------

    def run(
        self,
        train_records: list[dict],
        valid_records: list[dict],
        test_records: list[dict],
    ) -> list[CIKTResult]:
        """
        Full CIKT pipeline:
          1. Distillation
          2. Initial profiling (train + valid + test)
          3. Initial reasoning (evaluate all splits)
          4. Iteration loop (Stage 4 × n_iterations)
          5. Final evaluation on test set
        """
        t0 = time.time()

        # ---------- Stage 1: Distillation ----------
        self.stage1_distillation(train_records)

        # ---------- Stage 2: Profiling (initial) ----------
        train_profiles = self.stage2_profiling(train_records, "train")
        valid_profiles = self.stage2_profiling(valid_records, "valid")
        test_profiles = self.stage2_profiling(test_records, "test")

        # ---------- Stage 3: Initial Reasoning ----------
        train_result = self.stage3_reasoning(train_records, train_profiles, "train-iter0")
        valid_result = self.stage3_reasoning(valid_records, valid_profiles, "valid-iter0")
        test_result = self.stage3_reasoning(test_records, test_profiles, "test-iter0")

        iter0 = CIKTResult(
            iteration=0,
            train_metrics=train_result.train_metrics,
            valid_metrics=valid_result.valid_metrics,
            test_metrics=test_result.test_metrics,
            profiles=test_profiles,
            predictions=test_result.predictions,
            probs=test_result.probs,
        )
        self.results.append(iter0)
        self._save_result(iter0, test_records)

        # ---------- Stages 3+4: Iteration ----------
        for it in range(1, self.cfg.n_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {it}/{self.cfg.n_iterations}")
            print(f"{'='*60}")

            # Stage 4: update Analyst
            train_labels = [r["label"] for r in train_records]
            train_profiles = self.stage4_iteration(
                train_records,
                train_profiles,
                train_result.predictions,
                train_labels,
            )

            # Stage 3: re-predict with new profiles
            train_result = self.stage3_reasoning(
                train_records, train_profiles, f"train-iter{it}"
            )

            # Re-profile and re-predict valid and test with updated Analyst
            valid_profiles = self.stage2_profiling(valid_records, f"valid-iter{it}")
            test_profiles = self.stage2_profiling(test_records, f"test-iter{it}")

            valid_result = self.stage3_reasoning(
                valid_records, valid_profiles, f"valid-iter{it}"
            )
            test_result = self.stage3_reasoning(
                test_records, test_profiles, f"test-iter{it}"
            )

            result = CIKTResult(
                iteration=it,
                train_metrics=train_result.train_metrics,
                valid_metrics=valid_result.valid_metrics,
                test_metrics=test_result.test_metrics,
                profiles=test_profiles,
                predictions=test_result.predictions,
                probs=test_result.probs,
            )
            self.results.append(result)
            self._save_result(result, test_records)

        # ---------- Summary ----------
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"CIKT Complete  ({elapsed/60:.1f} min)")
        print(f"{'='*60}")
        self._print_summary(test_records)
        return self.results

    # ------------------------------------------------------------------
    # No-profile baseline (for comparison, mirrors "w/o Profile" ablation)
    # ------------------------------------------------------------------

    def run_no_profile_baseline(
        self,
        records: list[dict],
        label: str = "test-no-profile",
    ) -> dict:
        """Predict correctness without any profile (empty-profile predictor)."""
        print(f"\n=== Baseline: No-Profile Prediction ({label}) ===")
        empty_profile = (
            "<Knowledge State>\n(Profile not available)\n\n"
            "<Knowledge Acquisition>\nN/A\n\n"
            "<Misconception>\nNone."
        )
        profiles = [empty_profile] * len(records)
        predictions, probs = self.predictor.predict_batch(
            records, profiles, show_progress=True
        )
        labels = [r["label"] for r in records]
        return compute_metrics(labels, predictions, probs, records, label_name=label)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_result(self, result: CIKTResult, records: list[dict]) -> None:
        out_dir = Path(self.cfg.output_dir)
        payload = {
            "iteration": result.iteration,
            "train_metrics": result.train_metrics,
            "valid_metrics": result.valid_metrics,
            "test_metrics": result.test_metrics,
            "predictions": [
                {
                    "student_id": records[i]["student_id"],
                    "question_id": records[i]["question_id"],
                    "label": records[i]["label"],
                    "prediction": result.predictions[i] if i < len(result.predictions) else None,
                    "prob": result.probs[i] if i < len(result.probs) else None,
                    "profile": result.profiles[i] if i < len(result.profiles) else None,
                }
                for i in range(len(records))
            ],
        }
        path = out_dir / f"cikt_iter{result.iteration}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  Saved → {path}")

    def _print_summary(self, test_records: list[dict]) -> None:
        print("\nIteration | Test ACC  | Test ACC>15 | Test F1")
        print("-" * 50)
        for r in self.results:
            acc = r.test_metrics.get("acc", float("nan"))
            acc15 = r.test_metrics.get("acc_len15", float("nan"))
            f1 = r.test_metrics.get("f1", float("nan"))
            print(f"  {r.iteration:5d}   | {acc:.4f}   | {acc15:.4f}      | {f1:.4f}")

        if len(self.results) >= 2:
            compare_metrics(self.results[0].test_metrics, self.results[-1].test_metrics)
