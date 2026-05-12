"""
CIKT Analyst: generates textual student knowledge-state profiles from
interaction history, optionally conditioned on feedback examples (iteration).

Stage 1 (Distillation): fine-tune on GPT-4o-generated profiles.
Stage 2 (Profiling):    apply the trained Analyst to all sequences.
Stage 4 (Iteration):    refine via KTO-style feedback loop.

Since we use OpenAI API (not fine-tuning), we approximate stages 1 and 4
by injecting few-shot examples into the system prompt.  Stage 2 is a direct
API call.  The interface is otherwise identical to the paper's description.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import openai

from data_loader import format_history

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

BASE_SYSTEM = """\
You are an expert math education analyst.
Given a student's history of question–answer interactions annotated with \
Knowledge Components (KCs), generate a concise but informative student profile.

The profile must contain exactly three sections:

<Knowledge State>
For each KC in the history, state the mastery level:
  - unknown   : fewer than 2 attempts, or 0 % correct
  - confusion : 1–79 % correct, or any recurring misconception
  - mastered  : ≥ 80 % correct and no recurring misconception
Format each line: "<KC name>: <mastery level> (<n_correct>/<n_attempts>)"

<Knowledge Acquisition>
For each KC with ≥ 2 attempts, describe the direction of change only
(e.g., "held steady at mastered level", "shifted from confusion to mastered").
Use no timeline words (first, initially, most recent, etc.).

<Misconception>
List only RECURRING misconceptions (same error pattern in ≥ 2 different
incorrect responses).  If none, write "None".

Do NOT add any content outside these three sections.
"""


def _build_user_message(
    history: list[dict],
    target_question: str,
    target_kc: str,
) -> str:
    hist_str = format_history(history)
    return (
        f"Target question KC: {target_kc}\n"
        f"Target question: {target_question[:300] if target_question else '(not provided)'}\n\n"
        f"Student interaction history (oldest first):\n{hist_str}"
    )


def _build_system_with_examples(
    positive_examples: list[tuple[str, str]] | None,
    negative_examples: list[tuple[str, str]] | None,
) -> str:
    """Inject few-shot feedback examples into the system prompt (Stage 4)."""
    system = BASE_SYSTEM
    if positive_examples:
        pos_block = "\n\n---\nEXAMPLES OF GOOD PROFILES (led to correct predictions):\n"
        for i, (hist, profile) in enumerate(positive_examples[:3], 1):
            pos_block += f"\nExample {i} history:\n{hist}\nProfile:\n{profile}\n"
        system += pos_block
    if negative_examples:
        neg_block = "\n\n---\nEXAMPLES OF POOR PROFILES (led to wrong predictions; avoid their style):\n"
        for i, (hist, profile) in enumerate(negative_examples[:3], 1):
            neg_block += f"\nExample {i} history:\n{hist}\nProfile:\n{profile}\n"
        system += neg_block
    return system


class Analyst:
    """LLM-based Analyst: generates student profiles from interaction history."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        cache_path: Optional[str | Path] = None,
        positive_examples: list[tuple[str, str]] | None = None,
        negative_examples: list[tuple[str, str]] | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.cache_path = Path(cache_path) if cache_path else None
        self.positive_examples = positive_examples or []
        self.negative_examples = negative_examples or []
        self._cache: dict[str, str] = {}
        self._client = openai.OpenAI(api_key=OPENAI_API_KEY)
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as f:
                self._cache = json.load(f)

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)

    def _get_cache_key(self, user_msg: str) -> str:
        import hashlib
        h = hashlib.md5(
            f"{self.model}|{self.temperature}|{len(self.positive_examples)}pos|"
            f"{len(self.negative_examples)}neg|{user_msg}".encode()
        ).hexdigest()
        return h

    def generate_profile(
        self,
        history: list[dict],
        target_question: str = "",
        target_kc: str = "",
    ) -> str:
        """Generate a textual student profile for the given history (Stage 2)."""
        if not history:
            return (
                "<Knowledge State>\n(No prior history)\n\n"
                "<Knowledge Acquisition>\nNo attempts yet.\n\n"
                "<Misconception>\nNone."
            )

        user_msg = _build_user_message(history, target_question, target_kc)
        cache_key = self._get_cache_key(user_msg)
        if cache_key in self._cache:
            return self._cache[cache_key]

        system = _build_system_with_examples(self.positive_examples, self.negative_examples)
        for attempt in range(5):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=512,
                )
                profile = resp.choices[0].message.content.strip()
                self._cache[cache_key] = profile
                self._save_cache()
                return profile
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Analyst] API error ({e}), retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError("Analyst: OpenAI API failed after 5 retries")

    def generate_profiles_batch(
        self,
        records: list[dict],
        show_progress: bool = True,
    ) -> list[str]:
        """Generate profiles for a list of sequence records (Stage 2 at scale)."""
        profiles = []
        iterator = records
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(records, desc="Generating profiles")
        for rec in iterator:
            # Use existing_profile if available (from distillation / qa_profile data)
            if rec.get("existing_profile"):
                profiles.append(rec["existing_profile"])
            else:
                profile = self.generate_profile(
                    rec["history"],
                    target_question=rec.get("target_question", ""),
                    target_kc=rec.get("kc", ""),
                )
                profiles.append(profile)
        return profiles

    # ------------------------------------------------------------------
    # Stage 4: update feedback examples from Predictor results
    # ------------------------------------------------------------------

    def update_from_feedback(
        self,
        records: list[dict],
        profiles: list[str],
        predictions: list[bool],
        labels: list[bool],
        n_examples: int = 5,
    ) -> None:
        """
        Approximate KTO update: collect profiles that led to correct /
        incorrect predictions and inject them as few-shot examples.
        """
        correct_indices = [
            i for i, (p, l) in enumerate(zip(predictions, labels)) if p == l
        ]
        wrong_indices = [
            i for i, (p, l) in enumerate(zip(predictions, labels)) if p != l
        ]

        def _make_example(idx: int) -> tuple[str, str]:
            return format_history(records[idx]["history"]), profiles[idx]

        # Sample diverse examples (spread across history lengths)
        def _sample(indices: list[int], k: int) -> list[int]:
            if len(indices) <= k:
                return indices
            # Sort by history length to get diverse examples
            indices.sort(key=lambda i: len(records[i]["history"]))
            step = max(1, len(indices) // k)
            return [indices[j] for j in range(0, len(indices), step)][:k]

        pos_idx = _sample(correct_indices, n_examples)
        neg_idx = _sample(wrong_indices, n_examples)

        self.positive_examples = [_make_example(i) for i in pos_idx]
        self.negative_examples = [_make_example(i) for i in neg_idx]
        print(
            f"[Analyst] Updated feedback: {len(self.positive_examples)} positive, "
            f"{len(self.negative_examples)} negative examples"
        )


if __name__ == "__main__":
    analyst = Analyst(model="gpt-4o-mini", cache_path="/tmp/analyst_cache.json")
    sample_history = [
        {"question_id": 1, "kc": "Basic Angle Facts", "question": "What is 30+60?", "label": True},
        {"question_id": 2, "kc": "Basic Angle Facts", "question": "Find angle x.", "label": False},
    ]
    profile = analyst.generate_profile(
        sample_history,
        target_question="Find the missing angle.",
        target_kc="Basic Angle Facts",
    )
    print("Generated profile:")
    print(profile)
