"""
CIKT Predictor: forecasts binary correctness given student history, profile,
and target question.

Stage 3 (Reasoning): train/fine-tune on (history, profile, target) → label.
Stage 4 (Iteration): re-train on improved profiles after Analyst update.

We implement this with OpenAI API prompting (instead of LoRA fine-tuning)
and track probability via log-odds from the API's logprob output.
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

SYSTEM_PROMPT = """\
You are a student performance prediction system.

Given:
1. A student's interaction history (past questions and correctness),
2. A textual student knowledge profile summarising their current state,
3. A target question the student is about to answer,

Output exactly one word: "True" if the student will answer correctly, \
"False" otherwise.

Do NOT output anything else — just the single word True or False.
"""


def _build_predictor_user_message(
    history: list[dict],
    profile: str,
    target_question: str,
    target_kc: str,
) -> str:
    hist_str = format_history(history)
    return (
        f"## Student Profile\n{profile}\n\n"
        f"## Interaction History\n{hist_str}\n\n"
        f"## Target Question\n"
        f"KC: {target_kc}\n"
        f"Question: {target_question[:400] if target_question else '(not provided)'}\n\n"
        "Will the student answer the target question correctly? Reply True or False."
    )


class Predictor:
    """LLM-based Predictor: predicts binary correctness from profile + history."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        cache_path: Optional[str | Path] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
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
        return hashlib.md5(
            f"{self.model}|{self.temperature}|{user_msg}".encode()
        ).hexdigest()

    def predict(
        self,
        history: list[dict],
        profile: str,
        target_question: str = "",
        target_kc: str = "",
    ) -> tuple[bool, float]:
        """
        Returns (prediction: bool, prob_correct: float).
        prob_correct is the estimated probability of a correct answer.
        """
        user_msg = _build_predictor_user_message(
            history, profile, target_question, target_kc
        )
        cache_key = self._get_cache_key(user_msg)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached["prediction"], cached["prob"]

        for attempt in range(5):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_tokens=5,
                    logprobs=True,
                    top_logprobs=5,
                )
                answer = resp.choices[0].message.content.strip().lower()
                prediction = answer.startswith("t")

                # Compute probability from logprobs
                prob = _extract_true_prob(resp)
                if prob is None:
                    prob = 0.9 if prediction else 0.1

                self._cache[cache_key] = {"prediction": prediction, "prob": prob}
                self._save_cache()
                return prediction, prob

            except (openai.RateLimitError, openai.APIConnectionError) as e:
                wait = 2 ** attempt
                print(f"[Predictor] API error ({e}), retry in {wait}s")
                time.sleep(wait)

        raise RuntimeError("Predictor: OpenAI API failed after 5 retries")

    def predict_batch(
        self,
        records: list[dict],
        profiles: list[str],
        show_progress: bool = True,
    ) -> tuple[list[bool], list[float]]:
        """Predict for a batch of records."""
        predictions, probs = [], []
        iterator = list(zip(records, profiles))
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Predicting")
        for rec, profile in iterator:
            pred, prob = self.predict(
                rec["history"],
                profile,
                target_question=rec.get("target_question", ""),
                target_kc=rec.get("kc", ""),
            )
            predictions.append(pred)
            probs.append(prob)
        return predictions, probs


def _extract_true_prob(resp) -> Optional[float]:
    """Parse P(True) from the response logprobs."""
    import math
    try:
        lp_content = resp.choices[0].logprobs.content
        if not lp_content:
            return None
        first_token = lp_content[0]
        true_logprob, false_logprob = None, None
        for top in first_token.top_logprobs:
            token_lower = top.token.strip().lower()
            if token_lower.startswith("t") and true_logprob is None:
                true_logprob = top.logprob
            if token_lower.startswith("f") and false_logprob is None:
                false_logprob = top.logprob
        if true_logprob is None:
            return None
        if false_logprob is None:
            return math.exp(true_logprob)
        # Softmax over {true, false}
        t_p = math.exp(true_logprob)
        f_p = math.exp(false_logprob)
        return t_p / (t_p + f_p)
    except Exception:
        return None


if __name__ == "__main__":
    pred = Predictor(model="gpt-4o-mini", cache_path="/tmp/predictor_cache.json")
    history = [
        {"question_id": 1, "kc": "Basic Angle Facts", "question": "Q1", "label": True},
        {"question_id": 2, "kc": "Basic Angle Facts", "question": "Q2", "label": False},
    ]
    profile = (
        "<Knowledge State>\nBasic Angle Facts: confusion (1/2)\n\n"
        "<Knowledge Acquisition>\nBasic Angle Facts: unstable performance.\n\n"
        "<Misconception>\nNone."
    )
    result, prob = pred.predict(
        history, profile,
        target_question="Find angle x in the diagram.",
        target_kc="Basic Angle Facts",
    )
    print(f"Prediction: {result}, P(correct)={prob:.3f}")
