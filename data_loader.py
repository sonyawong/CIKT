"""
Data loading for CIKT. Builds student interaction sequences from:
  - qa_profile_*.csv (annotated sample with pre-generated profiles)
  - combined_data.csv (full 670-student Eedi dataset)

Each sequence record represents a student's history up to time t-1 and
the target interaction at time t (QuestionId, kc, IsCorrect).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

QA_PROFILE_DIR = Path("/home/shuang/dialogue-kt-raw/data/annotated")
# Prefer the version with enriched KC annotations
COMBINED_DATA_PATH = Path("/home/shuang/student_simulation/data/combined_data_with_kc.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_kc(kc_raw) -> str:
    if pd.isna(kc_raw):
        return "Unknown"
    s = str(kc_raw).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, list):
                return ", ".join(str(x) for x in lst)
        except (ValueError, SyntaxError):
            s = s[1:-1].replace("'", "").replace('"', "")
            return ", ".join(x.strip() for x in s.split(","))
    return s


def correctness_label(val) -> Optional[bool]:
    if pd.isna(val):
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("true", "1", "correct"):
        return True
    if s in ("false", "0", "incorrect"):
        return False
    return None


# ---------------------------------------------------------------------------
# Loading qa_profile format files
# ---------------------------------------------------------------------------

def load_qa_profile_csv(path: str | Path) -> pd.DataFrame:
    """Load a qa_profile CSV and normalise columns."""
    df = pd.read_csv(path)
    df["kc_clean"] = df["kc"].apply(clean_kc)
    df["label"] = df["IsCorrect"].apply(correctness_label)
    df = df.sort_values(["studentID", "Unnamed: 0"]).reset_index(drop=True)
    return df


def build_qa_profile_sequences(df: pd.DataFrame) -> list[dict]:
    """
    For each row t with a non-null label, build a sequence record:
      {
        student_id, question_id, kc, label,
        history:  [{question_id, kc, question, label}, ...],   # rows 0..t-1
        target_question: str,
        existing_profile: str | None   # qa_generated_profile if present
      }
    Rows with null label are kept in history but skipped as targets.
    """
    records = []
    for sid, grp in df.groupby("studentID"):
        grp = grp.reset_index(drop=True)
        history: list[dict] = []
        for _, row in grp.iterrows():
            label = correctness_label(row["IsCorrect"])
            kc = clean_kc(row.get("kc", row.get("KC", np.nan)))
            q_text = str(row.get("question", "")).strip()
            if label is not None:
                records.append({
                    "student_id": sid,
                    "question_id": row["QuestionId"],
                    "kc": kc,
                    "label": label,
                    "history": [h.copy() for h in history],
                    "target_question": q_text,
                    "existing_profile": row.get("qa_generated_profile", None)
                    if pd.notna(row.get("qa_generated_profile", np.nan)) else None,
                })
            # Always add to history regardless of label
            history.append({
                "question_id": row["QuestionId"],
                "kc": kc,
                "question": q_text,
                "label": label,
            })
    return records


# ---------------------------------------------------------------------------
# Loading from the full combined_data.csv
# ---------------------------------------------------------------------------

def load_combined_data(path: str | Path = COMBINED_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Prefer lowercase 'kc' column (enriched annotations) over 'KC'
    if "kc" in df.columns:
        df["kc_clean"] = df["kc"].apply(clean_kc)
    elif "KC" in df.columns:
        df["kc_clean"] = df["KC"].apply(clean_kc)
    else:
        df["kc_clean"] = "Unknown"
    df["label"] = df["IsCorrect"].apply(correctness_label)
    # Sort chronologically
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["studentID", "timestamp"])
    else:
        df = df.sort_values(["studentID"])
    df = df.reset_index(drop=True)
    return df


def build_combined_sequences(
    df: pd.DataFrame,
    min_history: int = 1,
    max_seq_len: int = 50,
    predict_last_only: bool = False,
) -> list[dict]:
    """
    Build sequence records from combined_data.
    For each student, we slide a window and for each position t≥min_history,
    we create a record predicting interaction t given history 0..t-1.

    If predict_last_only=True, only create a record for the final interaction
    (useful for evaluation).
    """
    records = []
    for sid, grp in df.groupby("studentID"):
        grp = grp.reset_index(drop=True)
        # Only keep rows with valid labels
        valid_mask = grp["label"].notna()
        grp = grp[valid_mask].reset_index(drop=True)
        if len(grp) < min_history + 1:
            continue

        # Cap sequence length
        if len(grp) > max_seq_len:
            grp = grp.iloc[-max_seq_len:].reset_index(drop=True)

        history: list[dict] = []
        positions = [len(grp) - 1] if predict_last_only else range(min_history, len(grp))
        for t in range(len(grp)):
            row = grp.iloc[t]
            if t in (positions if isinstance(positions, list) else list(positions)):
                records.append({
                    "student_id": sid,
                    "question_id": row.get("QuestionId", row.get("question_id")),
                    "kc": row["kc_clean"],
                    "label": row["label"],
                    "history": [h.copy() for h in history],
                    "target_question": str(row.get("question", "")).strip(),
                    "existing_profile": None,
                })
            history.append({
                "question_id": row.get("QuestionId", row.get("question_id")),
                "kc": row["kc_clean"],
                "question": str(row.get("question", "")).strip(),
                "label": row["label"],
            })
    return records


# ---------------------------------------------------------------------------
# Train / valid / test splits
# ---------------------------------------------------------------------------

def split_by_student(
    records: list[dict],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records by studentID so each student is in exactly one split."""
    rng = np.random.default_rng(seed)
    student_ids = sorted({r["student_id"] for r in records})
    rng.shuffle(student_ids)
    n = len(student_ids)
    # Ensure at least 1 student per non-empty split
    n_train = max(1, int(n * train_ratio))
    n_valid = max(1, int(n * valid_ratio)) if n >= 3 else 0
    n_test = n - n_train - n_valid
    if n_test <= 0:
        # Shrink train to give room for test
        n_train = max(1, n - max(1, n_valid) - 1)
        n_test = n - n_train - n_valid
    train_ids = set(student_ids[:n_train])
    valid_ids = set(student_ids[n_train: n_train + n_valid])
    test_ids = set(student_ids[n_train + n_valid:])
    train = [r for r in records if r["student_id"] in train_ids]
    valid = [r for r in records if r["student_id"] in valid_ids]
    test = [r for r in records if r["student_id"] in test_ids]
    return train, valid, test


def load_and_split_qa_profile() -> tuple[list[dict], list[dict], list[dict]]:
    """Load the qa_profile CSVs.
    All three files are identical (format demo for one student).
    We build one sequence and split it temporally so every position is usable.
    """
    df_list = []
    for split in ("train", "valid", "test"):
        df = load_qa_profile_csv(QA_PROFILE_DIR / f"qa_profile_{split}.csv")
        df_list.append(df)
    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates()
    records = build_qa_profile_sequences(df_all)

    # If only one student, split temporally (8:1:1)
    student_ids = list({r["student_id"] for r in records})
    if len(student_ids) <= 1:
        n = len(records)
        n_train = max(1, int(n * 0.8))
        n_valid = max(1, int(n * 0.1))
        train = records[:n_train]
        valid = records[n_train: n_train + n_valid]
        test = records[n_train + n_valid:]
        if not test:
            test = records[-1:]
        return train, valid, test

    return split_by_student(records, train_ratio=0.6, valid_ratio=0.2)


def load_and_split_combined(
    n_students: Optional[int] = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load combined_data.csv and split by student."""
    df = load_combined_data()
    if n_students is not None:
        rng = np.random.default_rng(seed)
        sids = list(df["studentID"].unique())
        rng.shuffle(sids)
        sids = sids[:n_students]
        df = df[df["studentID"].isin(sids)]
    records = build_combined_sequences(df)
    return split_by_student(records)


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def format_history(history: list[dict]) -> str:
    """Format history list into a human-readable string for LLM prompts."""
    if not history:
        return "(No prior history)"
    lines = []
    for i, h in enumerate(history, 1):
        corr = "Correct" if h["label"] is True else ("Incorrect" if h["label"] is False else "Unknown")
        lines.append(f"Q{i}. KC: {h['kc']} | Answer: {corr}")
    return "\n".join(lines)


def serialize_records(records: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)


def deserialize_records(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Loading qa_profile data...")
    tr, va, te = load_and_split_qa_profile()
    print(f"  train={len(tr)}, valid={len(va)}, test={len(te)}")

    print("Loading combined data (first 50 students)...")
    tr2, va2, te2 = load_and_split_combined(n_students=50)
    print(f"  train={len(tr2)}, valid={len(va2)}, test={len(te2)}")
