"""
Evaluation metrics for CIKT: ACC, ACC_{len>15}, F1-score.
Mirrors the metrics reported in Table 1 of the paper.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(
    labels: list[bool],
    predictions: list[bool],
    probs: Optional[list[float]] = None,
    records: Optional[list[dict]] = None,
    label_name: str = "",
) -> dict[str, float]:
    """
    Returns a dict with:
      acc       – overall accuracy
      acc_len15 – accuracy for samples where history length > 15
      f1        – macro F1
      auc       – AUROC (if probs provided and both classes present)
    """
    if not labels:
        return {}

    y_true = [int(l) for l in labels]
    y_pred = [int(p) for p in predictions]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ACC_{len>15}: filter to samples where history length > 15
    acc_len15 = None
    if records is not None:
        long_idx = [i for i, r in enumerate(records) if len(r.get("history", [])) > 15]
        if long_idx:
            acc_len15 = accuracy_score(
                [y_true[i] for i in long_idx],
                [y_pred[i] for i in long_idx],
            )

    auc = None
    if probs is not None:
        classes = set(y_true)
        if len(classes) == 2:
            try:
                auc = roc_auc_score(y_true, probs)
            except Exception:
                auc = None

    result = {"acc": acc, "f1": f1}
    if acc_len15 is not None:
        result["acc_len15"] = acc_len15
    if auc is not None:
        result["auc"] = auc

    prefix = f"[{label_name}] " if label_name else ""
    acc_str = f"{acc:.4f}"
    f1_str = f"{f1:.4f}"
    len15_str = f"{acc_len15:.4f}" if acc_len15 is not None else "N/A"
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    print(
        f"{prefix}ACC={acc_str}  ACC_len>15={len15_str}  F1={f1_str}  AUC={auc_str}"
        f"  (n={len(labels)})"
    )
    return result


def compare_metrics(baseline: dict, cikt: dict) -> None:
    """Print relative improvement of CIKT over a baseline."""
    print("\n--- Improvement (CIKT vs baseline) ---")
    for key in ("acc", "f1", "acc_len15", "auc"):
        if key in baseline and key in cikt:
            delta = cikt[key] - baseline[key]
            rel = delta / max(baseline[key], 1e-9) * 100
            print(f"  {key}: {baseline[key]:.4f} → {cikt[key]:.4f}  ({rel:+.2f}%)")
