import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from classifier import config


def evaluate_model(y_true, y_pred, y_proba=None, model_name="model"):
    class_names = config.CLASS_NAMES

    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*60}")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred)

    per_class_acc = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        if total > 0:
            per_class_acc[name] = float(cm[i, i] / total)
        else:
            per_class_acc[name] = 0.0

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
    }
