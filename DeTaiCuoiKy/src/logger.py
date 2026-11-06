"""Ghi nhật ký các lần chạy thực nghiệm."""

from datetime import datetime
from pathlib import Path
import csv
import uuid


LOG_PATH = Path("../experiments/experiment_log.csv")


def log_run(model, feature_set, cv, metrics, params="", notes="", log_path=LOG_PATH):
    """Ghi một dòng metric CV vào experiment log."""
    fields = ["run_id", "date", "stage", "model", "feature_set", "imbalance_method", "split", "cv", "params", "macro_f1", "balanced_acc", "mcc", "accuracy", "notes"]
    row = {
        "run_id": uuid.uuid4().hex[:12], "date": datetime.now().isoformat(timespec="seconds"),
        "stage": 3, "model": model, "feature_set": feature_set, "imbalance_method": "none",
        "split": "train_cv", "cv": cv, "params": params,
        "macro_f1": metrics["macro_f1_mean"], "balanced_acc": metrics["balanced_acc_mean"],
        "mcc": metrics["mcc_mean"], "accuracy": metrics["accuracy_mean"], "notes": notes,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(row)
    return row["run_id"]
