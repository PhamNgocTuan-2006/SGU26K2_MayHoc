"""Ghi và nâng cấp nhật ký các lần chạy thực nghiệm."""

from datetime import datetime
import csv
from pathlib import Path
import uuid

import pandas as pd


LOG_PATH = Path(__file__).resolve().parents[1] / "experiments" / "experiment_log.csv"
FIELDS = [
    "run_id", "date", "stage", "model", "feature_set", "imbalance_method", "split", "cv", "params",
    "macro_f1", "macro_f1_std", "balanced_acc", "mcc", "accuracy", "notes",
]


def _upgrade_log(path):
    """Nâng cấp log cũ, giữ nguyên số dòng và sao lưu trước khi ghi."""
    if not path.exists() or path.stat().st_size == 0:
        return
    current = pd.read_csv(path, encoding="utf-8-sig")
    if list(current.columns) == FIELDS:
        return
    before = len(current)
    backup = path.with_name("experiment_log_backup.csv")
    path.replace(backup)
    if "macro_f1_std" not in current.columns:
        current.insert(current.columns.get_loc("macro_f1") + 1, "macro_f1_std", "")
    current = current.reindex(columns=FIELDS)
    current.to_csv(path, index=False, encoding="utf-8-sig")
    after = len(current)
    print(f"Nâng cấp experiment log: trước={before} dòng, sau={after} dòng.")


def log_run(model, feature_set, cv, metrics, params="", notes="", stage=3, imbalance_method="none", log_path=LOG_PATH):
    """Ghi một dòng metric CV, tương thích với các notebook hiện có."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _upgrade_log(path)
    row = {
        "run_id": uuid.uuid4().hex[:12], "date": datetime.now().isoformat(timespec="seconds"),
        "stage": stage, "model": model, "feature_set": feature_set, "imbalance_method": imbalance_method,
        "split": "train_cv", "cv": cv, "params": params, "macro_f1": metrics.get("macro_f1_mean"),
        "macro_f1_std": metrics.get("macro_f1_std"), "balanced_acc": metrics.get("balanced_acc_mean"),
        "mcc": metrics.get("mcc_mean"), "accuracy": metrics.get("accuracy_mean"), "notes": notes,
    }
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    return row["run_id"]


def mark_invalid(run_id, reason, log_path=LOG_PATH):
    """Đánh dấu một dòng không hợp lệ mà không xóa dòng đó."""
    path = Path(log_path)
    data = pd.read_csv(path, encoding="utf-8-sig")
    matches = data["run_id"].astype(str) == str(run_id)
    if not matches.any():
        raise ValueError(f"Không tìm thấy run_id: {run_id}")
    note = f"KHÔNG HỢP LỆ: {reason}"
    index = data.index[matches][0]
    old = "" if pd.isna(data.at[index, "notes"]) else str(data.at[index, "notes"])
    data.at[index, "notes"] = f"{old}; {note}" if old else note
    data.to_csv(path, index=False, encoding="utf-8-sig")
