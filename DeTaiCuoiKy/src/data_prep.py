"""Các hàm đọc, làm sạch và xác định bộ đặc trưng cho RT-IoT2022."""

from pathlib import Path

import pandas as pd


def load_data(path):
    """Đọc CSV không đuôi từ đường dẫn được cung cấp."""
    return pd.read_csv(Path(path))


def clean_data(df):
    """Bỏ chỉ số, cột hằng và dòng trùng; trả về bảng sạch và thống kê."""
    work = df.copy()
    stats = {"rows_before": len(work), "columns_before": work.shape[1]}
    if "Unnamed: 0" in work.columns:
        work = work.drop(columns=["Unnamed: 0"])
    stats["dropped_index_column"] = "Unnamed: 0" in df.columns
    constant_cols = [c for c in work.columns if work[c].nunique(dropna=False) <= 1]
    work = work.drop(columns=constant_cols)
    stats["constant_columns"] = constant_cols
    stats["duplicates_removed"] = int(work.duplicated().sum())
    work = work.drop_duplicates().reset_index(drop=True)
    stats["rows_after"] = len(work)
    stats["columns_after"] = work.shape[1]
    return work, stats


def get_feature_sets(df):
    """Trả về FULL, NO_SRC_PORT và NO_PORTS, không gồm nhãn."""
    target = "Attack_type"
    full = [c for c in df.columns if c != target]
    return {
        "FULL": full,
        "NO_SRC_PORT": [c for c in full if c != "id.orig_p"],
        "NO_PORTS": [c for c in full if c not in {"id.orig_p", "id.resp_p"}],
    }
