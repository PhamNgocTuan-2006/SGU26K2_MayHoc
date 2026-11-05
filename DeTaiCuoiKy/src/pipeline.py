"""Tạo pipeline tiền xử lý chống rò rỉ cho RT-IoT2022."""

from math import floor

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def choose_cv_and_k(min_class_count, max_folds=5):
    """Chọn số fold và k theo lớp nhỏ nhất của tập train."""
    cv = min(max_folds, int(min_class_count))
    if cv < 2:
        raise ValueError("Lớp nhỏ nhất phải có ít nhất 2 mẫu để dùng CV.")
    # k = min(5, floor(lớp nhỏ nhất / số fold) - 1), tối thiểu 1.
    k_neighbors = max(1, min(5, floor(min_class_count / cv) - 1))
    return cv, k_neighbors


def make_pipeline(model, scale=False, sampler=None, drop_cols=None, feature_columns=None):
    """Tạo ColumnTransformer, sampler tùy chọn và model cuối pipeline."""
    return make_pipeline_from_columns(model, feature_columns, scale=scale, sampler=sampler, drop_cols=drop_cols)


def make_pipeline_from_columns(model, feature_columns, scale=False, sampler=None, drop_cols=None):
    """Tạo pipeline từ danh sách cột đã biết, phù hợp trước khi fit."""
    drop_cols = set(drop_cols or [])
    # Chọn cột khi fit để hàm dùng được trước khi biết schema.
    def categorical_cols(frame):
        return [c for c in frame.columns if c in {"proto", "service"} and c not in drop_cols]

    def numeric_cols(frame):
        return [c for c in make_column_selector(dtype_include="number")(frame) if c not in drop_cols]
    numeric_step = StandardScaler() if scale else "passthrough"
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", numeric_step, numeric_cols),
        ],
        remainder="drop",
    )
    steps = [("preprocess", transformer)]
    if sampler == "smote":
        # k sẽ được thay bằng giá trị tính từ train trong notebook trước khi fit.
        steps.append(("sampler", SMOTE(random_state=42, k_neighbors=1)))
    elif sampler is not None:
        raise ValueError("sampler chỉ nhận None hoặc 'smote'.")
    steps.append(("model", model))
    return Pipeline(steps)
