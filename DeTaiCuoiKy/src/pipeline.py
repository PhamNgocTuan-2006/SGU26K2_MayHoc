"""Tạo pipeline tiền xử lý chống rò rỉ cho RT-IoT2022."""

from math import floor

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["proto", "service"]


def choose_cv_and_k(min_class_count, max_folds=5):
    """Chọn số fold và k theo lớp nhỏ nhất của tập train."""
    cv = min(max_folds, int(min_class_count))
    if cv < 2:
        raise ValueError("Lớp nhỏ nhất phải có ít nhất 2 mẫu để dùng CV.")
    # k = min(5, floor(lớp nhỏ nhất / số fold) - 1), tối thiểu 1.
    k_neighbors = max(1, min(5, floor(min_class_count / cv) - 1))
    return cv, k_neighbors


def make_pipeline(
    model,
    feature_columns,
    scale=False,
    sampler=None,
    k_neighbors=None,
    sampling_strategy="auto",
    random_state=42,
):
    """Tạo pipeline có danh sách cột tường minh và có thể serialize bằng joblib."""
    cat_cols = [c for c in feature_columns if c in CATEGORICAL]
    num_cols = [c for c in feature_columns if c not in CATEGORICAL]
    numeric_step = StandardScaler() if scale else "passthrough"
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("numeric", numeric_step, num_cols),
        ],
        remainder="drop",
    )
    steps = [("preprocess", transformer)]
    if sampler == "smote":
        if k_neighbors is None:
            raise ValueError("k_neighbors phải được cung cấp khi sampler='smote'.")
        steps.append(
            (
                "sampler",
                SMOTE(
                    k_neighbors=k_neighbors,
                    sampling_strategy=sampling_strategy,
                    random_state=random_state,
                ),
            )
        )
    elif sampler is not None:
        raise ValueError("sampler chỉ nhận None hoặc 'smote'.")
    steps.append(("model", model))
    return Pipeline(steps)
