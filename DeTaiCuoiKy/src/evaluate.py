"""Đánh giá pipeline bằng Stratified K-Fold trên tập train."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict


def evaluate_cv(pipeline, X, y, random_state=42, n_splits=5):
    """Trả metric trung bình/độ lệch chuẩn, train-vs-CV và recall từng lớp."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {"macro_f1": "f1_macro", "balanced_acc": "balanced_accuracy", "mcc": "matthews_corrcoef", "accuracy": "accuracy", "weighted_f1": "f1_weighted"}
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    predictions = cross_val_predict(pipeline, X, y, cv=cv, method="predict", n_jobs=-1)
    recall = pd.Series(recall_score(y, predictions, labels=np.unique(y), average=None, zero_division=0), index=np.unique(y), name="recall")
    result = {}
    for name in scoring:
        result[f"{name}_mean"] = float(scores[f"test_{name}"].mean())
        result[f"{name}_std"] = float(scores[f"test_{name}"].std())
    result["train_macro_f1_mean"] = float(scores["train_macro_f1"].mean())
    result["train_macro_f1_std"] = float(scores["train_macro_f1"].std())
    result["train_vs_cv_gap"] = result["train_macro_f1_mean"] - result["macro_f1_mean"]
    result["recall_by_class"] = recall
    result["predictions"] = predictions
    return result
