# src/evaluation.py
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    balanced_accuracy_score,
    matthews_corrcoef, 
    recall_score
)
from .config import RANDOM_STATE, N_SPLITS, N_REPEATS, CV_DIR

def compute_metrics(y_true, y_pred, n_classes):
    # Tạo danh sách các nhãn lớp
    labels = list(range(n_classes))
    
    # Tính các độ đo tổng quan và lưu vào từ điển (dictionary)
    metrics_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    
    # Tính Recall cho từng lớp riêng biệt
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    for class_idx, recall_val in enumerate(recalls):
        metrics_dict[f"recall_c{class_idx}"] = recall_val
        
    return metrics_dict

def cv_evaluate(model, X, y, name, n_classes=12,
                n_splits=N_SPLITS, n_repeats=N_REPEATS, save=True):
    """Đánh giá mô hình bằng phương pháp Cross-Validation."""
    
    # Khởi tạo bộ chia dữ liệu
    cv_strategy = RepeatedStratifiedKFold(
        n_splits=n_splits, 
        n_repeats=n_repeats,
        random_state=RANDOM_STATE
    )
    
    all_fold_results = []
    
    # Tiến hành lặp qua từng fold
    for fold_idx, (train_index, val_index) in enumerate(cv_strategy.split(X, y)):
        # Sao chép mô hình để tránh ảnh hưởng giữa các fold
        current_model = clone(model)
        
        # Lấy dữ liệu train và validation cho fold này
        X_train_fold = X.iloc[train_index]
        y_train_fold = y.iloc[train_index]
        X_val_fold = X.iloc[val_index]
        y_val_fold = y.iloc[val_index]
        
        # Đo thời gian huấn luyện (fit)
        start_fit_time = time.time()
        current_model.fit(X_train_fold, y_train_fold)
        fit_duration = time.time() - start_fit_time
        
        # Đo thời gian dự đoán (predict)
        start_pred_time = time.time()
        predictions = current_model.predict(X_val_fold)
        pred_duration = time.time() - start_pred_time
        
        # Tính toán các chỉ số đánh giá
        fold_metrics = compute_metrics(y_val_fold, predictions, n_classes)
        
        # Bổ sung thông tin về model, fold và thời gian chạy vào kết quả
        fold_metrics.update({
            "model": name, 
            "fold": fold_idx, 
            "fit_time": fit_duration, 
            "pred_time": pred_duration
        })
        
        all_fold_results.append(fold_metrics)
        
    # Chuyển đổi danh sách kết quả thành DataFrame
    results_df = pd.DataFrame(all_fold_results)
    
    # Lưu file nếu được yêu cầu
    if save:
        output_file_path = CV_DIR / f"{name}.csv"
        results_df.to_csv(output_file_path, index=False)
        
    return results_df

def summarize(df):
    columns_to_summarize = ["macro_f1", "balanced_acc", "mcc", "accuracy", "weighted_f1", "fit_time"]
    summary_df = df.groupby("model")[columns_to_summarize].agg(["mean", "std"]).round(4)
    return summary_df