import numpy as np
import pandas as pd


def _metrics_from_cm(cm):
    # Đường chéo chính là số dự đoán đúng (TP) của từng lớp
    tp = np.diag(cm).astype(float)
    # Tổng theo hàng là số mẫu thực tế của từng lớp (support)
    sup = cm.sum(axis=1).astype(float)
    # Tổng theo cột là số lần mô hình dự đoán ra từng lớp
    pred = cm.sum(axis=0).astype(float)

    # Tính recall: nếu lớp không có mẫu thật (sup == 0) thì để NaN
    rec = np.full_like(tp, np.nan, dtype=float)
    has_sup = sup > 0
    rec[has_sup] = tp[has_sup] / sup[has_sup]

    # Tính precision: nếu không dự đoán lớp nào thì precision là 0
    prec = np.zeros_like(tp, dtype=float)
    has_pred = pred > 0
    prec[has_pred] = tp[has_pred] / pred[has_pred]

    # Thay NaN bằng 0 khi tính F1 cho từng lớp
    rec_for_f1 = np.nan_to_num(rec, nan=0.0)
    denom = prec + rec_for_f1

    f1 = np.zeros_like(tp, dtype=float)
    valid_denom = denom > 0
    f1[valid_denom] = 2 * prec[valid_denom] * rec_for_f1[valid_denom] / denom[valid_denom]

    return rec, prec, f1


def _build_cm(y_true, y_pred, num_classes):
    # Tạo ma trận nhầm lẫn kích thước k x k
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_cls, pred_cls in zip(y_true, y_pred):
        cm[true_cls, pred_cls] += 1
    return cm


def ci_report(y_true, y_pred, labels, n_boot=1000, seed=42, alpha=0.05):
    """Tính khoảng tin cậy bootstrap cho Macro-F1, Recall và Precision từng lớp."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    k = len(labels)
    n = len(y_true)

    # Tính chỉ số gốc trên toàn bộ tập test
    cm0 = _build_cm(y_true, y_pred, k)
    rec0, prec0, f10 = _metrics_from_cm(cm0)

    # Lặp bootstrap để ước lượng khoảng tin cậy
    rng = np.random.default_rng(seed)
    macro_list = []
    rec_list = []
    prec_list = []

    for _ in range(n_boot):
        # Lấy mẫu ngẫu nhiên có hoàn lại (bootstrap sample)
        sample_idx = rng.integers(0, n, size=n)
        y_true_boot = y_true[sample_idx]
        y_pred_boot = y_pred[sample_idx]

        cm_boot = _build_cm(y_true_boot, y_pred_boot, k)
        r_boot, p_boot, f_boot = _metrics_from_cm(cm_boot)

        macro_list.append(np.nanmean(f_boot))
        rec_list.append(r_boot)
        prec_list.append(p_boot)

    macro_arr = np.array(macro_list)
    rec_arr = np.array(rec_list)
    prec_arr = np.array(prec_list)

    # Tính cận dưới và cận trên theo mức ý nghĩa alpha
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    macro = {
        "macro_f1": float(np.nanmean(f10)),
        "ci_thap": float(np.percentile(macro_arr, lower_pct)),
        "ci_cao": float(np.percentile(macro_arr, upper_pct)),
    }

    table = pd.DataFrame({
        "lop": labels,
        "so_mau_test": cm0.sum(axis=1),
        "recall": rec0,
        "recall_ci_thap": np.nanpercentile(rec_arr, lower_pct, axis=0),
        "recall_ci_cao": np.nanpercentile(rec_arr, upper_pct, axis=0),
        "precision": prec0,
        "precision_ci_thap": np.nanpercentile(prec_arr, lower_pct, axis=0),
        "precision_ci_cao": np.nanpercentile(prec_arr, upper_pct, axis=0),
        "f1": f10,
    }).round(4)

    return macro, table