"""Tạo các hình v2 từ dữ liệu và log, không huấn luyện mô hình."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def main():
    # Đọc dữ liệu gốc để dựng lại các hình EDA.
    raw = pd.read_csv(ROOT / "data" / "RT_IOT2022")
    clean = raw.drop(columns=["Unnamed: 0"], errors="ignore")
    constant = [c for c in clean.columns if clean[c].nunique(dropna=False) <= 1]
    clean = clean.drop(columns=constant).drop_duplicates()
    labels = clean["Attack_type"].value_counts().sort_values()
    ax = labels.plot.bar(figsize=(12, 6), color="steelblue")
    ax.set_title("Phân bố nhãn sau loại trùng")
    ax.set_ylabel("Số mẫu")
    for bar, value in zip(ax.patches, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "01_phan_bo_nhan_sau_loai_trung_v2.png", dpi=150); plt.close()
    # Vẽ tám đặc trưng bằng log10(1+x), không dùng symlog.
    numeric = clean.select_dtypes(include=np.number).columns.tolist()
    chosen = [c for c in numeric if any(k in c.lower() for k in ["flow", "pkt", "payload", "iat", "flag", "header"])][:8]
    if len(chosen) < 8: chosen += [c for c in numeric if c not in chosen][:8 - len(chosen)]
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    for ax, col in zip(axes.flat, chosen):
        plot = clean[["Attack_type", col]].copy(); plot[col] = np.log10(1 + plot[col].clip(lower=0))
        sns.boxplot(data=plot, x="Attack_type", y=col, showfliers=False, ax=ax)
        ax.set_ylim(bottom=0); ax.tick_params(axis="x", rotation=75); ax.set_title(f"log10(1 + {col})")
    plt.tight_layout(); plt.savefig(FIG / "02_boxplot_log10_v2.png", dpi=150); plt.close()
    # Chỉ hiển thị các đặc trưng thuộc cặp tương quan cao để giữ đủ nhãn.
    corr = clean[numeric].corr(); high = set()
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1:]:
            if abs(corr.loc[c1, c2]) > .95: high.update([c1, c2])
    subset = sorted(high) if high else numeric
    plt.figure(figsize=(18, 14)); sns.heatmap(corr.loc[subset, subset], cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Tương quan đặc trưng thuộc cặp |r| > 0,95"); plt.tight_layout(); plt.savefig(FIG / "03_heatmap_tuong_quan_v2.png", dpi=150); plt.close()
    # Dùng log thật, loại các dòng bị đánh dấu không hợp lệ.
    log = pd.read_csv(ROOT / "experiments" / "experiment_log.csv", encoding="utf-8-sig")
    log = log[~log.notes.fillna("").str.startswith("KHÔNG HỢP LỆ")]
    summary = log.groupby("model", as_index=False)["macro_f1"].max().sort_values("macro_f1", ascending=False)
    ax = summary.plot.bar(x="model", y="macro_f1", legend=False, figsize=(11, 6), color="darkorange")
    ax.set_ylabel("Macro-F1"); ax.set_title("Macro-F1 theo mô hình (giá trị log thực tế)")
    for bar, value in zip(ax.patches, summary.macro_f1): ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=60, ha="right"); plt.tight_layout(); plt.savefig(FIG / "04_baseline_macro_f1_v2.png", dpi=150); plt.close()
    # Tách biểu đồ CV và Test, chỉ giữ dòng hợp lệ mới nhất trong nhóm trùng.
    valid = log[~log.notes.fillna("").str.contains("KHÔNG HỢP LỆ", case=False)]
    duplicate_keys = ["stage", "model", "feature_set", "imbalance_method", "split", "macro_f1"]
    valid = valid.sort_values("date").drop_duplicates(duplicate_keys, keep="last")
    cv_names = {"Dummy_most_frequent": "Dummy", "LogisticRegression": "LR", "DecisionTree_tuned": "DT_tuned", "RandomForest_tuned": "RF_tuned", "MLP_tuned": "MLP_tuned"}
    cv = valid[(valid["split"] == "train_cv") & ~valid["notes"].fillna("").str.contains("đánh giá test", case=False) & valid["model"].isin(cv_names)].copy()
    cv["display_model"] = cv["model"].map(cv_names)
    cv = cv.sort_values("date").drop_duplicates("display_model", keep="last").sort_values("display_model")
    test_names = {"Dummy": "Dummy", "Dummy_most_frequent": "Dummy", "LogisticRegression": "LR", "DecisionTree_tuned": "DT", "RandomForest_tuned": "RF", "MLP_tuned": "MLP"}
    test = valid[valid["notes"].fillna("").str.contains("đánh giá test", case=False) & valid["model"].isin(test_names)].copy()
    test["display_model"] = test["model"].map(test_names)
    test = test.sort_values("date").drop_duplicates("display_model", keep="last").sort_values("display_model")
    print("\nBảng dùng cho Hình 04a CV:")
    print(cv[["display_model", "macro_f1", "macro_f1_std"]].to_string(index=False))
    print("\nBảng dùng cho Hình 04b Test:")
    print(test[["display_model", "macro_f1"]].to_string(index=False))
    order = ["Dummy", "LR", "DT_tuned", "RF_tuned", "MLP_tuned"]
    cv = cv.set_index("display_model").reindex(order).dropna(subset=["macro_f1"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(cv.index, cv["macro_f1"], yerr=cv["macro_f1_std"].fillna(0), capsize=5, color="steelblue")
    ax.set_title("Macro-F1 CV 5-fold")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    for i, value in enumerate(cv["macro_f1"]): ax.text(i, value + .02, f"{value:.4f}", ha="center")
    fig.tight_layout(); fig.savefig(FIG / "04a_cv_macro_f1_v3.png", dpi=150); plt.close(fig)
    test_order = ["Dummy", "LR", "DT", "RF", "MLP"]
    test = test.set_index("display_model").reindex(test_order).dropna(subset=["macro_f1"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(test.index, test["macro_f1"], color="darkorange")
    ax.set_title("Macro-F1 Test")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    for i, value in enumerate(test["macro_f1"]): ax.text(i, value + .02, f"{value:.4f}", ha="center")
    fig.tight_layout(); fig.savefig(FIG / "04b_test_macro_f1_v3.png", dpi=150); plt.close(fig)
    print(f"Đã dùng {len(clean)} dòng sau loại trùng và tạo 4 hình v2.")


if __name__ == "__main__":
    main()
