"""Huấn luyện và lưu RF cuối để tái lập demo."""

from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestClassifier

from data_prep import load_data, clean_data, get_feature_sets
from pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]


def main():
    # Đọc dữ liệu, chỉ lấy train_idx đã chốt.
    df, _ = clean_data(load_data(ROOT / "data" / "RT_IOT2022"))
    split = __import__("numpy").load(ROOT / "experiments" / "split_indices.npz")
    train_idx = split["train_idx"]
    features = get_feature_sets(df)["NO_SRC_PORT"]
    X = df.drop(columns=["Attack_type"])
    y = df["Attack_type"]
    model = make_pipeline(
        RandomForestClassifier(n_estimators=250, max_depth=50, min_samples_split=5,
                               min_samples_leaf=1, max_features=None,
                               random_state=42, n_jobs=-1),
        features, random_state=42,
    )
    model.fit(X.iloc[train_idx][features], y.iloc[train_idx])
    output = ROOT / "models" / "final_rf.joblib"
    output.parent.mkdir(exist_ok=True)
    joblib.dump(model, output, compress=3)
    print(f"Đã lưu {output}; kích thước={output.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
