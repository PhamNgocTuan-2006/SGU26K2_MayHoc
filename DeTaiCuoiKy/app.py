"""Demo Streamlit cho RF cuối."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_assets():
    return joblib.load(ROOT / "models" / "final_rf.joblib"), json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))


def render_predictions(input_df):
    """Kiểm tra cột, dự đoán mọi dòng và hiển thị kết quả."""
    missing = [c for c in features if c not in input_df.columns]
    if missing:
        st.error("Thiếu cột đặc trưng: " + ", ".join(missing))
        return
    X = input_df[features]
    pred = model.predict(X)
    proba = model.predict_proba(X)
    groups = ["bình thường" if label in normal else "tấn công" for label in pred]
    confidence = [float(proba[i, list(model.classes_).index(label)]) for i, label in enumerate(pred)]
    result = input_df.copy()
    result.insert(0, "row_number", range(1, len(result) + 1))
    result["predicted_label"] = pred
    result["binary_group"] = groups
    result["confidence"] = np.round(confidence, 4)
    st.subheader("Tổng hợp dự đoán")
    summary_label = result["predicted_label"].value_counts().rename_axis("predicted_label").reset_index(name="count")
    summary_group = result["binary_group"].value_counts().rename_axis("binary_group").reset_index(name="count")
    st.dataframe(summary_label)
    st.dataframe(summary_group)
    st.subheader("Kết quả từng dòng")
    st.dataframe(result[["row_number", "predicted_label", "binary_group", "confidence"]])
    selected = st.selectbox("Xem chi tiết dòng số ...", result["row_number"].tolist())
    selected_index = selected - 1
    probs = pd.DataFrame({"label": model.classes_, "probability": proba[selected_index]}).sort_values("probability", ascending=False)
    probs["probability"] = probs["probability"].round(4)
    st.dataframe(probs)
    st.bar_chart(probs.set_index("label"), y="probability", horizontal=True, x_label="Xác suất", y_label="Nhãn lớp")
    st.download_button("Tải kết quả CSV", result.to_csv(index=False).encode("utf-8-sig"), "predictions.csv", "text/csv")

model, metadata = load_assets()
features = metadata["features"]
normal = set(metadata.get("normal_labels", []))
st.title("Phân lớp flow mạng IoT")
st.caption("Demo chỉ phân tích flow đã trích xuất đặc trưng, không chặn lưu lượng thật.")
sample_path = ROOT / "demo" / "sample_flows.csv"
sample = pd.read_csv(sample_path) if sample_path.exists() else None
tab_file, tab_sample, tab_about = st.tabs(["Tải CSV", "Dòng mẫu", "Về mô hình"])
with tab_file:
    uploaded = st.file_uploader("Chọn CSV", type="csv")
    input_df = pd.read_csv(uploaded) if uploaded else None
with tab_sample:
    input_df = sample if sample is not None else None
with tab_about:
    st.write(f"Macro-F1 CV: {metadata.get('cv_macro_f1', 'chưa có')}")
    st.write(f"Macro-F1 test: {metadata.get('test_macro_f1', 'chưa có')}")
    st.write("Mô hình học từ một testbed RT-IoT2022 và chỉ phân loại flow đã trích xuất đặc trưng.")
    st.warning("Không dùng để chặn lưu lượng thật. Lớp Metasploit_Brute_Force_SSH và NMAP_FIN_SCAN ít mẫu nên kết quả không ổn định.")
if input_df is not None:
    render_predictions(input_df)
