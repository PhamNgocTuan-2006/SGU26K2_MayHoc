import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR.parent / "results" / "models" / "final_model.joblib"
COLS_PATH = APP_DIR / "feature_columns.json"
LABELS_PATH = APP_DIR / "labels.json"
SAMPLE_PATH = APP_DIR / "sample_flows.csv"

LOP_BINH_THUONG = ["MQTT_Publish", "Thing_Speak", "Wipro_bulb"]

st.set_page_config(page_title="Phát hiện tấn công mạng IoT", layout="wide")


@st.cache_resource
def tai_mo_hinh():
    model = joblib.load(MODEL_PATH)
    cols_text = COLS_PATH.read_text(encoding="utf-8")
    labels_text = LABELS_PATH.read_text(encoding="utf-8")
    cols = json.loads(cols_text)
    labels = json.loads(labels_text)
    return model, cols, labels


def chuan_bi_dau_vao(df, cols):
    # One-hot proto và service nếu file đầu vào còn ở dạng gốc chưa mã hóa
    ma_hoa = []
    for c in ("proto", "service"):
        if c in df.columns:
            ma_hoa.append(c)

    if len(ma_hoa) > 0:
        df = pd.get_dummies(df, columns=ma_hoa)

    # Kiểm tra các cột đặc trưng bị thiếu so với danh sách cột lúc huấn luyện
    thieu = []
    for c in cols:
        if c not in df.columns:
            thieu.append(c)

    X = df.reindex(columns=cols, fill_value=0)
    X = X.astype(float)
    return X, thieu


st.title("Phát hiện và phân loại tấn công mạng IoT")
st.caption(
    "Mô hình HistGradientBoosting huấn luyện trên bộ dữ liệu RT-IoT2022. "
    "Ứng dụng chỉ minh họa, không thay thế hệ thống phát hiện xâm nhập thực tế."
)

try:
    model, cols, labels = tai_mo_hinh()
except Exception as e:
    st.error(
        "Chưa tải được mô hình. Hãy chạy experiments/07_final_eval.ipynb để tạo "
        f"results/models/final_model.joblib. Chi tiết lỗi: {e}"
    )
    st.stop()

nguon = st.radio("Nguồn dữ liệu", ["Dữ liệu mẫu", "Tải file CSV"], horizontal=True)

df = None
if nguon == "Dữ liệu mẫu":
    if SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH)
    else:
        st.warning("Chưa có app/sample_flows.csv, hãy tạo theo hướng dẫn trong README.")
else:
    f = st.file_uploader("Chọn file CSV chứa các luồng mạng", type="csv")
    if f is not None:
        df = pd.read_csv(f)

if df is not None:
    X, thieu = chuan_bi_dau_vao(df, cols)
    if len(thieu) > 0:
        st.warning(
            f"File thiếu {len(thieu)} trên {len(cols)} cột, các cột này được điền 0. "
            "Kết quả có thể kém tin cậy."
        )

    # Dự đoán xác suất từng lớp và chọn lớp có xác suất cao nhất
    proba = model.predict_proba(X)
    idx_max = proba.argmax(axis=1)
    pred = model.classes_[idx_max]

    ten_du_doan = []
    for i in pred:
        ten_du_doan.append(labels[int(i)])

    loai_luong = []
    for t in ten_du_doan:
        if t in LOP_BINH_THUONG:
            loai_luong.append("Bình thường")
        else:
            loai_luong.append("Tấn công")

    xac_suat_cao_nhat = proba.max(axis=1).round(4)

    kq = pd.DataFrame({
        "Dự đoán": ten_du_doan,
        "Loại": loai_luong,
        "Xác suất": xac_suat_cao_nhat,
    })

    if "nhan_that" in df.columns:
        kq.insert(0, "Nhãn thật", df["nhan_that"].values)

    so_luong_tan_cong = int((kq["Loại"] == "Tấn công").sum())
    do_tin_cay_tb = kq["Xác suất"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Số luồng", len(kq))
    c2.metric("Số luồng bị nghi tấn công", so_luong_tan_cong)
    c3.metric("Độ tin cậy trung bình", f"{do_tin_cay_tb:.3f}")

    st.subheader("Kết quả dự đoán")
    st.dataframe(kq, use_container_width=True)

    st.subheader("Phân bố nhãn dự đoán")
    dem_nhan = kq["Dự đoán"].value_counts()
    st.bar_chart(dem_nhan)

    st.subheader("Xác suất từng lớp của một luồng")
    i = st.number_input(
        "Chọn số thứ tự dòng",
        min_value=0,
        max_value=len(kq) - 1,
        value=0,
        step=1,
    )
    ten_cac_lop = []
    for c in model.classes_:
        ten_cac_lop.append(labels[int(c)])

    xac_suat_dong = pd.Series(proba[int(i)], index=ten_cac_lop)
    st.bar_chart(xac_suat_dong)