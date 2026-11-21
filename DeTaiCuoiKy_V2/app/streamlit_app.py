import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODEL_PATH = ROOT / "results" / "models" / "final_model.joblib"
COLS_PATH = APP_DIR / "feature_columns.json"
LABELS_PATH = APP_DIR / "labels.json"
SAMPLE_PATH = APP_DIR / "sample_flows.csv"
TEST_DIR = ROOT / "results" / "test"
CV_DIR = ROOT / "results" / "cv"
FIG_DIR = ROOT / "results" / "figures"

LOP_BINH_THUONG = {"MQTT_Publish", "Thing_Speak", "Wipro_bulb"}
MO_TA = {
    "ARP_poisioning": "Giả mạo bảng ARP để chen vào giữa các thiết bị trong mạng.",
    "DDOS_Slowloris": "Giữ nhiều kết nối HTTP mở dở để làm cạn tài nguyên máy chủ.",
    "DOS_SYN_Hping": "Gửi dồn dập gói SYN để làm tràn hàng đợi kết nối.",
    "MQTT_Publish": "Lưu lượng bình thường: thiết bị gửi dữ liệu qua giao thức MQTT.",
    "Metasploit_Brute_Force_SSH": "Dò mật khẩu dịch vụ SSH bằng công cụ Metasploit.",
    "NMAP_FIN_SCAN": "Quét cổng bằng gói FIN để tránh bị phát hiện.",
    "NMAP_OS_DETECTION": "Dò hệ điều hành của thiết bị bằng Nmap.",
    "NMAP_TCP_scan": "Quét cổng TCP bằng Nmap.",
    "NMAP_UDP_SCAN": "Quét cổng UDP bằng Nmap.",
    "NMAP_XMAS_TREE_SCAN": "Quét cổng bằng gói gắn nhiều cờ TCP cùng lúc.",
    "Thing_Speak": "Lưu lượng bình thường: thiết bị gửi dữ liệu lên nền tảng ThingSpeak.",
    "Wipro_bulb": "Lưu lượng bình thường của bóng đèn thông minh Wipro.",
}

st.set_page_config(page_title="Phát hiện tấn công mạng IoT", layout="wide")


@st.cache_resource
def tai_mo_hinh():
    model = joblib.load(MODEL_PATH)
    cols = json.loads(COLS_PATH.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, cols, labels


def doc_csv(path, **kw):
    return pd.read_csv(path, **kw) if Path(path).exists() else None


def chuan_bi_dau_vao(df, cols):
    # one-hot proto va service neu file con o dang goc chua ma hoa
    ma_hoa = [c for c in ("proto", "service") if c in df.columns]
    if ma_hoa:
        df = pd.get_dummies(df, columns=ma_hoa)
    thieu = [c for c in cols if c not in df.columns]
    X = df.reindex(columns=cols, fill_value=0).astype(float)
    return X, thieu


try:
    model, cols, labels = tai_mo_hinh()
except Exception as e:
    st.error(
        "Chưa tải được mô hình. Hãy chạy experiments/07_final_eval.ipynb để tạo "
        f"results/models/final_model.joblib. Chi tiết lỗi: {e}"
    )
    st.stop()

ten_lop = [labels[int(c)] for c in model.classes_]


def phan_loai(df):
    X, thieu = chuan_bi_dau_vao(df, cols)
    proba = model.predict_proba(X)
    ten = [ten_lop[i] for i in proba.argmax(axis=1)]
    return X, thieu, proba, ten


def hien_thi_mot(dong):
    X, thieu, proba, ten = phan_loai(dong)
    p = proba[0]
    t = ten[0]
    tan_cong = t not in LOP_BINH_THUONG

    c1, c2, c3 = st.columns(3)
    c1.metric("Kết quả phân loại", t)
    c2.metric("Xác suất", f"{p.max():.1%}")
    c3.metric("Loại luồng", "Tấn công" if tan_cong else "Bình thường")
    if tan_cong:
        st.error(f"Cảnh báo: luồng này bị nghi là tấn công ({t}).")
    else:
        st.success("Luồng này được xem là lưu lượng bình thường.")
    st.caption(MO_TA.get(t, ""))

    if "nhan_that" in dong.columns:
        that = dong["nhan_that"].iloc[0]
        if that == t:
            st.info(f"Nhãn thật: {that}. Mô hình dự đoán đúng.")
        else:
            st.warning(f"Nhãn thật: {that}. Mô hình dự đoán sai.")
    if thieu:
        st.warning(f"Thiếu {len(thieu)} trên {len(cols)} cột, các cột này được điền 0.")

    st.subheader("Xác suất từng lớp")
    top = pd.Series(p, index=ten_lop).sort_values(ascending=False)
    st.bar_chart(top)
    st.table(pd.DataFrame({"Xác suất": [f"{v:.2%}" for v in top.head(3).values]},
                          index=top.head(3).index))

    quan_trong = doc_csv(CV_DIR / "perm_importance_features.csv")
    if quan_trong is not None:
        ten_cot = [c for c in quan_trong["dac_trung"].head(8) if c in X.columns]
        st.subheader("Một số đặc trưng quan trọng của luồng này")
        st.table(X[ten_cot].T.rename(columns={X.index[0]: "Giá trị"}))


def hien_thi_nhieu(df):
    X, thieu, proba, ten = phan_loai(df)
    if thieu:
        st.warning(f"File thiếu {len(thieu)} trên {len(cols)} cột, các cột này được điền 0.")
    kq = pd.DataFrame({
        "Dự đoán": ten,
        "Loại": ["Bình thường" if t in LOP_BINH_THUONG else "Tấn công" for t in ten],
        "Xác suất": proba.max(axis=1).round(4),
    })
    if "nhan_that" in df.columns:
        kq.insert(0, "Nhãn thật", df["nhan_that"].values)

    c1, c2, c3 = st.columns(3)
    c1.metric("Số luồng", len(kq))
    c2.metric("Số luồng bị nghi tấn công", int((kq["Loại"] == "Tấn công").sum()))
    c3.metric("Độ tin cậy trung bình", f"{kq['Xác suất'].mean():.3f}")
    st.dataframe(kq)
    st.bar_chart(kq["Dự đoán"].value_counts())
    st.download_button("Tải kết quả CSV", kq.to_csv(index=False).encode("utf-8"),
                       "ket_qua_du_doan.csv", "text/csv")


def mo_phong(df):
    X, thieu, proba, ten = phan_loai(df)
    khung = st.empty()
    thanh = st.progress(0)
    dem_tan_cong = 0
    ds = []
    for i, t in enumerate(ten):
        tan_cong = t not in LOP_BINH_THUONG
        dem_tan_cong += int(tan_cong)
        dong = {"STT": i + 1, "Dự đoán": t,
                "Loại": "Tấn công" if tan_cong else "Bình thường",
                "Xác suất": f"{proba[i].max():.1%}"}
        if "nhan_that" in df.columns:
            dong["Nhãn thật"] = df["nhan_that"].iloc[i]
        ds.append(dong)
        khung.dataframe(pd.DataFrame(ds))
        thanh.progress((i + 1) / len(ten))
        time.sleep(0.4)
    st.success(f"Đã xử lý {len(ten)} luồng, phát hiện {dem_tan_cong} luồng nghi tấn công.")


st.title("Phát hiện và phân loại tấn công mạng IoT")
st.caption("Mô hình HistGradientBoosting huấn luyện trên bộ dữ liệu RT-IoT2022. "
           "Ứng dụng chỉ minh họa, không thay thế hệ thống phát hiện xâm nhập thực tế.")

with st.sidebar:
    st.header("Giới thiệu")
    st.write("Đầu vào: các đặc trưng thống kê của một luồng mạng, ví dụ số gói tin, "
             "kích thước, thời gian giữa các gói, cờ TCP.")
    st.write("Đầu ra: loại lưu lượng trong 12 lớp, gồm 3 lớp bình thường và 9 lớp tấn công.")
    st.write("Hạn chế: các lớp NMAP_FIN_SCAN và Metasploit_SSH rất ít mẫu nên dễ nhầm.")

tab1, tab2, tab3 = st.tabs(["Dự đoán", "Hiệu năng mô hình", "Đặc trưng quan trọng"])

with tab1:
    che_do = st.radio("Chọn cách nhập dữ liệu", ["Luồng mẫu", "Tải file CSV"], horizontal=True)
    if che_do == "Luồng mẫu":
        mau = doc_csv(SAMPLE_PATH)
        if mau is None:
            st.warning("Chưa có app/sample_flows.csv, hãy tạo theo hướng dẫn trong README.")
        else:
            lop = st.selectbox("Loại luồng cần thử", sorted(mau["nhan_that"].unique()))
            sub = mau[mau["nhan_that"] == lop].reset_index(drop=True)
            idx = st.slider("Mẫu số", 1, len(sub), 1) - 1
            hien_thi_mot(sub.iloc[[idx]])
            st.divider()
            if st.button("Mô phỏng giám sát luồng mạng"):
                mo_phong(mau.sample(min(15, len(mau)), random_state=7).reset_index(drop=True))
    else:
        f = st.file_uploader("Chọn file CSV chứa các luồng mạng", type="csv")
        if f is not None:
            df = pd.read_csv(f)
            if len(df) == 1:
                hien_thi_mot(df)
            else:
                hien_thi_nhieu(df)

with tab2:
    st.subheader("Kết quả trên tập test")
    m = doc_csv(TEST_DIR / "final_metrics.csv", index_col=0)
    if m is None:
        st.warning("Chưa có results/test/final_metrics.csv.")
    else:
        m = m["gia_tri"]
        a, b, c, d = st.columns(4)
        a.metric("Macro-F1", f"{m['macro_f1']:.4f}")
        b.metric("Balanced Accuracy", f"{m['balanced_acc']:.4f}")
        c.metric("MCC", f"{m['mcc']:.4f}")
        d.metric("Accuracy", f"{m['accuracy']:.4f}")
    ci = doc_csv(TEST_DIR / "final_macro_ci.csv", index_col=0)
    if ci is not None:
        ci = ci["gia_tri"]
        st.caption(f"Khoảng tin cậy 95% của Macro-F1 theo bootstrap: "
                   f"từ {ci['ci_thap']:.4f} đến {ci['ci_cao']:.4f}.")
    bang = doc_csv(TEST_DIR / "final_per_class_ci.csv")
    if bang is not None:
        st.subheader("Precision và Recall từng lớp")
        st.dataframe(bang)
        st.caption("Hai lớp NMAP_FIN_SCAN và Metasploit_SSH chỉ có vài mẫu test nên khoảng tin cậy rất rộng.")
    anh = FIG_DIR / "final_confusion_matrix.png"
    if anh.exists():
        st.subheader("Ma trận nhầm lẫn")
        st.image(str(anh))

with tab3:
    st.subheader("Mức giảm Macro-F1 khi xáo trộn từng nhóm đặc trưng")
    nhom = doc_csv(CV_DIR / "perm_importance_groups.csv")
    if nhom is None:
        st.warning("Chưa có results/cv/perm_importance_groups.csv.")
    else:
        st.bar_chart(nhom.set_index("nhom")["giam_macro_f1_tb"])
        st.dataframe(nhom)
    dt = doc_csv(CV_DIR / "perm_importance_features.csv")
    if dt is not None:
        st.subheader("Top 15 đặc trưng")
        st.dataframe(dt.head(15))