# Phân lớp tấn công mạng IoT

Đồ án phân lớp flow mạng IoT bằng RT-IoT2022, với bài toán đa lớp theo `Attack_type` và thí nghiệm nhị phân bình thường/tấn công. Câu hỏi chính: xử lý mất cân bằng ảnh hưởng Macro-F1 ra sao, Random Forest so với Decision Tree/MLP thế nào, và các lớp nào thường bị nhầm?

## Dữ liệu

Nguồn: [UCI RT-IoT2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022), DOI `10.24432/C5P338`, giấy phép CC BY 4.0. Đặt file không đuôi vào `data/RT_IOT2022`.

## Cài đặt và chạy

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

Chạy notebook theo thứ tự `01_eda.ipynb` đến `06_interpretation.ipynb`, sau đó có thể tái tạo model bằng `python src/train_final.py`. Demo chạy bằng `streamlit run app.py`, mở địa chỉ hiển thị trong terminal, thường là `http://localhost:8501`.

## Cấu trúc

`data/` dữ liệu; `notebooks/` phân tích; `src/` mã nguồn; `experiments/` split/log; `figures/` hình; `models/` model/metadata; `demo/` flow mẫu.

## Kết quả chính

RF tuned trên `NO_SRC_PORT`: Macro-F1 CV `0.9573`, test `0.9813`. Decision Tree tuned CV `0.9487`, test `0.9893`; MLP CV `0.9440`, test `0.9814`. Các số lấy từ kết quả notebook và `experiment_log.csv`, không tự suy diễn.

## Hạn chế

Dữ liệu đến từ một testbed, flow đã trích xuất đặc trưng và có lớp hiếm rất ít mẫu. Kết quả không đại diện cho mọi mạng IoT và demo không chặn lưu lượng thật.

## Công cụ AI

Nhóm sử dụng Kilo Code và công cụ AI để hỗ trợ viết mã, kiểm tra lỗi và diễn giải. Mọi con số trong báo cáo lấy từ các lần chạy notebook và `experiment_log.csv`.

## Tái lập

Dùng `random_state=42`; split cố định lưu tại `experiments/split_indices.npz`.
