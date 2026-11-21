# Phát hiện và phân loại tấn công mạng IoT trên bộ dữ liệu RT-IoT2022

Đề tài cuối kỳ môn Máy học. Bài toán phân lớp đa lớp trên dữ liệu luồng mạng IoT, mất cân bằng nặng, so sánh có kiểm soát các mô hình học máy và chọn một mô hình cuối theo quy tắc khai báo trước.

Video demo: [điền liên kết]

## 1. Mục tiêu và bài toán

- Đầu vào: các đặc trưng thống kê của một luồng mạng (số gói tin, kích thước, thời gian giữa các gói, cờ TCP, cửa sổ TCP, giao thức, dịch vụ).
- Đầu ra: loại lưu lượng trong 12 lớp, gồm 3 lớp bình thường (MQTT_Publish, Thing_Speak, Wipro_bulb) và 9 lớp tấn công.
- Loại bài toán: phân lớp đa lớp, mất cân bằng nặng.
- Metric chính: Macro-F1. Không dùng Accuracy làm thước đo chính vì lớp lớn nhất chiếm khoảng 76% dữ liệu.

## 2. Dữ liệu

- Nguồn: RT-IoT2022, UCI Machine Learning Repository, mã số 942.
  https://archive.ics.uci.edu/dataset/942/rt-iot2022
- Thông tin tác giả và bài báo gốc: xem trang dataset trên UCI, kiểm tra lại trước khi trích dẫn trong báo cáo.
- Quy mô gốc: 123.117 mẫu, 85 cột, không có giá trị thiếu.
- Tiền xử lý (chi tiết và lý do từng bước: `docs/preprocessing_log.md`):
  - bỏ cột `Unnamed: 0` vì là chỉ số hàng, không phải đặc trưng
  - xóa 5.195 dòng trùng, còn 117.922 mẫu
  - mã hóa nhãn bằng LabelEncoder, one-hot cho `proto` và `service`
  - chia 80/20 có stratify, `random_state=42`: train 94.337 mẫu, test 23.585 mẫu
  - lọc đa cộng tuyến theo ngưỡng tương quan 0,95 **tính trên tập train**, bỏ 24 cột, còn 70 đặc trưng
- Lớp hiếm nhất: NMAP_FIN_SCAN (28 mẫu, khoảng 22 mẫu train và 6 mẫu test), Metasploit_Brute_Force_SSH (khoảng 36 mẫu).
- Dữ liệu và mô hình đã huấn luyện không được commit lên git (xem `.gitignore`), được tái tạo bằng các notebook.

## 3. Cấu trúc thư mục

```
├── data/
│   ├── raw/              file gốc RT_IOT2022 (không commit)
│   └── processed/        X_train, X_test, y_train, y_test, label_names, scaler (không commit)
├── eda/                  eda-iot.ipynb
├── preprocessing/        preprocessing-iot.ipynb
├── docs/                 preprocessing_log.md
├── src/
│   ├── config.py         đường dẫn, seed, tham số CV
│   ├── data.py           load_data
│   ├── evaluation.py     cv_evaluate, compute_metrics, summarize
│   ├── pipelines.py      get_baselines, get_classic, get_advanced, get_xgb, get_final
│   ├── experiment_log.py ExperimentLogger, export_excel
│   ├── bootstrap.py      ci_report (bootstrap), exact_ci_table (Clopper Pearson)
│   └── feature_groups.py phân nhóm đặc trưng theo ý nghĩa mạng
├── experiments/          các notebook thí nghiệm, xem mục 5
├── results/
│   ├── cv/               kết quả CV theo fold của từng mô hình, kết quả tuning, summary_all.csv
│   ├── test/             kết quả trên tập test
│   ├── figures/          hình dùng trong báo cáo
│   ├── models/           mô hình đã lưu (không commit)
│   └── logs/             log_<người chạy>.csv và experiment_log.xlsx
├── app/                  demo Streamlit và dữ liệu minh họa
├── report/               báo cáo
├── slides/               slide trình bày
├── requirements.txt
└── requirements.lock.txt
```

## 4. Cài đặt

Khuyến nghị Python 3.10 trở lên.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` gồm: numpy, pandas, scikit-learn, imbalanced-learn, xgboost, matplotlib, seaborn, joblib, jupyter, streamlit, shap. Hai thư viện sau cũng cần cho mã nguồn nên phải có trong môi trường: `scipy` (kiểm định thống kê, khoảng tin cậy chính xác) và `openpyxl` (xuất nhật ký Excel). `requirements.lock.txt` ghi phiên bản đã dùng khi chạy.

## 5. Thứ tự chạy

Đặt file gốc `RT_IOT2022` vào `data/raw/`, sau đó chạy theo thứ tự.

| Bước | Notebook | Nội dung | Kết quả chính |
|---|---|---|---|
| 1 | `eda/eda-iot.ipynb` | Khám phá dữ liệu | hình EDA |
| 2 | `preprocessing/preprocessing-iot.ipynb` | Tiền xử lý, chia train và test | `data/processed/*` |
| 3 | `experiments/00_smoke_test.ipynb` | Kiểm tra hạ tầng bằng baseline Dummy | `results/cv/` |
| 4 | `01_baseline.ipynb` | Dummy, Logistic Regression, Decision Tree độ sâu 3 | `results/cv/` |
| 5 | `02_models_classic.ipynb` | KNN, Decision Tree, Random Forest mặc định | `results/cv/` |
| 6 | `02_models_advanced.ipynb` | MLP, HistGradientBoosting, XGBoost | `results/cv/` |
| 7 | `03_tuning.ipynb` | Tuning Random Forest, validation curve, `class_weight` trên RF, phân tích lỗi lớp hiếm trong CV | `tuning_rf_*.csv`, hình |
| 8 | `03_tuning_advanced.ipynb` | Tuning HGB và MLP | `tuning_hgb_random.csv`, `tuning_mlp_random.csv` |
| 9 | `04_imbalance.ipynb` | Xử lý mất cân bằng N1 đến N4 trên RF và HGB, chẩn đoán HGB, Precision/Recall từng lớp, kiểm định cặp | `imbalance_per_class_rf.csv` |
| 10 | `05_feature_importance.ipynb` | Permutation importance theo nhóm và theo đặc trưng | `perm_importance_*.csv` |
| 11 | `06_ablation_ports.ipynb` | Bỏ cổng, bỏ proto và service | `ablation_ports.csv` |
| 12 | `06b_ablation_groups.ipynb` | Bỏ từng nhóm đặc trưng trên Random Forest | `ablation_groups_rf.csv` |
| 13 | `06c_learning_curve.ipynb` | Đường cong học theo tỉ lệ dữ liệu huấn luyện | `learning_curve.csv` |
| 14 | `09_summary_all.ipynb` | Bảng tổng hợp của cả nhóm, kiểm định cặp giữa các ứng viên | `summary_all.csv` |
| 15 | `10_finalists_repeated_cv.ipynb` | CV lặp 3×5 cho hai ứng viên cuối, Precision/Recall từng lớp | `*_rep3.csv`, `finalists_per_class.csv` |
| 16 | `07_final_eval.ipynb` | Huấn luyện mô hình cuối, đánh giá test, bootstrap CI | `results/test/final_*.csv`, mô hình |
| 17 | `07e_exact_ci.ipynb` | Khoảng tin cậy chính xác theo lớp | `final_per_class_exact_ci.csv` |
| 18 | `08_error_analysis.ipynb` | Phân tích lỗi trên test, so sánh CV với test | `top_confusions.csv` |
| 19 | `08b_train_test_overlap.ipynb` | Kiểm tra bản sao đặc trưng giữa train và test | `train_test_overlap.csv` |
| 20 | `08c_nearest_neighbor_gap.ipynb` | Khoảng cách hàng xóm gần nhất từ test đến train | `nearest_neighbor_gap.csv` |
| 21 | `07d_demo_samples.ipynb` | Tạo dữ liệu minh họa cho demo | `app/sample_flows.csv`, `app/demo_upload.csv` |

Các notebook phụ:
- `07c_rebuild_app_files.ipynb`: tạo lại `app/feature_columns.json` và `app/labels.json` mà không cần chạy lại cell đánh giá test.
- `11_extra_models.ipynb` (tùy chọn): ExtraTrees và GaussianNB, chạy sau khi đã khóa mô hình, chỉ là kết quả bổ sung.

## 6. Giao thức thực nghiệm

| Hạng mục | Quy định |
|---|---|
| Chia dữ liệu | 80% train, 20% test, stratify, `random_state=42` |
| Kiểm định chéo | Stratified 5-fold trên tập train. CV lặp 3×5 cho các ứng viên cuối |
| Chống rò rỉ | Scaler, SMOTE và các bước tiền xử lý phụ thuộc dữ liệu nằm trong Pipeline, chỉ fit trên fold train. SMOTE dùng `imblearn.pipeline.Pipeline` |
| Metric chính | Macro-F1 |
| Metric phụ | Balanced Accuracy, MCC, Accuracy, Weighted-F1, Recall từng lớp, thời gian fit và predict |
| Tuning | Chỉ trên tập train (RandomizedSearchCV, GridSearchCV), scoring `f1_macro` |
| So sánh | Cùng split, cùng CV, cùng metric. Kiểm định cặp theo fold có hiệu chỉnh Nadeau và Bengio, ngưỡng |t| khoảng 2,78 với 5 fold |
| Seed | `random_state=42` ở mọi nơi |

## 7. Quy tắc làm việc nhóm

1. Tập test (`X_test`, `y_test`) chỉ dùng cho: (a) đánh giá cuối đúng một lần trong `07_final_eval.ipynb`, (b) phân tích và minh họa sau khi đã khóa mô hình (`07d`, `08b`, `08c`). **Tuyệt đối không dùng để chọn hay chỉnh mô hình.**
2. Mọi mô hình được đánh giá bằng `cv_evaluate` thông qua `ExperimentLogger`, tên mô hình là duy nhất (file kết quả CV lưu theo tên, đặt trùng sẽ ghi đè).
3. Scaler, SMOTE, PCA luôn nằm trong Pipeline.
4. Mỗi người làm trên nhánh riêng, tạo Pull Request vào `main`, người phụ trách hạ tầng duyệt và merge. Sau mỗi lần merge, chạy `git pull origin main` trước khi làm tiếp.
5. Mọi kết quả phải lưu ra `results/`, không chỉ in ra màn hình.
6. Ghi rõ phần có dùng công cụ AI hỗ trợ.

## 8. Quy tắc chọn mô hình cuối và mô hình đã khóa

Quy tắc (khai báo trước khi chạy test):

1. Chọn 4 đến 5 ứng viên đứng đầu theo Macro-F1 trên CV 5-fold của tập train.
2. So từng ứng viên với ứng viên đứng đầu bằng kiểm định cặp theo fold.
3. Nếu khác biệt không có ý nghĩa thống kê, ưu tiên theo thứ tự: Recall của lớp Wipro_bulb, độ ổn định giữa các fold, sự đơn giản và tốc độ.
4. Khóa mô hình. Sau đó chỉ chạy tập test một lần duy nhất.

Kết quả áp dụng:

- Mô hình đã khóa: `HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=42)`, các tham số còn lại mặc định. Xem `get_final()` trong `src/pipelines.py`.
- Cơ sở: mười cấu hình có Macro-F1 CV nằm trong khoảng 0,960 đến 0,969 và không khác nhau có ý nghĩa thống kê (kiểm định cặp có hiệu chỉnh, p từ 0,145 đến 0,477). Mô hình được chọn bằng tiêu chí phụ đã khai báo trước, không phải vì có bằng chứng vượt trội.
- Đã kiểm tra Precision từng lớp trên CV: Metasploit_SSH 0,808, NMAP_FIN_SCAN 0,900, không có suy giảm ở lớp hiếm.
- Dự phòng: `xgb_default`, Macro-F1 CV 0,9644 ± 0,0175.
- Phát hiện kèm theo: HistGradientBoosting với `learning_rate=0,1` và `l2_regularization=0` không ổn định (Macro-F1 khoảng 0,27, thấp hơn cả bộ phân loại luôn đoán lớp đa số). Hạ learning rate xuống 0,03 hoặc thêm `l2_regularization` đều khắc phục.

## 9. Kết quả chính

| Chỉ số | CV lặp 3×5 (train) | Test |
|---|---|---|
| Macro-F1 | 0,9661 ± 0,0143 | 0,9951 (bootstrap 95%: 0,9918 đến 0,9975) |
| Balanced Accuracy | 0,9558 | 0,9957 |
| MCC | 0,9960 | 0,9967 |
| Accuracy | 0,9984 | 0,9986 |

Cách đọc:
- Các lớp lớn cho kết quả test khớp với CV.
- Chênh lệch Macro-F1 giữa test và CV gần như hoàn toàn đến từ hai lớp ít mẫu, Metasploit_SSH (7 mẫu test) và NMAP_FIN_SCAN (6 mẫu test), cùng đạt Recall 100% trên test nhưng chỉ 0,72 và 0,82 trên CV. Khoảng tin cậy chính xác 95% của Recall hai lớp này lần lượt là [0,59 ; 1,00] và [0,54 ; 1,00]. Khoảng bootstrap của Macro-F1 hẹp hơn độ bất định thật vì bootstrap không tạo ra được lỗi khi mô hình đoán đúng hết.
- Hãy dùng CV 0,966 làm ước lượng bảo thủ và test 0,995 làm kết quả bổ sung.
- Bảng tổng hợp mọi mô hình: `results/cv/summary_all.csv`. Nhật ký thí nghiệm: `results/logs/experiment_log.xlsx`.

## 10. Hạn chế

- Dữ liệu sinh trong phòng thí nghiệm bằng các kịch bản lặp lại, nên các luồng cùng loại rất giống nhau. Bộ phân loại 1-NN đơn giản đã đạt 99,72% trên test, và khoảng cách trung vị từ một mẫu test tới mẫu train gần nhất nhỏ hơn 0,01 ở 6 trên 12 lớp. Mô hình cuối giảm số lỗi từ khoảng 65 xuống 32 so với 1-NN, nhưng điểm số cao phản ánh mức độ khớp với lưu lượng cùng kịch bản, không đảm bảo khái quát cho lưu lượng thực hoặc kiểu tấn công mới.
- Không có mẫu test nào trùng khớp tuyệt đối với mẫu train (0 trên 23.585), nhưng bản sao gần là phổ biến.
- Hai lớp Metasploit_SSH và NMAP_FIN_SCAN chỉ có 7 và 6 mẫu test, kết quả trên hai lớp này rất bất định.
- Kiểm định cặp theo fold hơi dễ dãi vì các fold dùng chung dữ liệu huấn luyện, nên dùng bản hiệu chỉnh và coi là chỉ báo.
- Ablation theo nhóm đặc trưng chạy trên Random Forest để tiết kiệm thời gian, không phải mô hình cuối.

## 11. Chạy demo

1. Chạy preprocessing để tạo `data/processed/`.
2. Chạy `experiments/07_final_eval.ipynb` để tạo `results/models/final_model.joblib`, `app/feature_columns.json` và `app/labels.json`.
3. Chạy `experiments/07d_demo_samples.ipynb` để tạo `app/sample_flows.csv` và `app/demo_upload.csv`.
4. Chạy lệnh từ thư mục gốc của repo:

```bash
streamlit run app/streamlit_app.py
```

Nếu app báo thiếu `feature_columns.json` hoặc `labels.json`, chạy `experiments/07c_rebuild_app_files.ipynb`.

Chức năng:
- Tab Dự đoán: chọn luồng mẫu theo loại, hoặc tải file CSV chứa các luồng; mô phỏng giám sát luồng mạng.
- Tab Hiệu năng mô hình: kết quả test, khoảng tin cậy, ma trận nhầm lẫn.
- Tab Đặc trưng quan trọng: kết quả permutation importance.

File tải lên cần có các cột đặc trưng giống `X_train.csv`. Nếu file còn ở dạng gốc với cột `proto` và `service` dạng chữ thì app tự one-hot. Cột thiếu được điền 0 kèm cảnh báo.

Lưu ý về dữ liệu minh họa: `app/sample_flows.csv` và `app/demo_upload.csv` được lấy mẫu từ tập test bằng `07d_demo_samples.ipynb`, chỉ để minh họa trong demo. Không có quyết định nào về mô hình dựa trên các mẫu này.

## 12. Tái lập và nhật ký thí nghiệm

- Mọi thí nghiệm ghi qua `ExperimentLogger("<tên>")`, mỗi người một file `results/logs/log_<tên>.csv` (Exp ID, batch, thời gian, seed, tham số, chiến lược, metric, Recall ba lớp hiếm, thời gian, mã commit).
- Xuất bản Excel tổng hợp bằng `export_excel()` trong `src/experiment_log.py`, mã thí nghiệm được ghép với tên người chạy để không trùng nhau.
- Chạy lại cùng cấu hình ở các commit khác nhau cho kết quả giống hệt đến 4 chữ số thập phân.

## 13. Ghi chú về việc dùng tập test

Cell đánh giá trên tập test trong `07_final_eval.ipynb` được chạy hai lần với cùng mô hình, cùng tham số và cùng seed, do mất file cấu hình của ứng dụng demo. Kết quả giống nhau. Không có quyết định nào về mô hình được đưa ra dựa trên kết quả test.

## 14. Phân công


| Thành viên | Nội dung |
|---|---|
| tuan | Hạ tầng dùng chung, tiền xử lý, mô hình cổ điển, tuning Random Forest, `class_weight` trên RF, CV lặp cho các ứng viên cuối, chọn và khóa mô hình |
| thai | MLP và HGB mặc định, tuning HGB, một phần thí nghiệm mất cân bằng (N1 trên HGB, N2 đến N4 trên RF) |
| bao | HGB cố định cấu hình tuned với các biến thể N0 đến N4, tuning MLP, XGBoost |
| khanh | Đánh giá cuối trên test, phân tích lỗi, permutation importance, ablation, đường cong học, demo Streamlit |


