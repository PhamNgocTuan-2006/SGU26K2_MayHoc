# Proposal – Phát hiện và phân loại tấn công mạng IoT bằng Machine Learning (RT-IoT2022)

Nov 4, 2026 · @Tuân

**Học phần:** Máy học · **Loại bài toán:** Phân lớp đa lớp (chính) và nhị phân (phụ) · **Lĩnh vực:** An ninh mạng, IoT · **Dữ liệu:** RT-IoT2022 (UCI)

## Thông tin nhóm

**Tên nhóm:** \[điền tên nhóm\] · **Nhóm trưởng:** \[điền họ tên và MSSV\]

| STT | Họ và tên | MSSV | Nhiệm vụ dự kiến |
| --- | --- | --- | --- |
| 1 | Phạm Ngọc Tuân | 3124411333 | Điều phối, thiết kế thực nghiệm, tổng hợp báo cáo |
| 2 | Nguyễn Phúc Gia Bảo | 3124411031 | Tìm hiểu dữ liệu, EDA và trực quan hóa |
| 3 | Tạ Văn Đức Khanh | 3124411126 | Tiền xử lý, xây dựng và tinh chỉnh mô hình |
| 4 | Lê Quốc Thái | 3124411268 | Đánh giá, phân tích lỗi, demo Streamlit và slide |

## 1. Giới thiệu bài toán

Mục tiêu của đề tài là xây dựng và so sánh các mô hình học có giám sát để phân loại luồng mạng IoT thành các nhãn bình thường và từng kiểu tấn công, dựa trên bộ dữ liệu công khai RT-IoT2022.

### 1.1. Bối cảnh và ý nghĩa

Internet vạn vật (IoT) kết nối cảm biến, thiết bị gia dụng, hệ thống giám sát và thiết bị công nghiệp qua mạng. Nhiều thiết bị có tài nguyên hạn chế nên luồng dữ liệu của chúng có thể bị lợi dụng cho các hoạt động tấn công như quét cổng, từ chối dịch vụ hay giả mạo ARP. Việc nhận diện luồng bất thường và phân loại kiểu tấn công vì vậy có ý nghĩa với giám sát và bảo vệ hệ thống IoT.

Đề tài dùng các đặc trưng luồng mạng đã trích xuất sẵn. Phạm vi chỉ là khả năng phân loại trên dữ liệu được cung cấp; nhóm không tuyên bố thay thế hệ thống phòng thủ triển khai thực tế. Demo chỉ phân tích tệp luồng đã trích xuất đặc trưng, không dùng để chặn lưu lượng trên mạng thật.

### 1.2. Mục tiêu nghiên cứu

1. Khảo sát cấu trúc, chất lượng và phân bố nhãn của RT-IoT2022, đặc biệt mức độ mất cân bằng giữa các lớp.
2. Xây dựng quy trình xử lý dữ liệu có kiểm soát rò rỉ dữ liệu và giảm ảnh hưởng của mất cân bằng lớp.
3. Huấn luyện, tinh chỉnh và so sánh công bằng các mô hình (Decision Tree, Random Forest và một mô hình khác họ) với mô hình mốc.
4. Phân tích ma trận nhầm lẫn, độ quan trọng đặc trưng và hiện tượng overfitting/underfitting.

### 1.3. Câu hỏi nghiên cứu

- **RQ1:** Khi xử lý mất cân bằng lớp bằng `class_weight='balanced'` hoặc SMOTE (áp dụng đúng cách, chỉ trong từng fold huấn luyện), phương pháp nào cải thiện Macro-F1 và recall của các lớp ít mẫu tốt hơn so với không xử lý?
- **RQ2:** Random Forest có cho Macro-F1 cao hơn và ổn định hơn Decision Tree không, khi cả hai được tinh chỉnh với cùng ngân sách?
- **RQ3:** Các loại luồng nào thường bị nhầm lẫn với nhau, và những đặc trưng nào đóng góp nhiều nhất cho dự đoán?

Mức độ mất cân bằng và chất lượng dữ liệu (câu hỏi khảo sát ban đầu) sẽ được trả lời ngay trong phần EDA, làm cơ sở cho ba câu hỏi trên.

### 1.4. Nghiên cứu liên quan

Bộ dữ liệu RT-IoT2022 do Sharmila và Nagapadma công bố và được dùng trong các nghiên cứu phát hiện xâm nhập cho thiết bị IoT hạn chế tài nguyên \[3\]. Các nghiên cứu IDS trên IoT thường dùng mô hình dựa trên cây (Decision Tree, Random Forest) và các biến thể boosting vì chúng xử lý tốt dữ liệu dạng bảng và dễ diễn giải. Nhóm sẽ bổ sung 2–3 bài báo IDS cho IoT và một tài liệu về xử lý mất cân bằng lớp (SMOTE) trong bản báo cáo cuối kỳ.

## 2. Xác định bài toán máy học

Bài toán chính là **phân lớp đa lớp có giám sát** theo cột `Attack_type`. Bài toán nhị phân (bình thường / tấn công) là thí nghiệm phụ để so sánh độ khó.

- **Đầu vào:** các đặc trưng của mỗi flow mạng IoT, ví dụ thời gian flow, số gói tin gửi/nhận, kích thước payload, số cờ TCP (SYN, ACK, FIN…), tốc độ gói tin, giao thức và dịch vụ. Tên cột cụ thể sẽ được liệt kê sau khi đọc file dữ liệu.
- **Đầu ra:** nhãn luồng, một trong các giá trị của `Attack_type` (một số dạng luồng bình thường và các kiểu tấn công như DoS SYN, ARP poisoning, các kiểu quét Nmap, Slowloris, brute-force SSH).
- **Nhãn nhị phân:** `Attack_type` chứa cả các lớp luồng bình thường, nên nhóm sẽ tự ánh xạ các nhãn sang "bình thường" hoặc "tấn công" và ghi rõ bảng ánh xạ trong báo cáo.
- **Phạm vi:** phân loại trên dữ liệu tĩnh đã trích xuất đặc trưng, không xử lý gói tin thô hay phát hiện thời gian thực.

## 3. Mô tả dữ liệu

RT-IoT2022 có 123.117 mẫu và 83 đặc trưng, không có giá trị thiếu theo mô tả của UCI, nhưng lớp lớn nhất là một loại tấn công chứ không phải luồng bình thường \[1\].

| Thông số | Giá trị | Ghi chú |
| --- | --- | --- |
| Nguồn | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/942/rt-iot2022) | Hiến tặng 01/2024; DOI 10.24432/C5P338; giấy phép CC BY 4.0 |
| Số mẫu | 123.117 | Mỗi dòng là một flow mạng |
| Số đặc trưng | 83 | Kiểu số thực, số nguyên và phân loại (ví dụ `proto`, `service`); nhóm sẽ đếm lại trên file CSV, kể cả cột chỉ số nếu có |
| Biến mục tiêu | `Attack_type` | Gồm các lớp luồng bình thường và các kiểu tấn công |
| Mất cân bằng | Rất nặng | `DOS_SYN_Hping` khoảng 94.659 mẫu (\~77%); `Metasploit_Brute_Force_SSH` 37 mẫu và `NMAP_FIN_SCAN` 28 mẫu |

Các con số theo từng lớp lấy từ trang UCI và sẽ được nhóm kiểm chứng lại trực tiếp trên file CSV trước khi đưa vào báo cáo. Trang UCI chỉ hiển thị một phần bảng biến, và một số cột (như `service`) bị gán kiểu chưa hợp lý, nên kiểu dữ liệu sẽ được kiểm tra thủ công.

Dữ liệu được sinh ra từ một testbed IoT công khai của nhóm tác giả, vì vậy kết quả có thể không tổng quát sang mạng khác; đây là một hạn chế sẽ nêu trong chương kết luận.

## 4. Khám phá và tiền xử lý dữ liệu

Mỗi thao tác tiền xử lý chỉ được thực hiện khi có căn cứ từ EDA, và lý do sẽ được ghi trong báo cáo.

### 4.1. Khám phá dữ liệu (EDA)

- Kiểm tra kích thước bảng, kiểu dữ liệu, thống kê mô tả, giá trị khuyết, cột hằng và giá trị bất thường.
- Đếm bản ghi trùng lặp và gần trùng lặp, vì flow sinh từ một kịch bản tấn công thường giống nhau.
- Vẽ phân bố nhãn (thang tuyến tính và log), phân bố đặc trưng theo lớp, ma trận tương quan giữa các đặc trưng số.

### 4.2. Tiền xử lý và kiểm soát rò rỉ

- **Loại bỏ trước khi chia dữ liệu:** cột chỉ số (index), bản ghi trùng lặp và cột hằng.
- **Kiểm tra cột có nguy cơ rò rỉ:** `id.orig_p` và `id.resp_p` (cổng) có thể "tiết lộ" loại tấn công do kịch bản sinh dữ liệu. Nhóm chạy thí nghiệm có và không có các cột này, rồi báo cáo sự khác biệt.
- **Mã hóa** các biến phân loại (`proto`, `service`) bằng One-Hot hoặc Ordinal tùy số giá trị khác nhau.
- **Chuẩn hóa** (Min-Max hoặc Standard) chỉ áp dụng cho mô hình nhạy với thang đo (Logistic Regression, KNN, SVM hoặc MLP); cây và Random Forest dùng dữ liệu gốc.
- **Lựa chọn đặc trưng:** loại đặc trưng tương quan rất cao (ngưỡng cho trước, ví dụ |r| > 0,95) để giảm dư thừa, và so sánh kết quả trước/sau. PCA chỉ dùng nếu cần cho trực quan hóa hoặc mô hình dùng khoảng cách.

### 4.3. Xử lý mất cân bằng lớp

Nhóm so sánh ba cách: không xử lý, `class_weight='balanced'`, và SMOTE (SMOTENC nếu còn cột phân loại). SMOTE chỉ sinh mẫu trên phần huấn luyện của từng fold, không bao giờ trên validation hay test. Với lớp chỉ 28–37 mẫu, SMOTE có thể sinh mẫu kém đa dạng, nên `k_neighbors` phải nhỏ hơn số mẫu của lớp hiếm trong fold.

### 4.4. Pipeline

Toàn bộ tiền xử lý (encoder, scaler, SMOTE) được đóng gói bằng `imblearn.pipeline.Pipeline`, vì `Pipeline` của scikit-learn không hỗ trợ bước lấy mẫu. Mọi bước chỉ `fit` trên dữ liệu huấn luyện để hạn chế rò rỉ dữ liệu.

## 5. Mô hình dự kiến

Nhóm dùng một mô hình mốc, hai mô hình dựa trên cây và một mô hình khác họ để so sánh có mục đích, không chạy nhiều thuật toán chỉ để tìm metric cao nhất.

| Mô hình | Vai trò | Lý do chọn | Cần chuẩn hóa |
| --- | --- | --- | --- |
| Dummy Classifier (`most_frequent`, `stratified`) | Mốc tối thiểu | Xác định mô hình có học được tín hiệu hữu ích hay không | Không |
| Logistic Regression | Mốc có học, tuyến tính | Đơn giản, cho biết dữ liệu có tách tuyến tính được không; hệ số dễ diễn giải | Có |
| Decision Tree | Mô hình dễ diễn giải | Cho phép xem quy tắc chia và so sánh với mô hình tổ hợp | Không |
| Random Forest | Mô hình chính | Kết hợp nhiều cây, mô hình hóa quan hệ phi tuyến, cung cấp độ quan trọng đặc trưng | Không |
| MLP cơ bản (hoặc KNN/SVM) | Mô hình khác họ | Đối chiếu với họ cây; chọn một trong các mô hình này theo thời gian chạy trên 123 nghìn mẫu | Có |

Nếu còn thời gian, nhóm có thể thêm Gradient Boosting/XGBoost như một thí nghiệm mở rộng. Giả định cần kiểm chứng: các mô hình cây sẽ vượt trội vì dữ liệu dạng bảng có nhiều ngưỡng cứng (cờ TCP, cổng), nhưng kết luận phải dựa trên số liệu thực nghiệm.

## 6. Kế hoạch thực nghiệm

Test set chỉ dùng một lần ở bước đánh giá cuối; mọi lựa chọn mô hình và siêu tham số dựa trên cross-validation trên phần huấn luyện.

### 6.1. Chia dữ liệu và kiểm soát rò rỉ

- Chia **stratified** 80% huấn luyện / 20% kiểm tra cuối (test), sau khi đã loại bản ghi trùng lặp. Với lớp 28 mẫu, test chỉ còn khoảng 6 mẫu, nên nhóm nêu rõ đây là hạn chế và báo cáo thêm kết quả Stratified K-Fold lặp lại (ví dụ 5×3) kèm độ lệch chuẩn.
- Tuning và so sánh mô hình dùng Stratified K-Fold trên tập huấn luyện, không tách thêm tập validation riêng để không làm lớp hiếm càng ít mẫu.
- Nếu EDA cho thấy các bản ghi liên hệ theo phiên, thiết bị hoặc thời gian, nhóm so sánh thêm cách chia theo nhóm hoặc theo thời gian để kiểm tra kết quả có bị lạc quan do near-duplicate hay không.
- Ghi lại `random_state`, phiên bản thư viện và cấu hình để tái lập.

### 6.2. Độ đo

- **Độ đo chính:** Macro-F1, vì mỗi lớp có trọng số bằng nhau và các lớp hiếm là phần khó.
- **Bổ sung:** balanced accuracy, MCC, weighted-F1, precision/recall theo từng lớp, accuracy và ma trận nhầm lẫn. Accuracy chỉ để tham khảo vì lớp lớn chiếm khoảng 77%.

### 6.3. Tinh chỉnh và so sánh công bằng

- Tuning **cả Decision Tree và Random Forest** bằng `RandomizedSearchCV`/`GridSearchCV` với cùng số lần thử, cùng CV, cùng độ đo, để phép so sánh công bằng. Tham số xem xét: `max_depth`, `min_samples_split`, `min_samples_leaf`, và với Random Forest thêm `n_estimators`, `max_features`.
- Các mô hình còn lại dùng lưới siêu tham số nhỏ, nhưng cùng quy trình CV.
- Tất cả mô hình được đánh giá trên cùng các fold và cùng test set.

### 6.4. Overfitting/underfitting và phân tích lỗi

- So sánh điểm huấn luyện với điểm CV và vẽ đường cong học (learning curve), đường cong theo `max_depth` để nhận diện overfitting.
- Đánh giá độ ổn định qua độ lệch chuẩn giữa các fold.
- Phân tích ma trận nhầm lẫn, các mẫu bị dự đoán sai, độ quan trọng đặc trưng (impurity và permutation importance), và cấu trúc cây quyết định.

## 7. Sản phẩm bàn giao và kết quả kỳ vọng

Nhóm nộp đủ năm sản phẩm theo đề bài, và kết quả được trình bày theo bằng chứng thực nghiệm chứ không cam kết trước mức độ chính xác.

- Báo cáo hoàn chỉnh (PDF) theo 5 chương quy định.
- Slide trình bày.
- Mã nguồn Jupyter Notebook/Colab kèm `requirements.txt`, chạy lại được từ đầu.
- README hướng dẫn cài đặt, chạy chương trình và mô tả cấu trúc mã nguồn.
- Demo Streamlit: nhận tệp CSV luồng đã trích xuất đặc trưng, trả về nhãn dự đoán và xác suất từng lớp.

**Kết quả kỳ vọng:** quy trình thực nghiệm minh bạch và tái lập được; so sánh có căn cứ giữa các mô hình; phân tích nguyên nhân nhầm lẫn qua ma trận nhầm lẫn; danh sách đặc trưng quan trọng nhất; và câu trả lời cho RQ1–RQ3 dựa trên dữ liệu.

## 8. Kế hoạch thực hiện dự kiến

Khối lượng được chia đều hơn cho cả bốn người, demo Streamlit bắt đầu từ Tuần 3 và Tuần 4 có thời gian dự phòng.

| Giai đoạn | Công việc | Phụ trách |
| --- | --- | --- |
| Tuần 1 | Đọc tài liệu, tải và kiểm tra dataset, chốt câu hỏi nghiên cứu, hoàn thiện tài liệu tham khảo | Phạm Ngọc Tuân, Nguyễn Phúc Gia Bảo |
| Tuần 2 | EDA, làm sạch, xây dựng pipeline và baseline | Nguyễn Phúc Gia Bảo, Tạ Văn Đức Khanh |
| Tuần 3 | Huấn luyện mô hình, tuning, đánh giá bằng CV; dựng khung demo Streamlit | Tạ Văn Đức Khanh, Phạm Ngọc Tuân, Lê Quốc Thái |
| Tuần 4 | Đánh giá trên test, phân tích lỗi, hoàn thiện báo cáo, slide, demo, README | Cả nhóm (Lê Quốc Thái chủ trì demo và slide; Phạm Ngọc Tuân tổng hợp báo cáo) |
| Dự phòng | 2–3 ngày rà soát, chạy lại mã nguồn từ đầu, tập trình bày | Cả nhóm |

## 9. Tài liệu tham khảo và cam kết trung thực học thuật

&#91;1\] B. S. và R. Nagapadma, "RT-IoT2022," UCI Machine Learning Repository, 2023. DOI: 10.24432/C5P338. <https://archive.ics.uci.edu/dataset/942/rt-iot2022>

&#91;2\] Scikit-learn Developers, *scikit-learn: Machine Learning in Python*. <https://scikit-learn.org/stable/>

&#91;3\] B. S. Sharmila và R. Nagapadma, "Quantized autoencoder (QAE) intrusion detection system for anomaly detection in resource-constrained IoT devices using RT-IoT2022 dataset," *Cybersecurity*, 2023. <https://cybersecurity.springeropen.com/articles/10.1186/s42400-023-00178-5> (nhóm kiểm tra lại thông tin trích dẫn trước khi nộp)

&#91;4\] Imbalanced-learn Developers, *imbalanced-learn documentation*. <https://imbalanced-learn.org/stable/>

&#91;5\] N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, 16, 2002.

**Cam kết:** nhóm sẽ nộp sản phẩm gốc, ghi rõ nguồn dữ liệu (CC BY 4.0), và ghi rõ các phần có tham khảo mã nguồn, notebook hoặc công cụ AI trong báo cáo và README. Kết quả có thể tái lập từ mã nguồn và dữ liệu đã công bố.
