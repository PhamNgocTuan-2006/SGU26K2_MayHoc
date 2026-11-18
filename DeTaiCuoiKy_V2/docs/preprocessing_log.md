# Nhật ký Tiền xử lý Dữ liệu (Data Preprocessing Log)

Tài liệu này ghi nhận chi tiết quy trình làm sạch, biến đổi và chia tách tập dữ liệu **RT-IoT2022**, phục vụ trực tiếp cho phần trình bày phương pháp thực nghiệm ở Chương 3 của báo cáo.

---

## 1. Tổng quan dữ liệu thô & Làm sạch cơ bản

- **Kích thước ban đầu:** $123.117$ dòng và $85$ cột (bao gồm $84$ cột đặc trưng/chỉ số và $1$ cột nhãn `Attack_type`).
- **Kiểm tra dữ liệu khuyết thiếu (Missing Values):** $0$ giá trị trống (`NaN`/`Null`) trên toàn bộ tập dữ liệu.
- **Loại bỏ cột chỉ số thừa:** Đã xóa cột `Unnamed: 0` vì đây chỉ là cột đánh số thứ tự hàng phát sinh trong quá trình xuất file CSV, không mang ý nghĩa đặc trưng mạng.
- **Loại bỏ cột hằng số (Zero-variance):** Đã xóa cột `bwd_URG_flag_count` do cột này chỉ chứa một giá trị duy nhất ($0$) trên toàn bộ dữ liệu, không đóng góp thông tin cho việc phân loại. _(Lưu ý: Cột này đã được loại bỏ chủ động trước khi chạy bước lọc tương quan)._

---

## 2. Xử lý dữ liệu trùng lặp (Deduplication)

- **Thực hiện:** Xóa các bản ghi trùng lặp hoàn toàn (`drop_duplicates()`) trước khi chia tập dữ liệu nhằm ngăn chặn tình trạng cùng một mẫu dữ liệu xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử.
- **Kết quả:**
  - Số dòng trước khi xóa trùng: **$123.117$**
  - Số dòng sau khi xóa trùng: **$117.922$** (loại bỏ $5.195$ bản ghi trùng lặp).
- **Tác động lên phân bố lớp:**
  - Lớp thiểu số `Wipro_bulb` giảm từ **$253$ xuống còn $219$ mẫu** (giảm $34$ mẫu, tương đương $\approx 13,4\%$).
  - Tỷ lệ chiếm hữu của lớp đa số `DOS_SYN_Hping` giảm nhẹ từ $76,89\%$ xuống còn **$76,40\%$** ($90.089 / 117.922$ mẫu).

---

## 3. Mã hóa và Chia tách tập dữ liệu (Encoding & Train-Test Split)

- **Mã hóa nhãn (`LabelEncoder`):** Chuyển đổi $12$ nhãn dạng chuỗi của cột `Attack_type` sang các giá trị số nguyên từ $0$ đến $11$ (theo thứ tự bảng chữ cái: `ARP_poisioning`: 0, ..., `Wipro_bulb`: 11).
- **Mã hóa biến phân loại (`One-Hot Encoding`):** Áp dụng biến đổi nhị phân ($0/1$) cho hai cột định danh `proto` và `service` nhằm tránh tạo ra thứ tự số học giả tạo giữa các giao thức/dịch vụ.
- **Chia tách dữ liệu:** Sử dụng kỹ thuật chia phân tầng (`stratify=y`) với tỷ lệ **$80\%$ Train / $20\%$ Test** và cố định `random_state=42` để đảm bảo tính tái lập của thực nghiệm:
  - **Tập huấn luyện (`X_train`):** **$94.337$ mẫu** ($80\%$).
  - **Tập kiểm thử (`X_test`):** **$23.585$ mẫu** ($20\%$).

---

## 4. Giảm chiều dữ liệu bằng Lọc tương quan (Correlation Filtering)

- **Nguyên tắc chống rò rỉ dữ liệu (Data Leakage Prevention):** Ma trận hệ số tương quan Pearson tuyệt đối được tính toán **độc lập chỉ trên tập huấn luyện (`X_train`)** (loại trừ các cột One-Hot `proto_*` và `service_*`).
- **Ngưỡng lọc:** Loại bỏ các đặc trưng có hệ số tương quan tuyệt đối $> 0,95$ (giữ lại cột xuất hiện trước và loại bỏ cột xuất hiện sau trong ma trận tam giác trên).
- **Kết quả:**
  - Số lượng cột bị loại bỏ: **$24$ cột** (danh sách chi tiết được lưu tại `data/processed/dropped_corr_cols.csv` để đưa vào phụ lục báo cáo).
  - Áp dụng loại bỏ đồng nhất $24$ cột này trên cả hai tập `X_train` và `X_test`.
  - **Số lượng đặc trưng cuối cùng:** Còn lại **$70$ đặc trưng** đầu vào (bao gồm các đặc trưng số gốc đã lọc và các cột One-Hot từ `proto`, `service`) _(Lưu ý: Con số chính xác tùy thuộc vào tổng số danh mục duy nhất của `proto` và `service` sau One-Hot)_.

---

## 5. Chuẩn hóa dữ liệu (Feature Scaling)

- Sử dụng `StandardScaler` đưa các đặc trưng về phân phối chuẩn ($\mu = 0, \sigma = 1$).
- Bộ chuẩn hóa chỉ thực hiện `fit()` trên `X_train` để học trung bình và độ lệch chuẩn, sau đó áp dụng `transform()` cho cả `X_train` và `X_test`.
- Dữ liệu được xuất ra hai phiên bản tại thư mục `data/processed/`: bản gốc chưa chuẩn hóa (dành cho các mô hình cây như Decision Tree, Random Forest) và đối tượng `scaler.pkl` (dành cho các mô hình nhạy cảm với khoảng cách như KNN, SVM, MLP, Logistic Regression).

---

## 6. Các hạn chế của dữ liệu và phương pháp tiền xử lý (Ghi nhận cho phần Thảo luận)

1. **Tồn tại các mẫu gần giống nhau (Near-duplicates):** Việc xóa trùng chỉ loại bỏ các luồng mạng giống hệt nhau $100\%$. Trong các cuộc tấn công tự động (đặc biệt là `DOS_SYN_Hping`), các gói tin sinh ra liên tục có đặc trưng gần như tương đồng vẫn có thể phân bố ở cả tập train và test, khiến kết quả đánh giá trên tập test có thể lạc quan hơn thực tế.
2. **Mất cân bằng lớp nghiêm trọng:** Hai lớp tấn công hiếm có số lượng mẫu rất ít trong tập kiểm thử (`Metasploit_Brute_Force_SSH` chỉ có **$7$ mẫu** và `NMAP_FIN_SCAN` chỉ có **$6$ mẫu**), dẫn đến chỉ số Recall của hai lớp này dễ biến động mạnh chỉ với một vài dự đoán sai.
3. **Thứ tự ưu tiên khi lọc tương quan:** Phương pháp lọc tam giác trên phụ thuộc vào thứ tự sắp xếp cột ban đầu trong bộ dữ liệu (ví dụ: giữ `flow_duration` và loại `flow_iat.tot` do `flow_duration` đứng trước). Ngoài ra, ma trận tương quan chịu sự chi phối lớn bởi lớp đa số `DOS_SYN_Hping` ($\approx 76,4\%$).
4. **Rủi ro học thuộc kịch bản từ cổng mạng (`id.orig_p`, `id.resp_p`):** Hai đặc trưng cổng nguồn và cổng đích được giữ lại trong pipeline chính nhưng có nguy cơ gắn chặt với kịch bản thu thập dữ liệu phòng thí nghiệm (ví dụ cổng `1883` gắn với `MQTT_Publish`). Vấn đề này sẽ được kiểm chứng bằng thực nghiệm đối chứng (có và không có hai cột cổng) ở phần đánh giá mô hình.
