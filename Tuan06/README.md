# Giải đáp Lý thuyết EDA Lab

## Phần 1.1.1: Thống kê mô tả
1. **Khác biệt giữa Thống kê mô tả và Thống kê suy luận:**
   - *Thống kê mô tả:* Tóm tắt, trình bày đặc tính cơ bản của tập dữ liệu thu thập được.
   - *Thống kê suy luận:* Đưa ra dự đoán, giả thuyết hoặc kết luận cho tổng thể (population) dựa trên mẫu (sample).
2. **Các thước đo & Khi nào dùng Trung vị:**
   - Dùng *Trung vị (Median)* khi dữ liệu có chứa giá trị ngoại lai (outliers) hoặc bị lệch (skewed), vì trung vị không bị ảnh hưởng mạnh bởi cực trị như trung bình.
3. **Phân bố dữ liệu:**
   - Xác định bằng Histogram hoặc KDE Plot. Các loại phổ biến: Phân bố chuẩn (Bell curve), lệch trái (Negative skew), lệch phải (Positive skew).
4. **Độ lệch chuẩn vs Phạm vi (Range):**
   - *Range:* Khoảng chênh lệch giữa Max và Min (nhạy cảm với outliers).
   - *Standard Deviation:* Mức độ phân tán của dữ liệu xung quanh giá trị trung bình.
5. **Q1, Q2, Q3 trong Boxplot:**
   - *Q1 (25%):* Tứ phân vị thứ nhất.
   - *Q2 (50%):* Trung vị (Median).
   - *Q3 (75%):* Tứ phân vị thứ ba. Khoảng IQR = Q3 - Q1.

## Phần 1.2.1: Trực quan hóa dữ liệu
- **Matplotlib:** Thư viện vẽ biểu đồ nền tảng, cho phép tùy biến chi tiết.
- **Seaborn:** Xây dựng trên Matplotlib, hỗ trợ vẽ biểu đồ thống kê đẹp mắt với cú pháp ngắn gọn.
- **Plotly:** Hỗ trợ biểu đồ tương tác (interactive charts).

## Phần 1.3.1: Phân tích đơn biến và hai biến
- **Đơn biến (Univariate):** Phân tích đặc trưng của từng biến độc lập (Histogram, Boxplot, Countplot).
- **Hai biến (Bivariate):** Phân tích mối quan hệ, sự tương quan giữa hai biến (Scatter plot, Heatmap, Crosstab).