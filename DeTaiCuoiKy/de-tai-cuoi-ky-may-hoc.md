# ĐỀ TÀI CUỐI KỲ MÔN MÁY HỌC

## 1. MỤC TIÊU

Đề tài cuối kỳ giúp sinh viên vận dụng tổng hợp kiến thức và kỹ năng của học phần Máy học để giải quyết một bài toán thực tế dựa trên dữ liệu. Sau khi hoàn thành đề tài, sinh viên có khả năng:

- Xác định bài toán học máy và xây dựng quy trình giải quyết phù hợp.
- Thực hiện các bước thu thập, khám phá, tiền xử lý và xây dựng/lựa chọn đặc trưng từ dữ liệu.
- Lựa chọn, cài đặt và thực nghiệm các thuật toán học máy phù hợp bằng Python và các thư viện như NumPy, Pandas, Scikit-learn hoặc PyTorch.
- Thiết kế thực nghiệm, lựa chọn độ đo, tinh chỉnh siêu tham số và so sánh các mô hình một cách hợp lý.
- Phân tích kết quả, nhận diện hiện tượng overfitting/underfitting, giải thích ưu – nhược điểm của các phương pháp và lựa chọn mô hình phù hợp.
- Trình bày kết quả một cách khoa học, khách quan; tuân thủ quy định về nguồn dữ liệu, bản quyền, trích dẫn và trung thực học thuật.
- Khuyến khích đề xuất cải tiến hoặc xây dựng ứng dụng minh họa cho mô hình.

## 2. HÌNH THỨC THỰC HIỆN

- Nhóm tối đa 4 sinh viên.
- Nếu thực hiện theo nhóm, yêu cầu mức độ phân tích, khối lượng thực nghiệm và chiều sâu thảo luận phải tương xứng với số lượng thành viên.
- Mỗi đề tài phải xác định rõ bài toán cần giải quyết, câu hỏi nghiên cứu hoặc mục tiêu dự đoán/phân nhóm; từ đó thiết kế pipeline học máy phù hợp và sử dụng kết quả thực nghiệm để trả lời câu hỏi đã đặt ra.
- Không đánh giá cao việc sử dụng nhiều thuật toán nếu thiếu mục tiêu và phân tích. Số lượng mô hình không phải tiêu chí chính; chất lượng thiết kế thực nghiệm, tính đúng đắn của pipeline, khả năng giải thích và lập luận dựa trên kết quả mới là trọng tâm.

## 3. NỘI DUNG YÊU CẦU

### 3.1. Xác định bài toán và dữ liệu

- Trình bày bối cảnh, mục tiêu và ý nghĩa của bài toán.
- Xác định input, output và loại bài toán học máy: hồi quy, phân lớp, gom cụm/học không giám sát hoặc mạng nơ-ron cơ bản.
- Nguồn dữ liệu phải rõ ràng, có liên kết hoặc tài liệu mô tả nguồn; ưu tiên dữ liệu công khai từ tổ chức, kho dữ liệu hoặc công trình có uy tín.
- Mô tả dataset: số mẫu, số thuộc tính, kiểu dữ liệu, thuộc tính mục tiêu (nếu có), ví dụ mẫu và các đặc điểm đáng chú ý.

### 3.2. Khám phá và tiền xử lý dữ liệu

- Thực hiện phân tích mô tả ban đầu bằng thống kê và trực quan hóa phù hợp.
- Kiểm tra và xử lý dữ liệu thiếu, trùng lặp, nhiễu, ngoại lệ, sai định dạng hoặc mất cân bằng lớp khi cần.
- Thực hiện encoding, chuẩn hóa/scale dữ liệu nếu mô hình yêu cầu.
- Thực hiện lựa chọn đặc trưng hoặc xây dựng đặc trưng khi phù hợp.
- Mỗi thao tác tiền xử lý cần có lý do và giải thích, tránh áp dụng máy móc.

### 3.3. Thiết kế thực nghiệm

- Xác định chiến lược chia dữ liệu: train/test hoặc train/validation/test; có thể dùng cross-validation khi phù hợp.
- Xây dựng ít nhất một mô hình baseline để làm mốc so sánh.
- Lựa chọn độ đo đánh giá phù hợp với loại bài toán và đặc điểm dữ liệu.
- Nêu rõ cách kiểm soát data leakage và cách đảm bảo phép so sánh giữa các mô hình là công bằng.

### 3.4. Xây dựng mô hình

- Chọn tối thiểu 02 mô hình phù hợp để thực nghiệm và so sánh.
- Các nhóm thuật toán có thể lựa chọn: Regression; Classification; Clustering/Unsupervised Learning; Neural Network cơ bản.
- Thuật toán tham khảo: Linear/Polynomial Regression, Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting/XGBoost, K-Means, DBSCAN, Hierarchical Clustering, PCA hỗ trợ giảm chiều, Perceptron/MLP cơ bản.
- Không bắt buộc một đề tài phải sử dụng tất cả các nhóm thuật toán; việc lựa chọn phải dựa trên bản chất bài toán.

### 3.5. Tinh chỉnh và đánh giá mô hình

- Tinh chỉnh siêu tham số đối với ít nhất một mô hình bằng phương pháp phù hợp.
- So sánh các mô hình bằng cùng dữ liệu đánh giá và cùng tiêu chí.
- Phân tích overfitting/underfitting và độ ổn định của mô hình khi có thể.
- Đối với phân lớp: có thể sử dụng Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC tùy bài toán.
- Đối với hồi quy: có thể sử dụng MAE, MSE/RMSE, R² tùy bài toán.
- Đối với gom cụm: có thể sử dụng Silhouette Score, Inertia và phân tích cấu trúc cụm.

### 3.6. Phân tích, diễn giải và thảo luận

- Không chỉ báo cáo metric; phải giải thích vì sao mô hình tốt hoặc kém hơn, mô hình sai ở đâu và kết quả có hợp lý hay không.
- Phân tích đặc trưng quan trọng, hệ số mô hình, cấu trúc cây, biểu diễn cụm hoặc các yếu tố ảnh hưởng khi phù hợp.
- Trả lời câu hỏi nghiên cứu hoặc mục tiêu đã đặt ra dựa trên bằng chứng từ dữ liệu và kết quả thực nghiệm.

### 3.7. Kết luận và hướng mở rộng

- Tóm tắt kết quả chính và lựa chọn mô hình cuối cùng.
- Nêu hạn chế của dữ liệu, phương pháp và thiết kế thực nghiệm.
- Đề xuất hướng cải tiến hoặc mở rộng trong tương lai.

### 3.8. Sản phẩm nộp

- Báo cáo hoàn chỉnh theo cấu trúc quy định.
- Slide trình bày.
- Mã nguồn minh chứng: Jupyter Notebook/Google Colab/Python script.
- README hướng dẫn cài đặt, chạy chương trình và mô tả cấu trúc mã nguồn.
- Demo ứng dụng bằng Streamlit, Gradio, API hoặc workflow tự động hóa.

## 4. CẤU TRÚC BÁO CÁO

### Chương 1. Giới thiệu

- Bối cảnh và ý nghĩa đề tài.
- Mục tiêu/câu hỏi nghiên cứu.
- Xác định bài toán, input/output và phạm vi nghiên cứu.
- Tổng quan ngắn gọn các nghiên cứu hoặc hướng tiếp cận liên quan.

### Chương 2. Cơ sở lý thuyết và nghiên cứu liên quan

- Các khái niệm và thuật toán được sử dụng.
- Nguyên lý, ưu điểm, hạn chế và lý do lựa chọn các phương pháp.

### Chương 3. Dữ liệu và phương pháp

- Mô tả dữ liệu và nguồn dữ liệu.
- EDA và tiền xử lý.
- Thiết kế pipeline, feature engineering/feature selection.
- Thiết lập thực nghiệm, chiến lược chia dữ liệu, baseline, metric và tuning.

### Chương 4. Thực nghiệm, kết quả và thảo luận

- Kết quả của từng mô hình.
- So sánh các mô hình.
- Phân tích lỗi, overfitting/underfitting, diễn giải kết quả.
- Trả lời câu hỏi nghiên cứu và thảo luận ý nghĩa kết quả.

### Chương 5. Kết luận và hướng phát triển

- Kết quả chính và mô hình được lựa chọn.
- Hạn chế.
- Hướng cải tiến/mở rộng.

## 5. QUY ĐỊNH VỀ DỮ LIỆU VÀ TRUNG THỰC HỌC THUẬT

- Dữ liệu phải có nguồn gốc rõ ràng. Không sử dụng dữ liệu tự tạo tùy tiện để làm đẹp kết quả.
- Nếu sử dụng dữ liệu tự thu thập hoặc dữ liệu nội bộ, phải mô tả cách thu thập, phạm vi sử dụng và các vấn đề riêng tư/bảo mật có liên quan.
- Phải nộp sản phẩm gốc của nhóm và chịu trách nhiệm về tính trung thực của kết quả.
- Được phép tham khảo mã nguồn, notebook, thư viện và công cụ AI, nhưng phải trích dẫn hoặc ghi rõ nguồn hỗ trợ khi có sử dụng.
- Không được sao chép báo cáo, mã nguồn hoặc kết quả thực nghiệm của nhóm khác.
- Kết quả phải có khả năng tái lập từ mã nguồn và dữ liệu đã công bố hoặc mô tả.

## 6. RUBRIC ĐÁNH GIÁ

| Tiêu chí                                              | Mô tả yêu cầu                                                                                                                               | Điểm tối đa |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| 1. Xác định bài toán, mục tiêu và dữ liệu             | Bài toán rõ ràng; xác định input/output; mục tiêu/câu hỏi cụ thể; dataset hợp lệ, có nguồn và mô tả đầy đủ.                                 | 10          |
| 2. Khám phá và tiền xử lý dữ liệu                     | EDA hợp lý; xử lý missing/outlier/encoding/scaling; feature selection/engineering khi cần; giải thích pipeline xử lý.                       | 15          |
| 3. Thiết kế thực nghiệm                               | Chia dữ liệu đúng; có baseline; chọn metric phù hợp; tránh data leakage; chiến lược validation hợp lý.                                      | 10          |
| 4. Xây dựng và cài đặt mô hình                        | Chọn tối thiểu 02 mô hình phù hợp; cài đặt đúng; mô tả thuật toán và tham số; mã nguồn có tổ chức.                                          | 20          |
| 5. Đánh giá, tuning và so sánh mô hình                | Có tinh chỉnh siêu tham số; đánh giá bằng metric phù hợp; so sánh công bằng; phân tích overfitting/underfitting; lựa chọn mô hình có cơ sở. | 20          |
| 6. Phân tích kết quả và thảo luận                     | Giải thích kết quả, phân tích lỗi/đặc trưng/cấu trúc cụm khi phù hợp; trả lời câu hỏi nghiên cứu dựa trên bằng chứng.                       | 10          |
| 7. Báo cáo và trình bày                               | Đúng cấu trúc; nội dung logic; biểu bảng rõ ràng; ngôn ngữ khoa học; trích dẫn đầy đủ.                                                      | 10          |
| 8. Mã nguồn, khả năng tái lập và trung thực học thuật | Nộp đủ source code/notebook và README; có thể chạy lại; ghi nguồn hỗ trợ; tuân thủ đạo đức học thuật.                                       | 5           |
| **Tổng điểm**                                         |                                                                                                                                             | **100**     |

**Điểm cộng** (tối đa +10 điểm, điểm học phần cuối cùng không vượt quá thang điểm quy định): dành cho đề tài có bài toán hoặc dữ liệu khó, phân tích sâu, cải tiến hợp lý, thực nghiệm bổ sung có giá trị, hoặc demo ứng dụng tốt và có ý nghĩa.

## 7. LƯU Ý KHI THỰC HIỆN

- Không chạy nhiều thuật toán chỉ để chọn mô hình có metric cao nhất. Cần giải thích lý do chọn mô hình và ý nghĩa của kết quả.
- Không sử dụng test set để lựa chọn mô hình hoặc tinh chỉnh siêu tham số.
- Không bắt buộc đạt độ chính xác cao nếu bài toán khó; cách thiết kế thực nghiệm, tính đúng đắn của phương pháp và khả năng phân tích kết quả được xem trọng.
- PCA là kỹ thuật giảm chiều hỗ trợ và có thể sử dụng khi phù hợp; không bắt buộc mọi đề tài phải dùng PCA.
- Demo bằng Streamlit/Gradio/API là nội dung khuyến khích, không thay thế yêu cầu cốt lõi về Machine Learning.
