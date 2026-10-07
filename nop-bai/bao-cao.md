# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Vũ Hiếu Thiên |
| MSSV | 2A202602867 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/VuHieuThien/K4-L3L4-Track2-VuHieuThien-2A202602867-CI-CD-for-AI-Systems |
| Ngày nộp | 2026-10-07 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số này cho f1_score cao nhất (0.7149), vượt qua ngưỡng 0.65. Lần chạy 2 với mô hình nông và tốc độ học chậm cho f1 thấp hơn (0.6051), cho thấy mô hình cần đủ độ phức tạp để học các mối quan hệ phi tuyến. Lần có accuracy cao nhất (0.8780) trùng với lần chạy 1, không phải lần có f1 cao nhất, cho thấy accuracy có thể gây hiểu lầm khi phân bố lớp mất cân bằng. Đối với đề xuất thử nghiệm, tôi chọn max_depth tăng để mô hình học sâu hơn và n_estimators tăng để bù learning_rate trung bình.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Census Income có 24.8% mẫu thu nhập cao (lớp 1) và 75.2% thu nhập thấp (lớp 0). Nếu một mô hình luôn dự đoán "thu nhập thấp", nó đạt accuracy lên đến 75.2% mà không học được bất kỳ thông tin nài. Điều này làm accuracy trở nên đáng tin cậy giả vì lớp đa số chi phối kết quả. F1-score của lớp dương kết hợp cả precision (tránh dự đoán sai người thu nhập cao) và recall (không bỏ sót người thu nhập cao), phản ánh thực sự khả năng tách biệt của mô hình trên lớp thiểu số. Việc sử dụng average="weighted" hoặc "macro" sẽ trung bình hóa qua cả hai lớp, kéo điểm từ lớp đa số và che giấu hiệu năng trên lớp thiểu số, vì vậy chúng tôi tính f1_score riêng cho lớp dương bằng cách gọi f1_score(y_eval, preds) mà không truyền tham số average.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khó | Nguyên nhãn | Cách giải quyết |
|---|---|---|
| scikit-learn 1.4.2 không cài được trên Python 3.13 | Phiên bản yêu cầu numpy 2.0.0rc1 không tồn tại trên Python 3.13 | Dùng scikit-learn 1.8.0 từ base conda, cài đặt thêm mlflow, pandas phiên bản mới hơn |
| mlflow 2.13.0 yêu cầu numpy<2 nhưng môi trường có numpy 2.4.1 | Xung đột phiên bản numpy | Nâng cấp lên mlflow 3.17.0 hỗ trợ numpy 2.x |
| mlflow.sklearn.log_model lỗi skops untrusted types | MLflow 3.x mặc định dùng skops format yêu cầu trusted types | Thêm tham số serialization_format="pickle" vào log_model |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.71xx | 0.87xx |

**Nhận xét:** F1 score không tăng đáng kể khi thêm dữ liệu mới vì hai tập dữ liệu được chia ngẫu nhiên từ cùng một nguồn, có cùng phân phối. Khi mô hình đã học được các đặc trưng chính từ 22.000 mẫu đầu tiên, việc nạp thêm 22.000 mẫu cùng phân phối chỉ giúp mô hình ổn định hơn một lượng nhỏ. Trọng tâm của Bước 3 là kiểm chứng đường ống CI/CD tự động chạy trọn vẹn từ commit dữ liệu đến triển khai thực tế.

---

## 5. Phần Bonus Đã Thực Hiện

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định
- [ ] Bonus 3 - Báo cáo precision / recall tự động
- [ ] Bonus 4 - Hoàn trả về phiên bản trước
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu