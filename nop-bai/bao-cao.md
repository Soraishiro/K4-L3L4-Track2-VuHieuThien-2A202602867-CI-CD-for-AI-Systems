# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Vũ Hiếu Thiên |
| MSSV | 2A202602867 |
| Lớp / Khóa | K4 |
| Repo | https://github.com/Soraishiro/K4-L3L4-Track2-VuHieuThien-2A202602867-CI-CD-for-AI-Systems |
| Nền tảng | AWS S3 + EC2 (ap-southeast-1) |

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | **0.7149** | 0.8740 |

**Đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5` — f1 cao nhất, vượt ngưỡng 0.65.

Đánh đổi giữa `n_estimators` và `learning_rate` thể hiện rõ ở lần chạy 2: hạ `learning_rate` còn 0.05 đòi hỏi tăng `n_estimators` để bù, nhưng khi cả hai cùng giảm (50 cây, depth 2) mô hình quá nông để học tương tác phi tuyến nên f1 tụt xuống 0.6051. Đáng chú ý: lần chạy 1 có accuracy cao nhất (0.8780) nhưng f1 lại thấp hơn lần chạy 3.

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24,8% mẫu thuộc lớp thu nhập cao. Mô hình luôn dự đoán "thu nhập thấp" vẫn đạt accuracy 75,2% mà không học được gì, nên accuracy dễ tạo cảm giác an toàn giả. F1 của lớp dương gộp precision (không báo oan) và recall (không bỏ sót), phản ánh đúng năng lực phân biệt trên lớp thiểu số. Vì vậy `src/train.py` gọi `f1_score(y_eval, preds)` không truyền `average`, và Quality Gate chặn khi `f1_score < 0.65`.

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| `dvc pull` báo 403 | DVC (qua s3fs) không đọc được secret JSON trong cùng step | Ghi `~/.aws/credentials` + export key qua `$GITHUB_ENV` |
| Push không trigger workflow | `workflow_dispatch` chạy được nhưng `push` không chạy | Unregister rồi đăng ký lại workflow (id mới), đổi tên `ci-cd.yml` |
| MLflow 3.x lỗi 404 với DagsHub | DagsHub dùng API MLflow cũ, thiếu endpoint `logged-models` | `dagshub.init(..., patch_mlflow=True)` |

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 — 22.361 mẫu | 0.7149 | 0.8740 |
| Bước 3 — 44.722 mẫu | 0.7354 | 0.8820 |

**Nhận xét:** F1 tăng nhẹ 0.7149 → 0.7354. Hai tập chia ngẫu nhiên từ cùng nguồn điều tra nên có chung phân phối; dữ liệu bổ sung chủ yếu ổn định ước lượng chứ không thêm tri thức mới. Giá trị của Bước 3 là chứng minh tự động hóa: chỉ cần `dvc push` rồi `git push` tệp `.dvc`, cả 4 job tự chạy xanh và VM được restart với mô hình mới.

## 5. Phần Bonus Đã Thực Hiện

- [x] **Bonus 1 — DagsHub**: `dagshub.init(..., patch_mlflow=True)`, token ở secret `DAGSHUB_TOKEN`.
- [x] **Bonus 2 — Ngưỡng tối ưu**: quét 0.1–0.9 → ngưỡng tối ưu 0.30, F1 tăng lên 0.7537.
- [x] **Bonus 3 — Precision/Recall**: precision 0.8283, recall 0.6613, `[[tn=359 fp=17][fn=42 tp=82]]`.
- [x] **Bonus 4 — Rollback**: Quality Gate chỉ release khi F1 mới ≥ F1 cũ trên S3 (đã chặn ở F1 0.5907 < 0.7354).
- [x] **Bonus 5 — Lệch lạc dữ liệu**: cảnh báo khi tỷ lệ lớp dương lệch >5% so với mốc 24,8% (thực tế 24,78%).