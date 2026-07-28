# Credit Card Fraud Detection - Web Dashboard

> Hệ thống phát hiện giao dịch gian lận thẻ tín dụng thời gian thực sử dụng Machine Learning (LightGBM) kết hợp với Giao diện Web Dashboard (Glassmorphic Design).

---

## 📋 Mục lục

- [Tổng quan](#tong-quan)
- [Điểm nổi bật](#diem-noi-bat)
- [Kiến trúc hệ thống](#kien-truc-he-thong)
- [Công nghệ sử dụng](#cong-nghe-su-dung)
- [Thuật toán và Pipeline](#thuat-toan-va-pipeline)
- [Kết quả Mô hình](#ket-qua-mo-hinh)
- [Cấu trúc Project](#cau-truc-project)
- [Hướng dẫn cài đặt và chạy](#huong-dan-cai-dat-va-chay)
- [Lợi ích và Ứng dụng](#loi-ich-va-ung-dung)

---

## 📌 Tổng quan

Dự án xây dựng một hệ thống **phát hiện giao dịch gian lận thẻ tín dụng (Credit Card Fraud Detection)** hoàn chỉnh End-to-End: từ khâu tiền xử lý dữ liệu thô, xử lý mất cân bằng dữ liệu cực đoan, huấn luyện mô hình học máy LightGBM, tối ưu ngưỡng cảnh báo (threshold), cho đến đóng gói dịch vụ REST API và triển khai giao diện **Web Dashboard** hiện đại cho người dùng cuối.

Hệ thống có khả năng phân tích và đưa ra dự đoán chính xác xác xuất gian lận của từng giao dịch dựa trên **14 đặc trưng (features)** thời gian thực.

---

## ✨ Điểm nổi bật

- **Mô hình LightGBM tối ưu:** Cho độ chính xác cao, khả năng dự đoán cực nhanh (< 1ms per transaction).
- **Xử lý mất cân bằng dữ liệu bằng Hybrid Resampling:** Kết hợp **SMOTE** (Oversampling) và **RandomUnderSampler** (Undersampling) giúp mô hình không bị thiên vị.
- **Tối ưu hóa ngưỡng phát hiện (Threshold Scanning):** Quét F1-optimal tự động tìm ngưỡng tối ưu (Best Threshold = 0.87) thay vì dùng mặc định 0.50, giúp giảm thiểu báo động giả (False Positives).
- **Web Dashboard Glassmorphism:** Giao diện Web hiện đại, trực quan với chế độ Dark Mode, đồng hồ đo rủi ro (Risk Gauge), biểu đồ phân tích ROC Curve, Confusion Matrix và Feature Importance tương tác thời gian thực (Chart.js + Flask REST API).

---

## 🏗️ Kiến trúc hệ thống

```
+------------------------------------------------------------------+
|                        DATA PIPELINE                             |
|                                                                  |
|  RawDataSet.csv --> processing.py --> CleanData.csv               |
|                          |                                       |
|                     (Drop NA, Duplicates)                        |
|                          v                                       |
|                      EDA.py --> dataset_processed.csv             |
|                  (EDA, Drop cc_num, Validate, Outlier Analysis)  |
+----------------------------------+-------------------------------+
                                   |
                                   v
+------------------------------------------------------------------+
|                     TRAINING PIPELINE                            |
|                                                                  |
|  dataset_processed.csv                                           |
|        |                                                         |
|        v                                                         |
|  features.py --> Feature Engineering + StandardScaler            |
|        |                                                         |
|        v                                                         |
|  train_model.py                                                  |
|    +-- Train/Test Split (70/30, stratified)                      |
|    +-- SMOTE (sampling_strategy=0.3)                             |
|    +-- RandomUnderSampler (sampling_strategy=0.5)                |
|    +-- LightGBM Training (800 estimators)                        |
|    +-- Threshold Optimization (F1-optimal scan)                  |
|    +-- Save: model.joblib, scaler.joblib, threshold.txt          |
+----------------------------------+-------------------------------+
                                   |
                                   v
+------------------------------------------------------------------+
|                    INFERENCE / WEB SERVING                       |
|                                                                  |
|                      app_web.py (Flask Server)                   |
|                        /                 \                       |
|                       v                   v                      |
|             REST API Endpoints     Web Dashboard UI              |
|             - /api/predict (POST)  - Glassmorphic Interface      |
|             - /api/model/evaluate  - Real-time Risk Gauge        |
|             - /api/sample/<type>   - ROC & Confusion Matrix      |
+------------------------------------------------------------------+
```

---

## 🛠️ Công nghệ sử dụng

### Machine Learning & Data Science

| Công nghệ | Mục đích sử dụng |
|---|---|
| **Python** | Ngôn ngữ lập trình chính |
| **LightGBM** | Thuật toán Gradient Boosting Decision Tree chính cho bài toán phân loại |
| **scikit-learn** | Tiền xử lý (StandardScaler), chia dữ liệu (train_test_split), tính toán metrics |
| **imbalanced-learn** | Cân bằng dữ liệu (SMOTE + RandomUnderSampler) |
| **pandas & NumPy** | Thao tác, biến đổi và tính toán dữ liệu dạng bảng |
| **joblib** | Lưu trữ và tải mô hình (`.joblib`) |

### Web Application & Frontend

| Công nghệ | Mục đích sử dụng |
|---|---|
| **Flask** | Python Web Framework xây dựng REST API & Web Server |
| **HTML5 & CSS3** | Cấu trúc giao diện và Hệ thống Design Glassmorphism (Dark Mode) |
| **JavaScript (ES6+)** | Xử lý logic Frontend, gọi REST API bất đồng bộ (Fetch API) |
| **Chart.js** | Biểu diễn biểu đồ tương giác (ROC Curve, Feature Importance) |
| **Google Fonts (Inter)** | Typography hiện đại và tối ưu hiển thị |

---

## ⚙️ Thuật toán và Pipeline

### 1. Tiền xử lý dữ liệu (Data Preprocessing)

#### `processing.py` - Làm sạch dữ liệu thô
- **Missing Values:** Phát hiện và loại bỏ các dòng chứa giá trị trống.
- **Deduplication:** Xóa bỏ hoàn toàn các bản ghi bị lặp.
- **Output:** `Data/CleanData.csv`

#### `EDA.py` - Phân tích khám phá dữ liệu
- Thống kê mô tả và phân tích phân phối thuộc tính target `is_fraud`.
- Kiểm tra ma trận tương quan giữa các đặc trưng.
- Phát hiện Outlier bằng phương pháp IQR (Interquartile Range).
- Loại bỏ các thông tin không có tính chất dự đoán (ví dụ: `cc_num`).
- **Output:** `Data/dataset_processed.csv`

### 2. Feature Engineering (`features.py`)

Hệ thống sử dụng **14 đặc trưng (Features)** chính:

| Feature | Kiểu | Mô tả |
|---|---|---|
| `amt` | Numeric | Số tiền giao dịch (USD) |
| `city_pop` | Numeric | Dân số thành phố nơi diễn ra giao dịch |
| `lat` | Numeric | Vĩ độ của chủ thẻ |
| `long` | Numeric | Kinh độ của chủ thẻ |
| `merch_lat` | Numeric | Vĩ độ địa điểm cửa hàng |
| `merch_long` | Numeric | Kinh độ địa điểm cửa hàng |
| `unix_time` | Numeric | Thời gian giao dịch (Unix Timestamp) |
| `distance` | Numeric | Khoảng cách địa lý giữa chủ thẻ và cửa hàng |
| `merchant` | Categorical (Encoded) | Mã định danh cửa hàng |
| `category` | Categorical (Encoded) | Phân loại ngành hàng giao dịch |
| `hour` | Temporal | Giờ thực hiện giao dịch (0 - 23) |
| `day` | Temporal | Ngày trong tháng (1 - 31) |
| `month` | Temporal | Tháng trong năm (1 - 12) |
| `gender` | Binary | Giới tính chủ thẻ (0 = Nữ, 1 = Nam) |

#### Standard Scaling
Chuẩn hóa **8 thuộc tính số** (`amt`, `city_pop`, `distance`, `lat`, `long`, `merch_lat`, `merch_long`, `unix_time`) bằng **StandardScaler**:
$$\displaystyle z = \frac{x - \mu}{\sigma}$$

### 3. Xử lý dữ liệu mất cân bằng (Hybrid Resampling)

Tỷ lệ giao dịch gian lận trong thực tế rất thấp (~0.5%). Nếu huấn luyện trực tiếp, mô hình sẽ bị lệch về lớp giao dịch bình thường.

Hệ thống áp dụng chiến lược **Hybrid Resampling**:
1. **SMOTE (`sampling_strategy=0.3`):** Sinh thêm dữ liệu nhân tạo cho lớp Fraud để đạt tỷ lệ 30% so với Normal.
2. **RandomUnderSampler (`sampling_strategy=0.5`):** Giảm ngẫu nhiên số lượng mẫu Normal xuống còn 2 lần số lượng Fraud.

-> Kết quả tạo ra tỷ lệ cân bằng lý tưởng **Normal : Fraud = 2 : 1** trên tập Train mà không gây Overfitting hay mất mát thông tin quá mức.

### 4. Mô hình LightGBM (`models.py`)

Sử dụng cấu hình **LightGBM Classifier**:
- `n_estimators`: 800
- `learning_rate`: 0.03
- `num_leaves`: 48
- `subsample`: 0.7
- `colsample_bytree`: 0.7
- `reg_alpha`: 0.5, `reg_lambda`: 0.8

### 5. Tối ưu ngưỡng phát hiện (`util_threshold.py`)

Thay vì áp dụng ngưỡng cố định `0.50`, hệ thống sử dụng thuật toán quét F1-optimal scan để tìm ra ngưỡng tối ưu:
$$\text{Best Threshold} = 0.87$$

Giao dịch chỉ bị gán nhãn **Fraud (1)** khi xác suất rủi ro $\ge 87\%$, giúp loại bỏ đa số cảnh báo sai (False Positives) trong vận hành thực tế.

---

## 📊 Kết quả Mô hình

### So sánh hiệu năng theo Threshold

| Metric | Ngưỡng 0.50 | Ngưỡng 0.87 (Tối ưu) |
|---|---|---|
| **Precision (Fraud)** | Thấp hơn (Nhiều cảnh báo sai) | **Cao vượt trội** |
| **Recall (Fraud)** | Cao | Cân bằng |
| **F1-Score (Fraud)** | Thấp hơn | **Tối ưu nhất** |
| **False Positives (FP)** | Nhiều báo động giả | **Giảm đáng kể** |

---

## 📁 Cấu trúc Project

```
Credit-Card-Fraud-Detection/
│
├── Data/
│   ├── CleanData.csv              # Dữ liệu sau bước làm sạch thô
│   └── dataset_processed.csv      # Dữ liệu hoàn chỉnh sẵn sàng huấn luyện
│
├── models/
│   ├── fraud_model.joblib         # Mô hình LightGBM đã huấn luyện
│   ├── scaler.joblib              # Object StandardScaler đã fit
│   ├── features.joblib            # Danh sách danh mục thuộc tính
│   └── best_threshold.txt         # Ngưỡng tối ưu (0.87)
│
├── templates/
│   └── index.html                 # Giao diện Web Dashboard (HTML5)
│
├── static/
│   ├── style.css                  # Design system (Glassmorphism, Dark mode)
│   └── app.js                     # Logic Frontend (API Fetching, Chart.js)
│
├── .gitignore                     # Cấu hình bỏ qua file rác / pycache
├── processing.py                  # Bước 1: Làm sạch dữ liệu thô
├── EDA.py                         # Bước 2: Phân tích khám phá & Tiền xử lý
├── features.py                    # Bước 3: Feature Engineering & Scaler
├── models.py                      # Bước 4: Định nghĩa mô hình LightGBM
├── train_model.py                 # Bước 5: Pipeline huấn luyện hoàn chỉnh
├── util_threshold.py              # Bước 6: Tối ưu hóa ngưỡng phát hiện
├── predict.py                     # Module suy luận (CLI Inference)
├── app_web.py                     # Flask Web Server & REST API
├── requirements.txt               # Danh sách thư viện phụ thuộc
└── README.md                      # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng dẫn cài đặt và chạy

### 1. Cài đặt môi trường & thư viện

Mở terminal trong thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```

### 2. Chạy Pipeline dữ liệu & Huấn luyện (Tùy chọn nếu muốn train lại)

```bash
# Tiền xử lý & EDA dữ liệu
python processing.py
python EDA.py

# Huấn luyện mô hình & lưu artifacts
python train_model.py
```

### 3. Khởi chạy Web Dashboard

Chạy ứng dụng Flask Web Server:

```bash
python app_web.py
```

Sau khi khởi chạy thành công, truy cập trình duyệt tại địa chỉ:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 💡 Lợi ích và Ứng dụng

### Lợi ích Doanh nghiệp / Ngân hàng
- **Giảm thiệt hại tài chính:** Cảnh báo và ngăn chặn giao dịch bất thường trong thời gian thực.
- **Nâng cao trải nghiệm khách hàng:** Giảm bớt cảnh báo sai (False Positives) làm gián đoạn thẻ của người dùng hợp pháp nhờ ngưỡng tối ưu 0.87.
- **Tự động hóa giám sát:** REST API dễ dàng tích hợp trực tiếp vào hệ thống Core Banking hoặc Payment Gateway.

### Lợi ích Kỹ thuật
- **Kiến trúc Clean Code & Modular:** Phân tách rõ ràng giữa Data Pipeline, Model Training, REST API và Web Frontend.
- **Trực quan hóa sinh động:** Web Dashboard cung cấp các biểu đồ ROC, Feature Importance và Risk Gauge thời gian thực.

---
*Dự án được phát triển phục vụ nghiên cứu và ứng dụng Machine Learning trong bài toán Phát hiện Gian lận Tài chính.*
