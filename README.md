# 🛡️ Credit Card Fraud Detection

> Hệ thống phát hiện gian lận thẻ tín dụng sử dụng Machine Learning (LightGBM) với giao diện Web Dashboard thời gian thực.

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Thuật toán & Pipeline](#-thuật-toán--pipeline)
- [Kết quả Model](#-kết-quả-model)
- [Cấu trúc Project](#-cấu-trúc-project)
- [Hướng dẫn cài đặt & chạy](#-hướng-dẫn-cài-đặt--chạy)
- [Lợi ích & Ứng dụng](#-lợi-ích--ứng-dụng)

---

## 🎯 Tổng quan

Project xây dựng một hệ thống **phát hiện giao dịch gian lận thẻ tín dụng** hoàn chỉnh từ khâu xử lý dữ liệu, huấn luyện mô hình, đến triển khai giao diện web cho người dùng cuối. Hệ thống có khả năng dự đoán một giao dịch là **gian lận (Fraud)** hay **bình thường (Normal)** dựa trên 14 đặc trưng của giao dịch.

### Điểm nổi bật

- 🚀 **Mô hình LightGBM** với hiệu suất cao, tốc độ dự đoán nhanh
- ⚖️ **Xử lý dữ liệu mất cân bằng** bằng kết hợp SMOTE + Random UnderSampling
- 🎯 **Tối ưu hoá threshold** tự động (F1-optimal) thay vì mặc định 0.5
- 🌐 **Web Dashboard** với giao diện glassmorphism hiện đại (Flask + Chart.js)
- 🖥️ **Desktop GUI** bằng Tkinter cho môi trường offline

---

## 🏗️ Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
│                                                                  │
│  RawDataSet.csv ──► processing.py ──► CleanData.csv              │
│                          │                                       │
│                     (Drop NA, Duplicates)                        │
│                          ▼                                       │
│                      EDA.py ──► dataset_processed.csv            │
│                  (EDA, Drop cc_num, Validate, Outlier Analysis)  │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
│                                                                  │
│  dataset_processed.csv                                           │
│        │                                                         │
│        ▼                                                         │
│  features.py ──► Feature Engineering + StandardScaler            │
│        │                                                         │
│        ▼                                                         │
│  train_model.py                                                  │
│    ├── Train/Test Split (70/30, stratified)                      │
│    ├── SMOTE (sampling_strategy=0.3)                             │
│    ├── RandomUnderSampler (sampling_strategy=0.5)                │
│    ├── LightGBM Training (800 estimators)                        │
│    ├── Threshold Optimization (F1-optimal scan)                  │
│    └── Save: model.joblib, scaler.joblib, threshold.txt          │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    INFERENCE / SERVING                            │
│                                                                  │
│  ┌─────────────┐              ┌──────────────────┐               │
│  │  App.py     │              │  app_web.py      │               │
│  │  (Tkinter)  │              │  (Flask Server)  │               │
│  │  Desktop GUI│              │  REST API + Web  │               │
│  └──────┬──────┘              └────────┬─────────┘               │
│         │                              │                         │
│         ▼                              ▼                         │
│    predict.py                  /api/predict (POST)               │
│    (Load model ──► Predict)    /api/model/info (GET)             │
│                                /api/model/evaluate (GET)         │
│                                /api/sample/<type> (GET)          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Công nghệ sử dụng

### Machine Learning & Data Science

| Công nghệ | Phiên bản | Mục đích |
|---|---|---|
| **Python** | 3.x | Ngôn ngữ lập trình chính |
| **LightGBM** | latest | Thuật toán Gradient Boosting cho classification |
| **scikit-learn** | latest | Preprocessing (StandardScaler), metrics, train/test split |
| **imbalanced-learn** | latest | Xử lý dữ liệu mất cân bằng (SMOTE, RandomUnderSampler) |
| **pandas** | latest | Xử lý và phân tích dữ liệu dạng bảng |
| **NumPy** | latest | Tính toán số học, xử lý mảng |
| **joblib** | latest | Serialize/deserialize model và scaler |

### Web Application

| Công nghệ | Mục đích |
|---|---|
| **Flask** | Web framework Python - REST API server |
| **HTML5 / CSS3** | Giao diện web với glassmorphism design system |
| **JavaScript (ES6+)** | Frontend logic, API calls, dynamic rendering |
| **Chart.js** | Biểu đồ tương tác (ROC Curve, Feature Importance) |
| **Google Fonts (Inter)** | Typography hiện đại |

### Desktop Application

| Công nghệ | Mục đích |
|---|---|
| **Tkinter** | GUI framework tích hợp sẵn trong Python |

---

## 🧠 Thuật toán & Pipeline

### 1. Tiền xử lý dữ liệu (Data Preprocessing)

#### `processing.py` — Làm sạch dữ liệu thô
- **Xử lý giá trị thiếu (Missing Values):** Phát hiện và loại bỏ các dòng chứa `null`/`NaN`
- **Loại bỏ bản sao (Deduplication):** Xóa các dòng trùng lặp hoàn toàn
- **Output:** `CleanData.csv`

#### `EDA.py` — Phân tích khám phá dữ liệu
- Thống kê mô tả (Descriptive Statistics)
- Phân tích phân phối biến mục tiêu `is_fraud` (tỷ lệ imbalance)
- Tương quan (Correlation) giữa các features với biến mục tiêu
- Phát hiện Outlier bằng phương pháp IQR (Interquartile Range)
- Loại bỏ cột không cần thiết (`cc_num` — thông tin định danh, không có giá trị dự đoán)
- **Output:** `dataset_processed.csv`

### 2. Feature Engineering (`features.py`)

#### 14 Features sử dụng

| Feature | Kiểu | Mô tả |
|---|---|---|
| `amt` | Numeric | Số tiền giao dịch (USD) |
| `city_pop` | Numeric | Dân số thành phố nơi giao dịch |
| `lat` | Numeric | Vĩ độ chủ thẻ |
| `long` | Numeric | Kinh độ chủ thẻ |
| `merch_lat` | Numeric | Vĩ độ cửa hàng |
| `merch_long` | Numeric | Kinh độ cửa hàng |
| `unix_time` | Numeric | Thời gian giao dịch (Unix timestamp) |
| `distance` | Numeric | Khoảng cách giữa chủ thẻ và cửa hàng |
| `merchant` | Categorical (encoded) | Mã cửa hàng (đã mã hoá) |
| `category` | Categorical (encoded) | Loại giao dịch (đã mã hoá) |
| `hour` | Temporal | Giờ giao dịch (0–23) |
| `day` | Temporal | Ngày trong tháng (1–31) |
| `month` | Temporal | Tháng (1–12) |
| `gender` | Binary | Giới tính (0 = Nữ, 1 = Nam) |

#### Chuẩn hoá dữ liệu (StandardScaler)
- Áp dụng **StandardScaler** cho 8 cột numeric: `amt`, `city_pop`, `distance`, `lat`, `long`, `merch_lat`, `merch_long`, `unix_time`
- Công thức: `z = (x - μ) / σ` — đưa dữ liệu về phân phối chuẩn (mean=0, std=1)
- Scaler được lưu lại để dùng khi inference (đảm bảo consistency)

### 3. Xử lý dữ liệu mất cân bằng (Imbalanced Data Handling)

Dữ liệu gian lận thẻ tín dụng **cực kỳ mất cân bằng** — giao dịch gian lận chỉ chiếm ~0.5% tổng số giao dịch. Nếu không xử lý, model sẽ thiên vị dự đoán tất cả là Normal.

#### Chiến lược: SMOTE + Random UnderSampling (Hybrid Resampling)

```
Dữ liệu gốc:    Normal: ~99.5%  |  Fraud: ~0.5%   (Ratio ~1:200)
                         │
                         ▼
          ┌─────── SMOTE (step 1) ───────┐
          │ sampling_strategy = 0.3      │
          │ Tăng Fraud lên = 30% Normal  │
          │ (Tạo synthetic samples)      │
          └──────────────┬───────────────┘
                         │
                         ▼
     ┌── RandomUnderSampler (step 2) ──┐
     │ sampling_strategy = 0.5         │
     │ Giảm Normal xuống = 2x Fraud    │
     │ (Loại bỏ random samples)        │
     └──────────────┬──────────────────┘
                    │
                    ▼
Dữ liệu cân bằng:  Normal : Fraud ≈ 2:1
```

**Tại sao kết hợp 2 phương pháp?**
- **Chỉ SMOTE:** Tạo quá nhiều synthetic samples → Overfitting
- **Chỉ UnderSample:** Mất quá nhiều dữ liệu Normal → Underfitting
- **Hybrid (SMOTE + Under):** Cân bằng tốt nhất giữa data diversity và sample size

> ⚠️ **Lưu ý quan trọng:** Resampling chỉ áp dụng trên **tập Train**, KHÔNG áp dụng trên tập Test để đảm bảo đánh giá chính xác.

### 4. Thuật toán LightGBM (`models.py`)

#### LightGBM (Light Gradient Boosting Machine) là gì?

LightGBM là thuật toán **Gradient Boosting Decision Tree (GBDT)** được phát triển bởi Microsoft, tối ưu cho tốc độ và hiệu suất cao trên dữ liệu lớn.

#### Nguyên lý hoạt động

```
Input X ──► Tree 1 ──► Residual ──► Tree 2 ──► Residual ──► ... ──► Tree N
              │                        │                               │
              ▼                        ▼                               ▼
           Pred 1     +             Pred 2     + ... +              Pred N
                                                                       │
                                                                       ▼
                                                              Final Prediction
                                                           P(fraud) = σ(Σ pred)
```

- **Ensemble Learning:** Kết hợp nhiều cây quyết định yếu (weak learners) thành một mô hình mạnh
- **Gradient Boosting:** Mỗi cây mới học từ sai số (residuals) của các cây trước
- **Leaf-wise Growth:** LightGBM phát triển cây theo chiều lá (best-first) thay vì theo chiều ngang (level-wise) → nhanh hơn, chính xác hơn

#### Hyperparameters đã cấu hình

| Parameter | Giá trị | Mô tả |
|---|---|---|
| `objective` | `binary` | Bài toán phân loại nhị phân |
| `boosting_type` | `gbdt` | Gradient Boosting Decision Tree |
| `n_estimators` | `800` | Số lượng cây quyết định (trees) |
| `learning_rate` | `0.03` | Tốc độ học — nhỏ để tăng generalization |
| `num_leaves` | `48` | Số lá tối đa mỗi cây — giới hạn complexity |
| `max_depth` | `-1` | Không giới hạn độ sâu (controlled bởi num_leaves) |
| `min_child_samples` | `50` | Số mẫu tối thiểu trong mỗi lá — chống overfitting |
| `subsample` | `0.7` | Sử dụng 70% dữ liệu cho mỗi cây (row sampling) |
| `colsample_bytree` | `0.7` | Sử dụng 70% features cho mỗi cây (column sampling) |
| `reg_alpha` | `0.5` | L1 Regularization — tăng sparsity |
| `reg_lambda` | `0.8` | L2 Regularization — kiểm soát overfitting |

> 💡 **Design Decision:** Không sử dụng `class_weight` vì đã xử lý imbalance bằng SMOTE — tránh "double-dipping" (xử lý imbalance 2 lần sẽ gây thiên lệch ngược).

### 5. Tối ưu Threshold (`util_threshold.py`)

Mặc định, mô hình phân loại sử dụng **threshold = 0.5** để quyết định Fraud/Normal. Tuy nhiên với dữ liệu mất cân bằng, threshold tối ưu thường **khác 0.5**.

#### 3 phương pháp tìm threshold

| Phương pháp | Mô tả | Ưu điểm |
|---|---|---|
| **F1-scan** (mặc định) | Quét threshold từ 0.05→0.95, chọn F1 cao nhất | Cân bằng tốt Precision-Recall |
| **PR-curve** | Dựa trên Precision-Recall curve, maximize F1 | Phù hợp dữ liệu imbalanced |
| **Youden's J** | Maximize (TPR − FPR) trên ROC curve | Tối ưu khoảng cách đến random |

#### Kết quả threshold tối ưu

```
Best Threshold (F1-optimal) = 0.87
```

> Threshold = 0.87 nghĩa là: model chỉ cảnh báo Fraud khi xác suất ≥ 87%, giúp **giảm False Positive** (cảnh báo sai) đáng kể so với threshold mặc định 0.5.

### 6. Đánh giá & Metrics

Model được đánh giá trên **tập Test** (30% dữ liệu, stratified split) với các metrics:

| Metric | Ý nghĩa |
|---|---|
| **Accuracy** | Tỷ lệ dự đoán đúng tổng thể |
| **Precision (Fraud)** | Trong các giao dịch model cảnh báo Fraud, bao nhiêu % thực sự là Fraud |
| **Recall (Fraud)** | Trong tổng số giao dịch Fraud thực tế, model phát hiện được bao nhiêu % |
| **F1-Score** | Trung bình điều hoà của Precision & Recall |
| **Balanced Accuracy** | Accuracy cân bằng giữa 2 class |
| **MCC (Matthews Correlation Coefficient)** | Metric tổng hợp, đáng tin cậy nhất cho imbalanced data |
| **ROC-AUC** | Diện tích dưới đường cong ROC — khả năng phân biệt 2 class |
| **Confusion Matrix** | Ma trận nhầm lẫn: TN, FP, FN, TP |

---

## 📊 Kết quả Model

### Hiệu suất tổng quan

Model đã được huấn luyện và đánh giá trên tập test (30% dữ liệu, ~555,000 giao dịch):

| Metric | Giá trị |
|---|---|
| **Threshold tối ưu** | `0.87` (F1-optimal) |
| **Số features** | 14 |
| **Số trees (estimators)** | 800 |
| **Thời gian train** | ~vài giây |

### Chiến lược Resampling

```
TRƯỚC Resample:
  Normal (0): ~1,289,000
  Fraud  (1): ~    6,400
  Ratio:      1 : 200

SAU Resample (SMOTE + UnderSampling):
  Normal (0): ~772,000
  Fraud  (1): ~386,000
  Ratio:      1 : 2
```

### Đánh giá chi tiết

Model được đánh giá với 2 threshold để so sánh:

#### Threshold mặc định (0.50) vs Threshold tối ưu (0.87)

| Metric | Threshold 0.50 | Threshold 0.87 |
|---|---|---|
| **Precision (Fraud)** | Thấp hơn (nhiều FP) | Cao hơn (ít FP) |
| **Recall (Fraud)** | Cao (bắt nhiều fraud) | Cân bằng hơn |
| **F1-Score (Fraud)** | Thấp hơn | **Cao nhất** |
| **False Positive** | Nhiều cảnh báo sai | Ít cảnh báo sai |

> 📈 Threshold = 0.87 cho F1-Score **tối ưu nhất**, đạt cân bằng lý tưởng giữa việc phát hiện fraud (Recall) và không gây phiền toái bằng cảnh báo sai (Precision).

### Web Dashboard hiển thị

Dashboard hiển thị đầy đủ:
- **Confusion Matrix** trực quan (TN, FP, FN, TP)
- **Classification Report** với Precision, Recall, F1 cho từng class
- **ROC Curve** với giá trị AUC
- **Feature Importance** biểu đồ thanh ngang

---

## 📁 Cấu trúc Project

```
Credit-Card-Fraud-Detection/
│
├── Data/
│   ├── CleanData.csv              # Dữ liệu đã làm sạch (150MB)
│   └── dataset_processed.csv      # Dữ liệu đã xử lý, sẵn sàng train (128MB)
│
├── models/
│   ├── fraud_model.joblib         # Model LightGBM đã train (~4.3MB)
│   ├── scaler.joblib              # StandardScaler đã fit
│   ├── features.joblib            # Danh sách features (đảm bảo consistency)
│   └── best_threshold.txt         # Threshold tối ưu (0.87)
│
├── templates/
│   └── index.html                 # Web Dashboard template (HTML5)
│
├── static/
│   ├── style.css                  # Design system (glassmorphism, dark mode)
│   └── app.js                     # Frontend logic (API calls, Chart.js)
│
├── processing.py                  # Bước 1: Làm sạch dữ liệu thô
├── EDA.py                         # Bước 2: Phân tích khám phá & tiền xử lý
├── features.py                    # Bước 3: Feature engineering + StandardScaler
├── models.py                      # Bước 4: Định nghĩa & cấu hình LightGBM
├── train_model.py                 # Bước 5: Training pipeline hoàn chỉnh
├── util_threshold.py              # Bước 6: Tìm threshold tối ưu
├── predict.py                     # Module dự đoán (inference)
├── app_web.py                     # 🌐 Flask Web Server + REST API
├── App.py                         # 🖥️ Tkinter Desktop GUI
├── requirements.txt               # Dependencies
└── README.md                      # Documentation (file này)
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Pipeline xử lý dữ liệu (nếu chạy từ đầu)

```bash
# Bước 1: Làm sạch dữ liệu thô
python processing.py

# Bước 2: EDA & tiền xử lý
python EDA.py
```

### 3. Huấn luyện model

```bash
python train_model.py
```

Output sẽ lưu vào thư mục `models/`:
- `fraud_model.joblib` — Model đã train
- `scaler.joblib` — Scaler đã fit
- `features.joblib` — Danh sách features
- `best_threshold.txt` — Threshold tối ưu

### 4. Chạy Web Dashboard

```bash
python app_web.py
```

Truy cập: **http://127.0.0.1:5000**

### 5. Chạy Desktop GUI (tuỳ chọn)

```bash
python App.py
```

---

## 💎 Lợi ích & Ứng dụng

### 🏦 Lợi ích cho ngành Tài chính — Ngân hàng

| Lợi ích | Mô tả |
|---|---|
| **Giảm thiệt hại tài chính** | Phát hiện gian lận nhanh, ngăn chặn giao dịch bất thường trước khi hoàn tất |
| **Bảo vệ khách hàng** | Cảnh báo tức thì khi phát hiện giao dịch nghi ngờ trên tài khoản |
| **Giảm chi phí vận hành** | Tự động hoá quy trình kiểm tra thay vì review thủ công mọi giao dịch |
| **Tuân thủ pháp luật** | Đáp ứng yêu cầu giám sát giao dịch theo quy định AML/KYC |

### 🔬 Lợi ích kỹ thuật — Học thuật

| Lợi ích | Mô tả |
|---|---|
| **End-to-End Pipeline** | Minh hoạ quy trình ML hoàn chỉnh: từ dữ liệu thô → model → deployment |
| **Xử lý Imbalanced Data** | Demo kỹ thuật SMOTE + UnderSampling — bài toán phổ biến trong thực tế |
| **Threshold Optimization** | Cho thấy tầm quan trọng của việc tối ưu threshold thay vì dùng mặc định |
| **Reproducibility** | Code có cấu trúc rõ ràng, dễ tái tạo kết quả (random_state=42) |
| **Multiple Interfaces** | Cung cấp cả Web API (Flask) và Desktop GUI (Tkinter) |

### 📈 Điểm mạnh của giải pháp

1. **Tốc độ cao:** LightGBM dự đoán trong < 1ms cho mỗi giao dịch
2. **Scalable:** Xử lý được dataset hàng triệu giao dịch (~1.8M records)
3. **Threshold linh hoạt:** Có thể điều chỉnh threshold tuỳ theo nhu cầu nghiệp vụ
   - Threshold **thấp** → Bắt nhiều fraud hơn nhưng nhiều cảnh báo sai
   - Threshold **cao** → Ít cảnh báo sai nhưng có thể bỏ sót fraud
4. **API-ready:** Flask REST API dễ dàng tích hợp vào hệ thống hiện có
5. **Trực quan:** Dashboard hiển thị Gauge, ROC, Feature Importance giúp diễn giải model

### 🌍 Ứng dụng thực tế

- **Ngân hàng:** Giám sát giao dịch thẻ tín dụng real-time
- **Fintech:** Tích hợp vào payment gateway để kiểm tra fraud tự động
- **E-commerce:** Phát hiện đơn hàng gian lận trước khi xử lý thanh toán
- **Bảo hiểm:** Phát hiện yêu cầu bồi thường gian lận (fraud claims)

---

## 📝 Ghi chú kỹ thuật

- **Train/Test Split:** 70/30 với `stratify=y` để đảm bảo tỷ lệ fraud được giữ nguyên
- **Random State:** Cố định `random_state=42` ở tất cả các bước để kết quả có thể tái tạo
- **Resampling:** Chỉ áp dụng trên **tập Train** — tập Test giữ nguyên phân phối gốc
- **Inference Pipeline:** `predict.py` load đúng feature schema từ `features.joblib` để đảm bảo thứ tự features khớp với lúc train

---

<p align="center">
  <strong>Được phát triển cho mục đích nghiên cứu và học tập trong lĩnh vực Machine Learning & Fraud Detection.</strong>
</p>
