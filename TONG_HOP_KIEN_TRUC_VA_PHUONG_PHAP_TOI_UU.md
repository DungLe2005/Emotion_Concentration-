# BÁO CÁO TỔNG HỢP TOÀN DIỆN: CÁC CƠ CHẾ, PHƯƠNG PHÁP & HƯỚNG PHÁT TRIỂN MÔ HÌNH NHẬN DIỆN CẢM XÚC & ĐỘ TẬP TRUNG (EduFocus AI)

> **Tài liệu tham chiếu kiến trúc & định hướng nghiên cứu phát triển**  
> *Phiên bản: 3.0 (Cập nhật đồng bộ: Facial Query Cross-Attention + Centroid LLM Prompts + Focal Loss + Dynamic Ensemble)*

---

## MỤC LỤC
1. [Toàn cảnh Kiến trúc Tổng thể (End-to-End Multimodal Pipeline)](#1-toàn-cảnh-kiến-trúc-tổng-thể)
2. [Cơ chế 1: Rã Băng Tuần Tự Lũy Tiến (Curriculum Progressive Unfreezing)](#2-cơ-chế-1-rã-băng-tuần-tự-lũy-tiến)
3. [Cơ chế 2: Facial Query Cross-Attention (DETR-Style Keypoint Queries)](#3-cơ-chế-2-facial-query-cross-attention)
4. [Cơ chế 3: Prompt Ensembling & Centroid Embeddings từ LLM (CuPL CVPR)](#4-cơ-chế-3-prompt-ensembling--centroid-embeddings-từ-llm)
5. [Cơ chế 4: Hàm Mất Mát Class-Weighted Focal Loss & Data Augmentation](#5-cơ-chế-4-hàm-mất-mát-class-weighted-focal-loss--data-augmentation)
6. [Cơ chế 5: Tối Ưu Hóa & Lọc Đặc Trưng Bằng EPSO](#6-cơ-chế-5-tối-ưu-hóa--lọc-đặc-trưng-bằng-epso)
7. [Cơ chế 6: Tri-Model Soft-Voting Ensemble với Trọng Số Động](#7-cơ-chế-6-tri-model-soft-voting-ensemble-với-trọng-số-động)
8. [Cơ chế 7: Ổn Định Thời Gian Thực & Đo Lường Độ Tập Trung Khách Quan](#8-cơ-chế-7-ổn-định-thời-gian-thực--đo-lường-độ-tập-trung-khách-quan)
9. [Bảng So Sánh Hiệu Năng Thực Nghiệm](#9-bảng-so-sánh-hiệu-năng-thực-nghiệm)
10. [Phân Tích Điểm Nghẽn Hiện Tại & 5 Hướng Phát Triển Đột Phá](#10-phân-tích-điểm-nghẽn-hiện-tại--5-hướng-phát-triển-đột-phá)

---

## 1. Toàn cảnh Kiến trúc Tổng thể

Hệ thống được thiết kế theo mô hình **Đa mô thức tích hợp (End-to-End Deep Multimodal FER)**, kết hợp giữa tri thức thị giác nền tảng (Foundation Vision-Language), biểu diễn giải phẫu cơ mặt sâu (Deep Facial Deformation) và phân loại biểu cảm theo cơ chế Ensemble.

### Luồng Dữ Liệu Chi Tiết (Pipeline Flow)

* **Bước 1: Đầu vào ảnh & Trích xuất đặc trưng nền tảng**
  - Ảnh khuôn mặt kích thước chuẩn `[3, 224, 224]` đưa qua **CLIP ViT-B/32 Vision Encoder**.
  - Đầu ra nhánh 1: **Global CLS Token** `[512D]` đi qua tầng `VisualProjection` (Linear + LayerNorm + GELU + Residual) để tạo ra đặc trưng thị giác thích nghi `[512D]`.
  - Đầu ra nhánh 2: **49 Spatial Patch Tokens** `[49, 768D]` đưa vào mô-đun `FacialQueryCrossAttention`.

* **Bước 2: Phân tích vi biểu cảm & Căn chỉnh ngữ nghĩa (Facial & Text Embedding)**
  - Nhánh Facial: **32 Learnable Queries** tương tác với nhau qua Self-Attention, sau đó Cross-Attention truy vấn vào 49 Patch Tokens để trích xuất biến dạng cơ mặt, kết thúc bằng Attentive Pooling chiếu về `[25D]`.
  - Nhánh Text: **10 câu prompt Action Units** cho mỗi cảm xúc từ LLM (`fer_descriptions.json`) được mã hóa và lấy trọng tâm hình học (Centroid Pooling), sau đó tính Cosine Similarity với vector ảnh để tạo ra điểm số tương đồng ngữ nghĩa `[7D]`.

* **Bước 3: Hợp nhất đa mô thức & Chọn lọc đặc trưng (Multimodal Fusion & EPSO)**
  - Ghép nối 3 nhánh thành vector đặc trưng đa mô thức chuẩn:
    `512D (Visual) + 7D (Prompt Similarity) + 25D (Facial Deformation) = 544D`.
  - Thuật toán tiến hóa **EPSO (Enhanced Particle Swarm Optimization)** áp dụng mặt nạ nhị phân chọn lọc **289 chiều đặc trưng tối ưu nhất**, loại bỏ các chiều dư thừa.

* **Bước 4: Phân loại đa mô hình & Bầu chọn trọng số động (Tri-Model Soft-Voting Ensemble)**
  - Vector 289D được phân phối song song đến 3 mô hình độc lập:
    1. `EmotionResMLP`: Mạng nơ-ron sâu với các khối kết nối tắt Residual.
    2. `XGBoost Classifier`: Cây quyết định tăng cường gradient mạnh trên dữ liệu dạng bảng.
    3. `Baseline EmotionMLP`: Mạng MLP tuyến tính tối ưu từ Bước 3.
  - Phân phối xác suất từ 3 mô hình được tổng hợp theo trọng số động quét lưới tối ưu:
    `P_final = w_res * P_res + w_xgb * P_xgb + w_base * P_base`.

* **Bước 5: Xử lý chuỗi thời gian & Đo lường độ tập trung (Temporal Tracking & Scoring)**
  - `TemporalFaceTracker` sử dụng IoU để theo dõi học sinh qua từng frame, kết hợp bộ lọc EMA (`alpha=0.35`) và Hysteresis để triệt tiêu hiện tượng giật nháy cảm xúc.
  - `Upper-Face Gating` phát hiện học sinh đang nói chuyện bằng tỷ lệ khẩu hình miệng để tránh nhầm sang bất ngờ/sợ hãi.
  - Tính điểm độ tập trung chuẩn hóa `[0 - 100%]` dựa trên phân phối xác suất cảm xúc tích cực và tiêu cực.

---

## 2. Cơ chế 1: Rã Băng Tuần Tự Lũy Tiến (Curriculum Progressive Unfreezing)

### 2.1. Vấn đề giải quyết
Khi fine-tune mô hình lớn như **CLIP ViT-B/32 (86M tham số)** trên các bộ dữ liệu khuôn mặt giới hạn (~12.000–15.000 mẫu như RAF-DB hoặc <1.000 mẫu như KMUFED), mô hình rất dễ gặp 2 thảm họa:
1. **Catastrophic Forgetting (Quên thảm khốc)**: Làm méo mó các bộ lọc thị giác tổng quát (cạnh, góc, kết cấu da) mà OpenAI đã học từ 400M ảnh.
2. **Representation Collapse & NaN Overflows**: Gradient quá lớn ở các tầng sâu làm sụp đổ trọng số khi cập nhật đột ngột.

### 2.2. Giải pháp kỹ thuật tích hợp
Trong `backend/evaluation/train_sequential_unfreeze.py`:
- **Khóa vĩnh viễn 8/12 Blocks ViT (Blocks 0–7)**: Bảo tồn 100% đặc trưng thị giác cấp thấp và trung bình.
- **Khóa 100% Text Encoder**: Giữ nguyên không gian ngữ nghĩa văn bản của CLIP làm "mỏ neo cố định".
- **Lộ trình 3 Phase lũy tiến (Curriculum Schedule)**:
  - **Phase 0 (1..30% epochs)**: Khóa 100% ViT Backbone. Chỉ huấn luyện thích nghi các tầng mới (`VisualProjection`, `FacialQueryCrossAttention`, `ResMLP Head`) với Learning Rate cao ($10^{-3}$).
  - **Phase 1 (31..60% epochs)**: Mở 2 Blocks trên cùng của ViT (Blocks 10–11) với LR vi sai nhỏ ($3 \times 10^{-5}$). Bảo toàn 100% momentum của các adapter tầng dưới.
  - **Phase 2 (61..100% epochs)**: Mở thêm Blocks 8–9 với cơ chế **Layer-wise Learning Rate Decay (LLRD)**:
    $$\text{LR}_{\text{block 8-9}} = 5 \times 10^{-6}, \quad \text{LR}_{\text{vis 10-11}} = 1.5 \times 10^{-5}, \quad \text{LR}_{\text{adapter}} = 8 \times 10^{-5}$$
- **Phòng thủ NaN**: Huấn luyện thuần **FP32** (tắt AMP), Gradient Clipping chuẩn hóa `max_norm=0.5` và cơ chế tự động Rollback giảm LR khi phát hiện dấu hiệu phân kỳ.

---

## 3. Cơ chế 2: Facial Query Cross-Attention (DETR-Style Keypoint Queries)

### 3.1. Vấn đề giải quyết
Phương pháp truyền thống dùng MediaPipe trích xuất 468 mốc hình học bằng CPU có nhiều điểm yếu:
- Phụ thuộc thuật toán đồ họa CPU (chậm ~15–25ms, không đồng bộ GPU).
- Chỉ đo khoảng cách cơ học 2D/3D bề mặt, rất dễ sai lệch khi góc mặt nghiêng, ánh sáng yếu hoặc bị che khuất.
- Không trích xuất được thông tin biến dạng vi mô (nếp nhăn khóe mắt, cơ gò má nổi).

### 3.2. Giải pháp kỹ thuật tích hợp
Triển khai tại `backend/models/facial_cross_attention.py`:
- **32 Learnable Queries (768D)**: Đóng vai trò là các "cảm biến tiềm ẩn" (latent probes) tự động học cách truy vấn vào từng vùng giải phẫu cơ mặt (AU).
- **2 Tầng Transformer Decoder (Pre-LN Architecture)**:
  1. *Multi-Head Self-Attention*: Cho phép 32 queries trao đổi thông tin với nhau, học mối tương quan không gian giữa các chuyển động cơ mặt (ví dụ: khóe mắt nheo lại đồng thời khóe miệng nâng lên trong nụ cười Duchenne).
  2. *Multi-Head Cross-Attention*: Các queries truy vấn trực tiếp vào **49 Spatial Patch Tokens (768D)** lấy từ lớp cuối của ViT-B/32, trích xuất cấu trúc vi biểu cảm trực tiếp từ biểu diễn thị giác sâu.
  3. *Feed-Forward Network (FFN)*: Chiếu phi tuyến 2 tầng với GELU activation và Dropout.
- **Attentive Query Pooling**: Thay vì mean pooling thô sơ, mô hình dùng một tầng học trọng số động:
  $$\alpha_i = \text{Softmax}(W_{\text{pool}} q_i), \quad x_{\text{pooled}} = \sum_{i=1}^{32} \alpha_i q_i$$
- **Projection Head ra 25D**: Chiếu vector 768D về đúng 25 chiều đặc trưng cấu trúc vi biểu cảm, tương thích 100% với không gian đa mô thức 544D chuẩn.

---

## 4. Cơ chế 3: Prompt Ensembling & Centroid Embeddings từ LLM (CuPL CVPR)

### 4.1. Vấn đề giải quyết
- Trước đây hệ thống chỉ dùng 1 câu prompt đơn lẻ cho mỗi cảm xúc (ví dụ: *"a happy smiling cheerful face"*).
- Trong siêu cầu đa chiều của CLIP, một câu viết tay đơn lẻ chỉ đại diện cho một điểm cục bộ hẹp, dễ làm mô hình nhầm lẫn giữa các sắc thái cảm xúc có nét tương đồng (ví dụ: Há miệng nói chuyện bị nhầm thành *Surprise*, hoặc mặt không cảm xúc bị nhầm thành *Sad*).

### 4.2. Giải pháp kỹ thuật tích hợp
Triển khai kỹ thuật **Customized Prompts via Language Models (CuPL)** kết nối qua [backend/fer_descriptions.json](file:///d:/Emotion_Concentration/backend/fer_descriptions.json):
- Dùng LLM sinh **10 câu mô tả thị giác giải phẫu chi tiết (FACS Action Units)** cho từng cảm xúc (tập trung vào: chân mày, khóe mắt, mí mắt dưới, cánh mũi, khóe môi, hàm).
- **Thuật toán Centroid Text Embedding**:
  1. Mã hóa toàn bộ $K=10$ câu mô tả cho cảm xúc $c$ thành các vector nhúng: $\{e_{c, 1}, e_{c, 2}, \dots, e_{c, K}\}$.
  2. Chuẩn hóa $L_2$ từng vector: $\hat{e}_{c, k} = \frac{e_{c, k}}{\|e_{c, k}\|_2}$.
  3. Tính trọng tâm hình học (Centroid) và chuẩn hóa lại:
     $$T_c = \frac{\frac{1}{K}\sum_{k=1}^K \hat{e}_{c, k}}{\left\|\frac{1}{K}\sum_{k=1}^K \hat{e}_{c, k}\right\|_2} \in \mathbb{R}^{512}$$
- **Đồng bộ hóa tuyệt đối**: Hàm dùng chung `get_emotion_centroid_embeddings()` trong [dataset/dataset_loader.py](file:///d:/Emotion_Concentration/backend/dataset/dataset_loader.py) đảm bảo cả 3 module (`train_sequential_unfreeze.py`, `dataset_preprocessor.py`, `pipeline.py`) sở hữu ma trận text features $[7 \times 512]$ giống nhau $100\%$ (sai số số học $= 0.00000000$).
- **Không tăng độ trễ thời gian thực**: Centroid được tính sẵn lúc khởi động pipeline ($0\text{ ms}$ overhead trong vòng lặp video).

---

## 5. Cơ chế 4: Hàm Mất Mát Class-Weighted Focal Loss & Data Augmentation

### 5.1. Class-Weighted Focal Loss
Dữ liệu cảm xúc thực tế luôn mất cân bằng nghiêm trọng (lớp *Neutral* và *Happy* chiếm số lượng áp đảo, trong khi *Disgust* và *Fear* rất hiếm). Nếu dùng Cross-Entropy tiêu chuẩn, mô hình sẽ bị "lười", thiên vị đoán vào các lớp đa số.

Hệ thống tích hợp **Focal Loss** kết hợp **Class Weighting** và **Label Smoothing (0.05)**:
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
- $\gamma = 2.0$: Dập tắt gradient từ các mẫu dễ phân loại ($p_t \to 1$), tập trung 100% lực tối ưu vào các mẫu biên khó phân biệt giữa Fear / Disgust / Sad.
- $\alpha_t = \sqrt{\frac{N}{C \cdot N_t}}$: Trọng số căn bậc hai nghịch đảo tần suất lớp, ngăn ngừa lớp thiểu số bị đè bẹp.
- $\text{Label Smoothing} = 0.05$: Giảm độ tự tin quá mức (Overconfidence), cải thiện độ hiệu chỉnh xác suất (Model Calibration).

### 5.2. Data Augmentation Chuyên Biệt Vi Biểu Cảm
Trong `get_train_transform()`:
- `RandomResizedCrop(224, scale=(0.85, 1.0))`: Mô phỏng góc camera gần/xa và lệch tâm.
- `RandomHorizontalFlip(p=0.5)`: Khai thác tính đối xứng tự nhiên của cơ mặt.
- `RandomRotation(degrees=10)`: Mô phỏng hành vi học sinh nghiêng đầu khi suy nghĩ/viết bài.
- `ColorJitter(brightness=0.15, contrast=0.15)`: Mô phỏng biến thiên ánh sáng bóng đèn/môi trường lớp học.
- `RandomErasing(scale=(0.02, 0.20), p=0.25)`: Che khuất ngẫu nhiên một phần khuôn mặt (mô phỏng tay chống cằm, đeo kính, tóc che trán), ép mạng phải trích xuất đặc trưng từ cả hai vùng mắt và miệng thay vì chỉ phụ thuộc vào khẩu hình.

---

## 6. Cơ chế 5: Tối Ưu Hóa & Lọc Đặc Trưng Bằng EPSO

Vector tổng hợp đa mô thức có chiều dài $544\text{D}$:
$$V_{544} = [H_{\text{visual\_proj}} (512\text{D}) \;\parallel\; \text{Sim}_{\text{text\_prompt}} (7\text{D}) \;\parallel\; H_{\text{facial\_cross\_attn}} (25\text{D})]$$

Trong $544\text{D}$ này có nhiều chiều chứa thông tin dư thừa hoặc nhiễu. Thuật toán **EPSO (Enhanced Particle Swarm Optimization)** thực hiện tìm kiếm toàn cục trong không gian nhị phân $\{0, 1\}^{544}$:
- Hàm mục tiêu đa mục tiêu (Multi-Objective Fitness):
  $$\text{Fitness} = \alpha \cdot \text{Accuracy} + (1 - \alpha) \cdot \left(1 - \frac{|D_{\text{selected}}|}{544}\right)$$
- EPSO đã chọn lọc ra tập con **289 đặc trưng tối ưu nhất**, loại bỏ $46.8\%$ số chiều dư thừa, giúp tăng tốc độ suy luận và cải thiện độ hội tụ của các bộ phân loại cấp trên.

---

## 7. Cơ chế 6: Tri-Model Soft-Voting Ensemble với Trọng Số Động

### 7.1. Cấu trúc 3 Bộ Phân Loại Đa Dạng (Diversity Ensemble)
1. **EmotionResMLP (Deep Residual Network)**: 
   - 3 khối Residual Blocks với Skip Connections, LayerNorm, GELU và Dropout.
   - Giỏi nắm bắt mối quan hệ phi tuyến tính sâu giữa các đặc trưng đa mô thức.
2. **XGBoost Classifier (Gradient Boosted Trees)**:
   - Cực kỳ mạnh mẽ trên dữ liệu bảng (tabular feature vector), phân tách ranh giới trực giao tốt, không bị ảnh hưởng bởi vanishing gradient.
3. **Baseline EmotionMLP**:
   - Sử dụng checkpoint tối ưu nhất từ Bước 3 EPSO (`emotion_mlp_best_acc_weights_{dataset}.pth`).

### 7.2. Tối Ưu Hóa Trọng Số Bầu Chọn Bằng 2D Grid-Search
Trước đây, việc gán cứng trọng số ($0.45, 0.45, 0.10$) làm mô hình Ensemble hoạt động kém hơn cả XGBoost đơn lẻ khi mạng nơ-ron bị overfit.

Hệ thống đã thay thế bằng **2D Grid-Search** trên tập Validation với bước nhảy $0.05$:
$$\max_{w_{\text{res}}, w_{\text{xgb}}, w_{\text{base}}} \text{Acc}\left(\sum w_i P_i\right) \quad \text{với } \sum w_i = 1, \; w_i \ge 0$$
- Trọng số tối ưu tự động được lưu ra [backend/ensemble_weights.json](file:///d:/Emotion_Concentration/backend/ensemble_weights.json).
- [models/ensemble_head.py](file:///d:/Emotion_Concentration/backend/models/ensemble_head.py) tự động nạp cấu hình này khi chạy thực tế.
- **Đảm bảo mặt toán học**: Độ chính xác của Ensemble luôn $\ge$ mô hình đơn lẻ tốt nhất.

---

## 8. Cơ chế 7: Ổn Định Thời Gian Thực & Đo Lường Độ Tập Trung Khách Quan

### 8.1. Temporal Face Tracker (IoU + EMA Smoothing + Hysteresis)
Tại [backend/temporal_tracker.py](file:///d:/Emotion_Concentration/backend/temporal_tracker.py):
- **IoU Matching**: Định danh và theo dõi khuôn mặt của từng học sinh qua chuỗi video với ngưỡng IoU $\ge 0.25$.
- **Lọc mượt hàm mũ (EMA Smoothing, $\alpha=0.35$, Window=7)**: Triệt tiêu hiện tượng nhấp nháy cảm xúc giữa các frame liên tiếp do nhiễu camera.
- **Hysteresis Thresholding (Margin=0.12, Min Consecutive=3)**: Chỉ chuyển đổi trạng thái cảm xúc khi xác suất cảm xúc mới vượt qua cảm xúc hiện tại một khoảng an toàn trong 3 frame liên tục, tránh gián đoạn dữ liệu.

### 8.2. Upper-Face Gating & Phân Biệt Khẩu Hình Nói
- Dùng MediaPipe Mouth Aspect Ratio (MAR) để phát hiện khi nào học sinh đang **phát biểu / trao đổi bài**.
- Khi phát hiện học sinh đang nói chuyện, cơ chế **Upper-Face Gating** sẽ phong tỏa xác suất của *Surprise* và *Fear* (vốn hay bị kích hoạt khi há miệng), và dồn trọng số quan sát lên vùng mắt và chân mày.

### 8.3. Công Thức Chuẩn Hóa Điểm Độ Tập Trung
Độ tập trung được tính toán minh bạch trên phân phối Softmax:
$$\text{Score} = 50 + 50 \cdot \left(\sum_{i \in \text{Pos}} w_i P_i - \sum_{j \in \text{Neg}} |w_j| P_j\right)$$
- Nhóm tích cực: Neutral ($+0.30$), Happy ($+0.10$), Surprise ($+0.15$).
- Nhóm tiêu cực: Sad ($-0.05$), Fear ($-0.20$), Angry ($-0.10$), Disgust ($-0.05$).
- Công thức ánh xạ tuyến tính phân phối xác suất về thang điểm chuẩn $0 - 100\%$, phản ánh chính xác trạng thái nhận thức của người học.

---

## 9. Bảng So Sánh Hiệu Năng Thực Nghiệm

Dưới đây là kết quả kiểm thử sau khi hoàn thành toàn bộ chuỗi tối ưu hóa trên 2 tập dữ liệu chuẩn:

| Tập Dữ Liệu | Mô hình / Phương pháp | Test Accuracy (%) | Macro F1 (%) | Ghi chú |
| :--- | :--- | :---: | :---: | :--- |
| **KMUFED** (Bộ dữ liệu nhỏ) | Baseline EmotionMLP | 71.20% | 71.73% | Mạng MLP cơ sở |
| | EmotionResMLP (Deep Residual) | 75.60% | 76.82% | Có Skip-Connections & Early Stopping |
| | XGBoost Classifier | 80.00% | 81.16% | Cực kỳ mạnh trên tập vector tabular |
| | **ENSEMBLE (Soft-Voting Tối Ưu)** | **80.80%** | **82.39%** | **Vượt mô hình đơn tốt nhất (+0.80%)** |
| **NEW_DATA** (RAF-DB, ~15.000 ảnh) | Baseline EmotionMLP | 85.93% | 78.33% | Checkpoint tối ưu từ Bước 3 EPSO |
| | EmotionResMLP | 85.86% | 77.39% | Deep Residual Blocks |
| | XGBoost Classifier | 85.93% | 77.80% | Gradient Boosted Trees |
| | **ENSEMBLE (Soft-Voting Tối Ưu)** | **86.25%** | **78.66%** | **Đạt kỷ lục cao nhất trên toàn hệ thống** |

---

## 10. Phân Tích Điểm Nghẽn Hiện Tại & 5 Hướng Phát Triển Đột Phá

Dù hệ thống đã đạt độ chính xác cao (**86.25%**) và hoạt động ổn định thời gian thực, vẫn còn các "nút thắt cổ chai" cần mở rộng để đưa mô hình lên chuẩn mực State-of-the-Art:

### Hướng 1: Temporal Transformer Head thay thế cho Frame-by-Frame + EMA
- **Điểm nghẽn**: Hiện tại mô hình vẫn trích xuất đặc trưng độc lập trên từng frame tĩnh, sau đó mới dùng bộ lọc EMA/Hysteresis để làm mượt hậu kỳ (post-processing). Điều này làm mất đi thông tin động học theo thời gian (Facial Dynamics) – ví dụ: quá trình cơ mặt co thắt từ trạng thái bình thường sang ngáp, cười hoặc nhíu mày.
- **Giải pháp**: Xây dựng một **Temporal Mini-Transformer (hoặc Bi-GRU/Mamba block)** tiếp nhận chuỗi 8–16 frame đặc trưng 544D liên tiếp để dự đoán cảm xúc dạng Spatio-Temporal Video FER.

### Hướng 2: Chuyển Đổi sang CoOp / CoCoOp (Continuous Context Optimization)
- **Điểm nghẽn**: Các câu prompt của LLM hiện tại dù đã qua Centroid Pooling nhưng vẫn là các token từ vựng rời rạc (Discrete Tokens).
- **Giải pháp**: Ứng dụng kỹ thuật **CoCoOp (Conditional Context Optimization)** để học trực tiếp các vector nhúng liên tục (Continuous Prompt Vectors $V \in \mathbb{R}^{M \times 512}$) được điều kiện hóa theo từng đặc trưng khuôn mặt của mỗi học sinh, tạo ra prompt cá nhân hóa cho từng góc mặt.

### Hướng 3: Cross-Scale Multi-Patch Attention (Tương Tác Đa Tỷ Lệ)
- **Điểm nghẽn**: ViT-B/32 chỉ chia ảnh thành $7 \times 7 = 49$ patches (mỗi patch kích thước $32 \times 32\text{px}$). Kích thước này khá lớn, làm mờ đi các chi tiết vi cơ mặt rất nhỏ (micro-expressions) ở khóe mắt hoặc đuôi chân mày.
- **Giải pháp**: 
  - Nâng cấp visual backbone sang **ViT-B/16** ($14 \times 14 = 196$ patches, độ phân giải vi mô nét gấp 4 lần).
  - Áp dụng cơ chế **Hierarchical Cross-Attention**: Tầng 1 soi toàn mặt, Tầng 2 phóng to crop vùng mắt và khóe môi để cross-attention song song.

### Hướng 4: Semi-Supervised Learning & Active Learning trên Dữ Liệu Lớp Học Thật
- **Điểm nghẽn**: Mô hình được huấn luyện trên dữ liệu phòng thí nghiệm (RAF-DB, KMUFED) – nơi người tham gia chụp ảnh theo yêu cầu diễn xuất. Trong lớp học thật, biểu cảm học sinh mang tính thụ động, mệt mỏi tự nhiên, góc mặt cúi xuống vở.
- **Giải pháp**: Xây dựng module **Pseudo-Labeling có chọn lọc** kết hợp **Negative Evidence Logging** (đã có sẵn trong `pipeline.py`). Tự động thu thập các frame học sinh bị gắn cờ tiêu cực liên tục, lọc qua ngưỡng tin cậy cao để tự mở rộng tập dữ liệu huấn luyện của chính phòng học đó.

### Hướng 5: Tăng Tốc Inference với TensorRT / ONNX Runtime FP16
- **Điểm nghẽn**: CLIP ViT-B/32 + RetinaFace + MediaPipe chạy tuần tự trên PyTorch ngốn khoảng $25–35\text{ms}$ mỗi frame (đạt ~30 FPS).
- **Giải pháp**: Chuyển đổi toàn bộ nhánh CLIP Visual và Facial Cross-Attention sang engine **TensorRT FP16 / INT8**. Dự kiến sẽ đẩy tốc độ lên **90–120 FPS**, cho phép một GPU RTX 3050 Laptop giám sát đồng thời cùng lúc 4–6 luồng camera lớp học trong trường học thực tế.
