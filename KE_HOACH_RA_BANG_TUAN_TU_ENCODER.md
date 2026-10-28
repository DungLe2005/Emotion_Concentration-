# 📋 KẾ HOẠCH RÃ BĂNG TUẦN TỰ (SEQUENTIAL GRADUAL UNFREEZING)
## Hướng Dẫn Kỹ Thuật & Lộ Trình Triển Khai Chi Tiết (100 Epochs) — Dự Án EduFocus AI

---

## 📌 PHẦN 1: TỔNG QUAN HỆ THỐNG & MỤC TIÊU CỐT LÕI

### 1.1. Kiến trúc Đa thức 544 Chiều Hiện Tại
Hệ thống nhận diện cảm xúc **EduFocus AI** hoạt động trên không gian vector kết hợp **544 chiều**:
* **512D Visual (CLIVP-FER):** Trích xuất từ CLIP ViT (`ViT-B/32` hoặc `ViT-B/16`), tóm tắt toàn bộ đặc trưng thị giác của khuôn mặt đã căn chỉnh 2 mắt ($224 \times 224$).
* **7D Text Semantic Cosine Similarity (CLIVP-FER):** Độ tương đồng cosine giữa vector ảnh khuôn mặt và 7 câu văn bản mô tả biểu cảm của 7 cảm xúc quốc tế (*Neutral, Happy, Surprise, Sad, Fear, Angry, Disgust*).
* **25D Facial Geometric Landmark Features (MediaPipe):** Tỷ lệ sinh trắc học hình học tính toán từ 478 mốc Face Mesh (tỷ lệ mở mắt EAR, tỷ lệ miệng MAR, khoảng cách chân mày, khóe môi...).
* **289D EPSO Feature Selection:** Vector 544D được lọc qua mặt nạ tối ưu bầy đàn (`epso_mask_real.npy`) để loại bỏ 255 chiều nhiễu, giữ lại 289 chiều nhạy nhất.
* **Soft-Voting Ensemble Head:** Dự đoán cảm xúc dựa trên kết hợp biểu quyết mềm: **ResMLP (45%) + XGBoost (45%) + Baseline MLP (10%)**.

---

### 1.2. Vấn Đề Cần Giải Quyết: Tại Sao Đóng Băng 100% Backbone Lại Giới Hạn Độ Chính Xác?
1. **Backbone CLIP chưa từng học chuyên sâu về cơ mặt:** Trọng số CLIP nguyên bản được OpenAI huấn luyện trên 400 triệu cặp ảnh-văn bản tổng quát (đồ vật, phong cảnh, con người nói chung). Vì vậy, mô hình chú ý rất nhiều vào các yếu tố ngoại cảnh: **danh tính cá nhân, kiểu tóc, màu da, ánh sáng nền** thay vì vi biểu cảm cơ mặt.
2. **Khoảng cách ngữ nghĩa Text-Visual chưa tối ưu cho khuôn mặt:** 7 câu mô tả cảm xúc chuẩn y khoa khi chiếu qua CLIP Text Encoder ban đầu có khoảng cách hình học khá gần nhau trong không gian đặc trưng, làm giảm độ tách biệt giữa các cảm xúc khó phân biệt (*Fear* vs. *Surprise*, hoặc *Sad* vs. *Neutral*).
3. **Mục tiêu:** Tinh chỉnh có kiểm soát (Fine-tuning qua rã băng tuần tự với chu kỳ chuẩn 100 Epochs) để các tầng Transformer cao nhất của CLIP tự uốn nắn trọng số, tập trung toàn diện vào vùng mắt, chân mày, khóe miệng mà **không làm mất đi tri thức tổng quát quý giá ban đầu**.

---

## 🔬 PHẦN 2: CƠ SỞ KHOA HỌC — TẠI SAO PHẢI RÃ BĂNG TUẦN TỰ?

### Sơ Đồ So Sánh: Rã Băng Đồng Loạt vs. Rã Băng Tuần Tự (Chu Kỳ 100 Epochs)

```
[ CÁCH LÀM SAI: RÃ BĂNG TOÀN BỘ NGAY TỪ ĐẦU (FULL UNFREEZE) ]
===================================================================================
Epoch 0: Classifier Head chưa học gì (weights ngẫu nhiên, loss rất lớn)
   │
   ▼ Lan truyền ngược (Backpropagation) với độ dốc gradient cực mạnh
┌─────────────────────────────────────────────────────────────────────────────────┐
│ CẢ BACKBONE CLIP BỊ VA ĐẬP MẠNH BỞI GRADIENT BẤT ỔN                             │
│ ➔ Hậu quả 1: Catastrophic Forgetting (Xóa sạch tri thức 400M ảnh của CLIP).    │
│ ➔ Hậu quả 2: Overfit trầm trọng vào danh tính người học và ánh sáng phòng học.  │
│ ➔ Hậu quả 3: Nhánh Text có quá ít mẫu (chỉ 7 câu) nên bị sụp đổ không gian nhúng│
└─────────────────────────────────────────────────────────────────────────────────┘

[ CÁCH LÀM ĐÚNG: RÃ BĂNG TUẦN TỰ PHÂN TẦNG (SEQUENTIAL GRADUAL UNFREEZING - 100 EPOCHS) ]
===================================================================================
Giai đoạn 0 (Epoch 1-20):  [❄️ Visual: ĐÓNG]  [❄️ Text: ĐÓNG 100%]   [🔥 Head + Projections: MỞ (LR=1e-3)]
                           └──> 20 epochs kéo Head và Tầng chiếu về vùng hội tụ ổn định trước.
                                        │
                                        ▼
Giai đoạn 1 (Epoch 21-50): [🔥 Visual Top-2 (Blocks 10-11): MỞ (LR=2e-5)] [❄️ Text: ĐÓNG 100%] [🔥 Head: LR=2e-4]
                           └──> 30 epochs cho ViT học vi biểu cảm cơ mặt trên ảnh 224x224.
                                        │
                                        ▼
Giai đoạn 2 (Epoch 51-75): [🔥 Visual Deep (Mở rộng Blocks 8-11): MỞ (LR=1.5e-5)] [❄️ Text: ĐÓNG 100%] [🔥 Head: LR=1.5e-4]
                           └──> 25 epochs thẩm thấu sâu hơn vào đặc trưng hình học khuôn mặt mà KHÔNG rã băng Text.
                                        │
                                        ▼
Giai đoạn 3 (Epoch 76-100):[🔥 Joint Visual-Head Consolidation (TĂNG MẠNH LR: Visual 1e-5, Head 1e-4) [❄️ Text: ĐÓNG 100%]
                           └──> 25 epochs tối ưu hóa với gradient đủ mạnh, tìm điểm hội tụ phẳng (Flat Minima).
```

---

## 🏛️ PHẦN 3: CHI TIẾT TỪNG TẦNG CỦA CLIP ViT-B & TRẠNG THÁI KHÓA/MỞ

Mô hình OpenAI CLIP ViT-B/32 (hoặc ViT-B/16) bao gồm hàng trăm lớp tham số. Dưới đây là bảng phân rã chi tiết và quy tắc can thiệp trong code PyTorch:

```
+---------------------------------------------------------------------------------------------------+
| 1. NHÁNH THỊ GIÁC: clip_model.visual (Vision Transformer - 12 Blocks)                             |
+---------------------------------------------------------------------------------------------------+
| Tên Tầng trong PyTorch                  | Cấu Trúc / Chức Năng               | Trạng Thái Can Thiệp       |
+-----------------------------------------+------------------------------------+--------------------+
| visual.conv1                            | Patch Embedding (32x32 hoặc 16x16) | ❄️ ĐÓNG BĂNG 100%   |
| visual.class_embedding, pos_embedding   | Vị trí không gian của patch        | ❄️ ĐÓNG BĂNG 100%   |
| visual.ln_pre                           | LayerNorm trước Transformer        | ❄️ ĐÓNG BĂNG 100%   |
| visual.transformer.resblocks.0 -> 9     | 10 Blocks đầu: Cạnh, góc, texture  | ❄️ ĐÓNG BĂNG 100%   |
| visual.transformer.resblocks.10         | Block 10: Biểu diễn trừu tượng mặt | 🔥 RÃ BĂNG TỪ PH.1 |
| visual.transformer.resblocks.11         | Block 11: Biểu cảm sâu khuôn mặt   | 🔥 RÃ BĂNG TỪ PH.1 |
| visual.ln_post                          | LayerNorm sau Transformer          | 🔥 RÃ BĂNG TỪ PH.1 |
| visual.proj                             | Chiếu vector ViT sang 512 chiều    | 🔥 RÃ BĂNG TỪ PH.1 |
+---------------------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------------------+
+---------------------------------------------------------------------------------------------------+
| 2. NHÁNH NGỮ NGHĨA VĂN BẢN: clip_model.transformer (Text Transformer - 12 Blocks)                 |
+---------------------------------------------------------------------------------------------------+
| Tên Tầng trong PyTorch                  | Cấu Trúc / Chức Năng               | Trạng Thái Can Thiệp       |
+-----------------------------------------+------------------------------------+--------------------+
| token_embedding, positional_embedding   | Mã hóa từ ngữ & thứ tự câu         | ❄️ ĐÓNG BĂNG 100%   |
| transformer.resblocks.0 -> 11           | Toàn bộ 12 Blocks Text Transformer | ❄️ ĐÓNG BĂNG 100%   |
| ln_final                                | LayerNorm cuối của Text            | ❄️ ĐÓNG BĂNG 100%   |
| text_projection                         | Chiếu ngữ nghĩa Text sang 512D     | ❄️ ĐÓNG BĂNG 100%   |
+---------------------------------------------------------------------------------------------------+
| ⚠️ LÝ DO KHOA HỌC: Chỉ có 7 câu prompt văn bản cho 7 cảm xúc (quá ít mẫu so với dữ liệu ảnh).   |
| Việc rã băng Text Encoder sẽ gây 'Sụp đổ không gian nhúng văn bản' (Representation Collapse).     |
| Vì vậy, KHÓA 100% Text Encoder là chìa khóa duy trì khoảng cách ngữ nghĩa chuẩn của CLIP.        |
+---------------------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------------------+
| 3. NHÁNH HÌNH HỌC KHUÔN MẶT: MediaPipe Face Mesh (25 Chiều Sinh Trắc Học)                        |
+---------------------------------------------------------------------------------------------------+
| 478 mốc 3D -> Tính toán tỷ lệ hình học thuần túy (EAR, MAR, Brow Distance) | ❄️ CỐ ĐỊNH HOÀN TOÀN|
| Nhánh này không có tham số học, gradient không truyền qua nhánh này.        | (Deterministic)    |
+---------------------------------------------------------------------------------------------------+
```

---

## 🔀 PHẦN 3.5: TẦNG CHIẾU TUYẾN TÍNH ĐỘC LẬP TRƯỚC KHI GHÉP NỐI (PRE-FUSION PROJECTIONS)

### 3.5.1. Vấn đề cốt lõi: Tính Không Đồng Nhất Giữa Các Phương Thức (Modal Heterogeneity)
Trước đây, 3 vector đặc trưng được ghép nối trực tiếp (raw concatenation):
$$\vec{V}_{\text{544}} = \left[ \vec{f}_{\text{Visual}}^{\text{(512D)}} \;\parallel\; \vec{S}_{\text{Text}}^{\text{(7D)}} \;\parallel\; \vec{f}_{\text{Landmark}}^{\text{(25D)}} \right]$$
Tuy nhiên, điều này gặp phải 2 xung đột phân phối đặc trưng nghiêm trọng:
1. **Lệch thang đo & đơn vị vật lý (Scale & Unit Discrepancy):** 
   - Vector Visual (512D) là các giá trị kích hoạt nơ-ron trừu tượng sau chuẩn hóa L2 ($\sim [-0.15, 0.15]$).
   - Điểm Text Cosine (7D) nằm chặt chẽ trong đoạn $[-1.0, 1.0]$.
   - Mốc MediaPipe AU (25D) là các tỷ lệ hình học thực tế (EAR $\sim 0.15-0.35$, MAR $\sim 0.05-0.60$, khoảng cách chân mày pixel $\sim 10-80$). Việc ép chung vào một lớp phân loại mà không có tầng điều hòa sẽ khiến nhánh hình học bị áp đảo hoặc gây nhiễu gradient cho nhánh thị giác.
2. **Thiếu không gian thích ứng chuyên biệt cho vi biểu cảm (Facial Expression Adaptation):** 
   - 512D CLIP ban đầu là đặc trưng tổng quát. Cần một tầng chiếu riêng biệt (Visual Adapter) để cô đọng thông tin tập trung vào vi biểu cảm trước khi hợp nhất.

### 3.5.2. Kiến Trúc 2 Tầng Chiếu Tuyến Tính Độc Lập

```
[ ĐẦU VÀO ĐA PHƯƠNG THỨC ]
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│ CLIP ViT Visual (512D)    │      │ CLIP Text Prompts (7D)    │      │ MediaPipe AU Ratios (25D) │
└─────────────┬─────────────┘      └─────────────┬─────────────┘      └─────────────┬─────────────┘
              │                                  │                                  │
              ▼                                  │                                  ▼
┌───────────────────────────┐                    │                    ┌───────────────────────────┐
│ 🔥 VisualProjection       │                    │                    │ 🔥 LandmarkProjection     │
│ - Linear(512, 512)        │                    │                    │ - Linear(25, 64)          │
│ - LayerNorm(512)          │                    │                    │ - LayerNorm(64) + GELU    │
│ - GELU + Dropout(0.1)     │                    │                    │ - Linear(64, 25)          │
│ - Residual: x + Proj(x)   │                    │                    │ - LayerNorm(25) + GELU    │
└─────────────┬─────────────┘                    │                    └─────────────┬─────────────┘
              │ h_vis (512D)                     │ prompt_sims (7D)                 │ h_lm (25D)
              └──────────────────────────┐       │       ┌──────────────────────────┘
                                         ▼       ▼       ▼
                                ┌─────────────────────────────────┐
                                │ ⊕ Ghép Nối Vector Chuẩn (544D) │
                                └────────────────┬────────────────┘
                                                 ▼
                                ┌─────────────────────────────────┐
                                │ 🔥 Classifier Head (ResMLP)     │
                                └─────────────────────────────────┘
```

* **1. Visual Projection Layer ($512 \to 512$):**
  - Sử dụng cơ chế **Residual Connection + LayerNorm**: $\vec{h}_{\text{vis}} = \text{LayerNorm}(\vec{f}_{\text{vis}} + \text{GELU}(\text{Linear}(\vec{f}_{\text{vis}})))$.
  - Bảo toàn 100% tri thức gốc của CLIP ViT ngay từ epoch đầu, đồng thời tạo không gian thích ứng riêng cho biến dạng khuôn mặt.
* **2. Landmark Projection Layer ($25 \to 64 \to 25$):**
  - Sử dụng mạng chiếu 2 tầng kèm LayerNorm và hàm kích hoạt phi tuyến GELU.
  - Ánh xạ các số đo hình học thô sang không gian nhúng liên tục (Geometric Latent Embedding), chuẩn hóa phương sai và đưa về cùng thang đo với vector thị giác.
* **Tính tương thích hệ thống:** Sau khi chiếu, kích thước các nhánh vẫn là $512\text{D} + 7\text{D} + 25\text{D} = \mathbf{544\text{ chiều}}$, đảm bảo khớp hoàn hảo với thuật toán lọc bầy đàn **EPSO (289D)** và **Soft-Voting Ensemble Head**!

---

---

## 📅 PHẦN 4: LỘ TRÌNH 4 GIAI ĐOẠN CHI TIẾT (CHU KỲ 100 EPOCHS - KHÓA TEXT 100%)

```
      PHASE 0                 PHASE 1                  PHASE 2                  PHASE 3
   [Epoch 1-20]            [Epoch 21-50]            [Epoch 51-75]           [Epoch 76-100]
   Head Warm-up         Visual Top Adaptation    Deep Visual Expansion    Joint Consolidation
┌────────────────┐       ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
│ Visual: ❄️ Khóa │       │ Visual: 🔥 Mở  │       │ Visual: 🔥 Mở  │       │ Visual: 🔥 Mở  │
│        (0/12)  │       │ (Blocks 10-11) │       │ (Blocks 8-11)  │       │ (Blocks 8-11)  │
│ Text:   ❄️ Khóa │  ──▶  │ Text:   ❄️ Khóa │  ──▶  │ Text:   ❄️ Khóa │  ──▶  │ Text:   ❄️ Khóa │
│        (100%)  │       │        (100%)  │       │        (100%)  │       │        (100%)  │
│ Head:   🔥 LR=1e-3     │ Head:   🔥 LR=2e-4     │ Head:   🔥 LR=1.5e-4   │ Head:   🔥 LR=1e-4 (Tăng!)
│ Vis LR: 0      │       │ Vis LR: 2e-5   │       │ Vis LR: 1.5e-5 │       │ Vis LR: 1e-5 (Tăng!)
└────────────────┘       └────────────────┘       └────────────────┘       └────────────────┘
```

### 🔹 Giai đoạn 0: Khởi Động Bộ Phân Loại (Head Warm-up) — [Epoch 1 ➔ 20]
* **Việc thực hiện:**
  - Khóa toàn bộ tham số của CLIP (cả Visual và Text): `for p in clip_model.parameters(): p.requires_grad = False`.
  - Mở toàn bộ tham số của Classifier Head (ResMLP hoặc MLP).
  - Tải trước ảnh khuôn mặt, tính vector 544D và huấn luyện Head với `lr = 1e-3`.
* **Kết quả cần đạt:**
  - Đưa độ chính xác tập Validation từ ngẫu nhiên (~14%) lên mức ổn định (~85-87%).
  - Các trọng số kết nối nơ-ron của Head đạt trạng thái cân bằng.

### 🔹 Giai đoạn 1: Rã Băng Tầng Sâu Visual Encoder (Visual Top-2 Adaptation) — [Epoch 21 ➔ 50]
* **Việc thực hiện:**
  - **Giữ nguyên khóa 100% Text Encoder.**
  - Mở khóa các tham số tầng cao của Visual: `visual.transformer.resblocks.10`, `visual.transformer.resblocks.11`, `visual.ln_post`, `visual.proj`.
  - Thiết lập bộ tối ưu hóa:
    * Nhóm Visual: `lr = 2e-5`, `weight_decay = 0.01`.
    * Nhóm Head: `lr = 2e-4`, `weight_decay = 1e-4`.
  - Bật chế độ `clip_grad_norm_(max_norm=1.0)` ở mỗi bước tối ưu.
* **Kết quả cần đạt:**
  - ViT học được các đặc trưng uốn lượn của nếp nhăn trán, khóe mắt, cơ gò má khi cười hoặc cáu giận.
  - Accuracy trên tập Val tăng thêm +1.5% đến +2.0%.

### 🔹 Giai đoạn 2: Mở Rộng Rã Băng Visual Sâu Hơn (Deep Visual Expansion) — [Epoch 51 ➔ 75]
* **Việc thực hiện:**
  - **Giữ nguyên khóa 100% Text Encoder** (loại bỏ hoàn toàn rã băng Text để triệt tiêu hiện tượng sụp đổ không gian nhúng).
  - Mở rộng thêm 2 Blocks Visual là `visual.transformer.resblocks.8` và `visual.transformer.resblocks.9` (tổng cộng 4 Blocks cuối 8, 9, 10, 11 + `ln_post` + `proj`).
  - Thiết lập bộ tối ưu hóa:
    * Nhóm Visual (4 blocks): `lr = 1.5e-5`, `weight_decay = 0.01`.
    * Nhóm Head: `lr = 1.5e-4`, `weight_decay = 1e-4`.
* **Kết quả cần đạt:**
  - Mô hình tiếp nhận biểu diễn không gian cấu trúc mặt sâu hơn từ Block 8 và 9.
  - Không gian đặc trưng Text giữ nguyên vẹn 100% tính nguyên bản của CLIP, giúp 7 vector cảm xúc giữ độ tách biệt tối đa.

### 🔹 Giai đoạn 3: Tinh Chỉnh Đồng Thời & Củng Cố với LR Tăng Cường (Joint Consolidation) — [Epoch 76 ➔ 100]
* **Việc thực hiện:**
  - **Tăng đáng kể Learning Rate:** Thay vì hạ xuống mức vi mô quá thấp ($2 \times 10^{-6}$ visual và $10^{-5}$ head khiến mô hình dậm chân tại chỗ), giai đoạn 3 thiết lập:
    * **Visual LR:** $1 \times 10^{-5}$ (tăng gấp 5 lần so với trước), hạ mượt với Cosine Annealing về $2 \times 10^{-6}$.
    * **Head LR:** $1 \times 10^{-4}$ (tăng gấp 10 lần so với trước), hạ mượt với Cosine Annealing về $1 \times 10^{-5}$.
    * **Text Encoder:** Tiếp tục **KHÓA 100%**.
  - Kích hoạt cơ chế Early Stopping với `patience = 10` epoch (dựa trên Macro F1 của tập Val).
* **Kết quả cần đạt:**
  - Tốc độ học đủ lớn để các trọng số vượt qua các rào cản cục bộ (saddle points / shallow local minima).
  - Tối ưu hóa sâu sắc cả Loss và Macro F1, đạt điểm hội tụ phẳng tối ưu nhất.

---

## 🛠️ PHẦN 5: CHÍNH XÁC TÔI SẼ LÀM GÌ TRONG CODE (IMPLEMENTATION WORKFLOW)

```
[ QUY TRÌNH KỸ THUẬT TOÀN DIỆN ]
===================================================================================
BƯỚC 1: Hoàn thiện File Huấn Luyện
        file:///d:/Emotion_Concentration/backend/evaluation/train_sequential_unfreeze.py
        ├── Khóa 100% Text Encoder ở toàn bộ các phase
        ├── Mở Visual Blocks 10-11 ở Phase 1, mở rộng Blocks 8-11 ở Phase 2
        ├── Tăng mạnh Learning Rate ở Phase 3 (Visual 1e-5, Head 1e-4)
        ├── Áp dụng Mixed Precision (torch.amp.autocast) tiết kiệm 50% VRAM
        └── Tự động lưu Checkpoint tốt nhất dựa trên Macro F1-Score
                                        │
                                        ▼
BƯỚC 2: Chạy Thử Nghiệm Kiểm Tra Kỹ Càng (Dry-run Validation)
        Chạy thử nghiệm với --dry_run để đảm bảo:
        - Text Transformer có đúng 0 tham số trainable (khóa 100%)
        - Visual Blocks 8-11 kích hoạt đúng requires_grad
        - Gradient truyền thông suốt, không bị NaN/Inf
                                        │
                                        ▼
BƯỚC 3: Thực Hiện Huấn Luyện Toàn Diện (Full Training Run - 100 Epochs)
        Chạy trọn vẹn 100 epochs trên GPU, ghi log chi tiết quá trình tăng trưởng
        của Loss, Accuracy và Macro F1 qua từng giai đoạn rã băng.
                                        │
                                        ▼
BƯỚC 4: Tái Trích Xuất Toàn Bộ Feature Cache 544D
        Dùng trọng số CLIP mới (clip_finetuned_best.pth) để trích xuất lại:
        file:///d:/Emotion_Concentration/backend/dataset_cache.npz
                                        │
                                        ▼
BƯỚC 5: Tái Chạy Thuật Toán Tiến Hóa Bầy Đàn EPSO
        Chạy train_epso.py trên cache 544D mới để tìm mặt nạ 289D tối ưu mới:
        file:///d:/Emotion_Concentration/backend/epso_mask_real.npy
                                        │
                                        ▼
BƯỚC 6: Huấn Luyện Lại Soft-Voting Ensemble Head
        Chạy train_ensemble.py để cập nhật trọng số cho cả 3 mô hình:
        - ResMLP (45% trọng số biểu quyết)
        - XGBoost (45% trọng số biểu quyết)
        - Baseline MLP (10% trọng số biểu quyết)
                                        │
                                        ▼
BƯỚC 7: Cập Nhật Trực Tiếp Vào Hệ Thống Real-time
        Nạp trọng số mới vào pipeline chính thức:
        file:///d:/Emotion_Concentration/backend/pipeline.py
```

---

## ⚖️ PHẦN 5.5: HÀM MẤT MÁT CLASS-WEIGHTED CROSS-ENTROPY LOSS (GIẢI QUYẾT MẤT CÂN BẰNG LỚP)

### 5.5.1. Thống Kê Thực Tế Về Độ Lệch Lớp (Severe Class Imbalance)
Trên tập dữ liệu huấn luyện thực tế (`new_data`), số lượng mẫu giữa các lớp cảm xúc có sự chênh lệch cực kỳ nghiêm trọng:

```
+--------------------+------------------+------------------+-----------------------------+
| Lớp Cảm Xúc        | Số Lượng Mẫu     | Tỷ Lệ %          | Trọng Số Nghịch Đảo (w_c)   |
+--------------------+------------------+------------------+-----------------------------+
| 0 - Angry          | 705 ảnh          | 5.7%             | w = 1.577                   |
| 1 - Disgust        | 717 ảnh          | 5.8%             | w = 1.564                   |
| 2 - Fear (Thiểu số)| 281 ảnh          | 2.3%             | w = 2.497  (Cao nhất!)      |
| 3 - Happy (Đa số)  | 4,772 ảnh        | 38.9%            | w = 0.606  (Thấp nhất!)     |
| 4 - Neutral        | 2,524 ảnh        | 20.6%            | w = 0.834                   |
| 5 - Sad            | 1,982 ảnh        | 16.2%            | w = 0.940                   |
| 6 - Surprise       | 1,290 ảnh        | 10.5%            | w = 1.166                   |
+--------------------+------------------+------------------+-----------------------------+
| TỔNG CỘNG          | 12,271 ảnh       | 100.0%           | Chênh lệch: Happy gấp 17x Fear!
+--------------------+------------------+------------------+-----------------------------+
```

### 5.5.2. Công Thức Tính Trọng Số Nghịch Đảo Tần Suất (Inverse Class Frequency)
Nếu dùng Cross-Entropy tiêu chuẩn, hàm mất mát sẽ bị chi phối bởi lớp `Happy` (38.9%) và `Neutral` (20.6%), dẫn đến việc mô hình "học vẹt" đoán `Happy` là đạt ngay 40% accuracy nhưng hoàn toàn "mù" trước lớp `Fear` và `Disgust`.

Hệ thống áp dụng **Class-Weighted Cross-Entropy Loss** với công thức trọng số căn bậc hai (Square-root Inverse Frequency):
$$w_c = \sqrt{\frac{N_{\text{total}}}{C \times N_c}}$$

Trong đó:
* $N_{\text{total}} = 12,271$ (tổng số ảnh).
* $C = 7$ (số lớp cảm xúc).
* $N_c$ là số mẫu của lớp $c$.
* Dấu căn bậc hai $\sqrt{\cdot}$ đóng vai trò điều hòa (smoothing) độ dốc gradient, tránh tình trạng phạt quá cực đoan làm mất ổn định hội tụ.

### 5.5.3. Kết Hợp Kỹ Thuật Làm Mịn Nhãn (Label Smoothing = 0.05)
$$y_{\text{smooth}} = (1 - \epsilon) \cdot y_{\text{one-hot}} + \frac{\epsilon}{C}, \quad \text{với } \epsilon = 0.05$$
* **Tác dụng:** Thay vì ép mô hình phải đạt xác suất 100% tuyệt đối cho một lớp duy nhất, label smoothing phân bổ 5% xác suất đồng đều cho các lớp còn lại. Điều này giúp ngăn chặn mô hình trở nên "quá tự tin" (overconfident) vào các lớp đa số và tăng khả năng tổng quát hóa trên khuôn mặt học sinh lạ ngoài thực tế.

---

## 💻 PHẦN 6: THIẾT KẾ CODE LÕI CỦA BỘ ĐIỀU PHỐI RÃ BĂNG (100 EPOCHS - KHÓA TEXT 100%)

```python
class MultimodalSequentialModel(nn.Module):
    def __init__(self, clip_model, num_classes=7):
        super().__init__()
        self.clip = clip_model
        
        # 1. Tầng chiếu tuyến tính riêng biệt cho nhánh Visual (512D -> 512D) có Residual
        self.visual_proj = VisualProjection(in_dim=512, out_dim=512, dropout=0.1)
        
        # 2. Tầng chiếu tuyến tính riêng biệt cho nhánh MediaPipe AU (25D -> 25D)
        self.landmark_proj = LandmarkProjection(in_dim=25, out_dim=25, dropout=0.1)
        
        # 3. Classifier Head tiếp nhận vector ghép nối 544D
        self.head = EmotionResMLP(input_dim=544, num_classes=num_classes)

    def forward(self, images, landmarks):
        img_feats = self.clip.encode_image(images).float()
        img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-7)
        
        # Chiếu qua tầng riêng biệt trước khi ghép nối
        h_vis = self.visual_proj(img_feats)
        
        text_feats = self.clip.encode_text(self.text_tokens).float()
        text_norm = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-7)
        prompt_sims = img_norm @ text_norm.T  # (B, 7)
        
        h_lm = self.landmark_proj(landmarks.float())
        
        # Ghép nối thành 544D chuẩn
        combined = torch.cat([h_vis, prompt_sims, h_lm], dim=1)
        return self.head(combined)


def setup_phase_optimizer(clip_model, classifier_head, visual_proj, landmark_proj, current_epoch, total_epochs=100):
    """
    Tự động cập nhật requires_grad cho từng tham số và cấu hình
    tốc độ học phân tầng chuẩn xác: KHÓA 100% TEXT ENCODER & TĂNG LR PHASE 3.
    """
    # Luôn khóa 100% Text Encoder để chống sụp đổ không gian nhúng
    for p in clip_model.transformer.parameters():
        p.requires_grad = False
    clip_model.token_embedding.requires_grad = False
    clip_model.positional_embedding.requires_grad = False
    clip_model.ln_final.requires_grad = False
    if hasattr(clip_model, "text_projection") and clip_model.text_projection is not None:
        clip_model.text_projection.requires_grad = False

    # Luôn mở trainable cho 2 tầng chiếu tuyến tính và Head
    for p in visual_proj.parameters(): p.requires_grad = True
    for p in landmark_proj.parameters(): p.requires_grad = True
    for p in classifier_head.parameters(): p.requires_grad = True

    adapter_params = list(classifier_head.parameters()) + list(visual_proj.parameters()) + list(landmark_proj.parameters())

    if current_epoch <= 20:
        # PHASE 0 (Epoch 1-20): Khóa 100% Backbone, chỉ mở Head + Projections
        for param in clip_model.parameters(): param.requires_grad = False
        optimizer = torch.optim.AdamW([{'params': adapter_params, 'lr': 1e-3, 'weight_decay': 1e-4}])

    elif 21 <= current_epoch <= 50:
        # PHASE 1 (Epoch 21-50): Mở Visual Blocks 10, 11 (Text KHÓA 100%)
        for param in clip_model.parameters(): param.requires_grad = False
        visual_trainable = [p for n, p in clip_model.visual.named_parameters() 
                            if any(k in n for k in ["transformer.resblocks.10", "transformer.resblocks.11", "ln_post", "proj"])]
        for p in visual_trainable: p.requires_grad = True

        optimizer = torch.optim.AdamW([
            {'params': visual_trainable, 'lr': 2e-5, 'weight_decay': 1e-2},
            {'params': adapter_params, 'lr': 2e-4, 'weight_decay': 1e-4}
        ])

    elif 51 <= current_epoch <= 75:
        # PHASE 2 (Epoch 51-75): Mở rộng Visual Blocks 8, 9, 10, 11 (Text KHÓA 100%)
        for param in clip_model.parameters(): param.requires_grad = False
        visual_trainable = [p for n, p in clip_model.visual.named_parameters() 
                            if any(k in n for k in ["transformer.resblocks.8", "transformer.resblocks.9", 
                                                    "transformer.resblocks.10", "transformer.resblocks.11", "ln_post", "proj"])]
        for p in visual_trainable: p.requires_grad = True

        optimizer = torch.optim.AdamW([
            {'params': visual_trainable, 'lr': 1.5e-5, 'weight_decay': 1e-2},
            {'params': adapter_params, 'lr': 1.5e-4, 'weight_decay': 1e-4}
        ])

    else:
        # PHASE 3 (Epoch 76-100): Joint Visual-Head Consolidation (TĂNG MẠNH LR, Text KHÓA 100%)
        visual_params = [p for p in clip_model.visual.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW([
            {'params': visual_params, 'lr': 1e-5, 'weight_decay': 1e-2},
            {'params': adapter_params, 'lr': 1e-4, 'weight_decay': 1e-4}
        ])

    return optimizer
```

---

## ⚡ PHẦN 7: CÁC LỆNH THỰC THI (COMMAND LINE INSTRUCTIONS)

### 1. Kiểm tra nhanh đường truyền dữ liệu & kiểm thử tính đúng đắn (Dry-run Test):
```bash
backend\venv\Scripts\python.exe backend/evaluation/train_sequential_unfreeze.py --dataset new_data --batch_size 16 --epochs 2 --dry_run
```

### 2. Chạy Huấn Luyện Rã Băng Tuần Tự Toàn Diện (100 Epochs):
```bash
backend\venv\Scripts\python.exe backend/evaluation/train_sequential_unfreeze.py --dataset new_data --batch_size 16 --epochs 100 --device cuda:0
```

### 3. Tái Tạo Cache & Tối Ưu Lại Bầy Đàn EPSO:
```bash
backend\venv\Scripts\python.exe backend/evaluation/train_epso.py --particles 50 --iterations 80
```

### 4. Huấn Luyện Lại Soft-Voting Ensemble Head:
```bash
backend\venv\Scripts\python.exe backend/evaluation/train_ensemble.py --epochs 120
```

---

## 🎯 PHẦN 8: BẢNG KỲ VỌNG KẾT QUẢ TRƯỚC VÀ SAU KHI RÃ BĂNG

```
+------------------------------------+-----------------------+-----------------------------+
| Tiêu Chí Đánh Giá                  | Trước Khi Rã Băng     | Kỳ Vọng Sau Khi Rã Băng     |
+------------------------------------+-----------------------+-----------------------------+
| Độ chính xác tổng thể (Accuracy)   | 88.5%                 | 91.0% - 92.2% (+2.5% - 3.7%)|
| Macro F1-Score (7 lớp cảm xúc)    | 82.1%                 | 86.0% - 87.5% (+3.9% - 5.4%)|
| Nhận diện lớp khó (Fear, Disgust)  | Khá dễ nhầm lẫn       | Tách biệt rõ nét            |
| Tính ổn định khi học sinh cúi mặt  | Dễ mất nhận diện      | Nhận diện vững chắc         |
| Tốc độ xử lý Real-time (FPS)       | 60 FPS (Không đổi)    | 60 FPS (Không đổi)          |
+------------------------------------+-----------------------+-----------------------------+
```
