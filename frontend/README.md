# 🖥️ EduFocus AI - Frontend Dashboard

Giao diện người dùng Web Application hiện đại dành cho hệ thống **Phân tích Cảm xúc & Mức độ Tập trung Học sinh (EduFocus AI)**.

Được xây dựng trên nền tảng **React 19**, **Vite** và **TailwindCSS 4**, ứng dụng cung cấp trải nghiệm theo dõi thời gian thực mượt mà, trực quan với thiết kế Light Enterprise Theme tối ưu cho giáo viên và nhà quản lý giáo dục.

---

## ✨ Tính Năng Giao Diện

- **🎥 Phân Tích Luồng Trực Tiếp (Live Stream)**: Hỗ trợ linh hoạt cả **Webcam trực tiếp** (`getUserMedia`) và **Tải file Video bài học** (`.mp4`, `.webm`, `.avi`).
- **📊 Dashboard Buổi Học Toàn Diện (SessionDashboard)**:
  - Hiển thị tỷ lệ tập trung tổng thể và xếp loại theo thang Likert 5 bậc (Xuất sắc, Tốt, Trung bình, Yếu, Rất yếu).
  - Phân tích phân phối xác suất 7 cảm xúc cơ bản (*Neutral, Happy, Surprise, Sad, Fear, Angry, Disgust*).
  - Biểu đồ biến thiên mức độ tập trung theo các mốc thời gian (Session Timeline).
- **⏱️ Dải Ảnh Thumbnail Minh Bạch 30s**: Hiển thị dải ảnh chụp các khoảnh khắc lấy mẫu của từng học sinh theo chu kỳ 30 giây để giáo viên đối chiếu trực quan.
- **🧬 Giám Sát Tiến Trình AI (PipelineVisualizer)**: Trực quan hóa 5 tầng xử lý của Pipeline AI (*Decode → Dual Detection → 544D Multimodal Extraction → EPSO 289D Selection → Ensemble Classification*) kèm độ trễ (latency ms) chi tiết từng bước.
- **📜 Bảng Nhật Ký Thời Gian Thực (LogConsole)**: Ghi nhận chi tiết luồng xử lý và các sự kiện bắt đầu/kết thúc buổi học tự động.
- **📑 Tự Động Tải Báo Cáo Excel (.xlsx)**: Nhận file Excel 2-sheet chi tiết do backend xuất ra ngay khi buổi học kết thúc.

---

## 🏗️ Cấu Trúc Thư Mục Frontend

```text
frontend/
├── src/
│   ├── assets/                 # Hình ảnh & icon tĩnh
│   ├── components/
│   │   ├── SessionDashboard.jsx    # Dashboard giám sát buổi học & thống kê
│   │   ├── PipelineVisualizer.jsx  # Trực quan hóa tiến trình 5 bước AI
│   │   └── LogConsole.jsx          # Bảng nhật ký xử lý real-time
│   ├── App.jsx                 # Điều hướng ứng dụng & phân trang tab
│   ├── App.css                 # Custom styles bổ trợ
│   ├── index.css               # Design System & TailwindCSS 4 configuration
│   └── main.jsx                # Entry point của React App
├── public/                     # Static public assets
├── package.json                # Dependencies (React 19, Lucide React, TailwindCSS 4)
├── vite.config.js              # Cấu hình Vite build tool & proxy
└── README.md                   # Tài liệu hướng dẫn Frontend
```

---

## 🛠️ Công Nghệ Sử Dụng

- **Core**: [React 19](https://react.dev/) + [Vite 6](https://vitejs.dev/)
- **Styling**: [TailwindCSS 4](https://tailwindcss.com/) + Custom CSS Design Tokens
- **Icons**: [Lucide React](https://lucide.dev/)
- **Giao tiếp API**: Fetch API / RESTful JSON gửi frame base64 tới FastAPI Backend (`http://127.0.0.1:8000`)

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Yêu cầu tiên quyết
- **Node.js**: Phiên bản `18.x` hoặc `20.x` trở lên
- **NPM**: Phiên bản `9.x` trở lên

### 2. Cài đặt thư viện dependencies
```bash
cd frontend
npm install
```

### 3. Chạy môi trường phát triển (Dev Server)
```bash
npm run dev
```
Ứng dụng sẽ khởi chạy tại: `http://localhost:5173` (hoặc cổng được Vite chỉ định).

### 4. Build phiên bản Production
```bash
npm run build
```
Thư mục bundle tĩnh được tạo tại `frontend/dist/`.
