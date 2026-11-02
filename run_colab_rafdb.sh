#!/bin/bash
# Script thực thi toàn bộ pipeline huấn luyện EduFocus AI trên RAF-DB (100 Epochs, Patience 15)
# Sử dụng trên Google Colab GPU

set -e

echo "================================================================================"
echo "🚀 EDUFOCUS AI — BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN RAF-DB (100 EPOCHS, PATIENCE 15)"
echo "================================================================================"

# Xác định thư mục dự án và clone từ GitHub nếu chưa có
PROJECT_ROOT="${PROJECT_ROOT:-/content/Emotion_Concentration}"
export PROJECT_ROOT

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "📥 Đang tải mã nguồn từ GitHub..."
    git clone https://github.com/DungLe2005/Emotion_Concentration-.git "$PROJECT_ROOT"
fi

echo "📁 Thư mục dự án: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Kiểm tra thư mục dữ liệu RAF_DB
if [ ! -d "$PROJECT_ROOT/RAF_DB/DATASET" ]; then
    echo "🔍 Không tìm thấy RAF_DB/DATASET, đang kiểm tra file nén..."
    if [ -f "$PROJECT_ROOT/RAF_DB.zip" ]; then
        echo "📦 Đang giải nén RAF_DB.zip..."
        unzip -q "$PROJECT_ROOT/RAF_DB.zip" -d "$PROJECT_ROOT"
    elif [ -f "/content/drive/MyDrive/RAF_DB.zip" ]; then
        echo "📦 Đang giải nén /content/drive/MyDrive/RAF_DB.zip..."
        unzip -q "/content/drive/MyDrive/RAF_DB.zip" -d "$PROJECT_ROOT"
    fi
fi

# Cài đặt thư viện phụ thuộc
echo "📦 Kiểm tra và cài đặt dependencies..."
pip install git+https://github.com/openai/CLIP.git -q
pip install ftfy regex tqdm pandas matplotlib scikit-learn opencv-python-headless mediapipe retina-face xgboost openpyxl seaborn -q

# Thực thi lệnh chính
echo "🚀 Chạy lệnh run_complete_pipeline.py..."
python backend/evaluation/run_complete_pipeline.py --dataset rafdb --epochs 100 --patience 15

echo "================================================================================"
echo "🎉 HOÀN THÀNH HUẤN LUYỆN TOÀN BỘ PIPELINE TRÊN RAF-DB!"
echo "================================================================================"
