import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmotionMLP(nn.Module):
    """
    Baseline EmotionMLP Classifier cho vector đặc trưng đa mô thức (647D).
    
    Cấu trúc:
    - Input: 647 chiều (512D Visual Proj + 7D Text Similarity + 128D Facial Query Cross-Attention)
    - Tầng 1: Linear(647 -> 512) + BatchNorm1d(512) + LeakyReLU(0.1) + Dropout(0.3)
    - Tầng 2: Linear(512 -> 256) + BatchNorm1d(256) + LeakyReLU(0.1) + Dropout(0.3)
    - Tầng 3: Linear(256 -> num_classes=7)
    
    Mô hình đã được thực nghiệm chứng minh trên tập RAF-DB vượt trội hơn kiến trúc
    Residual và các bộ phân loại tổ hợp phức tạp nhờ khả năng tổng quát hóa cao và tránh overfitting.
    """
    def __init__(self, input_dim=647, num_classes=7):
        super(EmotionMLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1 or not self.training:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

    def predict_probs(self, x):
        """
        Dự đoán phân phối xác suất mềm (Softmax probabilities).
        Nhận vào: torch.Tensor hoặc np.ndarray (1D hoặc 2D).
        Trả về: np.ndarray shape (batch_size, num_classes).
        """
        self.eval()
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, x):
        """
        Dự đoán lớp cảm xúc, độ tin cậy và phân phối xác suất cho 1 mẫu.
        Trả về: (predicted_class_idx, confidence, probabilities_array)
        """
        probs = self.predict_probs(x)
        pred_class = int(np.argmax(probs[0]))
        confidence = float(probs[0, pred_class])
        return pred_class, confidence, probs[0]

