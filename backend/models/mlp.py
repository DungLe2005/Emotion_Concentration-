import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionMLP(nn.Module):
    def __init__(self, input_dim=128, num_classes=8):
        super(EmotionMLP, self).__init__()
        # Mở rộng mạng nơ-ron để học được nhiều đặc trưng phức tạp hơn
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Giảm tỷ lệ rơi vãi (Dropout) từ 0.5 xuống 0.3 vì vector đầu vào khá nhỏ (~60 chiều)
        self.dropout = nn.Dropout(0.3)
        # Thay ReLU bằng LeakyReLU để tránh hiện tượng "chết" nơ-ron
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Lớp đầu ra (Chưa có softmax vì CrossEntropyLoss tự động có LogSoftmax)
        x = self.fc3(x)
        return x

    def predict_probs(self, x):
        """Trả về xác suất của từng cảm xúc"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        return probs
