import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import EmotionMLP

class ResBlock(nn.Module):
    """
    Residual Block với Skip Connection, BatchNorm, LeakyReLU và Dropout.
    """
    def __init__(self, dim, dropout=0.3):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out + residual)
        return out

class EmotionResMLP(nn.Module):
    """
    Deep Residual MLP Classifier cho vector đặc trưng đa mô thức 647D.
    Hỗ trợ Residual Connections giúp học sâu hơn mà không làm biến dạng thông tin.
    """
    def __init__(self, input_dim=647, num_classes=7):
        super(EmotionResMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        self.res1 = ResBlock(512, dropout=0.3)
        self.transition = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        self.res2 = ResBlock(256, dropout=0.3)
        self.out_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.transition(x)
        x = self.res2(x)
        x = self.out_layer(x)
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
        Dự đoán lớp cảm xúc, độ tin cậy và phân phối xác suất.
        """
        probs = self.predict_probs(x)
        pred_class = int(np.argmax(probs[0]))
        confidence = float(probs[0, pred_class])
        return pred_class, confidence, probs[0]
