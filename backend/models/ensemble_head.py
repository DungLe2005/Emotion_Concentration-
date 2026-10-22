import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.mlp import EmotionMLP
from models.resmlp import EmotionResMLP

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class EmotionEnsembleHead:
    """
    Soft-Voting Ensemble Classifier Head kết hợp:
    1. EmotionResMLP (Deep Residual Network - Trọng số 0.45)
    2. XGBoost Classifier (Gradient Boosted Decision Trees - Trọng số 0.45)
    3. Baseline EmotionMLP (3-Layer FC Network - Trọng số 0.10)
    """

    def __init__(self, input_dim=289, num_classes=7, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 1. Baseline MLP Head
        self.mlp = EmotionMLP(input_dim=input_dim, num_classes=num_classes).to(self.device)
        mlp_path = os.path.join(os.path.dirname(__file__), "..", "emotion_mlp_epso_weights.pth")
        if os.path.exists(mlp_path):
            self.mlp.load_state_dict(
                torch.load(mlp_path, map_location=self.device, weights_only=True)
            )
        self.mlp.eval()

        # 2. EmotionResMLP Head (Residual Skip Connections)
        self.resmlp = EmotionResMLP(input_dim=input_dim, num_classes=num_classes).to(self.device)
        resmlp_path = os.path.join(os.path.dirname(__file__), "..", "emotion_resmlp_weights.pth")
        if os.path.exists(resmlp_path):
            self.resmlp.load_state_dict(
                torch.load(resmlp_path, map_location=self.device, weights_only=True)
            )
        self.resmlp.eval()

        # 3. XGBoost Classifier Head
        self.xgb = None
        if HAS_XGBOOST:
            xgb_path = os.path.join(os.path.dirname(__file__), "..", "xgboost_model.json")
            if os.path.exists(xgb_path):
                try:
                    self.xgb = XGBClassifier()
                    self.xgb.load_model(xgb_path)
                except Exception as e:
                    print(f"[EnsembleHead Warning] Không nạp được XGBoost model từ file: {e}")
                    self.xgb = None

    def predict_probs(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Tính xác suất Softmax hợp nhất (Soft-Voting Ensemble) cho 1 mẫu hoặc 1 batch vector (289D).
        Output: numpy array (batch_size, 7)
        """
        if feature_vector.ndim == 1:
            X_2d = feature_vector.reshape(1, -1)
        else:
            X_2d = feature_vector

        tensor_in = torch.tensor(X_2d, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            mlp_logits = self.mlp(tensor_in)
            mlp_probs = F.softmax(mlp_logits, dim=1).cpu().numpy()

            res_logits = self.resmlp(tensor_in)
            res_probs = F.softmax(res_logits, dim=1).cpu().numpy()

        if self.xgb is not None:
            try:
                raw_xgb_probs = self.xgb.predict_proba(X_2d)
                xgb_probs = np.zeros((len(X_2d), self.num_classes), dtype=np.float32)
                cols = min(raw_xgb_probs.shape[1], self.num_classes)
                xgb_probs[:, :cols] = raw_xgb_probs[:, :cols]
            except Exception:
                xgb_probs = mlp_probs
        else:
            xgb_probs = mlp_probs

        # Soft-Voting Ensemble (Weighted Average)
        ensemble_probs = 0.45 * res_probs + 0.45 * xgb_probs + 0.10 * mlp_probs
        return ensemble_probs
