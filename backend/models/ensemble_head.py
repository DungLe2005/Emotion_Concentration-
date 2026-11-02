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

try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class EmotionEnsembleHead:
    """
    Soft-Voting Ensemble Classifier Head kết hợp:
    1. EmotionResMLP (Deep Residual Network)
    2. XGBoost Classifier (Gradient Boosted Decision Trees)
    3. Chosen Third Branch (RBF-SVM Classifier hoặc Baseline EmotionMLP theo Cơ Chế 5)
    """

    def __init__(self, input_dim=289, num_classes=7, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.third_model_type = "mlp"

        # 1. Baseline MLP Head
        self.mlp = EmotionMLP(input_dim=input_dim, num_classes=num_classes).to(
            self.device
        )
        candidate_paths = [
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "emotion_mlp_best_acc_weights_new_data.pth",
            ),
            os.path.join(
                os.path.dirname(__file__), "..", "emotion_mlp_best_acc_weights.pth"
            ),
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "emotion_mlp_epso_weights_new_data.pth",
            ),
            os.path.join(
                os.path.dirname(__file__), "..", "emotion_mlp_epso_weights.pth"
            ),
        ]
        loaded_mlp = False
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    sd = torch.load(p, map_location=self.device, weights_only=True)
                    self.mlp.load_state_dict(sd)
                    print(
                        f"[EnsembleHead] ✅ Đã nạp Baseline MLP từ: {os.path.basename(p)}"
                    )
                    loaded_mlp = True
                    break
                except Exception:
                    continue
        if not loaded_mlp:
            print(
                "[EnsembleHead Warning] Chưa tìm thấy checkpoint khớp kích thước cho Baseline MLP."
            )
        self.mlp.eval()

        # 1b. RBF-SVM Classifier Head (Cơ Chế 5)
        self.svm = None
        svm_path = os.path.join(
            os.path.dirname(__file__), "..", "rbf_svm_model.joblib"
        )
        if os.path.exists(svm_path) and HAS_JOBLIB:
            try:
                self.svm = joblib.load(svm_path)
                print(
                    f"[EnsembleHead] ✅ Đã nạp RBF-SVM Model (Cơ Chế 5) từ: {os.path.basename(svm_path)}"
                )
            except Exception as e:
                print(f"[EnsembleHead Warning] Không nạp được RBF-SVM model: {e}")

        # 2. EmotionResMLP Head (Residual Skip Connections)
        self.resmlp = EmotionResMLP(input_dim=input_dim, num_classes=num_classes).to(
            self.device
        )
        resmlp_path = os.path.join(
            os.path.dirname(__file__), "..", "emotion_resmlp_weights.pth"
        )
        if os.path.exists(resmlp_path):
            self.resmlp.load_state_dict(
                torch.load(resmlp_path, map_location=self.device, weights_only=True)
            )
            print(
                f"[EnsembleHead] ✅ Đã nạp EmotionResMLP từ: {os.path.basename(resmlp_path)}"
            )
        self.resmlp.eval()

        # 3. XGBoost Classifier Head
        self.xgb = None
        if HAS_XGBOOST:
            xgb_path = os.path.join(
                os.path.dirname(__file__), "..", "xgboost_model.json"
            )
            if os.path.exists(xgb_path):
                try:
                    self.xgb = XGBClassifier()
                    self.xgb.load_model(xgb_path)
                    print(
                        f"[EnsembleHead] ✅ Đã nạp XGBoost từ: {os.path.basename(xgb_path)}"
                    )
                except Exception as e:
                    print(
                        f"[EnsembleHead Warning] Không nạp được XGBoost model từ file: {e}"
                    )
                    self.xgb = None

        # 4. Trọng số Ensemble & Cấu hình nhánh thứ 3 từ ensemble_weights.json
        self.ensemble_weights = [0.34, 0.33, 0.33]
        ens_cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "ensemble_weights.json"
        )
        if os.path.exists(ens_cfg_path):
            try:
                import json

                with open(ens_cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "weights" in cfg and len(cfg["weights"]) == 3:
                        self.ensemble_weights = [float(w) for w in cfg["weights"]]
                    if "third_model" in cfg:
                        self.third_model_type = cfg["third_model"]
                    third_label = cfg.get(
                        "third_name", self.third_model_type.upper()
                    )
                    print(
                        f"[EnsembleHead] 🎯 Đã nạp trọng số Ensemble tối ưu: ResMLP={self.ensemble_weights[0]*100:.1f}%, XGB={self.ensemble_weights[1]*100:.1f}%, {third_label}={self.ensemble_weights[2]*100:.1f}%"
                    )
            except Exception as e:
                print(
                    f"[EnsembleHead Warning] Lỗi đọc ensemble_weights.json ({e}), dùng mặc định."
                )

    def predict_probs(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Tính xác suất Softmax hợp nhất (Soft-Voting Ensemble) cho 1 mẫu hoặc 1 batch vector.
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

        # Xác suất nhánh thứ 3: RBF-SVM hoặc Baseline EmotionMLP (Cơ Chế 5)
        if self.third_model_type == "rbf_svm" and self.svm is not None:
            try:
                raw_svm_probs = self.svm.predict_proba(X_2d)
                third_probs = np.zeros((len(X_2d), self.num_classes), dtype=np.float32)
                for i, c in enumerate(self.svm.classes_):
                    if c < self.num_classes:
                        third_probs[:, c] = raw_svm_probs[:, i]
            except Exception:
                third_probs = mlp_probs
        else:
            third_probs = mlp_probs

        # Soft-Voting Ensemble (Optimal Weighted Average)
        w_res, w_xgb, w_third = self.ensemble_weights
        ensemble_probs = (
            w_res * res_probs + w_xgb * xgb_probs + w_third * third_probs
        )
        return ensemble_probs
