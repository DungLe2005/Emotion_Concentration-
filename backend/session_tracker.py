import os
import json
import time
from typing import Dict, List, Any, Optional

EMOTION_KEYS = ["neutral", "happy", "surprise", "sad", "fear", "angry", "disgust"]

DEFAULT_WEIGHTS = {
    "neutral": 0.30,
    "happy": 0.10,
    "surprise": 0.15,
    "sad": -0.05,
    "fear": -0.20,
    "angry": -0.10,
    "disgust": -0.05,
}

# ============================================================================
# Nhóm cảm xúc theo chiều Emotional Engagement (Fredricks et al., 2004)
# - Tích cực (Positive): neutral (tập trung bình thường), happy (hứng thú), surprise (tò mò)
# - Tiêu cực (Negative): sad (buồn), fear (lo lắng), angry (bực bội), disgust (chán ghét)
# ============================================================================
POSITIVE_EMOTIONS = ["neutral", "happy", "surprise"]
NEGATIVE_EMOTIONS = ["sad", "fear", "angry", "disgust"]

# Tên cảm xúc tiếng Việt cho báo cáo
EMOTION_LABELS_VI = {
    "neutral": "Bình thường (Neutral)",
    "happy": "Vui vẻ / Hứng thú (Happy)",
    "surprise": "Tò mò / Bất ngờ (Surprise)",
    "sad": "Buồn / Khó hiểu (Sad)",
    "fear": "Lo lắng / Áp lực (Fear)",
    "angry": "Bực bội / Thất vọng (Angry)",
    "disgust": "Chán ghét / Phản cảm (Disgust)",
}

# ============================================================================
# Thang Xếp Loại 5 Bậc Likert (Likert, 1932)
# Phân khoảng đều (Equal Interval Method) trên thang 0–100%
# ============================================================================
ENGAGEMENT_LEVELS = [
    {
        "min": 80,
        "max": 100,
        "level_vi": "Xuất sắc",
        "level_en": "Excellent",
        "color": "emerald",
        "description": "Lớp học rất tập trung, cảm xúc tích cực chiếm ưu thế tuyệt đối.",
    },
    {
        "min": 60,
        "max": 79.9,
        "level_vi": "Tốt",
        "level_en": "Good",
        "color": "blue",
        "description": "Phần lớn học sinh duy trì trạng thái tập trung ổn định, ít biểu hiện tiêu cực.",
    },
    {
        "min": 40,
        "max": 59.9,
        "level_vi": "Trung bình",
        "level_en": "Average",
        "color": "amber",
        "description": "Mức tập trung dao động, có dấu hiệu phân tán chú ý trong lớp.",
    },
    {
        "min": 20,
        "max": 39.9,
        "level_vi": "Yếu",
        "level_en": "Below Average",
        "color": "orange",
        "description": "Phần lớn học sinh thiếu tập trung, biểu hiện cảm xúc tiêu cực chiếm tỷ lệ cao.",
    },
    {
        "min": 0,
        "max": 19.9,
        "level_vi": "Rất yếu",
        "level_en": "Poor",
        "color": "red",
        "description": "Lớp học hoàn toàn mất tập trung, cần can thiệp sư phạm khẩn cấp.",
    },
]

# Khuyến nghị sư phạm theo mức xếp loại
RECOMMENDATIONS = {
    "Xuất sắc": [
        "Duy trì phương pháp giảng dạy hiện tại — đây là buổi học hiệu quả cao.",
        "Có thể tăng độ khó nội dung hoặc bổ sung hoạt động thảo luận nhóm nâng cao.",
        "Ghi nhận và khen ngợi sự tập trung của lớp để duy trì động lực học tập.",
    ],
    "Tốt": [
        "Phương pháp giảng dạy hiệu quả, tiếp tục phát huy.",
        "Có thể bổ sung hoạt động tương tác ngắn (quiz, hỏi đáp) để nâng mức tập trung lên Xuất sắc.",
        "Chú ý đến nhóm học sinh có biểu hiện phân tán để hỗ trợ kịp thời.",
    ],
    "Trung bình": [
        "Cần xem xét điều chỉnh nhịp độ giảng dạy — xen kẽ giữa lý thuyết và thực hành.",
        "Tăng cường hoạt động tương tác: câu hỏi mở, bài tập nhóm ngắn, hoặc demo trực quan.",
        "Kiểm tra liệu nội dung có quá khó hoặc quá dễ dẫn đến mất hứng thú.",
        "Sử dụng hình ảnh, video minh họa để tăng sự chú ý.",
    ],
    "Yếu": [
        "Cần thay đổi đáng kể phương pháp giảng dạy cho buổi học tiếp theo.",
        "Ưu tiên hoạt động thực hành, trò chơi hóa (gamification), và học qua trải nghiệm.",
        "Chia nhỏ nội dung bài giảng, tăng tần suất nghỉ giải lao ngắn (micro-breaks).",
        "Tìm hiểu nguyên nhân gốc rễ: nội dung quá khó? Thời điểm học không phù hợp? Môi trường lớp?",
    ],
    "Rất yếu": [
        "KHẨN CẤP: Buổi học hoàn toàn không hiệu quả — cần trao đổi trực tiếp với giáo viên.",
        "Đánh giá lại toàn bộ phương pháp giảng dạy, nội dung, và cách tổ chức lớp học.",
        "Xem xét yếu tố ngoại cảnh: lớp học quá đông, nhiệt độ phòng, tiếng ồn, thời gian biểu.",
        "Tham khảo ý kiến chuyên gia giáo dục hoặc tổ trưởng bộ môn.",
        "Triển khai khảo sát phản hồi trực tiếp từ học sinh để hiểu nguyên nhân.",
    ],
}


def calculate_concentration_score(
    prob_dict: Dict[str, float], weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Hàm tính toán điểm độ tập trung (0 - 100%) chuẩn hóa từ dict xác suất cảm xúc.
    Sử dụng thống nhất cho cả per-frame (Single Frame / Video Realtime) và Session Tracker.
    """
    w_dict = weights if weights is not None else DEFAULT_WEIGHTS

    # Chuẩn hóa keys về lowercase
    p_dict = {k.lower(): float(v) for k, v in prob_dict.items()}

    sum_w_p = sum(w_dict.get(k, 0.0) * p_dict.get(k, 0.0) for k in EMOTION_KEYS)

    min_possible = min(w_dict.values())
    max_possible = max(w_dict.values())

    if max_possible > min_possible:
        score_normalized = (
            (sum_w_p - min_possible) / (max_possible - min_possible)
        ) * 100.0
    else:
        score_normalized = 50.0

    return max(0.0, min(100.0, round(score_normalized, 1)))


def generate_qualitative_assessment(summary_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sinh đánh giá văn bản khoa học (Qualitative Assessment) từ dữ liệu tổng kết buổi học.

    Cơ sở khoa học:
    - Xếp loại 5 bậc theo thang Likert (Likert, 1932) với phân khoảng đều.
    - Phân nhóm cảm xúc tích cực/tiêu cực theo mô hình Emotional Engagement (Fredricks et al., 2004).
    - Phương pháp đánh giá tự động từ biểu cảm khuôn mặt (Whitehill et al., 2014).

    Args:
        summary_dict: Dict chứa final_score, emotions_percent, duration_seconds, ...

    Returns:
        Dict gồm engagement_level, dominant_emotions, overall_comment, recommendations, ...
    """
    final_score = summary_dict.get("final_score", 0.0)
    emotions_pct = summary_dict.get("emotions_percent", {})
    duration_sec = summary_dict.get("duration_seconds", 0)
    total_faces = summary_dict.get("total_faces", 0)
    avg_faces = summary_dict.get("average_faces_per_frame", 0)

    # ── 1. Xếp Loại Mức Độ Tập Trung (5-Level Likert Scale) ──
    engagement_info = ENGAGEMENT_LEVELS[-1]  # Default: Rất yếu
    for level in ENGAGEMENT_LEVELS:
        if level["min"] <= final_score <= level["max"]:
            engagement_info = level
            break

    # ── 2. Phân Tích Cảm Xúc Chủ Đạo (Dominant Emotion Profile) ──
    # Sắp xếp theo % giảm dần, lấy top-2
    sorted_emotions = sorted(emotions_pct.items(), key=lambda x: x[1], reverse=True)
    dominant_emotions = []
    for emo_key, emo_pct in sorted_emotions[:2]:
        dominant_emotions.append(
            {
                "emotion": emo_key,
                "label_vi": EMOTION_LABELS_VI.get(emo_key, emo_key),
                "percent": round(emo_pct, 2),
            }
        )

    # Tính tỷ lệ nhóm tích cực vs tiêu cực (Fredricks et al., 2004)
    positive_sum = sum(emotions_pct.get(e, 0.0) for e in POSITIVE_EMOTIONS)
    negative_sum = sum(emotions_pct.get(e, 0.0) for e in NEGATIVE_EMOTIONS)
    total_sum = positive_sum + negative_sum
    positive_ratio = round((positive_sum / total_sum * 100) if total_sum > 0 else 0, 1)
    negative_ratio = round((negative_sum / total_sum * 100) if total_sum > 0 else 0, 1)

    # ── 3. Sinh Nhận Xét Tổng Quan (Overall Comment) ──
    duration_min = duration_sec // 60
    duration_s = duration_sec % 60

    # Mô tả cảm xúc chủ đạo
    if len(dominant_emotions) >= 2:
        dom_desc = (
            f"{dominant_emotions[0]['label_vi']} ({dominant_emotions[0]['percent']}%) "
            f"và {dominant_emotions[1]['label_vi']} ({dominant_emotions[1]['percent']}%)"
        )
    elif len(dominant_emotions) == 1:
        dom_desc = (
            f"{dominant_emotions[0]['label_vi']} ({dominant_emotions[0]['percent']}%)"
        )
    else:
        dom_desc = "không xác định"

    # Sinh nhận xét theo ngữ cảnh
    level_vi = engagement_info["level_vi"]

    if positive_ratio >= 75:
        sentiment_comment = (
            f"Tỷ lệ cảm xúc tích cực ({positive_ratio}%) vượt trội rõ rệt so với tiêu cực ({negative_ratio}%), "
            f"cho thấy môi trường học tập lành mạnh và hiệu quả."
        )
    elif positive_ratio >= 50:
        sentiment_comment = (
            f"Tỷ lệ cảm xúc tích cực ({positive_ratio}%) nhỉnh hơn tiêu cực ({negative_ratio}%), "
            f"nhưng vẫn có dư địa cải thiện để nâng cao chất lượng buổi học."
        )
    else:
        sentiment_comment = (
            f"Tỷ lệ cảm xúc tiêu cực ({negative_ratio}%) chiếm ưu thế so với tích cực ({positive_ratio}%), "
            f"cho thấy lớp học đang gặp vấn đề về sự gắn kết và hứng thú."
        )

    overall_comment = (
        f"Buổi học kéo dài {duration_min} phút {duration_s} giây, "
        f"với {total_faces} lượt khuôn mặt được phân tích "
        f"(trung bình {avg_faces} học sinh/khung hình). "
        f'Kết quả đánh giá đạt mức "{level_vi}" với chỉ số tập trung {final_score}%. '
        f"Cảm xúc chủ đạo trong buổi học là {dom_desc}. "
        f"{sentiment_comment}"
    )

    # ── 4. Khuyến Nghị Sư Phạm (Pedagogical Recommendations) ──
    recommendations = RECOMMENDATIONS.get(level_vi, RECOMMENDATIONS["Trung bình"])

    # ── 5. Tổng Hợp Kết Quả ──
    return {
        "engagement_level": level_vi,
        "engagement_level_en": engagement_info["level_en"],
        "engagement_color": engagement_info["color"],
        "engagement_description": engagement_info["description"],
        "dominant_emotions": dominant_emotions,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
        "overall_comment": overall_comment,
        "recommendations": recommendations,
    }


class SessionEmotionTracker:
    """
    Module tích lũy cảm xúc và đánh giá độ tập trung theo buổi học (Session Emotion Tracker).
    Hỗ trợ:
    - Cộng dồn xác suất cảm xúc theo thời gian real-time.
    - Lưu snapshot phân phối cảm xúc theo từng phút.
    - Tự động bắt đầu khi có khuôn mặt và tự động kết thúc khi không nhận được khuôn mặt 2-3 lần liên tiếp.
    - Xử lý mượt mà edge-case N=0 (chia cho 0).
    - Tính điểm số cuối buổi học theo công thức trọng số.
    - Sinh đánh giá văn bản khoa học cuối buổi (Qualitative Assessment).
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "emotion_config.json")

        self.config_path = config_path
        self.weights = DEFAULT_WEIGHTS.copy()
        self.auto_end_empty_limit = 3
        self.sampling_interval_seconds = 60
        self.min_faces_threshold = 2
        self.load_config()

        self.reset()

    def load_config(self):
        """Đọc file cấu hình trọng số riêng để không hard-code trong logic."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "weights" in cfg:
                        for k, v in cfg["weights"].items():
                            self.weights[k.lower()] = float(v)
                    if "auto_end_empty_limit" in cfg:
                        self.auto_end_empty_limit = int(cfg["auto_end_empty_limit"])
                    if "sampling_interval_seconds" in cfg:
                        self.sampling_interval_seconds = int(
                            cfg["sampling_interval_seconds"]
                        )
                    if "min_faces_threshold" in cfg:
                        self.min_faces_threshold = int(cfg["min_faces_threshold"])
            except Exception as e:
                print(
                    f"[SessionTracker] Warning loading config '{self.config_path}': {e}. Using default weights."
                )

    def reset(self):
        """Xóa toàn bộ dữ liệu để chuẩn bị cho buổi học mới."""
        self.emotion_sum: Dict[str, float] = {k: 0.0 for k in EMOTION_KEYS}
        self.total_faces: int = 0  # Tổng số lượt mẫu quan sát tích lũy
        self.latest_face_count: int = 0  # Sĩ số học sinh thực tế trong frame hiện tại
        self.total_frames_analyzed: int = 0  # Số khung hình đã quét thành công
        self.history: List[Dict[str, Any]] = []
        self.is_active: bool = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.empty_consecutive_count: int = 0
        self.last_snapshot_time: float = 0.0
        self.final_summary: Optional[Dict[str, Any]] = None
        self.negative_evidence_records: List[Dict[str, Any]] = []

    def record_negative_evidence(self, evidence_data: Dict[str, Any]):
        """Ghi nhận bản ghi bằng chứng chụp toàn cảnh khi học sinh có cảm xúc tiêu cực kéo dài >= 6 chu kỳ."""
        self.negative_evidence_records.append(evidence_data)
        print(
            f"📸 [SessionTracker Evidence] Đã ghi nhận chứng cứ học sinh #{evidence_data.get('track_id')} | "
            f"Cảm xúc: {evidence_data.get('emotion')} | Chu kỳ: {evidence_data.get('consecutive_cycles')} | "
            f"File: {evidence_data.get('image_filename')}"
        )

    def start_session(self) -> Dict[str, Any]:
        """Bắt đầu buổi học mới."""
        self.reset()
        self.is_active = True
        self.start_time = time.time()
        self.last_snapshot_time = self.start_time
        return {
            "status": "started",
            "message": "Đã tự động bắt đầu buổi học mới",
            "start_time": self.start_time,
        }

    def update(self, probs_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Nhận vào danh sách vector/dict xác suất của từng khuôn mặt detect được trong phút/frame hiện tại.
        - Tự động bắt đầu CHỈ KHI có hơn 5-10 người (min_faces_threshold).
        - Tự động kết thúc nếu số người giảm dưới ngưỡng N lần liên tiếp.
        """
        num_faces = len(probs_list)
        self.latest_face_count = num_faces
        self.total_frames_analyzed += 1

        print(
            f"\n🔍 [SessionTracker Sync] Frame #{self.total_frames_analyzed} | Sĩ số: {num_faces} người | Trạng thái: {'ĐANG HOẠT ĐỘNG' if self.is_active else 'CHƯA KÍCH HOẠT'}"
        )

        # Nếu chưa bắt đầu buổi học
        if not self.is_active:
            if num_faces >= self.min_faces_threshold:
                self.start_session()
                self.latest_face_count = num_faces
                self.total_frames_analyzed = 1
                print(
                    f"🚀 [SessionTracker Auto-Start] Đã phát hiện {num_faces} người (>= ngưỡng {self.min_faces_threshold}) -> TỰ ĐỘNG BẮT ĐẦU BUỔI HỌC!"
                )
            else:
                print(
                    f"⏸️ [SessionTracker Debug] Buổi học CHƯA KÍCH HOẠT. Đang chờ học sinh... ({num_faces}/{self.min_faces_threshold} người)"
                )
                return {
                    "session_active": False,
                    "waiting_for_students": True,
                    "detected_faces": num_faces,
                    "min_faces_threshold": self.min_faces_threshold,
                    "message": f"Chưa đủ số người để tự động bắt đầu buổi học ({num_faces}/{self.min_faces_threshold} người)",
                }

        # Nếu buổi học đang diễn ra mà số khuôn mặt < min_faces_threshold
        if num_faces < self.min_faces_threshold:
            self.empty_consecutive_count += 1
            print(
                f"⚠️ [SessionTracker Debug] SĨ SỐ LỚP RỖNG ({num_faces}/{self.min_faces_threshold}) - Lần rỗng {self.empty_consecutive_count}/{self.auto_end_empty_limit}"
            )
            if self.empty_consecutive_count >= self.auto_end_empty_limit:
                summary = self.end_session(reason="auto_insufficient_faces")
                print(
                    f"🛑 [SessionTracker Auto-End] Sĩ số rỗng {self.empty_consecutive_count} lần liên tiếp (>= giới hạn {self.auto_end_empty_limit}) -> TỰ ĐỘNG CHỐT & XUẤT BÁO CÁO BUỔI HỌC!"
                )
                return {
                    "session_active": False,
                    "auto_ended": True,
                    "message": f"Tự động kết thúc buổi học do không đủ số lượng người ({num_faces}/{self.min_faces_threshold}) {self.empty_consecutive_count} lần liên tiếp.",
                    "final_summary": summary,
                }
            return {
                "session_active": self.is_active,
                "empty_consecutive_count": self.empty_consecutive_count,
                "detected_faces": num_faces,
                "min_faces_threshold": self.min_faces_threshold,
                "message": f"Cảnh báo: Số người ít hơn ngưỡng ({num_faces}/{self.min_faces_threshold})",
            }

        # Reset bộ đếm khi có đủ số khuôn mặt hợp lệ (>= min_faces_threshold)
        if self.empty_consecutive_count > 0:
            print(
                f"✅ [SessionTracker Debug] Đã phát hiện lại {num_faces} người -> Reset bộ đếm rỗng về 0 (Bộ đếm cũ: {self.empty_consecutive_count})"
            )
        self.empty_consecutive_count = 0

        # Tích lũy xác suất từng cảm xúc cho tất cả khuôn mặt trong frame/phút này
        for face_probs in probs_list:
            for emotion_key in EMOTION_KEYS:
                # Tìm xác suất tương ứng (case-insensitive)
                val = 0.0
                for k, v in face_probs.items():
                    if k.lower() == emotion_key:
                        val = float(v)
                        break
                self.emotion_sum[emotion_key] += val

        self.total_faces += num_faces

        # Tự động snapshot nếu đủ khoảng thời gian (15s/30s) kể từ snapshot trước HOẶC là frame đầu tiên
        now = time.time()
        if len(self.history) == 0 or (
            now - self.last_snapshot_time >= self.sampling_interval_seconds
        ):
            self.snapshot()

        return {
            "session_active": self.is_active,
            "total_faces": self.total_faces,
            "added_faces": num_faces,
            "current_percents": self.get_current_percent(),
        }

    def get_current_percent(self) -> Dict[str, float]:
        """
        Trả về % cảm xúc tích lũy hiện tại (real-time).
        Xử lý edge case: Nếu total_faces == 0, trả về 0.0% cho tất cả cảm xúc thay vì crash (chia cho 0).
        """
        if self.total_faces == 0:
            return {k: 0.0 for k in EMOTION_KEYS}

        res = {}
        for k in EMOTION_KEYS:
            prob_avg = self.emotion_sum[k] / self.total_faces
            res[k] = round(prob_avg * 100.0, 2)
        return res

    def compute_score(
        self, weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Tính điểm cuối buổi theo công thức Σ(wᵢ × Pᵢ) / Σwᵢ hoặc normalized 0-100.
        Pᵢ là tỷ lệ/xác suất trung bình của cảm xúc i (trong khoảng [0, 1]).
        """
        if self.total_faces == 0:
            return {
                "score": 0.0,
                "raw_score": 0.0,
                "sum_w_p": 0.0,
                "sum_w": 0.0,
                "total_faces": self.total_faces,
                "emotions_percent": {k: 0.0 for k in EMOTION_KEYS},
            }

        w_dict = weights if weights is not None else self.weights

        percents = self.get_current_percent()
        # Chuyển % về xác suất Pᵢ in [0, 1]
        probs = {k: percents.get(k, 0.0) / 100.0 for k in EMOTION_KEYS}

        # 1. Tính theo công thức chuẩn Σ(wᵢ × Pᵢ) / Σwᵢ
        sum_w_p = sum(w_dict.get(k, 0.0) * probs[k] for k in EMOTION_KEYS)
        sum_w = sum(w_dict.get(k, 0.0) for k in EMOTION_KEYS)

        raw_score = sum_w_p / sum_w if sum_w != 0 else 0.0

        # 2. Quy đổi sang thang điểm trực quan 0 - 100 (Score Normalized)
        score_normalized = calculate_concentration_score(probs, w_dict)

        return {
            "score": score_normalized,
            "raw_score": round(raw_score, 4),
            "sum_w_p": round(sum_w_p, 4),
            "sum_w": round(sum_w, 4),
            "total_faces": self.total_faces,
            "emotions_percent": percents,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Lưu 1 điểm dữ liệu snapshot lịch sử theo từng phút."""
        now = time.time()
        start = self.start_time if self.start_time else now
        elapsed_sec = max(0, int(now - start))
        minute_index = len(self.history) + 1

        curr_percents = self.get_current_percent()
        curr_score_data = self.compute_score()

        snap_data = {
            "minute": minute_index,
            "elapsed_seconds": elapsed_sec,
            "timestamp": time.strftime("%H:%M:%S", time.localtime(now)),
            "total_faces": self.total_faces,
            "emotions_percent": curr_percents,
            "concentration_score": curr_score_data["score"],
        }

        self.history.append(snap_data)
        self.last_snapshot_time = now
        return snap_data

    def end_session(self, reason: str = "user_command") -> Dict[str, Any]:
        """Kết thúc buổi học, tính toán điểm số cuối cùng, sinh đánh giá văn bản và tạo báo cáo tổng kết."""
        if not self.is_active and self.final_summary is not None:
            return self.final_summary

        self.end_time = time.time()
        start_ts = self.start_time if self.start_time else self.end_time
        duration_sec = int(self.end_time - start_ts)

        # Tính toán score TRƯỚC KHI deactivate session
        final_score_info = self.compute_score()

        self.is_active = False

        # Thêm snapshot cuối cùng nếu chưa có snapshot nào HOẶC đã qua >= 30s kể từ snapshot gần nhất
        if (
            len(self.history) == 0 or (self.end_time - self.last_snapshot_time >= 30)
        ) and self.total_faces > 0:
            self.snapshot()

        start_time_fmt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
        end_time_fmt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time))
        avg_faces = (
            round(self.total_faces / max(1, self.total_frames_analyzed), 2)
            if self.total_frames_analyzed > 0
            else 0.0
        )

        self.final_summary = {
            "status": "ended",
            "end_reason": reason,
            "start_time_formatted": start_time_fmt,
            "end_time_formatted": end_time_fmt,
            "duration_seconds": duration_sec,
            "duration_formatted": f"{duration_sec // 60}m {duration_sec % 60}s",
            "total_faces": self.total_faces,
            "total_frames_analyzed": self.total_frames_analyzed,
            "average_faces_per_frame": avg_faces,
            "final_score": final_score_info["score"],
            "raw_score": final_score_info["raw_score"],
            "emotions_percent": final_score_info["emotions_percent"],
            "history": self.history,
            "weights_used": self.weights,
            "negative_evidence_records": list(self.negative_evidence_records),
        }

        # ★ Sinh đánh giá văn bản khoa học (Qualitative Assessment)
        self.final_summary["qualitative_assessment"] = generate_qualitative_assessment(
            self.final_summary
        )

        return self.final_summary

    def export_excel(self, output_path: Optional[str] = None) -> str:
        """
        Xuất file Excel (.xlsx) chứa dữ liệu thống kê trực quan, dễ hiểu về buổi học.
        Bao gồm 3 sheet:
        1. Báo Cáo Tổng Quan: Thống kê thông tin buổi học, điểm tập trung và phân bố 7 cảm xúc.
        2. Nhận Xét & Gợi Ý: Đánh giá định tính, nhận xét buổi học và khuyến nghị sư phạm (đã loại bỏ tài liệu tham khảo).
        3. Diễn Biến Theo Phút: Biến thiên độ tập trung và cảm xúc xuyên suốt buổi học theo từng phút.
        """
        import pandas as pd

        if self.final_summary is None:
            self.end_session(reason="export_trigger")

        summary = self.final_summary
        qa = summary.get("qualitative_assessment", {})

        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        if output_path is None:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                reports_dir, f"Bao_Cao_Buoi_Hoc_{timestamp_str}.xlsx"
            )

        em_pct = summary.get("emotions_percent", {})

        # ── 1. Sheet Báo Cáo Tổng Quan ──
        overview_data = [
            # Nhóm 1: Thông tin chung
            {
                "Hạng mục": "THÔNG TIN BUỔI HỌC",
                "Chỉ số thống kê": "Thời gian bắt đầu",
                "Giá trị": summary.get("start_time_formatted", "N/A"),
            },
            {
                "Hạng mục": "THÔNG TIN BUỔI HỌC",
                "Chỉ số thống kê": "Thời gian kết thúc",
                "Giá trị": summary.get("end_time_formatted", "N/A"),
            },
            {
                "Hạng mục": "THÔNG TIN BUỔI HỌC",
                "Chỉ số thống kê": "Thời lượng buổi học",
                "Giá trị": summary.get("duration_formatted", "0m 0s"),
            },
            {
                "Hạng mục": "THÔNG TIN BUỔI HỌC",
                "Chỉ số thống kê": "Sĩ số học sinh trung bình",
                "Giá trị": f"{summary.get('average_faces_per_frame', 0)} học sinh / khung hình",
            },
            {
                "Hạng mục": "THÔNG TIN BUỔI HỌC",
                "Chỉ số thống kê": "Tổng lượt học sinh quan sát",
                "Giá trị": f"{summary.get('total_faces', 0)} lượt khuôn mặt",
            },
            {
                "Hạng mục": "THÔNG TIN BUỔI HỌC",
                "Chỉ số thống kê": "Tổng số khung hình ghi nhận",
                "Giá trị": f"{summary.get('total_frames_analyzed', 0)} frames",
            },
            {
                "Hạng mục": "THÔNG TIN BUỔI HỌC",
                "Chỉ số thống kê": "Trạng thái kết thúc",
                "Giá trị": (
                    "Tự động chốt khi rời lớp"
                    if summary.get("end_reason") == "auto_insufficient_faces"
                    else "Hoàn thành buổi học"
                ),
            },
            # Nhóm 2: Đánh giá tập trung
            {
                "Hạng mục": "KẾT QUẢ ĐỘ TẬP TRUNG",
                "Chỉ số thống kê": "Chỉ số độ tập trung trung bình",
                "Giá trị": f"{summary.get('final_score', 0)}%",
            },
            {
                "Hạng mục": "KẾT QUẢ ĐỘ TẬP TRUNG",
                "Chỉ số thống kê": "Xếp loại buổi học",
                "Giá trị": f"{qa.get('engagement_level', 'N/A')}",
            },
            {
                "Hạng mục": "KẾT QUẢ ĐỘ TẬP TRUNG",
                "Chỉ số thống kê": "Mô tả xếp loại",
                "Giá trị": qa.get("engagement_description", ""),
            },
            {
                "Hạng mục": "KẾT QUẢ ĐỘ TẬP TRUNG",
                "Chỉ số thống kê": "Tỷ lệ cảm xúc tích cực (thúc đẩy tập trung)",
                "Giá trị": f"{qa.get('positive_ratio', 0)}%",
            },
            {
                "Hạng mục": "KẾT QUẢ ĐỘ TẬP TRUNG",
                "Chỉ số thống kê": "Tỷ lệ cảm xúc tiêu cực (gây phân tâm)",
                "Giá trị": f"{qa.get('negative_ratio', 0)}%",
            },
            # Nhóm 3: Chi tiết 7 cảm xúc
            {
                "Hạng mục": "CHI TIẾT 7 CẢM XÚC (%)",
                "Chỉ số thống kê": "Bình thường / Tập trung nghe giảng (Neutral)",
                "Giá trị": f"{em_pct.get('neutral', 0)}%",
            },
            {
                "Hạng mục": "CHI TIẾT 7 CẢM XÚC (%)",
                "Chỉ số thống kê": "Hào hứng / Vui vẻ (Happy)",
                "Giá trị": f"{em_pct.get('happy', 0)}%",
            },
            {
                "Hạng mục": "CHI TIẾT 7 CẢM XÚC (%)",
                "Chỉ số thống kê": "Tò mò / Chú ý bất ngờ (Surprise)",
                "Giá trị": f"{em_pct.get('surprise', 0)}%",
            },
            {
                "Hạng mục": "CHI TIẾT 7 CẢM XÚC (%)",
                "Chỉ số thống kê": "Buồn bã / Khó hiểu bài (Sad)",
                "Giá trị": f"{em_pct.get('sad', 0)}%",
            },
            {
                "Hạng mục": "CHI TIẾT 7 CẢM XÚC (%)",
                "Chỉ số thống kê": "Lo lắng / Áp lực căng thẳng (Fear)",
                "Giá trị": f"{em_pct.get('fear', 0)}%",
            },
            {
                "Hạng mục": "CHI TIẾT 7 CẢM XÚC (%)",
                "Chỉ số thống kê": "Bực bội / Khó chịu (Angry)",
                "Giá trị": f"{em_pct.get('angry', 0)}%",
            },
            {
                "Hạng mục": "CHI TIẾT 7 CẢM XÚC (%)",
                "Chỉ số thống kê": "Chán nản / Mất tập trung (Disgust)",
                "Giá trị": f"{em_pct.get('disgust', 0)}%",
            },
        ]
        df_overview = pd.DataFrame(overview_data)

        # ── 2. Sheet Nhận Xét & Gợi Ý Sư Phạm (Đã loại bỏ tài liệu tham khảo) ──
        assessment_data = [
            {
                "Nội dung đánh giá": "Xếp Loại Mức Độ Tập Trung",
                "Chi tiết nhận xét & khuyến nghị": f"{qa.get('engagement_level', 'N/A')} ({qa.get('engagement_level_en', '')})",
            },
            {
                "Nội dung đánh giá": "Mô Tả Xếp Loại",
                "Chi tiết nhận xét & khuyến nghị": qa.get("engagement_description", ""),
            },
            {
                "Nội dung đánh giá": "Tỷ Lệ Cảm Xúc Tích Cực",
                "Chi tiết nhận xét & khuyến nghị": f"{qa.get('positive_ratio', 0)}% (Học sinh tập trung, hào hứng, tò mò)",
            },
            {
                "Nội dung đánh giá": "Tỷ Lệ Cảm Xúc Tiêu Cực",
                "Chi tiết nhận xét & khuyến nghị": f"{qa.get('negative_ratio', 0)}% (Học sinh lo lắng, bực bội, buồn chán)",
            },
        ]

        # Thêm cảm xúc chủ đạo
        for i, dom in enumerate(qa.get("dominant_emotions", []), 1):
            assessment_data.append(
                {
                    "Nội dung đánh giá": f"Cảm Xúc Chủ Đạo #{i}",
                    "Chi tiết nhận xét & khuyến nghị": f"{dom.get('label_vi', '')} — chiếm {dom.get('percent', 0)}% thời lượng",
                }
            )

        # Nhận xét tổng quan buổi học
        assessment_data.append(
            {
                "Nội dung đánh giá": "NHẬN XÉT TỔNG QUAN",
                "Chi tiết nhận xét & khuyến nghị": qa.get("overall_comment", ""),
            }
        )

        # Khuyến nghị sư phạm cho giáo viên
        for i, rec in enumerate(qa.get("recommendations", []), 1):
            assessment_data.append(
                {
                    "Nội dung đánh giá": f"Gợi Ý Sư Phạm #{i}",
                    "Chi tiết nhận xét & khuyến nghị": rec,
                }
            )

        df_assessment = pd.DataFrame(assessment_data)

        # ── 3. Sheet Diễn Biến Theo Phút ──
        history_rows = []
        for item in summary.get("history", []):
            em = item.get("emotions_percent", {})
            history_rows.append(
                {
                    "Phút thứ": item.get("minute", 0),
                    "Thời điểm": item.get("timestamp", ""),
                    "Độ tập trung (%)": item.get("concentration_score", 0),
                    "Lượt mặt ghi nhận": item.get("total_faces", 0),
                    "Tập trung - Neutral (%)": em.get("neutral", 0),
                    "Hào hứng - Happy (%)": em.get("happy", 0),
                    "Tò mò - Surprise (%)": em.get("surprise", 0),
                    "Buồn - Sad (%)": em.get("sad", 0),
                    "Lo lắng - Fear (%)": em.get("fear", 0),
                    "Bực bội - Angry (%)": em.get("angry", 0),
                    "Chán nản - Disgust (%)": em.get("disgust", 0),
                }
            )
        df_history = pd.DataFrame(history_rows)

        # ── 4. Sheet Giám Sát Cảm Xúc Tiêu Cực ──
        evidence_records = summary.get("negative_evidence_records", [])
        evidence_rows = []
        if len(evidence_records) > 0:
            for idx, ev in enumerate(evidence_records, 1):
                evidence_rows.append(
                    {
                        "STT": idx,
                        "Mã học sinh": ev.get(
                            "student_label", f"Học sinh #{ev.get('track_id')}"
                        ),
                        "Thời điểm ghi nhận": ev.get("timestamp", ""),
                        "Thời lượng buổi học": ev.get("elapsed_formatted", ""),
                        "Cảm xúc tiêu cực": ev.get("emotion_vi", ev.get("emotion", "")),
                        "Độ tin cậy": f"{ev.get('confidence', 0)}%",
                        "Số chu kỳ liên tiếp": ev.get("consecutive_cycles", 6),
                        "Trạng thái giao tiếp": (
                            "Có cử động môi / Nói"
                            if ev.get("is_speaking")
                            else "Im lặng"
                        ),
                        "Gợi ý đối soát ngoại cảnh": ev.get(
                            "context_note",
                            "Kiểm tra bối cảnh toàn cảnh xung quanh: bạn bên cạnh, thiết bị riêng hoặc nội dung bài giảng.",
                        ),
                        "Tên tệp bằng chứng": ev.get("image_filename", ""),
                        "Ảnh chụp toàn cảnh hiện trường": "",  # Sẽ được chèn trực tiếp bằng Openpyxl Image
                    }
                )
        else:
            evidence_rows.append(
                {
                    "STT": 1,
                    "Mã học sinh": "Toàn bộ lớp học",
                    "Thời điểm ghi nhận": summary.get("end_time_formatted", ""),
                    "Thời lượng buổi học": summary.get("duration_formatted", ""),
                    "Cảm xúc tiêu cực": "Không xuất hiện tiêu cực kéo dài",
                    "Độ tin cậy": "100%",
                    "Số chu kỳ liên tiếp": "0",
                    "Trạng thái giao tiếp": "Bình thường",
                    "Gợi ý đối soát ngoại cảnh": "Không phát hiện học sinh nào duy trì cảm xúc tiêu cực từ 6 chu kỳ trở lên. Lớp học duy trì trạng thái tâm lý ổn định, tập trung và tích cực.",
                    "Tên tệp bằng chứng": "Không có",
                    "Ảnh chụp toàn cảnh hiện trường": "Không có vi phạm",
                }
            )
        df_negative_evidence = pd.DataFrame(evidence_rows)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_overview.to_excel(writer, sheet_name="Báo Cáo Tổng Quan", index=False)
            df_assessment.to_excel(writer, sheet_name="Nhận Xét & Gợi Ý", index=False)
            df_history.to_excel(writer, sheet_name="Diễn Biến Theo Phút", index=False)
            df_negative_evidence.to_excel(
                writer, sheet_name="Giám Sát Cảm Xúc Tiêu Cực", index=False
            )

            from openpyxl.drawing.image import Image as OpenpyxlImage
            from openpyxl.styles import Alignment

            # Tự động căn chỉnh độ rộng cột cho đẹp mắt, không bị che chữ
            for sheet_name in writer.sheets:
                ws = writer.sheets[sheet_name]
                for col in ws.columns:
                    col_letter = col[0].column_letter
                    max_len = max(len(str(cell.value or "")) for cell in col)
                    ws.column_dimensions[col_letter].width = min(
                        max(max_len + 4, 18), 70
                    )

            # Tinh chỉnh chuyên biệt cho sheet 'Giám Sát Cảm Xúc Tiêu Cực'
            ws_neg = writer.sheets["Giám Sát Cảm Xúc Tiêu Cực"]
            ws_neg.column_dimensions["K"].width = (
                34  # Độ rộng vừa vặn cho ảnh thumbnail
            )

            if len(evidence_records) > 0:
                for row_idx, ev in enumerate(evidence_records, start=2):
                    # Thiết lập chiều cao hàng đủ hiển thị ảnh (~92pt)
                    ws_neg.row_dimensions[row_idx].height = 94

                    # Căn giữa nội dung các ô văn bản
                    for col_idx in range(1, 12):
                        cell = ws_neg.cell(row=row_idx, column=col_idx)
                        cell.alignment = Alignment(
                            vertical="center", horizontal="center", wrap_text=True
                        )

                    # Chèn hình ảnh toàn cảnh chứng cứ trực tiếp vào ô tính K{row_idx}
                    img_path = ev.get("image_path")
                    if img_path and os.path.exists(img_path):
                        try:
                            img_obj = OpenpyxlImage(img_path)
                            # Resize thumbnail hiển thị gọn gàng trong ô tính Excel (200x112px)
                            img_obj.width = 200
                            img_obj.height = 112
                            cell_coord = f"K{row_idx}"
                            ws_neg.add_image(img_obj, cell_coord)
                        except Exception as img_err:
                            print(
                                f"[SessionTracker] Lỗi chèn ảnh Excel '{img_path}': {img_err}"
                            )
            else:
                ws_neg.row_dimensions[2].height = 45
                for col_idx in range(1, 12):
                    cell = ws_neg.cell(row=2, column=col_idx)
                    cell.alignment = Alignment(
                        vertical="center", horizontal="center", wrap_text=True
                    )

        return output_path
