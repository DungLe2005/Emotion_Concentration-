import time
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NEGATIVE_EMOTIONS = ["sad", "fear", "angry", "disgust"]
NEGATIVE_CYCLE_THRESHOLD = 6


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Tính Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2].
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_centroid_distance(box1: List[int], box2: List[int]) -> float:
    """
    Tính khoảng cách Euclidean giữa tâm của 2 bounding box.
    """
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return float(np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2))


def compute_embedding_similarity(
    emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]
) -> float:
    """
    Tính Cosine Similarity giữa 2 vector embedding khuôn mặt (CLIP 512D).
    """
    if emb1 is None or emb2 is None:
        return 0.0
    arr1 = np.array(emb1, dtype=np.float32).flatten()
    arr2 = np.array(emb2, dtype=np.float32).flatten()
    norm1 = float(np.linalg.norm(arr1))
    norm2 = float(np.linalg.norm(arr2))
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))


def apply_upper_face_gating(
    raw_probs: np.ndarray,
    landmarks: Optional[np.ndarray],
    idx_to_label: Dict[int, str],
) -> Tuple[np.ndarray, bool]:
    """
    Phương pháp 1: Upper-Face Gating (Nicolle et al., IEEE ACII 2013).
    Kiểm tra chéo nửa mặt trên (Mắt mở to AU5, Lông mày nhướng cao AU1+AU2)
    khi mô hình dự đoán xác suất có biểu hiện Ngạc nhiên (Surprise).

    Nếu chỉ mở miệng (AU25/AU26) mà Mắt và Lông mày ở trạng thái bình thường:
    -> Bác bỏ nhãn Surprise giả (do nói chuyện), chuyển xác suất về Neutral/Happy.

    Trả về: (adjusted_probs, is_speech_artifact_suppressed)
    """
    if landmarks is None or len(landmarks) < 20 or not np.any(landmarks != 0):
        return raw_probs, False

    # Tìm index của các cảm xúc quan trọng
    label_to_idx = {v.lower(): k for k, v in idx_to_label.items()}
    surprise_idx = label_to_idx.get("surprise")
    neutral_idx = label_to_idx.get("neutral")
    happy_idx = label_to_idx.get("happy")

    if surprise_idx is None:
        return raw_probs, False

    surprise_prob = float(raw_probs[surprise_idx])
    # Chỉ can thiệp nếu Surprise có xác suất đáng kể (> 0.20)
    if surprise_prob < 0.20:
        return raw_probs, False

    # Trích xuất các Action Unit proxies từ landmark vector
    mar = float(landmarks[0])  # Mouth Aspect Ratio (độ mở miệng)
    avg_ear = float(landmarks[13])  # Eye Aspect Ratio (độ mở mắt - AU5 proxy)
    left_brow = float(landmarks[11])  # Khoảng cách mắt - mày trái (AU1/AU2 proxy)
    right_brow = float(landmarks[12])  # Khoảng cách mắt - mày phải (AU1/AU2 proxy)
    avg_brow = (left_brow + right_brow) / 2.0

    # Ngưỡng sinh trắc học chuẩn FACS:
    # Trong ngạc nhiên thật sự, mắt mở to (EAR > 0.32) hoặc lông mày nhướng cong vút (Brow > 0.16)
    is_eye_widened = avg_ear > 0.32
    is_brow_raised = avg_brow > 0.16

    # Nếu miệng mở đáng kể (MAR > 0.18) NHƯNG mắt và lông mày không có biểu hiện ngạc nhiên
    if mar > 0.18 and not (is_eye_widened or is_brow_raised):
        adj_p = raw_probs.copy()

        # Triệt tiêu phần lớn xác suất Surprise giả
        suppressed_mass = adj_p[surprise_idx] * 0.85
        adj_p[surprise_idx] -= suppressed_mass

        # Bù lại cho Neutral (hoặc Happy nếu khóe miệng hơi nâng)
        if neutral_idx is not None:
            adj_p[neutral_idx] += suppressed_mass * 0.85
        if happy_idx is not None:
            adj_p[happy_idx] += suppressed_mass * 0.15

        # Chuẩn hóa tổng xác suất về 1.0
        norm_sum = np.sum(adj_p)
        if norm_sum > 0:
            adj_p = adj_p / norm_sum

        return adj_p, True

    return raw_probs, False


class FaceTrack:
    """
    Đại diện cho một khuôn mặt được theo dõi liên tục qua chuỗi thời gian (Video Stream).
    Bao gồm:
    1. Quản lý vị trí không gian (Bounding Box, Centroid, Track Age, Missed Frames)
    2. Nhận diện định danh Re-ID bằng 512D CLIP Embedding
    3. Bộ lọc làm mượt xác suất (EMA + Sliding Window Buffer)
    4. Phương pháp 1: Upper-Face Gating (Nicolle et al., IEEE 2013)
    5. Phương pháp 2: Lip Dynamics & Speaking Detection (Phát hiện tần số đóng mở môi)
    6. Máy trạng thái Ngưỡng trễ thời gian (Temporal Hysteresis Thresholding)
    """

    def __init__(
        self,
        track_id: int,
        box: List[int],
        initial_probs: np.ndarray,
        idx_to_label: Dict[int, str],
        landmarks: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        alpha: float = 0.35,
        window_size: int = 7,
        min_consecutive_to_switch: int = 3,
        instant_switch_threshold: float = 0.65,
        hysteresis_margin: float = 0.08,
        timestamp: Optional[float] = None,
    ):
        self.track_id = track_id
        self.box = box
        self.centroid = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.idx_to_label = idx_to_label
        self.num_classes = len(initial_probs)

        # 512D Feature Embedding Re-ID
        self.embedding: Optional[np.ndarray] = None
        if embedding is not None:
            arr = np.array(embedding, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(arr))
            if norm > 1e-6:
                self.embedding = arr / norm

        # Cấu hình Smoothing & Hysteresis
        self.alpha = alpha
        self.window_size = window_size
        self.min_consecutive_to_switch = min_consecutive_to_switch
        self.instant_switch_threshold = instant_switch_threshold
        self.hysteresis_margin = hysteresis_margin

        # Tracking state
        self.age = 1
        self.missed_frames = 0
        self.last_seen_time = timestamp or time.time()
        self.is_speaking = False

        # Lip Dynamics history (MAR buffer để nhận diện trạng thái nói chuyện)
        self.mar_history = deque(maxlen=self.window_size)
        if landmarks is not None and len(landmarks) > 0 and landmarks[0] > 0:
            self.mar_history.append(float(landmarks[0]))
        else:
            self.mar_history.append(0.0)

        # Áp dụng Upper-Face Gating cho frame đầu tiên
        gated_init_p, _ = apply_upper_face_gating(
            initial_probs, landmarks, self.idx_to_label
        )
        init_p = np.array(gated_init_p, dtype=np.float32)
        if np.sum(init_p) > 0:
            init_p = init_p / np.sum(init_p)

        self.raw_history = deque(maxlen=self.window_size)
        self.raw_history.append(init_p)
        self.smoothed_probs = init_p.copy()
        self.effective_probs = init_p.copy()

        # Hysteresis State Machine
        initial_top_idx = int(np.argmax(init_p))
        self.current_emotion_idx = initial_top_idx
        self.candidate_emotion_idx = initial_top_idx
        self.candidate_consecutive_frames = 1

        # Giám sát cảm xúc tiêu cực liên tiếp (Negative Emotion Tracking)
        init_emo_label = self.idx_to_label.get(
            self.current_emotion_idx, "Neutral"
        ).lower()
        self.consecutive_negative_cycles = (
            1 if init_emo_label in NEGATIVE_EMOTIONS else 0
        )
        self.is_negative_flagged = False
        self.last_evidence_capture_time = 0.0

    def update(
        self,
        raw_probs: np.ndarray,
        box: List[int],
        landmarks: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, str, float, bool]:
        """
        Cập nhật đặc trưng khuôn mặt với frame mới:
        1. Cập nhật Bounding Box, Centroid và Feature Embedding EMA.
        2. Áp dụng Upper-Face Gating (Nicolle et al., IEEE 2013).
        3. Phân tích Lip Dynamics để phát hiện trạng thái nói chuyện (Speaking Detection).
        4. Làm mượt xác suất qua EMA & Sliding Window Soft Voting.
        5. Kiểm tra ngưỡng trễ Hysteresis để quyết định chuyển trạng thái cảm xúc.

        Trả về: (effective_probs, stable_emotion_str, stable_confidence, is_speaking)
        """
        self.box = box
        self.centroid = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.missed_frames = 0
        self.age += 1
        self.last_seen_time = timestamp or time.time()

        # Cập nhật Feature Embedding qua EMA (chống trôi nhận diện)
        if embedding is not None:
            arr = np.array(embedding, dtype=np.float32).flatten()
            norm = float(np.linalg.norm(arr))
            if norm > 1e-6:
                norm_emb = arr / norm
                if self.embedding is None:
                    self.embedding = norm_emb
                else:
                    self.embedding = 0.80 * self.embedding + 0.20 * norm_emb
                    self.embedding /= max(1e-6, float(np.linalg.norm(self.embedding)))

        # --- BƯỚC 1: UPPER-FACE GATING (Nicolle et al., IEEE 2013) ---
        gated_probs, _ = apply_upper_face_gating(
            raw_probs, landmarks, self.idx_to_label
        )
        curr_p = np.array(gated_probs, dtype=np.float32)
        if np.sum(curr_p) > 0:
            curr_p = curr_p / np.sum(curr_p)

        # --- BƯỚC 2: LIP DYNAMICS & SPEAKING DETECTION ---
        if landmarks is not None and len(landmarks) > 0:
            current_mar = float(landmarks[0])
            self.mar_history.append(current_mar)

            if len(self.mar_history) >= 4:
                mar_arr = np.array(self.mar_history)
                mar_std = float(np.std(mar_arr))
                mar_mean = float(np.mean(mar_arr))
                mar_range = float(np.max(mar_arr) - np.min(mar_arr))

                if mar_std > 0.032 and mar_range > 0.07:
                    self.is_speaking = True
                elif mar_std < 0.018 and (mar_mean < 0.12 or mar_mean > 0.35):
                    self.is_speaking = False
            else:
                self.is_speaking = False
        else:
            self.is_speaking = False

        # --- BƯỚC 3: EXPONENTIAL MOVING AVERAGE (EMA) ---
        self.smoothed_probs = (
            self.alpha * curr_p + (1.0 - self.alpha) * self.smoothed_probs
        )
        if np.sum(self.smoothed_probs) > 0:
            self.smoothed_probs = self.smoothed_probs / np.sum(self.smoothed_probs)

        # --- BƯỚC 4: SLIDING WINDOW BUFFER SOFT VOTING ---
        self.raw_history.append(curr_p)
        window_weights = np.linspace(0.6, 1.0, len(self.raw_history))
        window_weights = window_weights / np.sum(window_weights)

        weighted_window_sum = np.zeros(self.num_classes, dtype=np.float32)
        for w, hist_p in zip(window_weights, self.raw_history):
            weighted_window_sum += w * hist_p

        self.effective_probs = 0.60 * self.smoothed_probs + 0.40 * weighted_window_sum
        if np.sum(self.effective_probs) > 0:
            self.effective_probs = self.effective_probs / np.sum(self.effective_probs)

        # Khóa Surprise nếu đang nói chuyện
        if self.is_speaking:
            label_to_idx = {v.lower(): k for k, v in self.idx_to_label.items()}
            surprise_idx = label_to_idx.get("surprise")
            neutral_idx = label_to_idx.get("neutral")
            if surprise_idx is not None and self.effective_probs[surprise_idx] > 0.15:
                surp_val = self.effective_probs[surprise_idx] * 0.70
                self.effective_probs[surprise_idx] -= surp_val
                if neutral_idx is not None:
                    self.effective_probs[neutral_idx] += surp_val
                if np.sum(self.effective_probs) > 0:
                    self.effective_probs = self.effective_probs / np.sum(
                        self.effective_probs
                    )

        # --- BƯỚC 5: TEMPORAL HYSTERESIS THRESHOLDING ---
        raw_top_idx = int(np.argmax(curr_p))
        if raw_top_idx != self.current_emotion_idx:
            if raw_top_idx == self.candidate_emotion_idx:
                self.candidate_consecutive_frames += 1
            else:
                self.candidate_emotion_idx = raw_top_idx
                self.candidate_consecutive_frames = 1
        else:
            self.candidate_emotion_idx = self.current_emotion_idx
            self.candidate_consecutive_frames = 0

        curr_eff_conf = float(self.effective_probs[self.current_emotion_idx])
        cand_eff_conf = float(self.effective_probs[self.candidate_emotion_idx])

        if self.candidate_emotion_idx != self.current_emotion_idx:
            if cand_eff_conf > (curr_eff_conf + self.hysteresis_margin):
                self.current_emotion_idx = self.candidate_emotion_idx
                self.candidate_consecutive_frames = 0
            elif self.candidate_consecutive_frames >= self.min_consecutive_to_switch:
                self.current_emotion_idx = self.candidate_emotion_idx
                self.candidate_consecutive_frames = 0
            elif cand_eff_conf >= self.instant_switch_threshold:
                self.current_emotion_idx = self.candidate_emotion_idx
                self.candidate_consecutive_frames = 0

        stable_emotion_str = self.idx_to_label.get(self.current_emotion_idx, "Neutral")
        stable_confidence = float(self.effective_probs[self.current_emotion_idx])

        # --- BƯỚC 6: GIÁM SÁT CẢM XÚC TIÊU CỰC LIÊN TIẾP (NEGATIVE EMOTION TRACKING) ---
        is_negative = stable_emotion_str.lower() in NEGATIVE_EMOTIONS
        trigger_evidence = False

        if is_negative:
            self.consecutive_negative_cycles += 1
            # Khi cảm xúc tiêu cực xuất hiện liên tục trong >= 6 chu kỳ và chưa bị đánh dấu
            if (
                self.consecutive_negative_cycles >= NEGATIVE_CYCLE_THRESHOLD
                and not self.is_negative_flagged
            ):
                trigger_evidence = True
                self.is_negative_flagged = True
                self.last_evidence_capture_time = self.last_seen_time
        else:
            # Khi đối tượng thoát khỏi cảm xúc tiêu cực -> xóa đánh dấu
            self.consecutive_negative_cycles = 0
            self.is_negative_flagged = False

        return (
            self.effective_probs,
            stable_emotion_str,
            stable_confidence,
            self.is_speaking,
            self.consecutive_negative_cycles,
            self.is_negative_flagged,
            trigger_evidence,
        )


class TemporalFaceTracker:
    """
    Bộ theo dõi đa khuôn mặt (Multi-Face Temporal Tracker) trên chuỗi video realtime:
    - Gán và duy trì Track ID qua IoU, Khoảng cách không gian và Cosine Embedding Re-ID (CLIP).
    - Bảo toàn định danh duy nhất cho từng học sinh, tuyệt đối không nhân bản 1 người thành 2 ID.
    - Quản lý vòng đời FaceTrack (Khởi tạo, Cập nhật, Xóa khi rời khỏi khung hình).
    - Tích hợp Upper-Face Gating & Lip Dynamics Speaking Detection.
    - Cung cấp kết quả làm mượt xác suất & cảm xúc ổn định cho toàn bộ Pipeline.
    """

    def __init__(
        self,
        idx_to_label: Optional[Dict[int, str]] = None,
        iou_threshold: float = 0.15,
        max_missed_frames: int = 25,
        alpha: float = 0.35,
        window_size: int = 7,
        min_consecutive_to_switch: int = 3,
        instant_switch_threshold: float = 0.65,
        hysteresis_margin: float = 0.08,
    ):
        if idx_to_label is None:
            self.idx_to_label = {i: name for i, name in enumerate(EMOTION_KEYS)}
        else:
            self.idx_to_label = idx_to_label

        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.alpha = alpha
        self.window_size = window_size
        self.min_consecutive_to_switch = min_consecutive_to_switch
        self.instant_switch_threshold = instant_switch_threshold
        self.hysteresis_margin = hysteresis_margin

        self.tracks: Dict[int, FaceTrack] = {}
        self.next_track_id: int = 1

    def reset(self):
        """Reset toàn bộ trạng thái tracking (khi bắt đầu buổi học mới)."""
        self.tracks.clear()
        self.next_track_id = 1

    def process_frame_faces(
        self,
        detections: List[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Khớp các khuôn mặt detect trong frame hiện tại với các track đã có:
        - detections: Danh sách dict [{'box': [x1,y1,x2,y2], 'raw_probs': np.ndarray, 'landmarks': np.ndarray, 'embedding': np.ndarray, ...}, ...]
        - Trả về danh sách detections đã được bổ sung 'track_id', 'smoothed_probs', 'stable_emotion', 'stable_confidence', 'is_speaking'.
        """
        now = timestamp or time.time()
        num_dets = len(detections)
        active_track_ids = list(self.tracks.keys())
        num_tracks = len(active_track_ids)

        matched_tracks = set()
        matched_dets = set()
        det_to_track_map: Dict[int, int] = {}

        if num_dets > 0 and num_tracks > 0:
            # TRƯỜNG HỢP 1: Chỉ có 1 người trong khung hình (Bảo toàn tuyệt đối 1 Track ID)
            if num_dets == 1 and num_tracks >= 1:
                d_box = detections[0]["box"]
                d_emb = detections[0].get("embedding", None)
                best_t_id = None
                best_score = -1.0

                for t_id, tr in self.tracks.items():
                    iou_val = compute_iou(d_box, tr.box)
                    dist_val = compute_centroid_distance(d_box, tr.box)
                    emb_sim = compute_embedding_similarity(d_emb, tr.embedding)

                    # Điểm tương đồng đa phương thức
                    score = (
                        0.35 * iou_val
                        + 0.35 * max(0.0, emb_sim)
                        + 0.30 * max(0.0, 1.0 - dist_val / 400.0)
                    )
                    if score > best_score:
                        best_score = score
                        best_t_id = t_id

                if best_t_id is not None and (best_score >= 0.15 or num_tracks == 1):
                    det_to_track_map[0] = best_t_id
                    matched_dets.add(0)
                    matched_tracks.add(best_t_id)

            # TRƯỜNG HỢP 2: Nhiều người trong khung hình (Matching Ma trận tối ưu)
            else:
                score_matrix = np.zeros((num_dets, num_tracks), dtype=np.float32)
                for d_idx, det in enumerate(detections):
                    d_box = det["box"]
                    d_emb = det.get("embedding", None)
                    for t_col, t_id in enumerate(active_track_ids):
                        tr = self.tracks[t_id]
                        iou_val = compute_iou(d_box, tr.box)
                        dist_val = compute_centroid_distance(d_box, tr.box)
                        emb_sim = compute_embedding_similarity(d_emb, tr.embedding)
                        score = (
                            0.40 * iou_val
                            + 0.35 * max(0.0, emb_sim)
                            + 0.25 * max(0.0, 1.0 - dist_val / 350.0)
                        )
                        score_matrix[d_idx, t_col] = score

                # Greedy matching theo Score cao nhất
                while True:
                    max_score = np.max(score_matrix)
                    if max_score < 0.18:
                        break
                    d_idx, t_col = np.unravel_index(
                        np.argmax(score_matrix), score_matrix.shape
                    )
                    t_id = active_track_ids[t_col]

                    det_to_track_map[d_idx] = t_id
                    matched_dets.add(d_idx)
                    matched_tracks.add(t_id)

                    score_matrix[d_idx, :] = -1.0
                    score_matrix[:, t_col] = -1.0

                # Fallback: Khớp khoảng cách tâm và embedding cho detection chưa khớp
                unmatched_dets = [d for d in range(num_dets) if d not in matched_dets]
                unmatched_tracks = [
                    t for t in active_track_ids if t not in matched_tracks
                ]

                for d_idx in unmatched_dets:
                    d_box = detections[d_idx]["box"]
                    d_emb = detections[d_idx].get("embedding", None)
                    best_dist = float("inf")
                    best_t_id = None

                    for t_id in unmatched_tracks:
                        if t_id in matched_tracks:
                            continue
                        tr = self.tracks[t_id]
                        dist = compute_centroid_distance(d_box, tr.box)
                        emb_sim = compute_embedding_similarity(d_emb, tr.embedding)
                        if (dist < best_dist and dist <= 300.0) or emb_sim >= 0.50:
                            best_dist = dist
                            best_t_id = t_id

                    if best_t_id is not None:
                        det_to_track_map[d_idx] = best_t_id
                        matched_dets.add(d_idx)
                        matched_tracks.add(best_t_id)

        # 3. Cập nhật các track đã match hoặc tạo mới
        enhanced_detections = []
        for d_idx, det in enumerate(detections):
            raw_p = det["raw_probs"]
            box = det["box"]
            landmarks = det.get("landmarks", None)
            embedding = det.get("embedding", None)

            if d_idx in det_to_track_map:
                t_id = det_to_track_map[d_idx]
                track = self.tracks[t_id]
                (
                    eff_probs,
                    stable_emotion,
                    stable_conf,
                    is_speaking,
                    consecutive_neg_cycles,
                    is_neg_flagged,
                    trigger_evidence,
                ) = track.update(
                    raw_p, box, landmarks=landmarks, embedding=embedding, timestamp=now
                )
            else:
                # Khởi tạo FaceTrack mới
                t_id = self.next_track_id
                self.next_track_id += 1
                new_track = FaceTrack(
                    track_id=t_id,
                    box=box,
                    initial_probs=raw_p,
                    idx_to_label=self.idx_to_label,
                    landmarks=landmarks,
                    embedding=embedding,
                    alpha=self.alpha,
                    window_size=self.window_size,
                    min_consecutive_to_switch=self.min_consecutive_to_switch,
                    instant_switch_threshold=self.instant_switch_threshold,
                    hysteresis_margin=self.hysteresis_margin,
                    timestamp=now,
                )
                self.tracks[t_id] = new_track
                matched_tracks.add(t_id)
                eff_probs = new_track.effective_probs
                stable_emotion = self.idx_to_label.get(
                    new_track.current_emotion_idx, "Neutral"
                )
                stable_conf = float(eff_probs[new_track.current_emotion_idx])
                is_speaking = new_track.is_speaking
                consecutive_neg_cycles = new_track.consecutive_negative_cycles
                is_neg_flagged = new_track.is_negative_flagged
                trigger_evidence = False

            # Tính toán phân phối xác suất làm mượt
            probs_list = eff_probs.tolist()
            emotions_distribution = sorted(
                [
                    {
                        "label": self.idx_to_label[i].capitalize(),
                        "score": round(p * 100, 1),
                    }
                    for i, p in enumerate(probs_list)
                ],
                key=lambda x: x["score"],
                reverse=True,
            )

            # Sao chép và bổ sung thông tin tracking
            out_det = dict(det)
            out_det["track_id"] = t_id
            out_det["id"] = t_id
            out_det["smoothed_probs"] = eff_probs
            out_det["emotion"] = stable_emotion.capitalize()
            out_det["confidence"] = round(stable_conf * 100, 1)
            out_det["emotions"] = emotions_distribution
            out_det["is_speaking"] = is_speaking
            out_det["consecutive_negative_cycles"] = consecutive_neg_cycles
            out_det["is_negative_flagged"] = is_neg_flagged
            out_det["trigger_negative_evidence"] = trigger_evidence

            enhanced_detections.append(out_det)

        # 4. Tăng missed_frames cho các track không xuất hiện trong frame hiện tại
        tracks_to_delete = []
        for t_id, track in list(self.tracks.items()):
            if t_id not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    tracks_to_delete.append(t_id)

        # Xóa các track đã quá hạn
        for t_id in tracks_to_delete:
            if t_id in self.tracks:
                del self.tracks[t_id]

        return enhanced_detections
