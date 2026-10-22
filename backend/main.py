import os
import json
import time
import torch
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pipeline import EmotionPipeline

app = FastAPI(title="Emotion & Concentration Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_instance: Optional[EmotionPipeline] = None
SESSIONS_DB_FILE = os.path.join(os.path.dirname(__file__), "sessions_history.json")


def get_pipeline() -> EmotionPipeline:
    """Lazy Singleton Pattern: Nạp mô hình AI vào GPU/RAM ở lần gọi đầu tiên."""
    global pipeline_instance
    if pipeline_instance is None:
        print("🚀 [Lazy Singleton] Đang khởi tạo AI Pipeline vào GPU/RAM...")
        pipeline_instance = EmotionPipeline()
        print("✅ [Lazy Singleton] AI Pipeline đã khởi tạo và sẵn sàng nhận request!")
    return pipeline_instance


def save_session_record(session_data: Dict[str, Any]):
    """Lưu vết thông tin buổi học đã kết thúc vào file lưu trữ dữ liệu lịch sử."""
    records = []
    if os.path.exists(SESSIONS_DB_FILE):
        try:
            with open(SESSIONS_DB_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    session_data["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    records.append(session_data)

    try:
        with open(SESSIONS_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Storage] Lỗi lưu session record: {e}")


@app.get("/")
def read_root():
    return {"message": "Welcome to Emotion & Concentration Analysis API (EduFocus AI)"}


@app.post("/analyze_emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        pipeline = get_pipeline()
        contents = await file.read()
        print(
            f"\n🎥 [API Request] /analyze_emotion - File: '{file.filename}', Size: {len(contents)/1024:.1f} KB"
        )
        result = pipeline.run(contents)

        # Zero-Touch Auto-Archive & Auto-Export Excel: Tự động lưu vết và tạo file Excel khi buổi học tự động kết thúc
        if isinstance(result, dict) and "session" in result:
            t_status = result["session"].get("tracker_status", {})
            if t_status.get("auto_ended") and "final_summary" in t_status:
                save_session_record(t_status["final_summary"])
                excel_path = pipeline.session_tracker.export_excel()
                excel_filename = os.path.basename(excel_path)
                t_status["final_summary"]["auto_excel_filename"] = excel_filename
                t_status["final_summary"]["auto_excel_path"] = excel_path
                print(
                    f"💾 [Zero-Touch Auto-Export] Đã tự động chốt & xuất báo cáo Excel server: {excel_path}"
                )

        return result
    except Exception as e:
        print(f"❌ [API Error] /analyze_emotion: {e}")
        return {"error": str(e)}


@app.post("/analyze_single_frame")
async def analyze_single_frame(file: UploadFile = File(...)):
    """Endpoint chuyên dụng riêng cho Phân Tích Khung Hình Đơn."""
    try:
        pipeline = get_pipeline()
        contents = await file.read()
        print(
            f"\n🖼️ [API Request] /analyze_single_frame - File: '{file.filename}', Size: {len(contents)/1024:.1f} KB"
        )
        result = pipeline.run_single_frame(contents)
        return result
    except Exception as e:
        print(f"❌ [API Error] /analyze_single_frame: {e}")
        return {"error": str(e)}


# ==================== SESSION TRACKER ENDPOINTS ==================== #


@app.post("/session/start")
def start_session():
    """Bắt đầu buổi học mới thủ công."""
    pipeline = get_pipeline()
    pipeline.reset_temporal_tracker()
    res = pipeline.session_tracker.start_session()
    return res


@app.get("/session/current")
def get_current_session():
    """Lấy thông tin % cảm xúc tích lũy thời gian thực và trạng thái hiện tại."""
    pipeline = get_pipeline()
    tracker = pipeline.session_tracker

    current_percents = tracker.get_current_percent()
    score_info = tracker.compute_score()

    return {
        "is_active": tracker.is_active,
        "latest_face_count": tracker.latest_face_count,
        "total_faces": tracker.total_faces,
        "total_face_samples": tracker.total_faces,
        "total_frames_analyzed": tracker.total_frames_analyzed,
        "empty_consecutive_count": tracker.empty_consecutive_count,
        "min_faces_threshold": tracker.min_faces_threshold,
        "current_percent": current_percents,
        "current_score": score_info["score"],
        "raw_score": score_info["raw_score"],
        "history_count": len(tracker.history),
        "negative_evidence_count": len(tracker.negative_evidence_records),
        "final_summary": tracker.final_summary,
    }


@app.get("/session/history")
def get_session_history():
    """Lấy danh sách các điểm snapshot lịch sử theo từng phút của buổi học."""
    pipeline = get_pipeline()
    tracker = pipeline.session_tracker
    return {
        "is_active": tracker.is_active,
        "total_faces": tracker.total_faces,
        "history": tracker.history,
    }


@app.get("/session/score")
def get_session_score():
    """Trả về điểm đánh giá tổng kết cuối buổi học."""
    pipeline = get_pipeline()
    tracker = pipeline.session_tracker
    score_info = tracker.compute_score()

    return {
        "is_active": tracker.is_active,
        "final_score": score_info["score"],
        "raw_score": score_info["raw_score"],
        "total_faces": score_info["total_faces"],
        "emotions_percent": score_info["emotions_percent"],
    }


@app.post("/session/end")
def end_session(reason: str = Query("user_command")):
    """Kết thúc buổi học, tính điểm cuối cùng và lưu lại lịch sử."""
    pipeline = get_pipeline()
    tracker = pipeline.session_tracker
    already_ended = not tracker.is_active and tracker.final_summary is not None
    summary = tracker.end_session(reason=reason)

    # Chỉ lưu bản ghi mới nếu buổi học vừa mới kết thúc
    if not already_ended:
        save_session_record(summary)

    return summary


@app.get("/session/export_excel")
def export_session_excel():
    """Xuất và tải về file báo cáo Excel (.xlsx) chứa dữ liệu đánh giá buổi học."""
    pipeline = get_pipeline()
    tracker = pipeline.session_tracker
    excel_file_path = tracker.export_excel()

    filename = os.path.basename(excel_file_path)
    return FileResponse(
        path=excel_file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/session/past_sessions")
def get_past_sessions():
    """Lấy danh sách các buổi học quá khứ đã được lưu trữ."""
    if os.path.exists(SESSIONS_DB_FILE):
        try:
            with open(SESSIONS_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return []


@app.on_event("shutdown")
def shutdown_event():
    """Giải phóng sạch sẽ tài nguyên MediaPipe C++ và GPU VRAM khi server reload hoặc tắt."""
    global pipeline_instance
    print("🛑 [Server Shutdown] Đang giải phóng tài nguyên AI Pipeline...")
    try:
        if pipeline_instance and hasattr(pipeline_instance, "landmark_extractor"):
            if hasattr(pipeline_instance.landmark_extractor, "landmarker"):
                try:
                    pipeline_instance.landmark_extractor.landmarker.close()
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[Shutdown Warning] {e}")
    print("✅ [Server Shutdown] Đã giải phóng toàn bộ tài nguyên hệ thống sạch sẽ.")
