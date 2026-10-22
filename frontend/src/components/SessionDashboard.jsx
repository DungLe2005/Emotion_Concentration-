import React, { useState, useEffect, useRef } from 'react';
import { Activity, Clock, Users, Award, TrendingUp, AlertCircle, ShieldCheck, Zap, Sparkles, BarChart3, Video, Camera, Play, Pause, Square, RefreshCw, CheckCircle2, Download } from 'lucide-react';

const EMOTION_STYLES = {
  neutral: { name: 'Neutral', label: 'Neutral', barColor: 'bg-blue-600', textColor: 'text-blue-700', badgeBg: 'bg-blue-50 text-blue-700 border-blue-200' },
  happy: { name: 'Happy', label: 'Happy', barColor: 'bg-emerald-500', textColor: 'text-emerald-700', badgeBg: 'bg-emerald-50 text-emerald-700 border-emerald-200' },
  surprise: { name: 'Surprise', label: 'Surprise', barColor: 'bg-amber-500', textColor: 'text-amber-700', badgeBg: 'bg-amber-50 text-amber-700 border-amber-200' },
  sad: { name: 'Sad', label: 'Sad', barColor: 'bg-indigo-500', textColor: 'text-indigo-700', badgeBg: 'bg-indigo-50 text-indigo-700 border-indigo-200' },
  fear: { name: 'Fear', label: 'Fear', barColor: 'bg-purple-500', textColor: 'text-purple-700', badgeBg: 'bg-purple-50 text-purple-700 border-purple-200' },
  angry: { name: 'Angry', label: 'Angry', barColor: 'bg-rose-500', textColor: 'text-rose-700', badgeBg: 'bg-rose-50 text-rose-700 border-rose-200' },
  disgust: { name: 'Disgust', label: 'Disgust', barColor: 'bg-orange-500', textColor: 'text-orange-700', badgeBg: 'bg-orange-50 text-orange-700 border-orange-200' }
};

export default function SessionDashboard({ apiBaseUrl = 'http://localhost:8000' }) {
  const [sessionInfo, setSessionInfo] = useState({
    is_active: false,
    current_score: 0,
    current_percent: {},
    total_faces: 0,
    empty_consecutive_count: 0,
    min_faces_threshold: 1
  });
  const [history, setHistory] = useState([]);
  const [finalSummary, setFinalSummary] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Input State: 'webcam' | 'video'
  const [inputMode, setInputMode] = useState('webcam');
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isVideoAnalyzing, setIsVideoAnalyzing] = useState(false);
  const [currentFrameFaces, setCurrentFrameFaces] = useState([]);
  const [analysisStatus, setAnalysisStatus] = useState('');

  // 30s Batch Evaluation State
  const [published30sResult, setPublished30sResult] = useState(null);
  const cycleBufferRef = useRef([]); // Accumulates raw detections during the active 30s cycle
  const cycleTimerRef = useRef(null);
  const lastFinalizeTimeRef = useRef(Date.now());

  // Refs for HTML5 elements
  const webcamVideoRef = useRef(null);
  const videoFileRef = useRef(null);
  const canvasRef = useRef(document.createElement('canvas'));
  const webcamStreamRef = useRef(null);
  const frameIntervalRef = useRef(null);
  const isSendingRef = useRef(false);
  const hasAutoDownloadedRef = useRef(false);

  // Tự động xuất & tải file báo cáo Excel về máy Admin ngay khi buổi học kết thúc (Zero-Touch Auto-Export)
  useEffect(() => {
    if (finalSummary && !hasAutoDownloadedRef.current) {
      hasAutoDownloadedRef.current = true;
      const triggerAutoDownload = async () => {
        try {
          console.log('📥 [Zero-Touch Auto-Export] Đang tự động tải file báo cáo Excel...');
          const response = await fetch(`${apiBaseUrl}/session/export_excel`);
          if (response.ok) {
            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = finalSummary.auto_excel_filename || `Bao_Cao_Buoi_Hoc_${Date.now()}.xlsx`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
            console.log('✅ [Zero-Touch Auto-Export] Đã tự động xuất & tải file báo cáo Excel về máy Admin thành công!');
          }
        } catch (err) {
          console.warn('Lỗi tự động tải file Excel:', err);
        }
      };
      triggerAutoDownload();
    }
  }, [finalSummary, apiBaseUrl]);

  useEffect(() => {
    fetchCurrentSession();
    const interval = setInterval(() => {
      fetchCurrentSession();
    }, 2000);
    return () => {
      clearInterval(interval);
      stopWebcam();
      stopFrameCapture();
      stopCycleTimer();
    };
  }, []);

  const fetchCurrentSession = async () => {
    try {
      const [currRes, histRes] = await Promise.all([
        fetch(`${apiBaseUrl}/session/current`),
        fetch(`${apiBaseUrl}/session/history`)
      ]);

      if (currRes.ok) {
        const currData = await currRes.json();
        setSessionInfo(currData);
        if (currData.final_summary) {
          setFinalSummary(currData.final_summary);
        }
      }

      if (histRes.ok) {
        const histData = await histRes.json();
        setHistory(histData.history || []);
      }
    } catch (err) {
      console.warn('Syncing API Session...', err.message);
    }
  };

  // ================= 30s BATCH FINALIZATION LOGIC ================= //
  const finalize30sCycle = () => {
    const buffered = [...cycleBufferRef.current];
    if (buffered.length === 0) {
      return;
    }

    // Nhóm tất cả các frame trong 30s theo từng người (track_id)
    const personMap = {};
    buffered.forEach(f => {
      const pid = f.track_id != null ? f.track_id : (f.id != null ? f.id : 1);
      if (!personMap[pid]) {
        personMap[pid] = {
          track_id: pid,
          faces: [],
          emotions: [],
          concentrations: [],
          confidences: [],
          speaking_count: 0,
        };
      }
      personMap[pid].faces.push(f);
      if (f.emotion) personMap[pid].emotions.push(f.emotion);
      if (f.concentration != null) personMap[pid].concentrations.push(f.concentration);
      if (f.confidence != null) personMap[pid].confidences.push(f.confidence);
      if (f.is_speaking) personMap[pid].speaking_count += 1;
    });

    const personResults = Object.values(personMap).map(p => {
      // Tìm cảm xúc chiếm ưu thế nhất trong 30s của học sinh này
      const counts = {};
      p.emotions.forEach(e => { counts[e] = (counts[e] || 0) + 1; });
      let dominant = 'Neutral';
      let maxC = 0;
      Object.entries(counts).forEach(([emo, c]) => {
        if (c > maxC) { maxC = c; dominant = emo; }
      });

      const avgConc = p.concentrations.length > 0 
        ? Math.round(p.concentrations.reduce((a, b) => a + b, 0) / p.concentrations.length)
        : 0;
      const avgConf = p.confidences.length > 0
        ? Math.round(p.confidences.reduce((a, b) => a + b, 0) / p.confidences.length)
        : 0;

      return {
        track_id: p.track_id,
        sample_count: p.faces.length,
        dominant_emotion: dominant,
        avg_concentration: avgConc,
        avg_confidence: avgConf,
        is_speaking_often: p.speaking_count >= 2,
        latest_image: p.faces[p.faces.length - 1]?.image || p.faces[0]?.image,
        all_faces: p.faces, // Toàn bộ frame chụp của học sinh này trong 30s
      };
    });

    console.log(`🎯 [30s Batch Finalized] Đã chốt chu kỳ 30s! Tổng số frame: ${buffered.length} | Số học sinh: ${personResults.length}`);
    setPublished30sResult(prev => ({
      cycleIndex: (prev?.cycleIndex || 0) + 1,
      completedAt: new Date().toLocaleTimeString(),
      totalFaces: buffered.length,
      allCapturedFaces: buffered,
      personResults: personResults,
    }));

    // Reset buffer và mốc thời gian cho chu kỳ 30s tiếp theo
    cycleBufferRef.current = [];
    lastFinalizeTimeRef.current = Date.now();
  };

  const startCycleTimer = () => {
    stopCycleTimer();
    cycleTimerRef.current = setInterval(() => {
      finalize30sCycle();
    }, 30000);
  };

  const stopCycleTimer = () => {
    if (cycleTimerRef.current) {
      clearInterval(cycleTimerRef.current);
      cycleTimerRef.current = null;
    }
  };

  // ================= WEBCAM CONTROLS ================= //
  const startWebcam = async () => {
    try {
      setAnalysisStatus('Đang mở Live Webcam...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } 
      });
      webcamStreamRef.current = stream;
      if (webcamVideoRef.current) {
        webcamVideoRef.current.srcObject = stream;
        await webcamVideoRef.current.play();
      }
      setIsWebcamActive(true);
      setAnalysisStatus('Live Webcam đang phát & Đang phân tích...');
      
      startFrameCapture(webcamVideoRef);
    } catch (err) {
      alert('Không thể truy cập Webcam: ' + err.message);
      setAnalysisStatus('Lỗi mở Webcam');
    }
  };

  const stopWebcam = () => {
    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach(track => track.stop());
      webcamStreamRef.current = null;
    }
    if (webcamVideoRef.current) {
      webcamVideoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
    stopFrameCapture();
    finalize30sCycle();
    setAnalysisStatus('Đã tắt Live Webcam');
  };

  // ================= VIDEO FILE CONTROLS ================= //
  const handleVideoFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      hasAutoDownloadedRef.current = false;
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setCurrentFrameFaces([]);
      setPublished30sResult(null);
      cycleBufferRef.current = [];
      lastFinalizeTimeRef.current = Date.now();
      setFinalSummary(null);
      setAnalysisStatus(`Đã nạp video: ${file.name}`);
      handleStartSession();
    }
  };

  const toggleVideoAnalysis = () => {
    if (!videoFileRef.current) return;

    if (isVideoAnalyzing) {
      videoFileRef.current.pause();
      setIsVideoAnalyzing(false);
      stopFrameCapture();
      finalize30sCycle();
      setAnalysisStatus('Đang tạm dừng phân tích video');
    } else {
      if (videoFileRef.current) {
        videoFileRef.current.muted = true;
      }
      videoFileRef.current.play();
      setIsVideoAnalyzing(true);
      startFrameCapture(videoFileRef);
      setAnalysisStatus('Đang phát video & Đang phân tích...');
    }
  };

  // ================= FRAME CAPTURE & API SENDING ================= //
  const startFrameCapture = (targetVideoRef) => {
    stopFrameCapture();
    isSendingRef.current = false;
    lastFinalizeTimeRef.current = Date.now(); // Bắt đầu bấm giờ 30s mới từ khoảnh khắc này
    cycleBufferRef.current = [];
    startCycleTimer();
    // Gửi ngay 1 frame đầu tiên khi bắt đầu
    captureAndSendFrame(targetVideoRef);
    // Tần suất lấy frame 2.5 giây / 1 lần
    frameIntervalRef.current = setInterval(() => {
      captureAndSendFrame(targetVideoRef);
    }, 2500);
  };

  const stopFrameCapture = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    stopCycleTimer();
    isSendingRef.current = false;
  };

  const captureAndSendFrame = async (targetVideoRef) => {
    const videoElem = targetVideoRef.current;
    if (!videoElem) return;

    if (inputMode === 'video' && (videoElem.paused || videoElem.ended)) {
      return;
    }

    if (isSendingRef.current) {
      return;
    }

    const MAX_WIDTH = 640;
    let vWidth = videoElem.videoWidth || 640;
    let vHeight = videoElem.videoHeight || 480;
    if (vWidth > MAX_WIDTH) {
      vHeight = Math.round((vHeight * MAX_WIDTH) / vWidth);
      vWidth = MAX_WIDTH;
    }

    const canvas = canvasRef.current;
    canvas.width = vWidth;
    canvas.height = vHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);

    isSendingRef.current = true;

    canvas.toBlob(async (blob) => {
      if (!blob) {
        isSendingRef.current = false;
        return;
      }

      try {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        const response = await fetch(`${apiBaseUrl}/analyze_emotion`, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          if (data.faces && data.faces.length > 0) {
            setCurrentFrameFaces(data.faces);
            const now = Date.now();
            const timeStr = new Date().toLocaleTimeString();
            const stamped = data.faces.map((f, idx) => ({
              ...f,
              timestamp: now,
              timeStr: timeStr,
              uniqueKey: `${f.track_id || f.id || idx}_${now}_${Math.random()}`
            }));
            // Tích lũy ngầm vào buffer của chu kỳ 30s hiện tại
            cycleBufferRef.current.push(...stamped);

            // Kiểm tra mốc 30 giây kể từ lần chốt trước
            const elapsed = Date.now() - lastFinalizeTimeRef.current;
            if (elapsed >= 30000) {
              lastFinalizeTimeRef.current = Date.now();
              finalize30sCycle();
            }
          }
          if (data.session) {
            setSessionInfo(prev => ({
              ...prev,
              is_active: data.session.is_active,
              total_faces: data.session.total_faces,
              current_percent: data.session.current_percent,
              current_score: data.session.current_score
            }));
            if (data.session.tracker_status?.final_summary) {
              setFinalSummary(data.session.tracker_status.final_summary);
              finalize30sCycle();
            }
          }
        }
      } catch (err) {
        console.warn('Lỗi gửi frame phân tích:', err.message);
      } finally {
        isSendingRef.current = false;
      }
    }, 'image/jpeg', 0.85);
  };

  const handleStartSession = async () => {
    setIsLoading(true);
    hasAutoDownloadedRef.current = false;
    setPublished30sResult(null);
    cycleBufferRef.current = [];
    setCurrentFrameFaces([]);
    setCycleSeconds(0);
    try {
      const res = await fetch(`${apiBaseUrl}/session/start`, { method: 'POST' });
      if (res.ok) {
        setFinalSummary(null);
        await fetchCurrentSession();
      }
    } catch (err) {
      alert('Lỗi bắt đầu buổi học: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEndSession = async () => {
    setIsLoading(true);
    finalize30sCycle();
    stopCycleTimer();
    try {
      const res = await fetch(`${apiBaseUrl}/session/end?reason=user_command`, { method: 'POST' });
      if (res.ok) {
        const summary = await res.json();
        setFinalSummary(summary);
        await fetchCurrentSession();
      }
    } catch (err) {
      alert('Lỗi kết thúc buổi học: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const isStreamLive = inputMode === 'webcam' ? isWebcamActive : isVideoAnalyzing;
  const isCurrentlyActive = sessionInfo.is_active && isStreamLive;

  const percents = sessionInfo.current_percent || {};
  const dominantKey = Object.keys(percents).length > 0 
    ? Object.keys(percents).reduce((a, b) => (percents[a] > percents[b] ? a : b), 'neutral')
    : 'neutral';

  const radius = 46;
  const circumference = 2 * Math.PI * radius;
  const activeScore = isCurrentlyActive ? (sessionInfo.current_score || 0) : 0;
  const scorePercent = Math.min(100, Math.max(0, activeScore));
  const strokeDashoffset = circumference - (scorePercent / 100) * circumference;

  return (
    <div className="space-y-6">
      
      {/* Top Banner Status Bar */}
      <div className="card-clean rounded-2xl p-5 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-white border border-slate-200 shadow-sm">
        <div className="flex items-center gap-3.5">
          <div className={`p-3 rounded-xl ${isCurrentlyActive ? 'bg-emerald-50 border border-emerald-200 text-emerald-600' : 'bg-amber-50 border border-amber-200 text-amber-600'}`}>
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-lg font-bold font-heading text-slate-900 tracking-tight">
                Giám Sát Buổi Học Tự Động
              </h2>
              <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold flex items-center gap-1.5 border ${
                isCurrentlyActive 
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200' 
                  : 'bg-amber-50 text-amber-700 border-amber-200'
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${isCurrentlyActive ? 'bg-emerald-500 animate-ping' : 'bg-amber-500'}`} />
                {isCurrentlyActive ? 'ĐANG TÍCH LŨY BUỔI HỌC REALTIME' : 'ĐANG CHỜ NẠP VIDEO / BẬT CAMERA'}
              </span>
            </div>
          </div>
        </div>

      </div>

      {/* Balanced 2-Column Layout (50% LEFT Video/Webcam Stream vs 50% RIGHT AI Analytics) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* LEFT COLUMN (6 Cols / 50% Width): Live Video / Webcam Feed & Face Gallery */}
        <div className="lg:col-span-6 space-y-6">
          
          {/* Card A: Nạp Video / Camera Feed Trực Tiếp */}
          <div className="card-clean rounded-2xl p-6 bg-white border border-slate-200 space-y-4">
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <h3 className="text-base font-bold font-heading text-slate-900 flex items-center gap-2">
                <Video className="w-5 h-5 text-blue-600" />
                Đầu Vào Luồng Video
              </h3>
            </div>

            {/* Selector Switch Mode */}
            <div className="flex p-1 rounded-xl bg-slate-100 border border-slate-200">
              <button
                onClick={() => { setInputMode('webcam'); stopFrameCapture(); }}
                className={`flex-1 py-2 text-xs font-bold font-heading rounded-lg transition-all flex items-center justify-center gap-2 ${
                  inputMode === 'webcam' ? 'bg-white text-blue-600 shadow-sm border border-slate-200' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Camera className="w-4 h-4" />
                Camera Trực Tiếp
              </button>

              <button
                onClick={() => { setInputMode('video'); stopWebcam(); }}
                className={`flex-1 py-2 text-xs font-bold font-heading rounded-lg transition-all flex items-center justify-center gap-2 ${
                  inputMode === 'video' ? 'bg-white text-blue-600 shadow-sm border border-slate-200' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Video className="w-4 h-4" />
                Video Bài Học (.mp4, .webm)
              </button>
            </div>

            {/* WEBCAM INPUT DISPLAY */}
            {inputMode === 'webcam' && (
              <div className="space-y-3">
                <div className="relative rounded-xl overflow-hidden bg-slate-900 aspect-video flex items-center justify-center border border-slate-200">
                  <video 
                    ref={webcamVideoRef} 
                    autoPlay 
                    playsInline 
                    muted 
                    className={`w-full h-full object-cover ${isWebcamActive ? 'block' : 'hidden'}`}
                  />

                  {!isWebcamActive && (
                    <div className="text-center p-6 text-slate-400 space-y-3">
                      <Camera className="w-10 h-10 text-slate-600 mx-auto" />
                      <p className="text-xs font-semibold text-slate-300">Live Camera hiện đang tắt</p>
                      <p className="text-[11px] text-slate-500">Ấn vào nút bên dưới để mở Webcam và tự động phân tích lớp học</p>
                    </div>
                  )}

                  {isWebcamActive && (
                    <div className="absolute top-3 left-3 bg-red-600 text-white text-[10px] font-bold font-mono px-2.5 py-1 rounded-full flex items-center gap-1.5 animate-pulse">
                      <span className="w-2 h-2 rounded-full bg-white" />
                      LIVE WEBCAM STREAMING
                    </div>
                  )}
                </div>

                {!isWebcamActive ? (
                  <button
                    onClick={startWebcam}
                    className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs tracking-wide uppercase shadow-sm transition-all flex items-center justify-center gap-2"
                  >
                    <Camera className="w-4 h-4" />
                    Bật Live Camera Trực Tiếp
                  </button>
                ) : (
                  <button
                    onClick={stopWebcam}
                    className="w-full py-3 rounded-xl bg-slate-800 hover:bg-slate-900 text-white font-bold text-xs tracking-wide uppercase shadow-sm transition-all flex items-center justify-center gap-2"
                  >
                    <Square className="w-4 h-4 text-red-400" />
                    Tắt Live Camera
                  </button>
                )}
              </div>
            )}

            {/* VIDEO FILE INPUT DISPLAY */}
            {inputMode === 'video' && (
              <div className="space-y-3">
                {videoUrl ? (
                  <div className="relative rounded-xl overflow-hidden bg-slate-900 aspect-video border border-slate-200">
                    <video 
                      ref={videoFileRef} 
                      src={videoUrl} 
                      controls 
                      muted 
                      playsInline
                      className="w-full h-full object-contain"
                      onLoadedMetadata={(e) => {
                        e.currentTarget.muted = true;
                      }}
                      onPlay={() => { 
                        if (videoFileRef.current) {
                          videoFileRef.current.muted = true;
                        }
                        if (!isVideoAnalyzing) toggleVideoAnalysis();
                      }}
                      onPause={() => { 
                        if (isVideoAnalyzing) toggleVideoAnalysis();
                      }}
                      onEnded={() => { 
                        setIsVideoAnalyzing(false); 
                        stopFrameCapture(); 
                        finalize30sCycle();
                        setAnalysisStatus('Video kết thúc — Đang tự động chốt báo cáo buổi học...'); 
                        handleEndSession();
                      }}
                    />
                  </div>
                ) : (
                  <div 
                    className="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-xl p-8 flex flex-col items-center justify-center text-slate-500 cursor-pointer transition-all bg-slate-50/50"
                    onClick={() => document.getElementById('videoFileInput').click()}
                  >
                    <input 
                      type="file" 
                      accept="video/mp4,video/webm,video/ogg,video/quicktime" 
                      className="hidden" 
                      id="videoFileInput" 
                      onChange={handleVideoFileChange} 
                    />
                    <Video className="w-10 h-10 text-blue-600 mb-2" />
                    <p className="text-xs font-semibold text-slate-700">Tải file video buổi học (.mp4, .webm, .mov)</p>
                    <p className="text-[11px] text-slate-400">Nhấp vào đây để chọn video từ máy tính</p>
                  </div>
                )}

                {videoUrl && (
                  <div className="flex gap-2">
                    <button
                      onClick={toggleVideoAnalysis}
                      className={`flex-1 py-3 rounded-xl font-bold text-xs tracking-wide uppercase transition-all flex items-center justify-center gap-2 text-white shadow-sm ${
                        isVideoAnalyzing ? 'bg-amber-600 hover:bg-amber-700' : 'bg-blue-600 hover:bg-blue-700'
                      }`}
                    >
                      {isVideoAnalyzing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
                      {isVideoAnalyzing ? 'Tạm Dừng Phân Tích' : 'Phát Video & Phân Tích Realtime'}
                    </button>

                    <button
                      onClick={() => document.getElementById('videoFileInput').click()}
                      className="px-4 py-3 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold text-xs border border-slate-200 transition-all flex items-center gap-1.5 shrink-0"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      Đổi Video
                    </button>
                    <input 
                      type="file" 
                      accept="video/mp4,video/webm,video/ogg,video/quicktime" 
                      className="hidden" 
                      id="videoFileInput" 
                      onChange={handleVideoFileChange} 
                    />
                  </div>
                )}
              </div>
            )}

            {analysisStatus && (
              <div className="text-[11px] text-slate-500 font-mono flex items-center gap-1.5 pt-1">
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping" />
                Trạng thái luồng: {analysisStatus}
              </div>
            )}
          </div>

          {/* Card B: Kết Quả Chốt Chu Kỳ 30s & Khung Hình Đã Chụp */}
          <div className="card-clean rounded-2xl p-6 bg-white border border-slate-200 space-y-4">
            
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 pb-3 border-b border-slate-100">
              <h3 className="text-base font-bold font-heading text-slate-900 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-blue-600" />
                Kết Quả Phân Tích Chốt Theo Chu Kỳ
              </h3>
            </div>

            {published30sResult ? (
              <div className="space-y-4">
                
                {/* 1. Kết Quả Cảm Xúc Cuối Cùng Của Từng Người Trong 30s */}
                <div className="space-y-2.5">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-600">
                      Cảm Xúc Từng Người Chốt Trong ({published30sResult.personResults?.length || 0} học sinh)
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono">
                      Mốc: {published30sResult.completedAt}
                    </span>
                  </div>

                  <div className="space-y-3">
                    {published30sResult.personResults?.map((person) => {
                      const emoStyle = EMOTION_STYLES[person.dominant_emotion.toLowerCase()] || {};
                      return (
                        <div key={person.track_id} className="p-3.5 rounded-xl bg-slate-50/80 border border-slate-200 space-y-2.5">
                          
                          {/* Person Summary Header */}
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2.5">
                              <img 
                                src={`data:image/jpeg;base64,${person.latest_image}`}
                                alt={`Học sinh #${person.track_id}`}
                                className="w-10 h-10 rounded-lg object-cover border border-slate-300 shadow-xs"
                              />
                              <div>
                                <strong className="text-xs text-slate-800 font-bold block">
                                  Học sinh #{person.track_id}
                                </strong>
                                <span className="text-[11px] text-slate-400 block">
                                  Đã đọc {person.sample_count} frames trong 30s
                                </span>
                              </div>
                            </div>

                            <div className="text-right space-y-1">
                              <span className={`inline-block px-2.5 py-0.5 rounded-md text-[11px] font-bold border uppercase tracking-wider ${emoStyle.badgeBg || 'bg-slate-100 text-slate-700'}`}>
                                {emoStyle.label || person.dominant_emotion}
                              </span>
                              <div className="text-[11px] text-emerald-600 font-bold">
                                Tập trung TB: {person.avg_concentration}%
                              </div>
                            </div>
                          </div>

                          {/* Speaking Tag if active speech detected */}
                          {person.is_speaking_often && (
                            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-50 border border-amber-200 text-[10px] font-semibold text-amber-700">
                              <span>🗣️ Phát hiện trao phát âm</span>
                            </div>
                          )}

                          {/* Horizontal Scrollable Strip of All Captured Frames for this Person */}
                          <div className="space-y-1 pt-1 border-t border-slate-200/60">
                            <span className="text-[10px] text-slate-400 font-medium block">
                              Toàn bộ frame đã chụp của học sinh #{person.track_id} ({person.all_faces?.length} ảnh):
                            </span>
                            <div className="flex gap-2 overflow-x-auto pb-1 pt-0.5">
                              {person.all_faces?.map((faceItem, fIdx) => (
                                <div key={fIdx} className="shrink-0 text-center space-y-0.5">
                                  <img 
                                    src={`data:image/jpeg;base64,${faceItem.image}`}
                                    alt={`Frame ${fIdx + 1}`}
                                    className="w-11 h-11 rounded-md object-cover border border-slate-200 shadow-2xs hover:scale-105 transition-all"
                                  />
                                  <span className="text-[9px] font-mono text-slate-400 block leading-tight">
                                    {faceItem.timeStr?.split(' ')[0] || `#${fIdx + 1}`}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>

                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>
            ) : (
              <div className="h-44 flex flex-col items-center justify-center border border-slate-100 rounded-xl text-xs text-slate-400 gap-1.5 bg-slate-50/50 text-center px-4">
                <Users className="w-7 h-7 text-slate-300 mb-1" />
                <span className="font-semibold text-slate-600">Đang phân tích dữ liệu buổi học...</span>
                <span className="text-[11px] text-slate-400 max-w-sm">
                  Kết quả cảm xúc và ảnh chụp của từng học sinh sẽ hiển thị sau mỗi 30 giây.
                </span>
              </div>
            )}

          </div>

        </div>

        {/* RIGHT COLUMN (6 Cols / 50% Width): AI Analytics & 7 Emotion Breakdown */}
        <div className="lg:col-span-6 space-y-6">
          
          {/* Card A: Focus Gauge Ring & Key Metrics */}
          <div className="card-clean rounded-2xl p-6 bg-white border border-slate-200 space-y-4">
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <h3 className="text-base font-bold font-heading text-slate-900 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-600" />
                Kết Quả Độ Tập Trung Cuối Cùng
              </h3>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-between gap-6 py-2">
              {/* Radial Score Ring */}
              <div className="relative w-32 h-32 flex items-center justify-center shrink-0">
                <svg className="w-32 h-32 transform -rotate-90">
                  <circle cx="64" cy="64" r={radius} className="stroke-slate-100" strokeWidth="10" fill="transparent" />
                  <circle
                    cx="64"
                    cy="64"
                    r={radius}
                    className="stroke-blue-600 transition-all duration-700"
                    strokeWidth="10"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    fill="transparent"
                  />
                </svg>
                <div className="absolute flex flex-col items-center justify-center text-center">
                  <span className="text-3xl font-extrabold font-heading text-slate-900">
                    {isCurrentlyActive ? `${sessionInfo.current_score}%` : '0%'}
                  </span>
                  <span className="text-[10px] text-slate-400 font-semibold">Độ Tập Trung</span>
                </div>
              </div>

              {/* Dominant Emotion & Stats */}
              <div className="space-y-3 flex-1 text-center sm:text-left">
                <div>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Cảm Xúc Thống Trị</span>
                  <div className="mt-1">
                    {isCurrentlyActive && (sessionInfo.total_faces > 0 || sessionInfo.total_face_samples > 0) ? (
                      <span className={`inline-block px-3 py-1 rounded-lg text-xs font-bold border uppercase tracking-wider ${EMOTION_STYLES[dominantKey]?.badgeBg || 'bg-slate-100 text-slate-700'}`}>
                        {EMOTION_STYLES[dominantKey]?.label || dominantKey} ({percents[dominantKey] || 0}%)
                      </span>
                    ) : (
                      <span className="inline-block px-3 py-1 rounded-lg text-xs font-semibold bg-slate-100 text-slate-500 border border-slate-200">
                        Đang chờ nạp Video / Bật Camera
                      </span>
                    )}
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-100 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-slate-400 block text-[10px]">Sĩ Số Lớp:</span>
                    <strong className="text-slate-800 font-bold">{currentFrameFaces.length || sessionInfo.latest_face_count || 0} người</strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px]">Lượt Mẫu Tích Lũy:</span>
                    <strong className="text-slate-800 font-bold">{sessionInfo.total_face_samples || sessionInfo.total_faces || 0} lượt</strong>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Card B: Phân Phối 7 Cảm Xúc Tích Lũy (%) */}
          <div className="card-clean rounded-2xl p-6 bg-white border border-slate-200 space-y-4">
            <div className="flex justify-between items-center pb-2 border-b border-slate-100">
              <h3 className="text-base font-bold font-heading text-slate-900 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                Phân Phối 7 Cảm Xúc Tích Lũy (%)
              </h3>
            </div>

            <div className="space-y-3">
              {Object.entries(EMOTION_STYLES).map(([key, style]) => {
                const val = percents[key] || 0;
                return (
                  <div key={key} className="space-y-1">
                    <div className="flex justify-between items-center text-xs font-semibold">
                      <span className="text-slate-700">{style.label}</span>
                      <span className={`font-mono text-sm font-bold ${style.textColor}`}>{val}%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                      <div 
                        className={`h-full ${style.barColor} rounded-full transition-all duration-500`}
                        style={{ width: `${val}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>

      </div>

      {/* ═══════════════════════════════════════════════════════════════════
           BÁO CÁO TỔNG KẾT BUỔI HỌC — ĐÁNH GIÁ VĂN BẢN KHOA HỌC
           Cơ sở: Fredricks et al. (2004), Likert (1932), Whitehill et al. (2014)
           ═══════════════════════════════════════════════════════════════════ */}
      {finalSummary && (() => {
        const qa = finalSummary.qualitative_assessment || {};
        const level = qa.engagement_level || 'N/A';
        const levelEn = qa.engagement_level_en || '';
        const icon = qa.engagement_icon || '📊';
        const color = qa.engagement_color || 'slate';

        // Tailwind color mapping
        const colorMap = {
          emerald: { badge: 'bg-emerald-100 text-emerald-800 border-emerald-300', header: 'from-emerald-50 via-white to-emerald-50/30 border-emerald-200', bar: 'bg-emerald-500', accent: 'text-emerald-700' },
          blue: { badge: 'bg-blue-100 text-blue-800 border-blue-300', header: 'from-blue-50 via-white to-blue-50/30 border-blue-200', bar: 'bg-blue-500', accent: 'text-blue-700' },
          amber: { badge: 'bg-amber-100 text-amber-800 border-amber-300', header: 'from-amber-50 via-white to-amber-50/30 border-amber-200', bar: 'bg-amber-500', accent: 'text-amber-700' },
          orange: { badge: 'bg-orange-100 text-orange-800 border-orange-300', header: 'from-orange-50 via-white to-orange-50/30 border-orange-200', bar: 'bg-orange-500', accent: 'text-orange-700' },
          red: { badge: 'bg-red-100 text-red-800 border-red-300', header: 'from-red-50 via-white to-red-50/30 border-red-200', bar: 'bg-red-500', accent: 'text-red-700' },
          slate: { badge: 'bg-slate-100 text-slate-800 border-slate-300', header: 'from-slate-50 via-white to-slate-50/30 border-slate-200', bar: 'bg-slate-500', accent: 'text-slate-700' },
        };
        const cm = colorMap[color] || colorMap.slate;

        return (
          <div className={`card-clean rounded-2xl p-6 bg-gradient-to-br ${cm.header} border-2 shadow-md space-y-5 animate-fade-in`}>

            {/* ── Header: Xếp Loại + Actions ── */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div className="flex items-center gap-4">
                <div className={`p-3.5 rounded-xl ${cm.badge} border text-2xl flex items-center justify-center`}>
                  {icon}
                </div>
                <div>
                  <h3 className="text-lg font-bold font-heading text-slate-900">Báo Cáo Tổng Kết Buổi Học</h3>
                  <div className="flex items-center gap-2 mt-1">
                    <span className={`px-3 py-1 rounded-full text-sm font-bold border ${cm.badge}`}>
                      {level} ({levelEn})
                    </span>
                    <span className="text-xs text-slate-400">
                      {finalSummary.end_reason === 'auto_insufficient_faces' ? 'Tự động chốt' : 'Hệ thống kết thúc'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* ── Metrics Grid ── */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 p-4 rounded-xl bg-white border border-slate-200 shadow-sm">
              <div>
                <span className="text-[11px] text-slate-400 uppercase font-semibold">Chỉ Số Tập Trung</span>
                <p className={`text-3xl font-extrabold font-heading mt-0.5 ${cm.accent}`}>{finalSummary.final_score}%</p>
              </div>
              <div>
                <span className="text-[11px] text-slate-400 uppercase font-semibold">Thời Lượng</span>
                <p className="text-xl font-bold text-slate-800 mt-1">{finalSummary.duration_formatted}</p>
              </div>
              <div>
                <span className="text-[11px] text-slate-400 uppercase font-semibold">Sĩ Số TB / Frame</span>
                <p className="text-xl font-bold text-slate-800 mt-1">{finalSummary.average_faces_per_frame || 0} người</p>
              </div>
              <div>
                <span className="text-[11px] text-slate-400 uppercase font-semibold">Tổng Khung Hình</span>
                <p className="text-xl font-bold text-slate-800 mt-1">{finalSummary.total_frames_analyzed || 0} frames</p>
              </div>
              <div>
                <span className="text-[11px] text-slate-400 uppercase font-semibold">Tổng Khuôn Mặt</span>
                <p className="text-xl font-bold text-slate-800 mt-1">{finalSummary.total_faces || 0} lượt</p>
              </div>
            </div>

            {/* ── Nhận Xét Tổng Quan (Overall Comment) ── */}
            {qa.overall_comment && (
              <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm space-y-2">
                <div className="flex items-center gap-2 text-sm font-bold text-slate-700">
                  <Sparkles className="w-4 h-4 text-amber-500" />
                  Nhận Xét Tổng Quan
                </div>
                <p className="text-sm text-slate-600 leading-relaxed">{qa.overall_comment}</p>
              </div>
            )}

            {/* ── Tỷ Lệ Cảm Xúc Tích Cực / Tiêu Cực ── */}
            {(qa.positive_ratio != null || qa.negative_ratio != null) && (
              <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm space-y-3">
                <div className="flex items-center gap-2 text-sm font-bold text-slate-700">
                  <BarChart3 className="w-4 h-4 text-blue-500" />
                  Phân Bổ Cảm Xúc
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs font-semibold text-emerald-700 w-28 shrink-0">Tích cực {qa.positive_ratio}%</span>
                  <div className="flex-1 h-4 rounded-full bg-slate-100 overflow-hidden flex">
                    <div
                      className="h-full bg-emerald-500 rounded-l-full transition-all duration-700"
                      style={{ width: `${qa.positive_ratio || 0}%` }}
                    />
                    <div
                      className="h-full bg-rose-400 rounded-r-full transition-all duration-700"
                      style={{ width: `${qa.negative_ratio || 0}%` }}
                    />
                  </div>
                  <span className="text-xs font-semibold text-rose-600 w-28 text-right shrink-0">Tiêu cực {qa.negative_ratio}%</span>
                </div>

                {/* Dominant Emotions */}
                {qa.dominant_emotions && qa.dominant_emotions.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-1">
                    <span className="text-xs text-slate-400">Cảm xúc chủ đạo:</span>
                    {qa.dominant_emotions.map((dom, i) => (
                      <span key={i} className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-slate-100 text-slate-700 border border-slate-200">
                        {dom.label_vi} — {dom.percent}%
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* ── Khuyến Nghị Sư Phạm (Pedagogical Recommendations) ── */}
            {qa.recommendations && qa.recommendations.length > 0 && (
              <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm space-y-3">
                <div className="flex items-center gap-2 text-sm font-bold text-slate-700">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                  Khuyến Nghị Sư Phạm
                </div>
                <ul className="space-y-2">
                  {qa.recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-sm text-slate-600">
                      <span className="w-5 h-5 rounded-full bg-slate-100 text-slate-500 text-[10px] font-bold flex items-center justify-center mt-0.5 shrink-0">
                        {i + 1}
                      </span>
                      <span className="leading-relaxed">{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* ── Timestamps + Cơ Sở Khoa Học ── */}
            <div className="space-y-2">
              {(finalSummary.start_time_formatted || finalSummary.end_time_formatted) && (
                <div className="flex flex-col sm:flex-row justify-between text-xs text-slate-500 bg-white/60 p-3 rounded-xl border border-slate-200/60">
                  <span><strong>Bắt đầu:</strong> {finalSummary.start_time_formatted || 'N/A'}</span>
                  <span><strong>Kết thúc:</strong> {finalSummary.end_time_formatted || 'N/A'}</span>
                </div>
              )}
            </div>

          </div>
        );
      })()}
    </div>
  );
}
