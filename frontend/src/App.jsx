import React, { useState } from 'react';
import { Upload, Camera, BrainCircuit, MonitorPlay, Layers, Sparkles, Cpu, Activity, Clock, Users, Zap, AlertTriangle, Square, Award, BarChart3, TrendingUp } from 'lucide-react';
import PipelineVisualizer from './components/PipelineVisualizer';
import LogConsole from './components/LogConsole';
import SessionDashboard from './components/SessionDashboard';

function App() {
  const [activeTab, setActiveTab] = useState('session'); // 'session' | 'frame'
  const [logs, setLogs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setLogs([]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setLogs([]);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) {
      alert('Vui lòng chọn hoặc kéo thả ảnh vào trước khi phân tích!');
      return;
    }

    setIsProcessing(true);
    setLogs([]);
    setResult(null);

    setLogs([{
      time: new Date().toLocaleTimeString(),
      stage: 'Image Decoding',
      message: 'Đang tải và giải mã ảnh...',
      latency_ms: 0,
      status: 'success'
    }]);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('http://localhost:8000/analyze_single_frame', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server Backend phản hồi lỗi ${response.status}`);
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      let currentLogIndex = 0;
      const backendLogs = data.logs || [];

      const runLogSimulation = () => {
        if (currentLogIndex < backendLogs.length) {
          const log = backendLogs[currentLogIndex];
          setTimeout(() => {
            setLogs(prev => {
              const filtered = prev.filter(p => p.stage !== log.stage);
              return [...filtered, {
                time: new Date().toLocaleTimeString(),
                stage: log.stage,
                message: log.message,
                latency_ms: log.latency_ms,
                status: 'success'
              }];
            });
            currentLogIndex++;
            runLogSimulation();
          }, 300);
        } else {
          setIsProcessing(false);
          setResult({
            emotion: data.emotion,
            concentrationScore: data.concentration,
            faces: data.faces,
          });
        }
      };

      runLogSimulation();

    } catch (error) {
      console.error(error);
      setLogs(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        stage: 'System Error',
        message: `Lỗi: ${error.message}`,
        latency_ms: 0,
        status: 'error'
      }]);
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-800 antialiased font-sans">
      
      {/* Top Enterprise Header */}
      <header className="sticky top-0 z-30 bg-white border-b border-slate-200/80 px-8 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-blue-600 text-white shadow-sm">
            <BrainCircuit className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold font-heading text-slate-900 tracking-tight flex items-center gap-2">
              EduFocus AI
            </h1>
            <p className="text-xs text-slate-500 font-medium">Hệ Thống Phân Tích Cảm Xúc & Độ Tập Trung Học Sinh</p>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-100 border border-slate-200">
          <button
            onClick={() => setActiveTab('session')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold font-heading transition-all ${
              activeTab === 'session'
                ? 'bg-white text-blue-600 shadow-sm border border-slate-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <MonitorPlay className="w-4 h-4 text-blue-600" />
            Dashboard Buổi Học
          </button>

          <button
            onClick={() => setActiveTab('frame')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold font-heading transition-all ${
              activeTab === 'frame'
                ? 'bg-white text-blue-600 shadow-sm border border-slate-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-4 h-4 text-blue-600" />
            Phân Tích Khung Hình Đơn
          </button>
        </div>
      </header>

      {/* Main Page Content */}
      <main className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
        {activeTab === 'session' ? (
          <SessionDashboard />
        ) : (
          /* Single Frame View */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-6 space-y-6">
              <div className="card-clean rounded-2xl p-6 bg-white border border-slate-200 space-y-4">
                <h2 className="text-base font-bold font-heading text-slate-900 flex items-center gap-2">
                  <Upload className="w-5 h-5 text-blue-600" />
                  Nạp Ảnh Single Frame
                </h2>
                
                <div 
                  className="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-xl p-6 flex flex-col items-center justify-center text-slate-500 cursor-pointer transition-all relative overflow-hidden h-48 bg-slate-50/50"
                  onClick={() => document.getElementById('fileInputSingleTab').click()}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                >
                  <input 
                    type="file" 
                    accept="image/*" 
                    className="hidden" 
                    id="fileInputSingleTab" 
                    onChange={handleFileChange} 
                  />
                  {previewUrl ? (
                    <img 
                      src={previewUrl} 
                      alt="Preview" 
                      className="absolute inset-0 w-full h-full object-cover"
                    />
                  ) : (
                    <div className="text-center space-y-2">
                      <Camera className="w-8 h-8 text-slate-400 mx-auto" />
                      <p className="text-xs font-semibold text-slate-700">Click to upload frame</p>
                    </div>
                  )}
                </div>
                
                <button 
                  onClick={analyzeImage}
                  disabled={isProcessing || !selectedFile}
                  className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs tracking-wide uppercase shadow-sm transition-all"
                >
                  {isProcessing ? 'Processing Pipeline...' : 'Run Pipeline Inference'}
                </button>
              </div>

              {result && (
                <div className="card-clean rounded-2xl p-6 bg-white border border-slate-200 space-y-4">
                  <div className="flex justify-between items-center pb-2 border-b border-slate-100">
                    <h3 className="text-base font-bold font-heading text-slate-900">
                      Khuôn Mặt Nhận Diện & Phân Phối Softmax ({result.faces?.length || 0})
                    </h3>
                    <span className="text-xs text-blue-600 font-bold bg-blue-50 px-2.5 py-0.5 rounded-full border border-blue-100">
                      TB Tập Trung: {result.concentration}%
                    </span>
                  </div>

                  <div className="space-y-4 max-h-[460px] overflow-y-auto pr-1">
                    {result.faces?.map(face => (
                      <div key={face.id} className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                        <div className="flex items-center gap-3">
                          <img 
                            src={`data:image/jpeg;base64,${face.image}`} 
                            className="w-14 h-14 rounded-xl object-cover border border-slate-300 shadow-sm shrink-0" 
                            alt={`Face ${face.id}`} 
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex justify-between items-center">
                              <span className="font-bold text-sm text-slate-900">{face.emotion}</span>
                              <span className="text-xs font-extrabold text-blue-600 bg-blue-50 px-2.5 py-1 rounded-lg border border-blue-100">
                                Tập Trung: {face.concentration}%
                              </span>
                            </div>
                            <p className="text-xs text-slate-500 mt-1">Độ tin cậy AI (Top-1 Softmax): {face.confidence}%</p>
                          </div>
                        </div>

                        {/* Phân Phối Xác Suất Softmax 7 Cảm Xúc */}
                        {face.emotions && face.emotions.length > 0 && (
                          <div className="pt-2.5 border-t border-slate-200/80 space-y-2">
                            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                              Bảng Phân Phối Xác Suất Softmax (7 Cảm Xúc)
                            </span>
                            <div className="grid grid-cols-1 gap-1.5">
                              {face.emotions.map(item => (
                                <div key={item.label} className="space-y-0.5">
                                  <div className="flex justify-between text-[11px] font-medium text-slate-600">
                                    <span>{item.label}</span>
                                    <span className="font-mono font-bold text-slate-800">{item.score}%</span>
                                  </div>
                                  <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                                    <div 
                                      className="bg-blue-600 h-full rounded-full transition-all duration-300"
                                      style={{ width: `${item.score}%` }}
                                    />
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="lg:col-span-6 space-y-6">
              <div className="card-clean rounded-2xl p-6 bg-white border border-slate-200">
                <h2 className="text-base font-bold font-heading mb-4 text-slate-900">AI Architecture Pipeline Status</h2>
                <PipelineVisualizer logs={logs} isProcessing={isProcessing} />
              </div>

              <div className="card-clean rounded-2xl bg-white border border-slate-200 h-64 flex flex-col overflow-hidden">
                <LogConsole logs={logs} />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
