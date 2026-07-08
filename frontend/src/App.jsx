import React, { useState, useEffect } from 'react';
import { Moon, Sun, Upload, Camera, Activity, BrainCircuit } from 'lucide-react';
import PipelineVisualizer from './components/PipelineVisualizer';
import LogConsole from './components/LogConsole';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [logs, setLogs] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const toggleTheme = () => setDarkMode(!darkMode);

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

    // Log khởi tạo
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

      const response = await fetch('http://localhost:8000/analyze_emotion', {
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

      // Giả lập hiển thị tuần tự từng bước xử lý cho sinh động
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
          }, 400); // Trễ 400ms giữa mỗi block để tăng tính trải nghiệm
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
    <div className="min-h-screen bg-light dark:bg-dark text-dark dark:text-light font-sans transition-colors duration-300">
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-4 bg-lightSurface dark:bg-darkSurface shadow-sm">
        <div className="flex items-center gap-3">
          <BrainCircuit className="text-accent w-8 h-8" />
          <h1 className="text-xl font-semibold tracking-wide">EduFocus AI</h1>
        </div>
        <button 
          onClick={toggleTheme}
          className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors"
        >
          {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>
      </header>

      {/* Main Content */}
      <main className="p-8 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Input & Results */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-lightSurface dark:bg-darkSurface rounded-2xl p-6 shadow-sm border border-gray-100 dark:border-gray-800">
            <h2 className="text-lg font-medium mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5 text-accent" />
              Input Source
            </h2>
            
            <div 
              className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl p-8 flex flex-col items-center justify-center text-gray-500 hover:border-accent dark:hover:border-accent cursor-pointer transition-colors relative overflow-hidden h-48" 
              onClick={() => document.getElementById('fileInput').click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input 
                type="file" 
                accept="image/*" 
                className="hidden" 
                id="fileInput" 
                onChange={handleFileChange} 
              />
              {previewUrl ? (
                <img 
                  src={previewUrl} 
                  alt="Preview" 
                  className="absolute inset-0 w-full h-full object-cover"
                />
              ) : (
                <>
                  <Camera className="w-10 h-10 mb-3 text-gray-400" />
                  <p className="text-sm font-medium">Click to upload image</p>
                  <p className="text-xs mt-1">or drag and drop</p>
                </>
              )}
            </div>
            
            <button 
              onClick={analyzeImage}
              disabled={isProcessing || !selectedFile}
              className="w-full mt-4 py-3 rounded-xl bg-accent hover:bg-blue-600 text-white font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? 'Processing...' : 'Analyze Now'}
            </button>
          </div>

          {result && (
            <div className="space-y-6 animate-fade-in">
              {/* Overall result card */}
              <div className="bg-lightSurface dark:bg-darkSurface rounded-2xl p-6 shadow-sm border border-gray-100 dark:border-gray-800">
                <h2 className="text-lg font-medium mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-accentMint" />
                  Overall Result
                </h2>
                
                <div className="text-center py-4">
                  <div className="text-5xl font-bold text-accentMint mb-2">
                    {result.concentrationScore}%
                  </div>
                  <p className="text-sm text-gray-500 dark:text-gray-400 uppercase tracking-widest font-semibold">
                    Avg Concentration Level
                  </p>
                </div>
                
                <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-800">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">Dominant Emotion:</span>
                    <span className="font-semibold text-accent">{result.emotion}</span>
                  </div>
                </div>
              </div>

              {/* Detected faces gallery card */}
              <div className="bg-lightSurface dark:bg-darkSurface rounded-2xl p-6 shadow-sm border border-gray-100 dark:border-gray-800">
                <h2 className="text-lg font-medium mb-4 flex items-center gap-2">
                  <BrainCircuit className="w-5 h-5 text-accent" />
                  Detected Faces ({result.faces?.length || 0})
                </h2>

                <div className="space-y-4 max-h-96 overflow-y-auto pr-1">
                  {result.faces && result.faces.length > 0 ? (
                    result.faces.map((face) => (
                      <div key={face.id} className="flex items-center gap-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-xl border border-gray-100 dark:border-gray-800">
                        {/* Cropped face preview */}
                        <div className="w-16 h-16 rounded-lg overflow-hidden shrink-0 border border-gray-200 dark:border-gray-700 bg-gray-200 dark:bg-gray-800">
                          <img 
                            src={`data:image/jpeg;base64,${face.image}`} 
                            alt={`Face ${face.id + 1}`} 
                            className="w-full h-full object-cover"
                          />
                        </div>
                        
                        {/* Face details */}
                        <div className="flex-1 min-w-0">
                          <div className="flex justify-between items-start">
                            <h4 className="font-semibold text-sm text-gray-800 dark:text-gray-200">
                              Face #{face.id + 1}
                            </h4>
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                              face.concentration >= 70 
                                ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' 
                                : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                            }`}>
                              Focus: {face.concentration}%
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            Emotion: <span className="font-semibold text-accent">{face.emotion}</span> ({face.confidence}%)
                          </p>

                          {/* Phân phối tất cả cảm xúc */}
                          {face.emotions && face.emotions.length > 0 && (
                            <div className="mt-2.5 space-y-1 pt-2 border-t border-gray-100 dark:border-gray-800/80">
                              <span className="text-[10px] font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider block mb-1">
                                Phân phối cảm xúc
                              </span>
                              {face.emotions.map((item, i) => (
                                <div key={i} className="flex items-center gap-2 text-[11px]">
                                  <span className="w-14 truncate font-medium text-gray-600 dark:text-gray-400">{item.label}</span>
                                  <div className="flex-1 bg-gray-100 dark:bg-gray-800 rounded-full h-1 overflow-hidden">
                                    <div 
                                      className={`h-full rounded-full transition-all duration-500 ${
                                        i === 0 ? 'bg-accent' : 'bg-gray-300 dark:bg-gray-700'
                                      }`} 
                                      style={{ width: `${item.score}%` }}
                                    />
                                  </div>
                                  <span className="w-8 text-right text-gray-500 dark:text-gray-400 font-semibold">{item.score}%</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-gray-500 text-center py-4">No faces processed.</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Pipeline & Logs */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="bg-lightSurface dark:bg-darkSurface rounded-2xl p-6 shadow-sm border border-gray-100 dark:border-gray-800 flex-1">
            <h2 className="text-lg font-medium mb-6">AI Pipeline Status</h2>
            <PipelineVisualizer logs={logs} isProcessing={isProcessing} />
          </div>
          
          <div className="bg-lightSurface dark:bg-darkSurface rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800 h-64 flex flex-col overflow-hidden">
            <LogConsole logs={logs} />
          </div>
        </div>

      </main>
    </div>
  );
}

export default App;
