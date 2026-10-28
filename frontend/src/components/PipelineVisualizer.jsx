import React from 'react';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';

const stages = [
  { id: 'upload', name: '1. Image Decoding & Tight Crop (Giải mã & Alignment 5%)', matchKey: 'Image Decoding' },
  { id: 'detect', name: '2. Face Detection (RetinaFace TF)', matchKey: 'Face Detection' },
  { id: 'extract', name: '3. Multimodal 544D (CLIP 512D + Prompt 7D + AU 25D)', matchKey: 'Feature Extraction' },
  { id: 'optimize', name: '4. EPSO Feature Reduction (544D → 289D)', matchKey: 'Feature Selection' },
  { id: 'classify', name: '5. Ensemble Classifier (ResMLP + XGBoost)', matchKey: 'Emotion Classification' }
];

export default function PipelineVisualizer({ logs = [], isProcessing }) {
  return (
    <div className="flex flex-col space-y-4 relative">
      {/* Connecting line */}
      <div className="absolute left-[15px] top-[24px] bottom-[24px] w-0.5 bg-slate-200 -z-10" />
      
      {stages.map((stage, idx) => {
        const logEntry = logs.find(l => l.stage && l.stage.includes(stage.matchKey));
        const isCompleted = Boolean(logEntry);
        const isCurrent = !isCompleted && isProcessing && (idx === 0 || logs.some(l => l.stage && l.stage.includes(stages[idx - 1].matchKey)));

        return (
          <div key={stage.id} className="flex items-start gap-4">
            <div className="mt-1 bg-white z-10">
              {isCompleted ? (
                <CheckCircle2 className="w-8 h-8 text-emerald-500" />
              ) : isCurrent ? (
                <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
              ) : (
                <Circle className="w-8 h-8 text-slate-300" />
              )}
            </div>
            <div className={`flex-1 ${!isCompleted && !isCurrent ? 'opacity-50' : ''}`}>
              <h3 className={`text-sm font-semibold ${isCurrent ? 'text-blue-600' : 'text-slate-800'}`}>{stage.name}</h3>
              <p className="text-xs text-slate-500 h-5 mt-0.5">
                {isCompleted && logEntry ? `Hoàn tất trong ${logEntry.latency_ms ?? logEntry.latency ?? 0}ms` : isCurrent ? 'Đang xử lý...' : 'Đang chờ'}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
