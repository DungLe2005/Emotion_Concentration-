import React from 'react';
import { CheckCircle2, Circle, Loader2 } from 'lucide-react';

const stages = [
  { id: 'upload', name: 'Image Decoding' },
  { id: 'detect', name: 'Face Detection (RetinaFace)' },
  { id: 'extract', name: 'Feature Extraction (CLIVP-FER)' },
  { id: 'optimize', name: 'Feature Selection (EPSO)' },
  { id: 'classify', name: 'Emotion Classification (MLP)' }
];

export default function PipelineVisualizer({ logs, isProcessing }) {
  const completedStages = logs.map(l => l.stage);
  const currentStageIndex = isProcessing ? completedStages.length : -1;

  return (
    <div className="flex flex-col space-y-4 relative">
      {/* Connecting line */}
      <div className="absolute left-[15px] top-[24px] bottom-[24px] w-0.5 bg-gray-200 dark:bg-gray-800 -z-10" />
      
      {stages.map((stage, idx) => {
        const isCompleted = completedStages.includes(stage.name);
        const isCurrent = idx === currentStageIndex && isProcessing;
        const logEntry = logs.find(l => l.stage === stage.name);

        return (
          <div key={stage.id} className="flex items-start gap-4">
            <div className="mt-1 bg-lightSurface dark:bg-darkSurface z-10">
              {isCompleted ? (
                <CheckCircle2 className="w-8 h-8 text-accentMint" />
              ) : isCurrent ? (
                <Loader2 className="w-8 h-8 text-accent animate-spin" />
              ) : (
                <Circle className="w-8 h-8 text-gray-300 dark:text-gray-700" />
              )}
            </div>
            <div className={`flex-1 ${!isCompleted && !isCurrent ? 'opacity-50' : ''}`}>
              <h3 className={`font-medium ${isCurrent ? 'text-accent' : ''}`}>{stage.name}</h3>
              <p className="text-sm text-gray-500 h-5">
                {isCompleted && logEntry ? `Completed in ${logEntry.latency_ms ?? logEntry.latency}ms` : isCurrent ? 'Processing...' : 'Waiting'}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
