import React, { useRef, useEffect } from 'react';
import { Terminal } from 'lucide-react';

export default function LogConsole({ logs }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <>
      <div className="bg-gray-50 dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800 px-4 py-3 flex items-center gap-2">
        <Terminal className="w-4 h-4 text-gray-500" />
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Debug Log</h3>
      </div>
      <div className="flex-1 overflow-y-auto p-4 font-mono text-xs bg-gray-950 text-green-400">
        {logs.length === 0 ? (
          <p className="text-gray-600">Waiting for system input...</p>
        ) : (
          logs.map((log, i) => (
            <div key={i} className="mb-1 flex gap-3">
              <span className="text-gray-500 shrink-0">[{log.time}]</span>
              <span className="text-blue-400 shrink-0">[{log.stage}]</span>
              <span className="text-green-400">{log.message}</span>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </>
  );
}
