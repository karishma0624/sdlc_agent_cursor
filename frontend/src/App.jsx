import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';

function App() {
  const [selectedRunId, setSelectedRunId] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('job_id') || null;
  });

  const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

  // Auto-init session if missing
  React.useEffect(() => {
    if (!selectedRunId) {
      handleNewChat();
    }
  }, []);

  const handleNewChat = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions/new`, { method: 'POST' });
      const data = await res.json();
      if (data.job_id) {
        setSelectedRunId(data.job_id);
        const newUrl = `${window.location.pathname}?job_id=${data.job_id}`;
        window.history.pushState({ path: newUrl }, '', newUrl);
      }
    } catch (e) {
      console.error("Failed to create session", e);
    }
  };

  const handleSelectRun = (id) => {
    setSelectedRunId(id);
    window.history.pushState({}, '', `/?job_id=${id}`);
  };

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar
        onNewChat={handleNewChat}
        onSelectRun={handleSelectRun}
        selectedRunId={selectedRunId}
      />

      <div className="flex-1 h-full min-w-0">
        <ChatInterface
          key={selectedRunId}
          runId={selectedRunId}
          setRunId={handleSelectRun}
        />
      </div>
    </div>
  );
}

export default App;
