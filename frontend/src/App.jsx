import React, { useState, useEffect } from 'react';
import Layout from './components/Layout';
import ChatInterface from './components/ChatInterface';
import LiveContext from './components/LiveContext';

function App() {
  const [runId, setRunId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'build'
  const [status, setStatus] = useState(null);

  const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

  // Load history
  const loadSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/runs`);
      const data = await res.json();
      setSessions(data.runs || []);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => { loadSessions(); }, [runId]);

  // Create new session
  const handleNewSession = async (mode = 'auto') => {
    try {
      const res = await fetch(`${API_BASE}/sessions/new`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setRunId(data.job_id);
      setActiveTab('chat');
      loadSessions();
    } catch (e) {
      console.error(e);
      alert("Failed to create session. Is the backend server running? Check terminal.");
    }
  };

  // Poll for status to drive everything
  useEffect(() => {
    if (!runId) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/sdlc/status?job_id=${runId}`);
        const data = await res.json();
        setStatus(data);

        // Update Sessions list with new status to keep sidebar in sync
        setSessions(prev => prev.map(s =>
          s.job_id === runId ? { ...s, status: data.status, prompt: data.prompt || s.prompt } : s
        ));
      } catch (e) {
        console.error(e);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [runId]);

  return (
    <Layout
      onNewSession={handleNewSession}
      sessions={sessions}
      currentSessionId={runId}
      onSelectSession={setRunId}
    >
      <div className="flex flex-col h-full bg-background-dark w-full relative">

        {/* Mobile/Toggle Controller (Visible on ALL screens now to support Toggle) */}
        <div className="px-4 py-3 bg-background-dark z-30 border-b border-border-dark flex justify-center">
          <div className="bg-surface-dark p-1 rounded-lg flex relative w-full max-w-md">
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-all text-center ${activeTab === 'chat' ? 'bg-border-dark text-white shadow-sm' : 'text-slate-400 hover:text-white'
                }`}
            >
              Agent Chat
            </button>
            <button
              onClick={() => setActiveTab('build')}
              className={`flex-1 py-2 text-sm font-medium rounded-md transition-all text-center flex items-center justify-center gap-2 ${activeTab === 'build' ? 'bg-border-dark text-white shadow-sm' : 'text-slate-400 hover:text-white'
                }`}
            >
              Build Context
              {status?.status === 'running' && (
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
              )}
            </button>
          </div>
        </div>

        {/* Content Area - Tabbed View */}
        <div className="flex-1 overflow-hidden relative">

          {/* Chat Tab */}
          <div className={`
              absolute inset-0 transition-all duration-300 bg-surface-dark z-20 
              ${activeTab === 'chat' ? 'opacity-100 pointer-events-auto translate-x-0' : 'opacity-0 pointer-events-none -translate-x-4'}
          `}>
            <ChatInterface key={runId} runId={runId} setRunId={setRunId} status={status} />
          </div>

          {/* Build Context Tab */}
          <div className={`
              absolute inset-0 transition-all duration-300 bg-background-dark z-20
              ${activeTab === 'build' ? 'opacity-100 pointer-events-auto translate-x-0' : 'opacity-0 pointer-events-none translate-x-4'}
          `}>
            <LiveContext status={status} onSwitchTab={setActiveTab} />
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default App;
