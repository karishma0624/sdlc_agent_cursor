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
  const handleNewSession = async () => {
    const res = await fetch(`${API_BASE}/sessions/new`, { method: 'POST' });
    const data = await res.json();
    setRunId(data.job_id);
    setActiveTab('chat');
    loadSessions();
  };

  // Poll for status to drive everything
  useEffect(() => {
    if (!runId) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/sdlc/status?job_id=${runId}`);
        const data = await res.json();
        setStatus(data);

        // Auto-switch to build tab if running for the first time? 
        // No, user requested explicit tabs. But we can show notification badges.
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
      <div className="flex flex-col h-full bg-background-dark max-w-4xl mx-auto border-x border-border-dark shadow-2xl relative">

        {/* Mobile/Toggle Switch */}
        <div className="px-4 py-3 bg-background-dark z-30 border-b border-border-dark">
          <div className="bg-surface-dark p-1 rounded-lg flex relative">
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

        {/* Content Area */}
        <div className="flex-1 overflow-hidden relative">
          {/* Chat Tab - Always rendered but hidden via CSS/Transform if inactive to maintain state */}
          <div className={`absolute inset-0 transition-opacity duration-300 ${activeTab === 'chat' ? 'opacity-100 z-10 pointer-events-auto' : 'opacity-0 z-0 pointer-events-none'}`}>
            <ChatInterface runId={runId} setRunId={setRunId} status={status} />
          </div>

          {/* Build Context Tab */}
          <div className={`absolute inset-0 transition-opacity duration-300 ${activeTab === 'build' ? 'opacity-100 z-10 pointer-events-auto' : 'opacity-0 z-0 pointer-events-none'}`}>
            <LiveContext status={status} />
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default App;
