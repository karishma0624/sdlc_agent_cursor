import React, { useState, useEffect } from 'react';
import { Menu, Plus, MessageSquare, Clock, Settings, User, Bot, Trash2 } from 'lucide-react';

export default function Layout({ children, onNewSession, sessions, currentSessionId, onSelectSession }) {
    const [sidebarOpen, setSidebarOpen] = useState(true);

    return (
        <div className="flex h-screen w-screen bg-background-dark text-slate-100 font-display overflow-hidden m-0 p-0" style={{ maxWidth: '100vw', width: '100vw' }}>
            {/* Sidebar */}
            <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-background-dark flex flex-col shrink-0 overflow-hidden`}>
                <div className="p-5 border-b border-border-dark flex items-center gap-2">
                    <div className="w-8 h-8 rounded bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center shrink-0">
                        <Bot className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-bold text-lg tracking-tight text-white whitespace-nowrap">SDLC Agent</span>
                </div>

                <div className="p-2">
                    <button
                        onClick={onNewSession}
                        className="w-full flex items-center justify-center gap-2 bg-primary hover:bg-primary-dark text-white py-2 px-3 rounded-lg font-medium transition-all shadow-lg shadow-primary/20 whitespace-nowrap text-sm"
                    >
                        <Plus className="w-4 h-4" />
                        New Project
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
                    <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2 px-1">Recent</div>
                    {sessions.map(session => (
                        <div
                            key={session.job_id}
                            className={`group flex items-center justify-between gap-1.5 px-2 py-1.5 rounded-lg transition-colors ${currentSessionId === session.job_id
                                ? 'bg-surface-dark border border-primary/30'
                                : 'border border-transparent hover:bg-surface-dark hover:border-border-dark'
                                }`}
                        >
                            <button
                                onClick={() => onSelectSession(session.job_id)}
                                className="grow flex flex-col gap-0.5 min-w-0 text-left"
                            >
                                <div className="flex justify-between items-start gap-2">
                                    <span className="font-medium text-slate-300 text-[11px] truncate">{session.prompt || "New Session"}</span>
                                    <span className={`px-1 py-0.5 rounded text-[9px] font-mono capitalize shrink-0 ${session.status === 'completed' ? 'bg-green-500/10 text-green-400' :
                                        session.status === 'failed' ? 'bg-red-500/10 text-red-400' :
                                            'bg-primary/20 text-primary-300'
                                        }`}>
                                        {session.status}
                                    </span>
                                </div>
                                <span className="text-[9px] text-slate-500 truncate">{session.started_at?.split('T')[0]}</span>
                            </button>
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (confirm('Delete this session?')) {
                                        fetch(`http://localhost:8000/runs/${session.job_id}`, { method: 'DELETE' })
                                            .then(() => window.location.reload());
                                    }
                                }}
                                style={{
                                    padding: '6px',
                                    color: '#ef4444',
                                    backgroundColor: 'transparent',
                                    border: '1px solid #ef4444',
                                    borderRadius: '4px',
                                    cursor: 'pointer',
                                    flexShrink: 0
                                }}
                                title="Delete"
                            >
                                <Trash2 style={{ width: '14px', height: '14px' }} />
                            </button>
                        </div>
                    ))}
                </div>

                <div className="p-4 border-t border-border-dark">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center">
                            <User className="w-4 h-4 text-slate-400" />
                        </div>
                        <div className="flex flex-col overflow-hidden">
                            <span className="text-sm font-medium text-white truncate">Dev User</span>
                            <span className="text-xs text-slate-400">Pro Plan</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0">
                <header className="h-16 shrink-0 b  g-background-dark/95 backdrop-blur-md flex items-center justify-between px-4 sticky top-0 z-40">
                    <div className="flex items-center gap-3">
                        <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 -ml-2 rounded-lg text-slate-400 hover:text-white hover:bg-surface-dark transition-colors">
                            <Menu className="w-5 h-5" />
                        </button>
                        <div className="flex flex-col">
                            <h1 className="text-sm font-bold text-white tracking-wide truncate max-w-[200px] md:max-w-md">
                                {sessions.find(s => s.job_id === currentSessionId)?.prompt || "SDLC Workspace"}
                            </h1>
                            <span className="text-[10px] font-mono text-primary uppercase tracking-wider">
                                Build #{currentSessionId?.substring(0, 6)}
                            </span>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1.5 px-2 py-1 bg-surface-dark border border-border-dark rounded-full">
                            <div className="w-2 h-2 rounded-full bg-success animate-pulse"></div>
                            <span className="text-[10px] font-bold text-slate-300">Orchestrator Online</span>
                        </div>
                    </div>
                </header>

                <main className="flex-1 overflow-hidden relative">
                    {children}
                </main>
            </div>
        </div>
    );
}
