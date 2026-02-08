import React, { useEffect, useState } from 'react';
import { Package, Plus, RefreshCw, Clock, Trash2 } from 'lucide-react';

export default function Sidebar({ onNewChat, onSelectRun, selectedRunId }) {
    const [runs, setRuns] = useState([]);
    const [loading, setLoading] = useState(false);

    const fetchRuns = async () => {
        try {
            setLoading(true);
            const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
            const res = await fetch(`${API_BASE}/runs`);
            const data = await res.json();
            if (data.runs) {
                setRuns(data.runs);
            }
        } catch (e) {
            console.error("Failed to fetch runs", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchRuns();
        const interval = setInterval(fetchRuns, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="w-64 bg-gray-900 border-r border-gray-800 text-gray-300 flex flex-col h-screen">
            {/* Header */}
            <div className="p-3 border-b border-gray-800 flex items-center gap-2">
                <Package className="w-5 h-5 text-blue-500" />
                <span className="font-bold text-white text-sm">SDLC Agent</span>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col p-2 overflow-hidden">
                {/* New Project Button */}
                <button
                    onClick={onNewChat}
                    className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-2 px-3 rounded-lg transition-colors font-medium text-sm mb-3 shadow-lg"
                >
                    <Plus className="w-4 h-4" /> New Project
                </button>

                {/* Section Header */}
                <div className="flex items-center justify-between text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2 px-1">
                    <span>Recent</span>
                    <button onClick={fetchRuns} className="hover:text-white transition-colors">
                        <RefreshCw className="w-3 h-3" />
                    </button>
                </div>

                {/* Sessions List */}
                <div className="flex-1 space-y-0.5 overflow-y-auto pr-1">
                    {runs.map((run) => (
                        <div
                            key={run.job_id}
                            className={`group flex items-center justify-between gap-1.5 px-2 py-1.5 rounded-md transition-all ${selectedRunId === run.job_id
                                ? 'bg-gray-800 text-white shadow-sm'
                                : 'hover:bg-gray-800/50'
                                }`}
                        >
                            {/* Session Info Button */}
                            <button
                                onClick={() => onSelectRun(run.job_id)}
                                className="grow text-left truncate flex items-center gap-1.5 overflow-hidden min-w-0"
                            >
                                <Clock className="w-3 h-3 shrink-0 opacity-40" />
                                <div className="flex flex-col overflow-hidden w-full min-w-0">
                                    <span className="truncate font-medium text-[11px] leading-tight">
                                        {run.prompt || "New Session"}
                                    </span>
                                    <div className="flex items-center gap-1.5 text-[9px] text-gray-500 mt-0.5">
                                        <span className="truncate">
                                            {run.started_at ? new Date(run.started_at).toLocaleDateString() : 'Now'}
                                        </span>
                                        <span
                                            className={`px-1 py-0.5 rounded-sm capitalize shrink-0 ${run.status === 'running'
                                                ? 'bg-blue-500/20 text-blue-400'
                                                : run.status === 'failed'
                                                    ? 'text-red-400'
                                                    : run.status === 'completed'
                                                        ? 'text-green-400'
                                                        : 'text-gray-400'
                                                }`}
                                        >
                                            {run.status}
                                        </span>
                                    </div>
                                </div>
                            </button>

                            {/* DELETE BUTTON - NUCLEAR OPTION WITH INLINE STYLES */}
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (confirm('Delete this session?')) {
                                        const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
                                        fetch(`${API_BASE}/runs/${run.job_id}`, { method: 'DELETE' })
                                            .then(() => fetchRuns());
                                    }
                                }}
                                style={{
                                    padding: '6px',
                                    color: '#ef4444',
                                    backgroundColor: 'transparent',
                                    border: '1px solid #ef4444',
                                    borderRadius: '4px',
                                    cursor: 'pointer',
                                    flexShrink: 0,
                                    opacity: 1,
                                    visibility: 'visible',
                                    display: 'block'
                                }}
                                title="Delete"
                            >
                                <Trash2 style={{ width: '14px', height: '14px' }} />
                            </button>
                        </div>
                    ))}
                    {runs.length === 0 && !loading && (
                        <div className="text-center p-3 text-gray-600 text-[10px] italic">
                            No builds yet.
                        </div>
                    )}
                </div>
            </div>

            {/* Footer */}
            <div className="mt-auto p-2 border-t border-gray-800 text-[10px] text-center text-gray-500">
                v2.0.0
            </div>
        </div>
    );
}
