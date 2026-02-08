import React, { useEffect, useState } from 'react';
import { Package, Plus, RefreshCw, Clock } from 'lucide-react';

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
        // Poll for history updates occasionally
        const interval = setInterval(fetchRuns, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="w-64 bg-gray-900 border-r border-gray-800 text-gray-300 flex flex-col h-screen">
            <div className="p-4 border-b border-gray-800 flex items-center gap-2">
                <Package className="w-6 h-6 text-blue-500" />
                <span className="font-bold text-white">SDLC Agent</span>
            </div>

            <div className="p-3">
                <button
                    onClick={onNewChat}
                    className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-lg transition-colors font-medium mb-4"
                >
                    <Plus className="w-4 h-4" /> New Project
                </button>

                <div className="flex items-center justify-between text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-1">
                    <span>History</span>
                    <button onClick={fetchRuns} className="hover:text-white"><RefreshCw className="w-3 h-3" /></button>
                </div>

                <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-180px)]">
                    {runs.map((run) => (
                        <button
                            key={run.job_id}
                            onClick={() => onSelectRun(run.job_id)}
                            className={`w-full text-left p-3 rounded-md text-sm truncate flex items-center gap-2 transition-colors ${selectedRunId === run.job_id
                                ? 'bg-gray-800 text-white'
                                : 'hover:bg-gray-800/50'
                                }`}
                        >
                            <Clock className="w-3 h-3 shrink-0 opacity-50" />
                            <span className="truncate">
                                <div className="flex flex-col overflow-hidden w-full">
                                    <span className="truncate font-medium">{run.prompt || "New Session"}</span>
                                    <div className="flex justify-between items-center text-[10px] text-gray-500 mt-1">
                                        <span>{run.started_at ? new Date(run.started_at).toLocaleDateString() : 'Just now'}</span>
                                        <span className={`px-1.5 py-0.5 rounded-sm capitalize ${run.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                                                run.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                                                    run.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                                                        'bg-gray-700 text-gray-400'
                                            }`}>
                                            {run.status}
                                        </span>
                                    </div>
                                </div>
                            </span>
                        </button>
                    ))}
                    {runs.length === 0 && !loading && (
                        <div className="text-center p-4 text-gray-600 text-xs italic">
                            No builds yet.
                        </div>
                    )}
                </div>
            </div>

            <div className="mt-auto p-4 border-t border-gray-800 text-xs text-center text-gray-500">
                v2.0.0 Agents
            </div>
        </div>
    );
}
