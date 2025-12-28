import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, Terminal, CheckCircle, AlertCircle, Loader2, Server, Globe, Map, ChevronRight } from 'lucide-react';
import Flowchart from './Flowchart';

export default function ChatInterface({ runId, setRunId, status }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

    // Sync Messages
    useEffect(() => {
        if (status && status.messages) {
            // Simple sync: if we have more messages on server, update.
            // In real app we'd merge smartly.
            if (status.messages.length >= messages.length) {
                setMessages(status.messages);
            }
        }
    }, [status]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = input;
        setInput('');

        // Optimistic update
        setMessages(prev => [...prev, { role: 'user', content: userMsg, timestamp: new Date().toISOString() }]);
        setLoading(true);

        try {
            const isFirstBuild = status?.status === 'idle' || !status?.run_id || status?.prompt === 'New Session';
            const endpoint = isFirstBuild ? `${API_BASE}/sdlc/build` : `${API_BASE}/chat`;
            const payload = isFirstBuild ? { prompt: userMsg, job_id: runId } : { message: userMsg, job_id: runId };

            await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            // Rely on polling in App.js to pick up response
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    if (!runId) return (
        <div className="flex items-center justify-center h-full text-slate-500">
            Select or create a new project to start.
        </div>
    );

    return (
        <div className="flex flex-col h-full bg-background-dark">
            <div className="flex-1 overflow-y-auto p-4 space-y-6 pb-20 scroll-smooth">
                {/* Date Separator */}
                <div className="flex justify-center">
                    <span className="text-[10px] font-mono text-slate-500 bg-surface-dark px-3 py-1 rounded-full border border-border-dark">
                        Today
                    </span>
                </div>

                {messages.map((m, i) => (
                    <div key={i} className={`flex gap-3 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        {/* Avatar */}
                        <div className={`w-8 h-8 rounded-lg shrink-0 flex items-center justify-center shadow-lg ${m.role === 'agent'
                            ? 'bg-gradient-to-br from-primary to-blue-600 shadow-primary/20'
                            : 'bg-slate-700'
                            }`}>
                            {m.role === 'agent' ? <Bot className="w-4 h-4 text-white" /> : <span className="text-xs text-white font-bold">U</span>}
                        </div>

                        <div className={`flex flex-col gap-1 max-w-[85%] ${m.role === 'user' ? 'items-end' : ''}`}>
                            <span className="text-xs font-bold text-slate-400 ml-1">{m.role === 'agent' ? 'SDLC Agent' : 'You'}</span>
                            <div className={`p-3.5 rounded-2xl text-sm leading-relaxed shadow-sm ${m.role === 'agent'
                                ? 'bg-surface-dark border border-border-dark text-slate-200 rounded-tl-sm'
                                : 'bg-primary text-white rounded-tr-sm shadow-primary/10'
                                }`}>
                                {m.content}
                            </div>
                        </div>
                    </div>
                ))}

                {/* Live Activity Indicator inside Chat (Bridge) */}
                {status && status.status === 'running' && (
                    <div className="flex gap-3 animate-fade-in-up">
                        <div className="w-8 h-8 rounded-lg bg-transparent shrink-0"></div>
                        <div className="flex flex-col gap-1 max-w-[85%]">
                            <div className="bg-background-dark border border-primary/40 rounded-lg p-3 flex items-center justify-between cursor-pointer hover:bg-surface-dark transition-colors group">
                                <div className="flex items-center gap-3">
                                    <div className="relative w-5 h-5">
                                        <div className="absolute inset-0 border-2 border-primary/30 rounded-full"></div>
                                        <div className="absolute inset-0 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                                    </div>
                                    <div className="flex flex-col">
                                        <span className="text-xs font-bold text-white group-hover:text-primary transition-colors">Build in progress...</span>
                                        <span className="text-[10px] text-slate-400">Current: {status.current_phase || 'Initializing...'}</span>
                                    </div>
                                </div>
                                <ChevronRight className="w-4 h-4 text-slate-500" />
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-3 bg-background-dark border-t border-border-dark w-full">
                <form
                    onSubmit={handleSubmit}
                    className="relative flex items-end gap-2 bg-surface-dark border border-border-dark rounded-xl p-2 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/50 transition-all shadow-lg"
                >
                    <textarea
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={e => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSubmit(e);
                            }
                        }}
                        placeholder="Describe your web app..."
                        className="w-full bg-transparent text-sm text-slate-300 placeholder:text-slate-500 border-0 focus:ring-0 resize-none py-2.5 min-h-[44px] max-h-32 disabled:opacity-50"
                        disabled={loading}
                        rows={1}
                        style={{ overflow: 'hidden', height: 'auto' }}
                        onInput={e => {
                            e.target.style.height = 'auto';
                            e.target.style.height = e.target.scrollHeight + 'px';
                        }}
                    />
                    <button
                        type="submit"
                        disabled={loading || !input.trim() || (status && status.status === 'running')}
                        className="p-2 mb-0.5 bg-primary hover:bg-primary-dark text-white rounded-lg transition-all shadow-sm disabled:bg-slate-700 disabled:text-slate-400 shrink-0"
                    >
                        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                    </button>
                </form>
                {status && status.status === 'running' && (
                    <div className="text-center mt-2 animate-pulse">
                        <span className="text-[10px] text-primary flex items-center justify-center gap-1 font-medium">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Autonomous Agent is building your app...
                        </span>
                    </div>
                )}
                {/* Error Toast (Simple) */}
                {status && status.error && (
                    <div className="mt-2 text-center">
                        <span className="text-[10px] text-error flex items-center justify-center gap-1">
                            <AlertCircle className="w-3 h-3" />
                            {status.error}
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
}
