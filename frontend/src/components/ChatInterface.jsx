import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, Terminal, CheckCircle, AlertCircle, Loader2, Server, Globe, Map, ChevronRight, Paperclip, X } from 'lucide-react';
import Flowchart from './Flowchart';

export default function ChatInterface({ runId, setRunId, status }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [attachments, setAttachments] = useState([]);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);

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
        const currentAttachments = [...attachments];

        setInput('');
        setAttachments([]);

        // Optimistic update
        setMessages(prev => [...prev, {
            role: 'user',
            content: userMsg,
            attachments: currentAttachments,
            timestamp: new Date().toISOString()
        }]);
        setLoading(true);

        try {
            const isFirstBuild = status?.status === 'idle' || !status?.run_id || status?.prompt === 'New Session';
            const endpoint = isFirstBuild ? `${API_BASE}/sdlc/build` : `${API_BASE}/chat`;

            // Build payload
            let payload = { job_id: runId };
            if (isFirstBuild) {
                payload.prompt = userMsg;
                // Note: First build usually implies prompt-only. 
                // We'll append attachments description to prompt if needed or ignore for V1 simplicity
            } else {
                payload.message = userMsg;
                payload.attachments = currentAttachments;
            }

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

    if (!runId && messages.length === 0) return (
        <div className="flex items-center justify-center h-full text-slate-500 flex-col gap-4">
            <Bot className="w-12 h-12 opacity-50" />
            <span>Select or create a new project to start.</span>
        </div>
    );

    // Auto-fix for lost jobs if we have no messages but a runId that failed
    if (status?.status === 'not_found') {
        return (
            <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-4 p-8 text-center">
                <AlertCircle className="w-12 h-12 text-error" />
                <h3 className="text-lg font-bold text-white">Session Not Found</h3>
                <p className="max-w-md">This session seems to have been deleted or the server was restarted without persistence. Please create a new session.</p>
                <button
                    onClick={() => setRunId(null)}
                    className="px-4 py-2 bg-primary text-white rounded hover:bg-primary-dark transition-colors"
                >
                    Go Back
                </button>
            </div>
        );
    }

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

                        <div className={`flex flex-col gap-1 max-w-full ${m.role === 'user' ? 'items-end' : ''}`}>
                            <span className="text-xs font-bold text-slate-400 ml-1">{m.role === 'agent' ? 'SDLC Agent' : 'You'}</span>
                            <div className={`p-3.5 rounded-2xl text-sm leading-relaxed shadow-sm ${m.role === 'agent'
                                ? 'bg-surface-dark border border-border-dark text-slate-200 rounded-tl-sm'
                                : 'bg-primary text-white rounded-tr-sm shadow-primary/10'
                                }`}>
                                {m.content}
                                {m.attachments && m.attachments.length > 0 && (
                                    <div className="mt-2 flex flex-wrap gap-2">
                                        {m.attachments.map((att, idx) => (
                                            <div key={idx} className="relative group border border-white/20 rounded overflow-hidden">
                                                {att.type.startsWith('image/') ? (
                                                    <img src={att.content} alt={att.name} className="h-20 w-auto object-cover" />
                                                ) : (
                                                    <div className="h-20 w-20 flex items-center justify-center bg-slate-800 text-xs p-2 text-center text-slate-300">
                                                        {att.name}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                )}
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
                {attachments.length > 0 && (
                    <div className="flex gap-2 mb-2 px-2">
                        {attachments.map((att, i) => (
                            <div key={i} className="relative bg-surface-dark border border-slate-700 rounded p-1 flex items-center gap-2">
                                <span className="text-[10px] text-slate-300 max-w-[100px] truncate">{att.name}</span>
                                <button onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))} className="text-slate-500 hover:text-white">
                                    <X className="w-3 h-3" />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
                <form
                    onSubmit={handleSubmit}
                    className="relative flex items-end gap-2 bg-surface-dark border border-border-dark rounded-xl p-2 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/50 transition-all shadow-lg"
                >
                    <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="p-2 mb-0.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
                    >
                        <Paperclip className="w-5 h-5" />
                    </button>
                    <input
                        type="file"
                        ref={fileInputRef}
                        className="hidden"
                        multiple
                        onChange={(e) => {
                            const files = Array.from(e.target.files);
                            files.forEach(file => {
                                const reader = new FileReader();
                                reader.onloadend = () => {
                                    setAttachments(prev => [...prev, {
                                        name: file.name,
                                        type: file.type,
                                        content: reader.result
                                    }]);
                                };
                                reader.readAsDataURL(file);
                            });
                            e.target.value = null; // reset
                        }}
                    />
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
                {status && status.error && !status.error.includes("Syntax error") && (
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
