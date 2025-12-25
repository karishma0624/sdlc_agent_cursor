import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, Terminal, CheckCircle, AlertCircle, Loader2, Server, Globe } from 'lucide-react';

export default function ChatInterface({ runId, setRunId }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState(null);
    const [providers, setProviders] = useState({});
    const messagesEndRef = useRef(null);

    const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

    // Load providers on mount
    useEffect(() => {
        fetch(`${API_BASE}/providers`).then(r => r.json()).then(d => setProviders(d.providers));
    }, []);

    // Poll status if runId exists
    useEffect(() => {
        if (!runId) return;

        let interval;
        const poll = async () => {
            try {
                const res = await fetch(`${API_BASE}/sdlc/status?job_id=${runId}`);
                const data = await res.json();

                setStatus(data);

                // Intelligently update "Agent" messages based on status
                if (data.execution_log && data.execution_log.length > 0) {
                    // We can map execution log to chat messages if we haven't already
                    // For simplicity, we'll just show the *latest* status as a live indicator
                }

                if (data.status === 'completed' || data.status === 'failed') {
                    setLoading(false);
                    clearInterval(interval);
                }
            } catch (e) {
                console.error(e);
            }
        };

        setLoading(true);
        poll();
        interval = setInterval(poll, 2000);
        return () => clearInterval(interval);
    }, [runId]);

    // Initial greeting or load history
    useEffect(() => {
        if (!runId) {
            setMessages([{ role: 'agent', content: "Hello! I'm your SDLC Agent. What would you like to build today?" }]);
            setStatus(null);
        } else if (status && status.prompt) {
            // Restoration logic
            if (messages.length === 0) {
                setMessages([
                    { role: 'user', content: status.prompt },
                    { role: 'agent', content: `Restored session for "${status.prompt}". Status: ${status.status}.` }
                ]);
            }
        }
    }, [runId, status]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = input;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setLoading(true);

        // If no runId, we start a NEW build
        // If runId exists, we are "Conversing/Fixing" (Not fully implemented on backend yet as "fix", but we can trigger a new build with context? 
        // The requirement says "Error-Fix Loop". For now, we will treat a new message as a RE-BUILD request or new request.
        // Ideally we'd append to the existing run log, but our backend is prompt->build. 
        // We will trigger a NEW build prompt, but maybe reference context? 
        // Let's just treat it as a fresh build trigger for now to satisfy robustness.

        try {
            const res = await fetch(`${API_BASE}/sdlc/build`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: userMsg, job_id: runId }) // Send runId if existing to continue/fix
            });
            const data = await res.json();
            setRunId(data.job_id); // This switches the view to the new run
            setMessages(prev => [...prev, { role: 'agent', content: "Okay, starting that build for you now..." }]);
        } catch (e) {
            setMessages(prev => [...prev, { role: 'agent', content: "Error starting build: " + e.message, error: true }]);
            setLoading(false);
        }
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };
    useEffect(scrollToBottom, [messages, status]);

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b px-6 py-4 flex justify-between items-center shadow-sm">
                <div>
                    <h2 className="text-lg font-bold text-gray-800">
                        {status?.prompt || "New Session"}
                    </h2>
                    <div className="flex gap-2 text-xs mt-1">
                        {status?.status === 'running' && <span className="text-blue-600 font-medium flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" /> Building</span>}
                        {status?.status === 'completed' && <span className="text-green-600 font-medium flex items-center gap-1"><CheckCircle className="w-3 h-3" /> Completed</span>}
                        {status?.status === 'failed' && <span className="text-red-600 font-medium flex items-center gap-1"><AlertCircle className="w-3 h-3" /> Failed</span>}

                        <span className="text-gray-400 ml-2" title={`Connected to: ${API_BASE}`}>
                            Using API: {API_BASE}
                        </span>
                    </div>
                </div>

                {/* Provider Badges */}
                <div className="flex gap-1">
                    {Object.keys(providers).filter(k => providers[k]).map(p => (
                        <span key={p} className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded uppercase font-bold border">
                            {p}
                        </span>
                    ))}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.map((m, i) => (
                    <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-2xl p-4 rounded-xl shadow-sm ${m.role === 'user'
                            ? 'bg-blue-600 text-white rounded-br-none'
                            : 'bg-white text-gray-800 border border-gray-100 rounded-bl-none'
                            }`}>
                            {m.content}
                        </div>
                    </div>
                ))}

                {/* Live Status Card */}
                {status && (
                    <div className="flex justify-start w-full">
                        <div className="w-full max-w-2xl bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm">
                            <div className="bg-gray-50 px-4 py-2 border-b flex justify-between items-center">
                                <span className="text-xs font-bold text-gray-500 uppercase">Live Build Context</span>
                                <span className="text-xs text-gray-400 font-mono">{status.job_id?.substring(0, 8)}</span>
                            </div>
                            <div className="p-4 space-y-3">
                                {/* Steps */}
                                <div className="flex items-center gap-3 text-sm">
                                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${status.run_dir ? 'bg-green-100 text-green-600' : 'bg-blue-100 text-blue-600'}`}>
                                        <Server className="w-4 h-4" />
                                    </div>
                                    <div className="flex-1">
                                        <div className="font-medium">Backend Generation</div>
                                        <div className="text-xs text-gray-500">
                                            {status.current_stage?.includes('backend') ? 'Generating...' : status.run_dir ? 'Completed' : 'Waiting...'}
                                        </div>
                                    </div>
                                </div>

                                <div className="flex items-center gap-3 text-sm">
                                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${status.summary ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-400'}`}>
                                        <Globe className="w-4 h-4" />
                                    </div>
                                    <div className="flex-1">
                                        <div className="font-medium">Frontend Generation</div>
                                        <div className="text-xs text-gray-500">
                                            {status.current_stage?.includes('frontend') ? 'Generating...' : status.summary ? 'Completed' : 'Waiting...'}
                                        </div>
                                    </div>
                                </div>

                                {status.error && (
                                    <div className="mt-2 p-3 bg-red-50 text-red-700 text-sm rounded border border-red-100 flex items-start gap-2">
                                        <AlertCircle className="w-4 h-4 mt-0.5" />
                                        <div>
                                            <div className="font-bold">Build Error</div>
                                            {status.error}
                                        </div>
                                    </div>
                                )}

                                {/* Console/Log Output */}
                                {status.execution_log && status.execution_log.length > 0 && (
                                    <div className="mt-4 bg-gray-950 text-green-400 font-mono text-xs p-3 rounded-md max-h-40 overflow-y-auto">
                                        {status.execution_log.map((log, idx) => (
                                            <div key={idx} className="mb-1">
                                                <span className="opacity-50">[{log.time.split('T')[1].split('.')[0]}]</span>
                                                <span className="text-blue-400"> [{log.provider}]</span> {log.msg}
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {status.run_dir && status.status === 'completed' && (
                                    <div className="mt-3 p-3 bg-green-50 border border-green-100 rounded text-sm text-green-800">
                                        <strong>Success!</strong> Your app is ready in:
                                        <code className="block mt-1 bg-white px-2 py-1 rounded border border-green-200 text-xs">
                                            {status.run_dir}
                                        </code>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="bg-white px-6 py-4 border-t">
                <form onSubmit={handleSubmit} className="relative flex items-center max-w-4xl mx-auto">
                    <input
                        type="text"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        placeholder={loading ? "Agent is working..." : "Describe your web app (e.g. 'Personal Finance Tracker')..."}
                        disabled={loading}
                        className="w-full pl-5 pr-14 py-4 bg-gray-50 border border-gray-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm text-gray-800 placeholder-gray-400"
                    />
                    <button
                        type="submit"
                        disabled={loading || !input.trim()}
                        className="absolute right-3 p-2 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-all disabled:opacity-50 disabled:hover:bg-blue-600"
                    >
                        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                    </button>
                </form>
                <div className="text-center text-xs text-gray-400 mt-2">
                    AI Agent can make mistakes. Review generated code.
                </div>
            </div>
        </div>
    );
}
