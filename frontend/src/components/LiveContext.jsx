import React from 'react';
import { CheckCircle, Loader2, Circle, FileText, Map, Terminal, AlertCircle } from 'lucide-react';
import Flowchart from './Flowchart';

const PreviewButton = ({ jobId }) => {
    const [loading, setLoading] = React.useState(false);

    const handlePreview = async () => {
        setLoading(true);
        try {
            const res = await fetch(`http://localhost:8000/sdlc/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId })
            });
            const data = await res.json();
            if (data.url) window.open(data.url, '_blank');
            else alert('Preview failed: ' + (data.error || 'Unknown error'));
        } catch (e) {
            alert('Preview error: ' + e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <button
            onClick={handlePreview}
            disabled={loading}
            style={{ backgroundColor: '#2563eb', padding: '2px 8px', borderRadius: '4px', border: '1px solid #3b82f6', color: 'white', fontSize: '10px', fontWeight: 'bold', marginLeft: '8px', cursor: loading ? 'not-allowed' : 'pointer', boxShadow: '0 0 10px rgba(59,130,246,0.5)', display: 'inline-flex', alignItems: 'center', gap: '4px', opacity: loading ? 0.7 : 1 }}
        >
            {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : '🚀'}
            {loading ? 'Starting...' : 'Preview App'}
        </button>
    );
};

export default function LiveContext({ status, onSwitchTab }) {
    if (!status) return <div className="p-10 text-center text-slate-500">No active build context</div>;

    const phases = [
        { id: 'requirements', label: '1. Requirements', icon: FileText },
        { id: 'planning', label: '2. Planning', icon: Map },
        { id: 'design', label: '3. System Design', icon: Map },
        { id: 'frontend', label: '4. Frontend Generation', icon: Terminal },
        { id: 'backend', label: '5. Backend Generation', icon: Terminal },
        { id: 'tests', label: '6. Testing (Disabled)', icon: CheckCircle },
        { id: 'deployment', label: '7. Deployment Prep (Disabled)', icon: CheckCircle },
    ];

    const currentPhaseIdx = phases.findIndex(p => p.id === status.current_phase);
    const progress = Math.max(5, ((currentPhaseIdx + (status.status === 'running' ? 0.5 : 1)) / phases.length) * 100);

    // Call to Action for Idle state checks
    // We render the full view even if idle, to match the "New Session" screenshot 
    // which shows the phases in "Waiting" state.

    return (
        <div className="absolute inset-0 flex flex-col bg-background-dark overflow-y-auto p-4 space-y-4 pb-24">
            {/* Status Banner */}
            <div className="rounded-xl bg-gradient-to-r from-surface-dark to-slate-900 border border-border-dark p-4 shadow-lg shrink-0">
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <h2 className="text-sm font-semibold text-white">Live Execution</h2>
                        <div className="flex items-center gap-2">
                            <p className="text-xs text-slate-400 mt-1">/usr/projects/{status.job_id?.substring(0, 8)}</p>
                            <button
                                onClick={() => {
                                    fetch('http://localhost:8000/open-folder', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ path: status.run_dir }) // Ensure run_dir is passed from status in main.py
                                    });
                                }}
                                className="ml-2 text-[10px] bg-slate-700 hover:bg-slate-600 text-white px-2 py-0.5 rounded border border-slate-600"
                            >
                                Open Output
                            </button>
                            {status.phases && status.phases.frontend === 'completed' && (
                                <PreviewButton jobId={status.job_id} />
                            )}
                        </div>
                    </div>
                    <span className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wide ${status.status === 'running' ? 'bg-primary/20 text-primary border border-primary/20 animate-pulse' :
                        status.status === 'completed' ? 'bg-green-500/20 text-green-400 border border-green-500/20' :
                            'bg-slate-700 text-slate-400'
                        }`}>
                        {status.status}
                    </span>
                </div>
                {/* Progress Bar */}
                <div className="w-full bg-slate-800 rounded-full h-1.5 mb-2 overflow-hidden">
                    <div className="bg-primary h-1.5 rounded-full transition-all duration-1000" style={{ width: `${progress}%` }}></div>
                </div>
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>Phase {currentPhaseIdx + 1}/{phases.length}</span>
                    <span>{status.current_phase}</span>
                </div>
            </div>

            {/* Phases List */}
            <div className="space-y-3">
                {
                    phases.map((phase, idx) => {
                        const phaseStatus = status.phases && status.phases[phase.id];
                        const isRunning = phaseStatus === 'running';
                        const isCompleted = phaseStatus === 'completed';
                        const isWaiting = !phaseStatus || phaseStatus === 'waiting';
                        const isFailed = phaseStatus === 'failed';

                        const providerInfo = status.providers_history?.find(h => h.phase === phase.id || (phase.id === 'tests' && h.phase === 'tests_gen'));

                        return (
                            <div key={phase.id} className={`rounded-lg border transition-all duration-300 overflow-hidden ${isRunning ? 'border-primary bg-surface-dark shadow-[0_0_15px_-3px_rgba(43,108,238,0.15)] relative z-10' :
                                isCompleted ? 'border-[#1e293b] bg-[#0f172a]' :
                                    'border-border-dark bg-background-dark/50 opacity-60'
                                }`}>
                                <div className={`p-4 flex items-center justify-between ${isCompleted ? 'bg-[#15803d]/10' : ''}`}>
                                    <div className="flex items-center gap-4">
                                        <div className={`w-8 h-8 rounded-full flex items-center justify-center border ${isRunning ? 'border-primary/30 border-t-primary animate-spin' :
                                            isCompleted ? 'bg-[#15803d]/20 border-[#15803d]' :
                                                'border-slate-700 bg-slate-800'
                                            }`}>
                                            {isRunning ? null :
                                                isCompleted ? <CheckCircle className="w-4 h-4 text-[#4ade80]" /> :
                                                    <span className="text-xs text-slate-500 font-mono">{idx + 1}</span>
                                            }
                                        </div>
                                        <div className="flex flex-col">
                                            <span className={`text-sm font-semibold ${isCompleted ? 'text-slate-200' : isRunning ? 'text-white' : 'text-slate-500'}`}>
                                                {phase.label}
                                            </span>
                                            {isRunning && <span className="text-[10px] text-primary animate-pulse font-mono mt-0.5">Executing...</span>}
                                            {isCompleted && providerInfo && (
                                                <span className="text-[10px] text-slate-400 flex items-center gap-1 mt-0.5 font-mono">
                                                    Generated by <span className="text-slate-300">{providerInfo.provider}</span>
                                                    <span className="text-slate-500">({providerInfo.model})</span>
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                    <span className={`text-[10px] font-bold px-3 py-1 rounded-full border capitalize tracking-wide ${isCompleted ? 'text-[#4ade80] bg-[#15803d]/20 border-[#15803d]/30 shadow-sm' :
                                        isRunning ? 'text-primary bg-primary/10 border-primary/20 shadow-[0_0_10px_-2px_rgba(59,130,246,0.5)]' :
                                            'text-slate-600 bg-slate-800/50 border-transparent'
                                        }`}>
                                        {phaseStatus || 'Waiting'}
                                    </span>
                                </div>

                                {/* Artifact Previews - Dark Box Design */}
                                {
                                    isCompleted && phase.id === 'planning' && (
                                        <div className="mx-4 mb-4 mt-2 p-3 bg-[#020617] border border-[#1e293b] rounded text-left">
                                            <div className="flex items-center gap-2 mb-1">
                                                <FileText className="w-3 h-3 text-slate-400" />
                                                <span className="text-xs text-slate-300 font-medium">Planning.json</span>
                                            </div>
                                            <div className="text-[10px] font-mono text-slate-500 truncate pl-5 mb-3">
                                                Requirements analysis and implementation roadmap generated.
                                            </div>

                                            {status.flowchart && (
                                                <div className="mt-4 pt-4 border-t border-[#1e293b]">
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <Map className="w-3 h-3 text-slate-400" />
                                                        <span className="text-xs text-slate-300 font-medium">Architecture Flowchart</span>
                                                    </div>
                                                    <Flowchart chart={status.flowchart} />
                                                </div>
                                            )}
                                        </div>
                                    )
                                }

                                {
                                    isCompleted && phase.id === 'design' && (
                                        <div className="mx-4 mb-4 mt-2 p-3 bg-[#020617] border border-[#1e293b] rounded text-left">
                                            <div className="flex items-center gap-2 mb-1">
                                                <Map className="w-3 h-3 text-slate-400" />
                                                <span className="text-xs text-slate-300 font-medium">System Design Context</span>
                                            </div>
                                            <div className="text-[10px] font-mono text-slate-500 truncate pl-5">
                                                UI Architecture and components prepared for Frontend Generation.
                                            </div>
                                        </div>
                                    )
                                }

                                {
                                    isCompleted && phase.id === 'backend' && (
                                        <div className="mx-4 mb-4 mt-2 p-3 bg-[#020617] border border-[#1e293b] rounded text-left">
                                            <div className="flex items-center gap-2 mb-1">
                                                <Terminal className="w-3 h-3 text-slate-400" />
                                                <span className="text-xs text-slate-300 font-medium">Backend Generated</span>
                                            </div>
                                            <div className="text-[10px] font-mono text-slate-500 truncate pl-5">
                                                FastAPI/Python structure created.
                                            </div>
                                        </div>
                                    )
                                }

                                {
                                    isCompleted && phase.id === 'tests' && status.test_report && (
                                        <div className="mx-4 mb-4 mt-2 p-3 bg-[#020617] border border-[#1e293b] rounded text-left font-mono text-xs">
                                            <div className="flex gap-4 mb-1">
                                                <div className="flex items-center gap-1 text-[#4ade80]">
                                                    <CheckCircle className="w-3 h-3" />
                                                    <span>Passed: {status.test_report.passed}</span>
                                                </div>
                                                {status.test_report.failed > 0 && (
                                                    <div className="flex items-center gap-1 text-red-400">
                                                        <AlertCircle className="w-3 h-3" />
                                                        <span>Failed: {status.test_report.failed}</span>
                                                    </div>
                                                )}
                                            </div>
                                            {status.test_report.error && (
                                                <div className="text-error mt-1">{status.test_report.error}</div>
                                            )}
                                        </div>
                                    )
                                }
                            </div>
                        );
                    })
                }
            </div >
            {/* Live Terminal */}
            < div className="rounded-xl border border-border-dark bg-[#0d1117] overflow-hidden flex flex-col shrink-0 mt-4" >
                <div className="flex items-center justify-between px-4 py-2 border-b border-border-dark bg-surface-dark">
                    <div className="flex items-center gap-2">
                        <Terminal className="w-4 h-4 text-slate-400" />
                        <span className="text-xs font-medium text-slate-300">Terminal Output</span>
                    </div>
                    {status.status === 'running' && <Loader2 className="w-3 h-3 text-primary animate-spin" />}
                </div>
                <div className="p-4 h-48 overflow-y-auto font-mono text-[10px] space-y-1 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                    {(status.execution_log || []).map((log, i) => (
                        <div key={i} className="flex gap-3 text-slate-400">
                            <span className="text-slate-600 shrink-0">{log.time.split('T')[1].split('.')[0]}</span>
                            <span className={`${log.status === 'failed' ? 'text-error' :
                                log.phase === 'system' ? 'text-blue-400' :
                                    'text-slate-300'
                                }`}>
                                <span className="opacity-50 mr-2">[{log.phase}]</span>
                                {log.message || log.msg}
                            </span>
                        </div>
                    ))}
                    {!status.execution_log?.length && (
                        <div className="text-slate-600 italic">Waiting for logs...</div>
                    )}
                </div>
            </div >
        </div >
    );
}
