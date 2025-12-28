import React from 'react';
import { CheckCircle, Loader2, Circle, FileText, Map, Terminal, AlertCircle } from 'lucide-react';
import Flowchart from './Flowchart';

export default function LiveContext({ status }) {
    if (!status) return <div className="p-10 text-center text-slate-500">No active build context</div>;

    const phases = [
        { id: 'planning', label: '1. Planning', icon: FileText },
        { id: 'design', label: '2. System Design', icon: Map },
        { id: 'backend', label: '3. Backend Generation', icon: Terminal },
        { id: 'frontend', label: '4. Frontend Generation', icon: Terminal },
        { id: 'tests', label: '5. Testing', icon: CheckCircle },
        { id: 'deployment', label: '6. Deployment Prep', icon: CheckCircle },
    ];

    const currentPhaseIdx = phases.findIndex(p => p.id === status.current_phase);
    const progress = Math.max(5, ((currentPhaseIdx + (status.status === 'running' ? 0.5 : 1)) / phases.length) * 100);

    return (
        <div className="absolute inset-0 flex flex-col bg-background-dark overflow-y-auto p-4 space-y-4 pb-24">
            {/* Status Banner */}
            <div className="rounded-xl bg-gradient-to-r from-surface-dark to-slate-900 border border-border-dark p-4 shadow-lg shrink-0">
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <h2 className="text-sm font-semibold text-white">Live Execution</h2>
                        <p className="text-xs text-slate-400 mt-1">/usr/projects/{status.run_id?.substring(0, 8)}</p>
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
                {phases.map((phase, idx) => {
                    const phaseStatus = status.phases && status.phases[phase.id];
                    const isRunning = phaseStatus === 'running';
                    const isCompleted = phaseStatus === 'completed';
                    const isWaiting = !phaseStatus || phaseStatus === 'waiting';
                    const isFailed = phaseStatus === 'failed';

                    const providerInfo = status.providers_history?.find(h => h.phase === phase.id || (phase.id === 'tests' && h.phase === 'tests_gen'));

                    return (
                        <div key={phase.id} className={`rounded-lg border transition-all duration-300 overflow-hidden ${isRunning ? 'border-primary bg-surface-dark shadow-[0_0_15px_-3px_rgba(43,108,238,0.15)]' :
                            isCompleted ? 'border-success/30 bg-surface-dark' :
                                'border-border-dark bg-background-dark/50 opacity-60'
                            }`}>
                            <div className={`p-3 flex items-center justify-between ${isCompleted ? 'bg-success/5' : ''}`}>
                                <div className="flex items-center gap-3">
                                    <div className={`w-6 h-6 rounded-full flex items-center justify-center border ${isRunning ? 'border-primary/30 border-t-primary animate-spin' :
                                        isCompleted ? 'bg-success/20 border-success/30' :
                                            'border-slate-700 bg-slate-800'
                                        }`}>
                                        {isRunning ? null :
                                            isCompleted ? <CheckCircle className="w-3 h-3 text-success" /> :
                                                <span className="text-[10px] text-slate-500 font-mono">{idx + 1}</span>
                                        }
                                    </div>
                                    <div className="flex flex-col">
                                        <span className={`text-sm font-medium ${isCompleted ? 'text-slate-200' : 'text-slate-500'}`}>{phase.label}</span>
                                        {isRunning && <span className="text-[10px] text-primary animate-pulse">Executing...</span>}
                                        {isCompleted && providerInfo && (
                                            <span className="text-[9px] text-slate-400 flex items-center gap-1">
                                                Generated by <span className="text-secondary-400 font-mono">{providerInfo.provider}</span>
                                                <span className="opacity-50">({providerInfo.model})</span>
                                            </span>
                                        )}
                                    </div>
                                </div>
                                <span className={`text-[10px] font-medium px-2 py-0.5 rounded border capitalize ${isCompleted ? 'text-success bg-success/10 border-success/20' :
                                    isRunning ? 'text-primary bg-primary/10 border-primary/20' :
                                        'text-slate-600 bg-slate-800/50 border-transparent'
                                    }`}>
                                    {phaseStatus || 'Waiting'}
                                </span>
                            </div>

                            {/* Artifact Previews */}
                            {isCompleted && phase.id === 'design' && status.flowchart && (
                                <div className="border-t border-border-dark p-3 bg-black/20">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Map className="w-3 h-3 text-slate-500" />
                                        <span className="text-xs text-slate-400 font-medium">Architecture.mmd</span>
                                    </div>
                                    <Flowchart chart={status.flowchart} />
                                </div>
                            )}

                            {isCompleted && phase.id === 'planning' && (
                                <div className="border-t border-border-dark p-3 bg-black/20">
                                    <div className="flex items-center gap-2 mb-2">
                                        <FileText className="w-3 h-3 text-slate-500" />
                                        <span className="text-xs text-slate-400 font-medium">Planning.json</span>
                                    </div>
                                    <div className="text-[10px] font-mono text-slate-500 truncate">
                                        Requirements analysis completed and stored.
                                    </div>
                                </div>
                            )}
                            {isCompleted && phase.id === 'tests' && status.test_report && (
                                <div className="border-t border-border-dark p-3 bg-black/20 font-mono text-xs">
                                    <div className="flex gap-4 mb-1">
                                        <div className="flex items-center gap-1 text-green-400">
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
                            )}
                        </div>
                    );
                })}
            </div>
            {/* Live Terminal */}
            <div className="rounded-xl border border-border-dark bg-[#0d1117] overflow-hidden flex flex-col shrink-0 mt-4">
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
            </div>
        </div>
    );
}
