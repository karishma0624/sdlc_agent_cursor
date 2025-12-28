import { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Clock, FileText, ChevronRight } from 'lucide-react';

export default function BuildStatus({ jobId }) {
    const [status, setStatus] = useState(null);
    const [report, setReport] = useState(null);
    const [polling, setPolling] = useState(true);

    useEffect(() => {
        if (!jobId) return;

        let intervalId;
        const fetchStatus = async () => {
            try {
                const res = await fetch(`/api/sdlc/status?job_id=${jobId}`);
                if (res.ok) {
                    const data = await res.json();
                    setStatus(data);

                    if (data.status === 'completed' || data.status === 'failed') {
                        setPolling(false);
                        if (data.status === 'completed') {
                            // Fetch final report
                            const repRes = await fetch(`/api/sdlc/report?job_id=${jobId}`);
                            if (repRes.ok) setReport(await repRes.json());
                        }
                    }
                }
            } catch (err) {
                console.error("Poll error", err);
            }
        };

        fetchStatus(); // Initial check
        if (polling) {
            intervalId = setInterval(fetchStatus, 2000);
        }

        return () => clearInterval(intervalId);
    }, [jobId, polling]);

    if (!jobId) return null;

    const isRunning = status?.status === 'running';
    const isCompleted = status?.status === 'completed';
    const isFailed = status?.status === 'failed';

    return (
        <div className="space-y-6">
            {/* Status Card */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-800">Build Status</h2>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-2
            ${isRunning ? 'bg-blue-100 text-blue-700' : ''}
            ${isCompleted ? 'bg-green-100 text-green-700' : ''}
            ${isFailed ? 'bg-red-100 text-red-700' : ''}
          `}>
                        {isRunning && <Clock size={16} className="animate-spin" />}
                        {isCompleted && <CheckCircle size={16} />}
                        {isFailed && <XCircle size={16} />}
                        <span className="uppercase">{status?.status || 'Waiting...'}</span>
                    </div>
                </div>

                {status && (
                    <div className="space-y-2 text-sm text-gray-600">
                        <div className="flex justify-between">
                            <span>Started:</span>
                            <span className="font-mono">{new Date(status.started_at).toLocaleTimeString()}</span>
                        </div>
                        {status.finished_at && (
                            <div className="flex justify-between">
                                <span>Finished:</span>
                                <span className="font-mono">{new Date(status.finished_at).toLocaleTimeString()}</span>
                            </div>
                        )}
                        {status.run_dir && (
                            <div className="mt-4 p-3 bg-gray-50 rounded border border-gray-100 break-all font-mono text-xs">
                                {status.run_dir}
                            </div>
                        )}
                        {status.error && (
                            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded border border-red-100">
                                {status.error}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Results Card */}
            {report && (
                <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 animate-in fade-in slide-in-from-bottom-4">
                    <h3 className="text-lg font-semibold mb-4 text-gray-800 flex items-center gap-2">
                        <FileText className="text-blue-600" />
                        Build Report
                    </h3>

                    <div className="grid gap-4 md:grid-cols-2">
                        {/* Artifacts Summary */}
                        <div className="border rounded-lg p-4">
                            <h4 className="font-medium text-gray-900 mb-2">Artifacts</h4>
                            <ul className="space-y-1 text-sm text-gray-600">
                                {Object.keys(report.artifacts || {}).map(k => (
                                    <li key={k} className="flex items-center gap-2">
                                        <ChevronRight size={14} className="text-gray-400" />
                                        <span className="capitalize">{k}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Commands */}
                        <div className="border rounded-lg p-4 bg-gray-900 text-gray-300 font-mono text-xs overflow-x-auto">
                            <h4 className="font-medium text-white mb-2 font-sans">Next Steps</h4>
                            {report.commands?.backend && (
                                <div className="mb-3">
                                    <div className="text-gray-500 mb-1"># Backend</div>
                                    {report.commands.backend.map((cmd, i) => (
                                        <div key={i}>{cmd}</div>
                                    ))}
                                </div>
                            )}
                            {report.commands?.frontend && (
                                <div>
                                    <div className="text-gray-500 mb-1"># Frontend</div>
                                    {report.commands.frontend.map((cmd, i) => (
                                        <div key={i}>{cmd}</div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="mt-4 text-center">
                        <p className="text-sm text-gray-500 mb-2">Project ready in output directory</p>
                        <button
                            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                            onClick={() => navigator.clipboard.writeText(report.run_dir)}
                        >
                            Copy Path to Clipboard
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
