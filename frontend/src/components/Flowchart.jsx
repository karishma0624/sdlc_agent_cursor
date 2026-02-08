import React, { useEffect, useRef, useMemo, useState } from 'react';
import mermaid from 'mermaid';

export default function Flowchart({ chart }) {
    const ref = useRef(null);
    const [error, setError] = useState(null);
    // Stable ID to prevent re-rendering issues
    const id = useMemo(() => `mermaid-${Math.random().toString(36).substr(2, 9)}`, []);

    useEffect(() => {
        // Guard against double initialization in StrictMode
        if (mermaid.dataset?.initialized) return;

        try {
            // CRITICAL: Suppress default error reporting (The Bomb Icon)
            mermaid.parseError = (err, hash) => {
                console.debug('Mermaid parse error suppressed (Custom Handler):', err);
            };
            if (mermaid.mermaidAPI) {
                mermaid.mermaidAPI.parseError = (err, hash) => {
                    console.debug('Mermaid API parse error suppressed (Custom Handler):', err);
                };
            }

            mermaid.initialize({
                startOnLoad: false,
                theme: 'dark',
                securityLevel: 'loose',
                fontFamily: 'Inter, sans-serif',
                logLevel: 'error', // Reduce noise
            });
            mermaid.dataset = { initialized: true };
        } catch (e) {
            console.warn("Mermaid init failed (likely already initialized):", e);
        }
    }, []);

    useEffect(() => {
        if (chart && ref.current) {
            // Safety: If chart contains multiple diagrams, take the first one (Graph usually)
            let safeChart = chart;
            if (safeChart.includes("graph ") && safeChart.includes("sequenceDiagram")) {
                safeChart = safeChart.split("sequenceDiagram")[0].trim();
            }

            // Basic validation to prevent "Syntax error" popups from empty strings
            if (!safeChart.trim() || safeChart.length < 5) return;

            setError(null);

            // Clean out previous SVG to prevent duplicates if render fails mid-way
            ref.current.innerHTML = '';

            const renderDiagram = async () => {
                try {
                    // We must use a unique ID for every render attempt in some mismatched versions, 
                    // but v10 usually handles it. We pass the container ID.
                    const { svg } = await mermaid.render(id, safeChart);
                    if (ref.current) {
                        ref.current.innerHTML = svg || '<div class="text-slate-500 text-xs p-2">Diagram rendered empty.</div>';
                    }
                } catch (error) {
                    console.error("Mermaid Render Failed:", error);
                    setError("Diagram syntax error");
                    if (ref.current) {
                        ref.current.innerHTML = `<div class="text-red-400 text-[10px] p-4 text-center border border-red-900/50 bg-red-900/10 rounded font-mono">
                            Failed to render diagram. <br/>
                            <span class="opacity-50 text-[8px]">${error.message?.slice(0, 50)}...</span>
                        </div>`;
                    }
                }
            };

            renderDiagram();
        }
    }, [chart, id]);

    return (
        <div className="w-full overflow-x-auto p-4 bg-surface-dark rounded border border-border-dark flex justify-center min-h-[100px]">
            {/* We use a div for the ref. mermaid.render will put SVG string into it manually */}
            <div ref={ref} className="w-full flex justify-center" />
        </div>
    );
}
