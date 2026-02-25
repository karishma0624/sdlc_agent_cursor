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
            mermaid.initialize({
                startOnLoad: false,
                theme: 'dark',
                securityLevel: 'loose',
                fontFamily: 'Inter, sans-serif'
            });
            mermaid.dataset = { initialized: true };
        } catch (e) {
            console.warn("Mermaid init failed:", e);
        }
    }, []);

    useEffect(() => {
        let isMounted = true;
        if (chart && ref.current) {
            let safeChart = chart.trim();
            // Sometimes it comes with markdown blocks if string slicing failed
            if (safeChart.startsWith("```mermaid")) safeChart = safeChart.replace("```mermaid", "");
            if (safeChart.startsWith("```")) safeChart = safeChart.replace("```", "");
            if (safeChart.endsWith("```")) safeChart = safeChart.substring(0, safeChart.length - 3);
            safeChart = safeChart.trim();

            if (safeChart.includes("graph ") && safeChart.includes("sequenceDiagram")) {
                safeChart = safeChart.split("sequenceDiagram")[0].trim();
            }

            if (!safeChart || safeChart.length < 5) return;

            setError(null);
            ref.current.innerHTML = '<span class="text-xs text-slate-500 animate-pulse">Rendering diagram...</span>';

            const renderDiagram = async () => {
                try {
                    const { svg } = await mermaid.render(id, safeChart);
                    if (isMounted && ref.current) {
                        ref.current.innerHTML = svg;
                    }
                } catch (error) {
                    console.error("Mermaid Render Failed:", error);
                    // Force a cleanup of mermaid's temporary bomb SVG from the DOM to prevent styling bleed
                    const erroredSvg = document.getElementById(id);
                    if (erroredSvg) erroredSvg.remove();

                    if (isMounted && ref.current) {
                        setError("Diagram syntax error");
                        ref.current.innerHTML = `<div class="text-red-400 text-xs p-4 border border-red-900/50 bg-red-900/10 rounded overflow-auto max-w-full"><div class="font-bold mb-1">Failed to parse Flowchart</div><pre class="text-[10px] opacity-70">${safeChart}</pre></div>`;
                    }
                }
            };
            renderDiagram();
        }
        return () => { isMounted = false; };
    }, [chart, id]);

    return (
        <div className="w-full overflow-x-auto p-4 bg-surface-dark rounded border border-border-dark min-h-[100px] flex items-center justify-center">
            <div ref={ref} className="w-full h-full flex justify-center" />
        </div>
    );
}
