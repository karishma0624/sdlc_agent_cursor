import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

export default function Flowchart({ chart }) {
    const ref = useRef(null);
    const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;

    useEffect(() => {
        mermaid.initialize({
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            fontFamily: 'Inter, sans-serif'
        });
    }, []);

    useEffect(() => {
        if (chart && ref.current) {
            mermaid.render(id, chart).then((result) => {
                ref.current.innerHTML = result.svg;
            }).catch(e => {
                console.warn("Mermaid render error:", e);
                ref.current.innerHTML = `<div class="text-slate-500 text-[10px] p-4 text-center border border-dashed border-slate-700 rounded">Analysis Diagram (Rendering...)</div>`;
            });
        }
    }, [chart, id]);

    return (
        <div className="w-full overflow-x-auto p-4 bg-surface-dark rounded border border-border-dark flex justify-center">
            <div ref={ref} />
        </div>
    );
}
