import os
import json
from services.adapters import call_mermaid

def execute_design_phase(planning_content: str, run_dir: str) -> None:
    """
    PHASE 4: DESIGN (Mermaid ONLY)
    Generates Architecture, component, and flow diagrams using Mermaid.js syntax.
    """
    
    # 1. Architecture Diagram
    arch_prompt = (
        "Create a High-Level System Architecture Diagram using Mermaid 'graph TD'.\n"
        "Include: User, Frontend (React), Backend (Future Python API), Database (Future SQLite).\n"
        "Show data flow direction."
    )
    arch_mmd = call_mermaid(arch_prompt, {"context": planning_content[:1000]})
    _save_mmd(run_dir, "architecture.mmd", arch_mmd)
    
    # 2. Component Diagram
    comp_prompt = (
        "Create a React Component Hierarchy Diagram using Mermaid 'graph TD'.\n"
        "Based on the Planning Doc, show the parent-child relationships of components.\n"
        "e.g. App -> Layout -> Dashboard"
    )
    comp_mmd = call_mermaid(comp_prompt, {"context": planning_content[:1000]})
    _save_mmd(run_dir, "components.mmd", comp_mmd)
    
    # 3. Data Flow Diagram
    flow_prompt = (
        "Create a Data Flow Diagram using Mermaid 'sequenceDiagram'.\n"
        "Show a typical user interaction (e.g. Login -> Dashboard Load -> Data Fetch)."
    )
    flow_mmd = call_mermaid(flow_prompt, {"context": planning_content[:1000]})
    _save_mmd(run_dir, "flow.mmd", flow_mmd)


def _save_mmd(run_dir: str, filename: str, content: str):
    # Sanitize content
    lines = content.split('\n')
    clean_lines = []
    in_block = False
    for line in lines:
        if "```" in line:
            in_block = not in_block
            continue
        clean_lines.append(line)
    
    final_content = "\n".join(clean_lines).strip()
    if not final_content.startswith("graph ") and not final_content.startswith("sequenceDiagram"):
        # Fallback check
        if "graph TD" in final_content or "sequenceDiagram" in final_content:
            pass 
        else:
            final_content = "graph TD;\nError[Invalid Mermaid Output];"

    path = os.path.join(run_dir, "design")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "w", encoding="utf-8") as f:
        f.write(final_content)
