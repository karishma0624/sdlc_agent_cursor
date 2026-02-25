import os
from services.adapters import call_mistral, call_gemini_text

def execute_planning_phase(requirements_content: str, run_dir: str) -> str:
    """
    PHASE 2: PLANNING (Mistral ONLY)
    Generates a professional implementation roadmap/engineering blueprint.
    """
    
    system_prompt = (
        "You are a Principal Software Architect.\n"
        "Based on the Requirements provided below, generate a detailed Engineering Implementation Plan.\n"
        "Output Format: Markdown.\n"
        "This is NOT high-level fluff. It must be actionable for developers.\n\n"
        "MUST INCLUDE THE FOLLOW SECTIONS:\n"
        "1. Technical Stack Selection (Be specific)\n"
        "2. Chosen Frontend Architecture\n"
        "3. Folder Structure Plan (ASCII tree preferred)\n"
        "4. Component Breakdown (List of React components + Props)\n"
        "5. State Management Strategy\n"
        "6. UI Sections Mapping\n"
        "7. Data Flow & Interaction Logic\n"
        "8. Milestone-based Implementation Plan\n"
        "9. Risk Assessment\n"
        "10. Flowchart (CRITICAL: MUST include a Mermaid.js `graph TD` architecture/flow diagram enclosed in ```mermaid ... ``` code blocks. Do not skip this!)\n\n"
        "Requirements Context:\n"
    )
    
    full_prompt = system_prompt + requirements_content
    
    try:
        content = call_mistral(full_prompt)
        # Validation: Check for key sections
        required_keywords = ["Stack", "Folder Structure", "Component", "State", "Milestone"]
        missing = [k for k in required_keywords if k.lower() not in content.lower()]
        
        if missing:
            # Strict mode: fail if very short or completely off.
            if len(content) < 200:
                raise ValueError("Planning output too short/invalid.")
    except Exception as e:
        print(f"Mistral failed in Planning Phase: {e}. Falling back to Gemini...")
        content = call_gemini_text(full_prompt, model="gemini-1.5-pro")
        if content.startswith("Error:"):
            raise RuntimeError(f"Planning Phase Failed (Both APIs): {content}")
        if len(content) < 200:
            raise RuntimeError("Planning Phase Failed: generated invalid/short planning content.")
            
    # Save Artifact
    path = os.path.join(run_dir, "planning.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        
    # Extract Flowchart to display in UI during planning phase
    flowchart_content = ""
    in_mermaid = False
    lines = content.split('\n')
    for line in lines:
        if "```mermaid" in line:
            in_mermaid = True
            continue
        elif "```" in line and in_mermaid:
            in_mermaid = False
            break
        if in_mermaid:
            flowchart_content += line + "\n"
    
    if flowchart_content.strip():
        flow_path = os.path.join(run_dir, "flowchart.mmd")
        with open(flow_path, "w", encoding="utf-8") as f:
            f.write(flowchart_content.strip())
        
    return content
