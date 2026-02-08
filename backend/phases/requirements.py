import os
from services.adapters import call_mistral

def execute_requirements_phase(prompt: str, run_dir: str) -> str:
    """
    PHASE 1: REQUIREMENTS (Mistral ONLY)
    Generates a clean Software Requirements Specification (SRS).
    """
    
    system_prompt = (
        "You are a Senior Technical Product Manager.\n"
        "Your goal: Generate a clean, professional Software Requirements Specification (SRS) based on the user's request.\n"
        "Output Format: Markdown.\n"
        "Tone: Professional, structured, no fluff.\n\n"
        "Must Include:\n"
        "1. Project Overview\n"
        "2. User Personas\n"
        "3. Functional Requirements\n"
        "4. Non-Functional Requirements\n"
        "5. System Constraints\n"
        "6. Assumptions\n"
        "7. Edge Cases\n\n"
        "User Request:\n"
    )
    
    full_prompt = system_prompt + prompt
    
    try:
        content = call_mistral(full_prompt)
        
        # Validation: content must be non-empty and have markdown headers
        if not content or len(content) < 100 or "# " not in content:
            raise ValueError("Mistral generated invalid/empty requirements content.")
            
        # Save Artifact
        path = os.path.join(run_dir, "requirements.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return content
        
    except Exception as e:
        raise RuntimeError(f"Requirements Phase Failed: {str(e)}")
