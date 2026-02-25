import os
from services.adapters import call_mistral, call_gemini_text

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
        "CRITICAL INSTRUCTION: Do NOT summarize away specific details! You must act as a precise translator. If the user asks for 'beauty parlor including makeup artist', you MUST include both the beauty parlor and the 'Makeup Artist' as a specific actor/feature. Capture ALL specific features and specifications mentioned in the prompt.\n\n"
        "Must Include:\n"
        "1. Project Overview\n"
        "2. User Personas (Include ALL specific roles mentioned)\n"
        "3. Functional Requirements (Detail EVERY single specific feature requested by the user, skipping nothing)\n"
        "4. Non-Functional Requirements\n"
        "5. System Constraints\n"
        "6. Assumptions\n"
        "7. Edge Cases\n\n"
        "User Request (READ IT CAREFULLY AND OMIT NOTHING):\n"
    )
    
    full_prompt = system_prompt + prompt
    
    try:
        content = call_mistral(full_prompt)
        # Validation: content must be non-empty and have markdown headers
        if not content or len(content) < 100 or "# " not in content:
            raise ValueError("Mistral generated invalid/empty requirements content.")
    except Exception as e:
        print(f"Mistral failed in Requirements Phase: {e}. Falling back to Gemini...")
        content = call_gemini_text(full_prompt, model="gemini-1.5-pro")
        if content.startswith("Error:"):
            raise RuntimeError(f"Requirements Phase Failed (Both APIs): {content}")
        if not content or len(content) < 100 or "# " not in content:
            raise RuntimeError("Requirements Phase Failed: generated invalid/empty requirements content.")
            
    # Save Artifact
    path = os.path.join(run_dir, "requirements.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        
    return content
