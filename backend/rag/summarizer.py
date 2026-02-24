import json
from services.adapters import InferenceRouter

router = InferenceRouter()

def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Splits text intelligently by paragraphs to respect max_chars limit."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 <= max_chars:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If a single paragraph is larger than max_chars, split it blindly (fallback)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                current_chunk = ""
            else:
                current_chunk = p + "\n\n"
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def generate_structured_summary(phase_name: str, content: str) -> str:
    """
    Generates a structured summary conforming to strict requirements:
    - Phase
    - Key Components
    - Architecture Patterns
    - Dependencies
    - Constraints
    - Important Decisions
    """
    chunks = chunk_text(content, 4000)
    summaries = []
    
    prompt_template = (
        "You are an expert technical summarizer summarizing SDLC outputs.\n"
        "Extract technical memory from the following text chunk for phase: {phase}.\n"
        "Output ONLY a structured summary with these exact bullet points (use N/A if missing):\n"
        "- Phase: {phase}\n- Key Components: \n- Architecture Patterns: \n"
        "- Dependencies: \n- Constraints: \n- Important Decisions: \n\n"
        "Content Chunk:\n{chunk}"
    )
    
    for chunk in chunks:
        prompt = prompt_template.format(phase=phase_name, chunk=chunk)
        # Using mistral or openai for reliable structured output
        res = router.generate_text(prompt, preference=["openai", "mistral", "gemini"])
        summaries.append(res.get("output", "").strip())
    
    # Merge summaries if multiple chunks
    if len(summaries) == 1:
        return summaries[0]
    
    # Final merge prompt
    merge_prompt = (
        "Merge the following sub-summaries into a single, unified structured summary.\n"
        "Maintain the EXACT output structure:\n"
        "- Phase: {phase}\n- Key Components: \n- Architecture Patterns: \n"
        "- Dependencies: \n- Constraints: \n- Important Decisions: \n\n"
        "Sub-summaries to merge:\n{summaries}"
    ).format(phase=phase_name, summaries="\n\n---\n\n".join(summaries))
    
    res_merge = router.generate_text(merge_prompt, preference=["openai", "mistral"])
    return res_merge.get("output", "").strip()
