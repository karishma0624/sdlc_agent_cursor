import logging
import time
from typing import Dict, Any, List
from .embedding_service import generate_embedding
from .rag_repository import retrieve_project_context, retrieve_session_context
from services.supabase_client import save_execution_log

logger = logging.getLogger(__name__)

def retrieve_and_merge_context(
    query_text: str, 
    project_id: str, 
    session_id: str, 
    iteration: int,
    session_limit: int = 3,
    project_limit: int = 5
) -> str:
    """
    Generates embedding for query_text, retrieves top session and project contexts,
    and merges them into a string. Returns empty string if it fails.
    Also logs retrieval latency to execution_logs.
    """
    start_time = time.time()
    
    embedding, tokens_used = generate_embedding(query_text)
    if not embedding:
        return ""
        
    session_docs = []
    if iteration > 0 and session_id:
        session_docs = retrieve_session_context(session_id, embedding, limit=session_limit)
        
    project_docs = []
    if project_id:
        project_docs = retrieve_project_context(project_id, embedding, limit=project_limit)
        
    latency_ms = int((time.time() - start_time) * 1000)
    total_docs = len(session_docs) + len(project_docs)
    
    # Log latency
    if session_id:
        save_execution_log(
            session_id=session_id,
            phase="RAG_Retrieval",
            provider="openai",
            success=True,
            error_output=f"Latency: {latency_ms}ms, Docs Retreived: {total_docs}",
            iteration=iteration
        )
        
    if not total_docs:
        return ""

    merged_context = []
    
    if session_docs:
        merged_context.append("--- RECENT SESSION CONTEXT ---")
        for doc in session_docs:
            merged_context.append(f"Prompt: {doc['prompt']}\nSummary: {doc['summary']}\n(Distance: {doc['distance']:.3f})")
            
    if project_docs:
        merged_context.append("--- PROJECT TECHNICAL MEMORY ---")
        for doc in project_docs:
            # Prevent duplication if the exact same doc was caught in session context
            if any(sd.get("id") == doc["id"] for sd in session_docs):
                continue
            merged_context.append(f"Prompt: {doc['prompt']}\nSummary: {doc['summary']}\n(Distance: {doc['distance']:.3f})")
            
    return "\n\n".join(merged_context)
