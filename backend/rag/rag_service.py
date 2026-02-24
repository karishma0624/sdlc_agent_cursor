import logging
from typing import Optional

from .summarizer import generate_structured_summary
from .embedding_service import generate_embedding
from .rag_repository import insert_summary
from services.supabase_client import save_provider_usage

logger = logging.getLogger(__name__)

def index_phase_output(
    project_id: str,
    session_id: str,
    phase_name: str,
    content: str,
    original_prompt: str,
    iteration: int
) -> bool:
    """
    Orchestrates the indexing process:
    1. Summarizes the content (chunking if needed).
    2. Embeds the summary.
    3. Saves to rag_summaries via pgvector pool.
    """
    try:
        if not project_id or not session_id:
            logger.warning("Missing project_id or session_id. Skipping RAG indexing.")
            return False
            
        logger.info(f"Generating structured summary for phase: {phase_name}")
        summary = generate_structured_summary(phase_name, content)
        if not summary:
            logger.error("Failed to generate summary.")
            return False
            
        logger.info("Generating embedding for summary.")
        embedding, tokens_used = generate_embedding(summary)
        if not embedding:
            logger.error("Failed to generate embedding.")
            return False
            
        if session_id:
            save_provider_usage(
                session_id=session_id,
                provider_name="openai",
                model_name="text-embedding-3-small",
                tokens_used=tokens_used,
                phase=f"{phase_name}_indexing",
                success=True
            )
            
        logger.info("Inserting summary into rag_summaries.")
        success = insert_summary(
            project_id=project_id,
            session_id=session_id,
            prompt=original_prompt,
            summary=summary,
            embedding=embedding,
            status="indexed",
            iteration=iteration
        )
        return success
    except Exception as e:
        logger.error(f"Error during index_phase_output: {e}")
        return False
