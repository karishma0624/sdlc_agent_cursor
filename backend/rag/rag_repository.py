import logging
from typing import List, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

def get_connection():
    """Lazily fetches a connection from the global pool initialized in main.py."""
    try:
        from app.main import db_pool
        if db_pool:
            return db_pool.getconn(), db_pool
    except ImportError:
        pass
        
    # Fallback to connect directly if standalone or pool not initialized (e.g. tests)
    import psycopg2
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        return conn, None
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None, None

def release_connection(conn, pool_ref):
    if pool_ref and conn:
        pool_ref.putconn(conn)
    elif conn:
        conn.close()

def insert_summary(
    project_id: str,
    session_id: str,
    prompt: str,
    summary: str,
    embedding: List[float],
    status: str,
    iteration: int
) -> bool:
    """Inserts a new summary with its embedding vector."""
    conn, pool_ref = get_connection()
    if not conn:
        return False
        
    query = """
        INSERT INTO rag_summaries 
        (project_id, session_id, prompt, summary, embedding, status, iteration) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                query, 
                (project_id, session_id, prompt, summary, embedding, status, iteration)
            )
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to insert summary into rag_summaries: {e}")
        conn.rollback()
        return False
    finally:
        release_connection(conn, pool_ref)

def retrieve_project_context(project_id: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves project-level context strictly isolated by project_id.
    Uses parameterized pgvector <-> operator for cosine distance.
    Filters by RAG_DISTANCE_THRESHOLD.
    """
    conn, pool_ref = get_connection()
    if not conn:
        return []

    threshold = settings.RAG_DISTANCE_THRESHOLD

    query = """
        SELECT id, session_id, prompt, summary, iteration, (embedding <-> %s::vector) as distance
        FROM rag_summaries
        WHERE project_id = %s
          AND (embedding <-> %s::vector) < %s
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    """
    
    results = []
    try:
        with conn.cursor() as cur:
            # We must pass the string representation of the list for pgvector casting
            vec_str = str(list(query_embedding))
            cur.execute(
                query,
                (vec_str, project_id, vec_str, threshold, vec_str, limit)
            )
            rows = cur.fetchall()
            for r in rows:
                results.append({
                    "id": r[0],
                    "session_id": r[1],
                    "prompt": r[2],
                    "summary": r[3],
                    "iteration": r[4],
                    "distance": r[5]
                })
    except Exception as e:
        logger.error(f"Failed to retrieve project context: {e}")
    finally:
        release_connection(conn, pool_ref)
        
    return results

def retrieve_session_context(session_id: str, query_embedding: List[float], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves session-level context strictly isolated by session_id.
    Uses parameterized pgvector <-> operator for cosine distance.
    Filters by RAG_DISTANCE_THRESHOLD.
    """
    conn, pool_ref = get_connection()
    if not conn:
        return []

    threshold = settings.RAG_DISTANCE_THRESHOLD

    query = """
        SELECT id, project_id, prompt, summary, iteration, (embedding <-> %s::vector) as distance
        FROM rag_summaries
        WHERE session_id = %s
          AND (embedding <-> %s::vector) < %s
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    """
    
    results = []
    try:
        with conn.cursor() as cur:
            vec_str = str(list(query_embedding))
            cur.execute(
                query,
                (vec_str, session_id, vec_str, threshold, vec_str, limit)
            )
            rows = cur.fetchall()
            for r in rows:
                results.append({
                    "id": r[0],
                    "project_id": r[1],
                    "prompt": r[2],
                    "summary": r[3],
                    "iteration": r[4],
                    "distance": r[5]
                })
    except Exception as e:
        logger.error(f"Failed to retrieve session context: {e}")
    finally:
        release_connection(conn, pool_ref)
        
    return results
