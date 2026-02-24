import os
import requests
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generates a 1536-dimensional embedding using OpenAI's text-embedding-3-small.
    Returns the vector as a list of floats.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set. Cannot generate embedding.")
        return None

    # text-embedding-3-small defaults to 1536 dimensions
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        embedding = data["data"][0]["embedding"]
        
        # Log approximate token usage (managed by caller or globally, but we can capture it here)
        tokens_used = data.get("usage", {}).get("prompt_tokens", 0)
        return embedding, tokens_used
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None, 0
