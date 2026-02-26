import time
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_config import (
    EMBEDDING_CACHE_ENABLE,
    EMBEDDING_CACHE_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME,
)

_EMBEDDER: Optional[SentenceTransformer] = None

# In-memory query embedding cache
_query_embedding_cache: Dict[str, Dict] = {}


def get_embedder() -> SentenceTransformer:
    """Load (or return cached) sentence-transformer embedder."""
    global _EMBEDDER
    if _EMBEDDER is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDER


def normalize_query(query: str) -> str:
    """Normalize query for caching: lowercase, strip punctuation, remove stopwords."""
    import string

    # Lowercase
    normalized = query.lower().strip()
    # Remove punctuation
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    return normalized


def find_similar_cached_embedding(
    query: str, threshold: float = EMBEDDING_CACHE_SIMILARITY_THRESHOLD
) -> Optional[np.ndarray]:
    """Find similar query embedding in cache using cosine similarity."""
    if not EMBEDDING_CACHE_ENABLE or not _query_embedding_cache:
        return None

    normalized = normalize_query(query)

    # Get embedder for similarity calculation
    embedder = get_embedder()
    query_prefixed = f"query: {query}"
    query_embedding = embedder.encode(
        [query_prefixed], normalize_embeddings=True
    )[0]

    best_match = None
    best_similarity = 0.0

    for cached_query, cache_data in _query_embedding_cache.items():
        cached_emb = cache_data["embedding"]
        similarity = float(np.dot(query_embedding, cached_emb))

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = cache_data

    if best_match and best_similarity >= threshold:
        best_match["hit_count"] += 1
        best_match["last_used"] = time.time()
        print(
            f"[EMBEDDING-CACHE] ✅ Cache hit! Similarity: {best_similarity:.3f} (hits: {best_match['hit_count']})"
        )
        return best_match["embedding"]

    return None


def cache_query_embedding(query: str, embedding: np.ndarray) -> None:
    """Cache query embedding for future reuse."""
    if not EMBEDDING_CACHE_ENABLE:
        return

    normalized = normalize_query(query)

    # Limit cache size (keep most recent 1000)
    if len(_query_embedding_cache) >= 1000:
        # Remove oldest entries
        sorted_items = sorted(
            _query_embedding_cache.items(), key=lambda x: x[1]["last_used"]
        )
        for key in sorted_items[:100]:  # Remove oldest 100
            del _query_embedding_cache[key[0]]

    _query_embedding_cache[normalized] = {
        "embedding": embedding,
        "last_used": time.time(),
        "hit_count": 0,
    }


def create_embeddings(text_chunks: List[str]) -> np.ndarray:
    """Create embeddings for a list of text chunks."""
    embedder = get_embedder()
    print(f"Creating embeddings for {len(text_chunks)} chunks...")
    prefixed = [f"passage: {chunk}" for chunk in text_chunks]
    embs = embedder.encode(
        prefixed, show_progress_bar=True, batch_size=32, normalize_embeddings=True
    )
    embs = np.array(embs).astype("float32")
    print("Embeddings created:", embs.shape)
    return embs

