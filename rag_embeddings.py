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
_query_embedding_cache: Dict[str, Dict] = {}


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDER


def normalize_query(query: str) -> str:
    import string
    normalized = (query or "").lower().strip()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = " ".join(normalized.split())
    return normalized


def find_similar_cached_embedding(
    query: str, threshold: float = EMBEDDING_CACHE_SIMILARITY_THRESHOLD
) -> Optional[np.ndarray]:
    if not EMBEDDING_CACHE_ENABLE or not _query_embedding_cache:
        return None

    embedder = get_embedder()
    query_prefixed = f"query: {query}"
    query_embedding = embedder.encode([query_prefixed], normalize_embeddings=True)[0]

    best = None
    best_sim = 0.0

    for _, cache_data in _query_embedding_cache.items():
        cached_emb = cache_data["embedding"]
        sim = float(np.dot(query_embedding, cached_emb))
        if sim > best_sim:
            best_sim = sim
            best = cache_data

    if best and best_sim >= threshold:
        best["hit_count"] += 1
        best["last_used"] = time.time()
        print(f"[EMBEDDING-CACHE] ✅ Cache hit! Similarity: {best_sim:.3f} (hits: {best['hit_count']})")
        return best["embedding"]

    return None


def cache_query_embedding(query: str, embedding: np.ndarray) -> None:
    if not EMBEDDING_CACHE_ENABLE:
        return

    normalized = normalize_query(query)

    if len(_query_embedding_cache) >= 1000:
        oldest = sorted(_query_embedding_cache.items(), key=lambda x: x[1]["last_used"])
        for key, _ in oldest[:100]:
            del _query_embedding_cache[key]

    _query_embedding_cache[normalized] = {
        "embedding": embedding,
        "last_used": time.time(),
        "hit_count": 0,
    }


def create_embeddings(text_chunks: List[str]) -> np.ndarray:
    embedder = get_embedder()
    print(f"Creating embeddings for {len(text_chunks)} chunks...")
    prefixed = [f"passage: {chunk}" for chunk in text_chunks]
    embs = embedder.encode(prefixed, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    embs = np.array(embs).astype("float32")
    print("Embeddings created:", embs.shape)
    return embs