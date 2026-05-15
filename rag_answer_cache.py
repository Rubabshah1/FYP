"""
rag_answer_cache.py
-------------------
Persistent semantic answer cache backed by Supabase pgvector.

Lookup:  embed the English query → cosine similarity search in cached_responses
         → return best hit if similarity >= ANSWER_CACHE_SIMILARITY_THRESHOLD

Store:   only cache results that pass three quality guards (see store())

Clear:   DELETE FROM cached_responses — called from /admin/cache/clear

Stats:   aggregate query on cached_responses — surfaced on admin dashboard

The table uses the same vector(768) type and cosine operator (<=>)
as the documents table, so no new infrastructure is needed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cosine similarity threshold for a cache hit.
# 0.88 catches close paraphrases while keeping precision high.
# Override with ANSWER_CACHE_SIMILARITY_THRESHOLD env var.
ANSWER_CACHE_SIMILARITY_THRESHOLD = float(
    os.getenv("ANSWER_CACHE_SIMILARITY_THRESHOLD", "0.88")
)

# Minimum combined_confidence a result must have before it is worth caching.
MIN_CONFIDENCE_TO_CACHE = float(os.getenv("ANSWER_CACHE_MIN_CONFIDENCE", "0.45"))

# ---------------------------------------------------------------------------
# Do-not-cache answer signals
# ---------------------------------------------------------------------------
_SKIP_PHRASES = (
    "i don't know",
    "i do not know",
    "mujhe maloom nahi",
    "mujhe maloom nahin",
    "that is an irrelevant question",
)


def _is_uncacheable_answer(answer: str) -> bool:
    a = (answer or "").lower().strip()
    return any(phrase in a for phrase in _SKIP_PHRASES)


# ---------------------------------------------------------------------------
# Supabase client helper
# ---------------------------------------------------------------------------

def _get_sb():
    from supabase_client import get_supabase_client
    return get_supabase_client()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup(query_en: str, embedding: np.ndarray, input_lang: str = "en") -> Optional[Dict[str, Any]]:
    """
    Search cached_responses for a semantically similar query.
    Returns a result dict on a hit, None on a miss.

    Uses the match_cached_responses RPC function (see create_cached_responses.sql).
    The embedding must be the L2-normalised float32 vector for query_en
    — the same one you would pass to the document retrieval step.
    """
    # Non-English embeddings cluster more tightly after translation —
    # use a stricter threshold to avoid cross-query false positives.
    effective_threshold = ANSWER_CACHE_SIMILARITY_THRESHOLD
    if input_lang in ("ur", "roman_ur"):
        effective_threshold = max(ANSWER_CACHE_SIMILARITY_THRESHOLD, 0.94)

    try:
        sb = _get_sb()
        result = sb.rpc(
            "match_cached_responses",
            {
                "query_embedding": embedding.astype("float64").tolist(),
                "similarity_threshold": effective_threshold,
                "match_count": 1,
                "filter_lang": input_lang,
            },
        ).execute()

        rows = result.data or []
        if not rows:
            print(f"[ANSWER-CACHE] ❌ Miss  query='{query_en[:60]}'", flush=True)
            return None

        row = rows[0]
        similarity = float(row.get("similarity", 0.0))
        print(
            f"[ANSWER-CACHE] ✅ Hit  sim={similarity:.3f}  "
            f"hits={row.get('hit_count', 0)}  query='{query_en[:60]}'",
            flush=True,
        )

        # Non-blocking hit_count + last_used_at update
        try:
            sb.table("cached_responses").update({
                "hit_count":    (row.get("hit_count") or 0) + 1,
                "last_used_at": _now_iso(),
            }).eq("cache_id", row["cache_id"]).execute()
        except Exception as _ue:
            print(f"[ANSWER-CACHE] hit_count update failed (non-fatal): {_ue}", flush=True)

        return {
            "answer":   row["answer"],
            "sources":  row.get("sources") or [],
            "confidence_scores": {
                "combined_confidence": float(row.get("confidence", 0.0))
            },
            "domain_classification": {
                "domain":     row.get("domain", "general"),
                "confidence": float(row.get("confidence", 0.0)),
                "all_scores": row.get("domain_scores") or {},
            },
            "selfrag_metrics": {},
            "input_lang":      row.get("input_lang", "en"),
            "cache_id":        row["cache_id"],
        }

    except Exception as e:
        print(f"[ANSWER-CACHE] Lookup error (non-fatal): {e}", flush=True)
        return None


def store(
    query_en: str,
    embedding: np.ndarray,
    answer: str,
    sources: list,
    confidence_scores: dict,
    domain_classification: dict,
    selfrag_metrics: dict,
    input_lang: str,
) -> bool:
    """
    Persist a RAG result in cached_responses.
    Returns True if stored, False if skipped or errored.

    Three quality guards — each is deliberate:
    1. Don't cache "I don't know" / irrelevant answers.
    2. Don't cache low-confidence results (< MIN_CONFIDENCE_TO_CACHE).
    3. Don't cache agent-routed responses.
    """
    if _is_uncacheable_answer(answer):
        print(f"[ANSWER-CACHE] Skip — uninformative answer", flush=True)
        return False

    conf = float((confidence_scores or {}).get("combined_confidence", 0.0))
    if conf < MIN_CONFIDENCE_TO_CACHE:
        print(f"[ANSWER-CACHE] Skip — low confidence ({conf:.3f})", flush=True)
        return False

    if (selfrag_metrics or {}).get("route_to_agent"):
        print(f"[ANSWER-CACHE] Skip — routed to agent", flush=True)
        return False

    try:
        sb = _get_sb()
        now = _now_iso()

        sb.table("cached_responses").insert({
            "query_en":     query_en,
            "embedding":    embedding.astype("float64").tolist(),
            "answer":       answer,
            "sources":      sources or [],
            "confidence":   conf,
            "domain":       (domain_classification or {}).get("domain", "general"),
            "domain_scores": (domain_classification or {}).get("all_scores", {}),
            "input_lang":   input_lang,
            "hit_count":    0,
            "created_at":   now,
            "last_used_at": now,
        }).execute()

        print(
            f"[ANSWER-CACHE] Stored  conf={conf:.3f}  lang={input_lang}  "
            f"query='{query_en[:60]}'",
            flush=True,
        )
        return True

    except Exception as e:
        print(f"[ANSWER-CACHE] Store error (non-fatal): {e}", flush=True)
        return False


def clear() -> int:
    """
    Delete all rows from cached_responses.
    Returns the number of entries deleted.
    Call this after updating the knowledge base so stale answers are not served.
    """
    try:
        sb = _get_sb()
        # Supabase REST requires a filter on DELETE; neq on PK covers all rows
        result = sb.table("cached_responses").delete().neq(
            "cache_id", "00000000-0000-0000-0000-000000000000"
        ).execute()
        count = len(result.data) if result.data else 0
        print(f"[ANSWER-CACHE] 🗑  Cleared {count} entries", flush=True)
        return count
    except Exception as e:
        print(f"[ANSWER-CACHE] Clear error: {e}", flush=True)
        raise


def stats() -> Dict[str, Any]:
    """Return cache statistics for the admin dashboard."""
    try:
        sb = _get_sb()
        result = sb.table("cached_responses").select(
            "cache_id, confidence, hit_count, created_at, domain, input_lang"
        ).execute()

        rows = result.data or []
        if not rows:
            return {
                "total_entries": 0,
                "total_hits": 0,
                "oldest_entry_age_seconds": None,
                "similarity_threshold": ANSWER_CACHE_SIMILARITY_THRESHOLD,
                "min_confidence_to_cache": MIN_CONFIDENCE_TO_CACHE,
            }

        total_hits = sum(r.get("hit_count", 0) for r in rows)

        oldest_ts = min(r["created_at"] for r in rows)
        try:
            oldest_dt = datetime.fromisoformat(oldest_ts.replace("Z", "+00:00"))
            age_seconds = round((datetime.now(timezone.utc) - oldest_dt).total_seconds(), 1)
        except Exception:
            age_seconds = None

        domain_counts: Dict[str, int] = {}
        for r in rows:
            d = r.get("domain") or "general"
            domain_counts[d] = domain_counts.get(d, 0) + 1

        lang_counts: Dict[str, int] = {}
        for r in rows:
            l = r.get("input_lang") or "en"
            lang_counts[l] = lang_counts.get(l, 0) + 1

        return {
            "total_entries":            len(rows),
            "total_hits":               total_hits,
            "oldest_entry_age_seconds": age_seconds,
            "domain_breakdown":         domain_counts,
            "language_breakdown":       lang_counts,
            "similarity_threshold":     ANSWER_CACHE_SIMILARITY_THRESHOLD,
            "min_confidence_to_cache":  MIN_CONFIDENCE_TO_CACHE,
        }

    except Exception as e:
        print(f"[ANSWER-CACHE] Stats error: {e}", flush=True)
        return {"error": str(e)}