import os

"""
Shared configuration/constants for the Alkhidmat RAG system.
Separated from `RAG_supabase.py` so other modules can import without
pulling in the entire RAG implementation.
"""

# ============ SUPABASE CONFIG ============
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")

# ============ MODELS ============
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
LLM_MODEL_FILENAME = os.environ.get(
    "GPT4ALL_MODEL", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)

# ============ CHUNKING / RETRIEVAL ============
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_DIM = 768

RELEVANCE_THRESHOLD = 0.7

# ============ SELF-RAG PARAMETERS ============
SELFRAG_ENABLE = True
SELFRAG_RETRIEVE_THRESHOLD = 0.5
SELFRAG_RELEVANCE_THRESHOLD = 0.6
SELFRAG_SUPPORT_THRESHOLD = 0.7
SELFRAG_MIN_CONFIDENCE = 0.6

# ============ BRAND / LANGUAGE HELPERS ============
BRAND_TERMS = [
    "EasyPaisa",
    "JazzCash",
    "Alkhidmat",
    "Alkhidmat Foundation",
    "Bank of Punjab",
    "Taqwa Islamic Banking",
]

ROMAN_URDU_MARKERS = {
    "kya",
    "kyu",
    "kyun",
    "kaise",
    "kesy",
    "kese",
    "kis",
    "kon",
    "ka",
    "ki",
    "ko",
    "mein",
    "main",
    "mera",
    "meri",
    "mere",
    "hum",
    "ham",
    "aap",
    "ap",
    "tum",
    "yeh",
    "nahi",
    "nai",
    "han",
    "haan",
    "hai",
    "hain",
    "tha",
    "thi",
    "thay",
    "kr",
    "kar",
    "karo",
    "kren",
    "karein",
    "krna",
    "hona",
    "hogya",
    "ho",
    "hoga",
    "please",
    "plz",
}

# If True, translate context into Urdu before generation for Urdu/Roman Urdu outputs
TRANSLATE_CONTEXT_FOR_URDU_OUTPUT = True

# ============ AGENTIC / CACHE SETTINGS ============
AGENTIC_ENABLE = True  # Enable agentic workflows
EMBEDDING_CACHE_ENABLE = True  # Enable query embedding cache
QUERY_ROUTER_ENABLE = True  # Enable query routing before embeddings
RETRIEVAL_RETRY_ENABLE = True  # Enable retrieval retry with reformulation
EVIDENCE_COVERAGE_ENABLE = True  # Enable claim-by-claim evidence checking
CONVERSATION_MEMORY_ENABLE = True  # Enable conversation state reuse

# Cache thresholds
EMBEDDING_CACHE_SIMILARITY_THRESHOLD = (
    0.95  # Reuse embedding if similarity >= this
)
DOMAIN_CENTROID_REUSE_THRESHOLD = (
    0.85  # Reuse domain embedding if similarity >= this
)
RETRIEVAL_RETRY_MAX_ATTEMPTS = 2  # Maximum retrieval retry attempts
RETRIEVAL_RETRY_RELEVANCE_THRESHOLD = (
    0.6  # Retry if relevance < this
)

