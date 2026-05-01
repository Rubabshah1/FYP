import os

"""
Shared configuration/constants for the Alkhidmat RAG system.
Keep this file lightweight: env + constants only.
"""

# ============ SUPABASE CONFIG ============
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")

# ============ MODELS ============
# Multilingual cross-lingual retrieval model (Urdu <-> English works)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

# Default (general) local model (GGUF path)
LLM_MODEL_FILENAME = os.environ.get("LLM_MODEL_FILENAME", os.environ.get("GPT4ALL_MODEL", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"))

# Urdu LLM options
URDU_LLM_ENABLE = os.environ.get("URDU_LLM_ENABLE", "1").lower() in ("1", "true", "yes", "y")

# (A) Local GGUF path (if you already downloaded it)
URDU_LLM_MODEL_FILENAME = os.environ.get("URDU_LLM_MODEL", "Alif-1.0-8B-Instruct.gguf")

# (B) Load GGUF directly from Hugging Face via llama-cpp-python
URDU_LLM_HF_REPO = os.environ.get("URDU_LLM_HF_REPO", "large-traversaal/Alif-1.0-8B-Instruct")
URDU_LLM_HF_FILENAME = os.environ.get("URDU_LLM_HF_FILENAME", "model-Q4_K.gguf")
URDU_LLM_LOAD_VIA_HF = os.environ.get("URDU_LLM_LOAD_VIA_HF", "1").lower() in ("1", "true", "yes", "y")

# Evidence handling for Urdu output
# - bilingual_evidence: keep context English, answer Urdu Nastaliq
# - translate_context: translate retrieved context to Urdu before generation (slower, noisier)
URDU_EVIDENCE_MODE = os.environ.get("URDU_EVIDENCE_MODE", "bilingual_evidence")  # bilingual_evidence | translate_context

# Retrieval controls
CROSS_LINGUAL_RETRIEVAL_ENABLE = os.environ.get("CROSS_LINGUAL_RETRIEVAL_ENABLE", "1").lower() in ("1", "true", "yes", "y")
DUAL_QUERY_RETRIEVAL_ENABLE = os.environ.get("DUAL_QUERY_RETRIEVAL_ENABLE", "1").lower() in ("1", "true", "yes", "y")

# If True, translate context into Urdu before generation for Urdu/Roman Urdu outputs
# For Option-3 bilingual-evidence mode, keep this False or leave it True but it will be overridden by URDU_EVIDENCE_MODE.
TRANSLATE_CONTEXT_FOR_URDU_OUTPUT = os.environ.get(
    "TRANSLATE_CONTEXT_FOR_URDU_OUTPUT", "0"
).lower() in ("1", "true", "yes", "y")

# ============ ROMAN URDU (LATIN URDU) ============
ROMAN_URDU_ENABLE = os.environ.get("ROMAN_URDU_ENABLE", "1").lower() in ("1", "true", "yes", "y")

# Romanization strategy:
# - "llm_then_fallback": try LLM romanization, else fallback transliteration
# - "fallback_only": never call LLM (fast, lower quality)
ROMAN_URDU_ROMANIZATION_MODE = os.environ.get("ROMAN_URDU_ROMANIZATION_MODE", "llm_then_fallback")

# Enforce strict Latin-only output (recommended)
ROMAN_URDU_STRICT_LATIN_ONLY = os.environ.get("ROMAN_URDU_STRICT_LATIN_ONLY", "1").lower() in ("1", "true", "yes", "y")

# Max tokens for romanization step
ROMAN_URDU_MAX_TOKENS = int(os.environ.get("ROMAN_URDU_MAX_TOKENS", "260"))

# If you want a stronger “Pakistani” feel (spellings + particles)
ROMAN_URDU_PAK_VERBIAGE = os.environ.get("ROMAN_URDU_PAK_VERBIAGE", "1").lower() in ("1", "true", "yes", "y")

# ============ CHUNKING / RETRIEVAL ============
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.7"))

# ============ SELF-RAG PARAMETERS ============
SELFRAG_ENABLE = os.environ.get("SELFRAG_ENABLE", "1").lower() in ("1", "true", "yes", "y")
SELFRAG_RETRIEVE_THRESHOLD = float(os.environ.get("SELFRAG_RETRIEVE_THRESHOLD", "0.5"))
SELFRAG_RELEVANCE_THRESHOLD = float(os.environ.get("SELFRAG_RELEVANCE_THRESHOLD", "0.6"))
SELFRAG_SUPPORT_THRESHOLD = float(os.environ.get("SELFRAG_SUPPORT_THRESHOLD", "0.7"))
SELFRAG_MIN_CONFIDENCE = float(os.environ.get("SELFRAG_MIN_CONFIDENCE", "0.6"))

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
    # Question words
    "kya","kyu","kyun","kaise","kesy","kese","kaisay","kis","kon","kaun","ka","ki","ko",
    # Pronouns / person markers
    "mujhe","mujh","meri","mera","mere","hum","ham","aap","ap","tum","yeh","ye","woh","wo",
    "main","mein","unhe","unka","unki","inhe","inka","inki",
    # Common verb roots and forms
    "hai","hain","tha","thi","thay","ho","hoga","hogi","hone","hona","hogya","hogyi",
    "kr","kar","karo","kren","karein","krna","karna","karta","karti","karte",
    "chahta","chahti","chahiye","chahye","chahte",
    "dena","dein","deta","deti","milna","milta","milti","milega","milegi",
    "jana","janna","jata","jati","jayein","jaein","gaya","gayi",
    "batao","bataein","batana","bata","pata","samjhao","samjhana",
    "lena","leta","leti","lijiye","lijye",
    # Connectors and prepositions
    "ke","se","par","pe","tak","saath","sath","baad","pehle","phir","aur","ya","lekin","magar",
    "bare","barey","baarey",
    # Common adverbs and particles
    "nahi","nai","nahin","han","haan","bilkul","zaroor","sirf","bas","abhi","ab","pls","plz","please",
    # Common nouns appearing in Roman Urdu support queries
    "sahoolat","sahooliat","mosool","madad","maloomat","malumat","jankari",
    "program","khidmat","ilaaj","sehat","zakat","sadqa","sadaqah","qurbani",
}

# ============ AGENTIC / CACHE SETTINGS ============
AGENTIC_ENABLE = os.environ.get("AGENTIC_ENABLE", "1").lower() in ("1", "true", "yes", "y")
EMBEDDING_CACHE_ENABLE = os.environ.get("EMBEDDING_CACHE_ENABLE", "1").lower() in ("1", "true", "yes", "y")
QUERY_ROUTER_ENABLE = os.environ.get("QUERY_ROUTER_ENABLE", "1").lower() in ("1", "true", "yes", "y")
RETRIEVAL_RETRY_ENABLE = os.environ.get("RETRIEVAL_RETRY_ENABLE", "1").lower() in ("1", "true", "yes", "y")
EVIDENCE_COVERAGE_ENABLE = os.environ.get("EVIDENCE_COVERAGE_ENABLE", "1").lower() in ("1", "true", "yes", "y")
CONVERSATION_MEMORY_ENABLE = os.environ.get("CONVERSATION_MEMORY_ENABLE", "1").lower() in ("1", "true", "yes", "y")

EMBEDDING_CACHE_SIMILARITY_THRESHOLD = float(os.environ.get("EMBEDDING_CACHE_SIMILARITY_THRESHOLD", "0.85"))
DOMAIN_CENTROID_REUSE_THRESHOLD = float(os.environ.get("DOMAIN_CENTROID_REUSE_THRESHOLD", "0.85"))
RETRIEVAL_RETRY_MAX_ATTEMPTS = int(os.environ.get("RETRIEVAL_RETRY_MAX_ATTEMPTS", "2"))
RETRIEVAL_RETRY_RELEVANCE_THRESHOLD = float(os.environ.get("RETRIEVAL_RETRY_RELEVANCE_THRESHOLD", "0.6"))