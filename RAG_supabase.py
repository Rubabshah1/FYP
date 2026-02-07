# #!/usr/bin/env python3
# """
# ALKHIDMAT RAG SYSTEM - SUPABASE CLIENT EDITION
# WITH DOMAIN CLASSIFICATION & CONFIDENCE SCORING
# Uses Supabase Python client (REST API) and Llama-cpp-python
# """

# from dotenv import load_dotenv
# load_dotenv()

# import os
# import re
# import json
# import time
# import sys
# from pathlib import Path
# from typing import List, Dict, Tuple, Any, Optional
# import uuid

# import gc

# import numpy as np
# import zipfile
# from scipy.stats import entropy
# from sklearn.metrics.pairwise import cosine_similarity

# # Embeddings
# from sentence_transformers import SentenceTransformer

# # LLM (Llama CPP)
# from llama_cpp import Llama

# # Supabase Client
# from supabase import create_client, Client

# # Urdu helpers
# from deep_translator import GoogleTranslator
# import langdetect

# # text splitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Import domain anchors
# from domain_anchors import DOMAIN_ANCHOR_QUERIES

# # ============ SUPABASE CONFIG ============
# SUPABASE_URL = os.environ.get("SUPABASE_URL") 
# SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY") 

# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
# LLM_MODEL_FILENAME = os.environ.get("GPT4ALL_MODEL", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

# # Chunking parameters
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 200
# EMBEDDING_DIM = 768

# # Retrieval parameters
# RELEVANCE_THRESHOLD = 0.7

# # Self-RAG parameters
# SELFRAG_ENABLE = True
# SELFRAG_RETRIEVE_THRESHOLD = 0.5
# SELFRAG_RELEVANCE_THRESHOLD = 0.6
# SELFRAG_SUPPORT_THRESHOLD = 0.7
# SELFRAG_MIN_CONFIDENCE = 0.6

# # Brand terms protection
# BRAND_TERMS = ["EasyPaisa", "JazzCash", "Alkhidmat", "Alkhidmat Foundation", "Bank of Punjab", "Taqwa Islamic Banking"]

# # Roman Urdu detection markers
# ROMAN_URDU_MARKERS = {
#     "kya", "kyu", "kyun", "kaise", "kesy", "kese", "kis", "kon", "ka", "ki", "ko",
#     "mein", "main", "mera", "meri", "mere", "hum", "ham", "aap", "ap", "tum", "yeh",
#     "nahi", "nai", "han", "haan", "hai", "hain", "tha", "thi", "thay",
#     "kr", "kar", "karo", "kren", "karein", "krna", "hona", "hogya", "ho", "hoga",
#     "please", "plz"
# }

# # If True, translate context into Urdu before generation for Urdu/Roman Urdu outputs
# TRANSLATE_CONTEXT_FOR_URDU_OUTPUT = True

# # ============================================================================
# # SELF-RAG REFLECTION TOKENS & CRITIC
# # ============================================================================

# class SelfRAGReflectionTokens:
#     RETRIEVE = "[Retrieve]"
#     NO_RETRIEVE = "[No Retrieval]"
#     RELEVANT = "[Relevant]"
#     IRRELEVANT = "[Irrelevant]"
#     FULLY_SUPPORTED = "[Fully supported]"
#     PARTIALLY_SUPPORTED = "[Partially supported]"
#     NO_SUPPORT = "[No support]"
#     UTILITY_5 = "[Utility:5]"
#     UTILITY_4 = "[Utility:4]"
#     UTILITY_3 = "[Utility:3]"
#     UTILITY_2 = "[Utility:2]"
#     UTILITY_1 = "[Utility:1]"

# class SelfRAGCritic:
#     def __init__(self, llm_model=None):
#         self.llm = llm_model

#     def _get_llm(self):
#         """Get LLM model, loading if necessary."""
#         if self.llm is None:
#             self.llm = load_llm()
#         return self.llm

#     def is_domain_relevant(self, query: str) -> Tuple[bool, float]:
#         prompt = f"""Determine if this question is about Alkhidmat Foundation Pakistan.

# Question: {query}

# You are a strict binary classifier.

# Task:
# Decide whether the user's query is SPECIFICALLY about the NGO
# "Alkhidmat Foundation Pakistan".

# Definition of RELEVANT:
# A query is RELEVANT ONLY IF it clearly and explicitly refers to:
# - Alkhidmat Foundation Pakistan by name or by obvious NGO context
# AND
# - Its services, operations, donations, healthcare, education, relief work,
#   offices, leadership, volunteers, or official programs.

# Examples of RELEVANT queries:
# - "What is Alkhidmat Foundation?"
# - "How can I donate to Alkhidmat?"
# - "Does Alkhidmat provide free medical tests?"
# - "Where is the Alkhidmat hospital in Lahore?"
# - "Who is the CEO of Alkhidmat Foundation?"

# Definition of IRRELEVANT:
# A query is IRRELEVANT if:
# - Alkhidmat Foundation is NOT clearly mentioned or strongly implied
# - The query is about general topics, even if related to charity, Islam, health, or Pakistan
# - The query is about people, celebrities, politics, sports, self-help, fitness, news, or current affairs
# - The query could apply to ANY NGO, not specifically Alkhidmat

# Examples of IRRELEVANT queries:
# - "Best charity in Pakistan"
# - "What is zakat?"
# - "Free hospitals in Karachi"
# - "How to stay healthy"
# - "Imran Khan latest news"
# - "Flood situation in Pakistan"

# Important rules:
# - Do NOT assume the query is about Alkhidmat unless it is clearly stated.
# - If the query is ambiguous or generic, mark it IRRELEVANT.
# - When in doubt, choose IRRELEVANT.

# Output:
# Respond with ONLY ONE of the following labels:
# [RELEVANT]
# [IRRELEVANT]

# Answer:"""

#         try:
#             model = self._get_llm()
#             output = model(prompt, max_tokens=10, temperature=0.1, stop=["\n"], echo=False)
#             response = output['choices'][0]['text'].strip().upper()
#             if "[RELEVANT]" in response:
#                 return True, 0.85
#             elif "[IRRELEVANT]" in response:
#                 return False, 0.85
#             return True, 0.4
#         except Exception as e:
#             print(f"[SelfRAG] Domain relevance check error: {e}")
#             return True, 0.5

#     def should_retrieve(self, query: str) -> Tuple[bool, float]:
#         prompt = f"""Determine if external knowledge retrieval is needed to answer this question.

# Question: {query}

# Consider:
# - Does this require specific factual information?
# - Is this about a specific organization, service, or policy?
# - Can this be answered with general knowledge alone?

# Respond with ONLY one of:
# {SelfRAGReflectionTokens.RETRIEVE} - if retrieval is needed
# {SelfRAGReflectionTokens.NO_RETRIEVE} - if general knowledge is sufficient

# Answer:"""
#         try:
#             model = self._get_llm()
#             output = model(prompt, max_tokens=10, temperature=0.1, stop=["\n"], echo=False)
#             response = output['choices'][0]['text'].strip()
#             if SelfRAGReflectionTokens.RETRIEVE in response:
#                 return True, 0.8
#             elif SelfRAGReflectionTokens.NO_RETRIEVE in response:
#                 return False, 0.8
#             return True, 0.5
#         except Exception as e:
#             print(f"[SelfRAG] Retrieval prediction error: {e}")
#             return True, 0.5

#     def assess_relevance(self, query: str, document: str) -> Tuple[bool, float]:
#         doc_preview = document[:500] + "..." if len(document) > 500 else document
#         prompt = f"""Assess if this document is relevant to answering the question.

# Question: {query}

# Document excerpt:
# {doc_preview}

# Is this document relevant for answering the question?

# Respond with ONLY one of:
# {SelfRAGReflectionTokens.RELEVANT} - if document is relevant
# {SelfRAGReflectionTokens.IRRELEVANT} - if document is not relevant

# Answer:"""
#         try:
#             model = self._get_llm()
#             output = model(prompt, max_tokens=10, temperature=0.1, stop=["\n"], echo=False)
#             response = output['choices'][0]['text'].strip()
#             if SelfRAGReflectionTokens.RELEVANT in response:
#                 return True, 0.8
#             elif SelfRAGReflectionTokens.IRRELEVANT in response:
#                 return False, 0.8
#             return True, 0.5
#         except Exception as e:
#             print(f"[SelfRAG] Relevance assessment error: {e}")
#             return True, 0.5

#     def check_answer_in_context(self, query: str, context: str) -> Tuple[bool, float]:
#         context_preview = context[:1000] + "..." if len(context) > 1000 else context
#         prompt = f"""Check if the provided context contains information to answer the question.

# Question: {query}

# Context:
# {context_preview}

# Can this question be answered using ONLY the information in the context above?

# Respond with ONLY one of:
# [CAN_ANSWER] - context contains the answer
# [CANNOT_ANSWER] - context does NOT contain the answer

# Answer:"""
#         try:
#             model = self._get_llm()
#             output = model(prompt, max_tokens=10, temperature=0.1, stop=["\n"], echo=False)
#             response = output['choices'][0]['text'].strip().upper()
#             if "[CAN_ANSWER]" in response:
#                 return True, 0.8
#             elif "[CANNOT_ANSWER]" in response:
#                 return False, 0.8
#             return True, 0.4
#         except Exception as e:
#             print(f"[SelfRAG] Answer presence check error: {e}")
#             return True, 0.5

#     def verify_support(self, query: str, answer: str, context: str) -> Tuple[str, float]:
#         prompt = f"""Verify if the answer is supported by the provided context.

# Question: {query}

# Context:
# {context[:800]}...

# Answer:
# {answer}

# IMPORTANT: Check if the answer facts come from the context, or if the answer is making up information.

# Respond with ONLY one of:
# {SelfRAGReflectionTokens.FULLY_SUPPORTED} - answer is fully supported by context
# {SelfRAGReflectionTokens.PARTIALLY_SUPPORTED} - answer is partially supported
# {SelfRAGReflectionTokens.NO_SUPPORT} - answer is NOT supported or makes up information

# Answer:"""
#         try:
#             model = self._get_llm()
#             output = model(prompt, max_tokens=20, temperature=0.1, stop=["\n"], echo=False)
#             response = output['choices'][0]['text'].strip()
#             if SelfRAGReflectionTokens.FULLY_SUPPORTED in response:
#                 return "fully_supported", 0.9
#             elif SelfRAGReflectionTokens.PARTIALLY_SUPPORTED in response:
#                 return "partially_supported", 0.6
#             elif SelfRAGReflectionTokens.NO_SUPPORT in response:
#                 return "no_support", 0.9
#             return "uncertain", 0.4
#         except Exception as e:
#             print(f"[SelfRAG] Support verification error: {e}")
#             return "uncertain", 0.4

#     def evaluate_utility(self, query: str, answer: str) -> Tuple[int, float]:
#         prompt = f"""Evaluate how useful this answer is for the question.

# Question: {query}

# Answer:
# {answer}

# Rate the utility on a scale of 1-5:
# 5 = Excellent, complete, and directly answers the question
# 4 = Good, mostly answers the question
# 3 = Acceptable, provides some useful information
# 2 = Poor, barely addresses the question
# 1 = Very poor, does not answer the question

# Respond with ONLY one of:
# {SelfRAGReflectionTokens.UTILITY_5}
# {SelfRAGReflectionTokens.UTILITY_4}
# {SelfRAGReflectionTokens.UTILITY_3}
# {SelfRAGReflectionTokens.UTILITY_2}
# {SelfRAGReflectionTokens.UTILITY_1}

# Answer:"""
#         try:
#             model = self._get_llm()
#             output = model(prompt, max_tokens=15, temperature=0.1, stop=["\n"], echo=False)
#             response = output['choices'][0]['text'].strip()
#             if SelfRAGReflectionTokens.UTILITY_5 in response:
#                 return 5, 0.8
#             elif SelfRAGReflectionTokens.UTILITY_4 in response:
#                 return 4, 0.8
#             elif SelfRAGReflectionTokens.UTILITY_3 in response:
#                 return 3, 0.8
#             elif SelfRAGReflectionTokens.UTILITY_2 in response:
#                 return 2, 0.8
#             elif SelfRAGReflectionTokens.UTILITY_1 in response:
#                 return 1, 0.8
#             return 3, 0.4
#         except Exception as e:
#             print(f"[SelfRAG] Utility evaluation error: {e}")
#             return 3, 0.4

# # ============================================================================
# # DOMAIN CLASSIFICATION CLASS (FROM ORIGINAL RAG)
# # ============================================================================

# class DomainClassifier:
#     """
#     Classifies queries into domains using embedding similarity with anchor queries.
#     Uses pre-computed domain centroids from representative queries.
#     """
    
#     _domain_embeddings_cache = None
#     _embedding_model = None
    
#     @staticmethod
#     def initialize_domain_embeddings(model_name: str = None):
#         """
#         Pre-compute embeddings for all anchor queries and create domain centroids.
#         OPTIMIZED: Reuses main embedding model instead of loading a separate one.
#         """
#         if DomainClassifier._domain_embeddings_cache is not None:
#             return
        
#         if os.environ.get('BATCH_MODE') != 'True':
#             print("\n🔄 Initializing domain embeddings from anchor queries...")
        
#         # OPTIMIZATION: Reuse the main embedding model instead of loading a new one
#         # This saves ~1-2 seconds on first query
#         # Get the embedder function from the module namespace (avoid circular import)
#         import sys
#         current_module = sys.modules[__name__]
#         model = current_module.get_embedder()  # Reuse main embedding model
#         DomainClassifier._embedding_model = model
        
#         domain_embeddings = {}
        
#         for domain, queries in DOMAIN_ANCHOR_QUERIES.items():
#             # Use the same prefix format as query encoding for consistency
#             prefixed_queries = [f"query: {q}" for q in queries]
#             embeddings = model.encode(prefixed_queries, show_progress_bar=False, normalize_embeddings=True)
#             centroid = np.mean(embeddings, axis=0)
#             domain_embeddings[domain] = centroid
            
#             if os.environ.get('BATCH_MODE') != 'True':
#                 print(f"  ✓ {domain}: {len(queries)} anchor queries → centroid computed")
        
#         DomainClassifier._domain_embeddings_cache = domain_embeddings
        
#         if os.environ.get('BATCH_MODE') != 'True':
#             print("✅ Domain embeddings initialized!\n")
    
#     @staticmethod
#     def classify_domain(query: str) -> Dict[str, any]:
#         """
#         Classify query using embedding similarity to domain centroids.
#         OPTIMIZED: Reuses main embedding model.
#         """
#         if DomainClassifier._domain_embeddings_cache is None:
#             DomainClassifier.initialize_domain_embeddings()
        
#         # Use same prefix format as query encoding for consistency
#         query_prefixed = f"query: {query}"
#         query_embedding = DomainClassifier._embedding_model.encode([query_prefixed], normalize_embeddings=True)[0]
        
#         similarities = {}
#         for domain, centroid in DomainClassifier._domain_embeddings_cache.items():
#             query_reshaped = query_embedding.reshape(1, -1)
#             centroid_reshaped = centroid.reshape(1, -1)
#             similarity = cosine_similarity(query_reshaped, centroid_reshaped)[0][0]
#             similarities[domain] = float(similarity)
        
#         winning_domain = max(similarities, key=similarities.get)
#         confidence = similarities[winning_domain]
        
#         return {
#             'domain': winning_domain,
#             'confidence': confidence,
#             'all_scores': similarities
#         }
    
#     @staticmethod
#     def get_domain_emoji(domain: str) -> str:
#         """Get emoji representation for domain."""
#         emoji_map = {
#             'donation': '💰',
#             'healthcare': '🏥',
#             'general': '📋'
#         }
#         return emoji_map.get(domain, '📋')

# # ============================================================================
# # CONFIDENCE SCORING CLASS (FROM ORIGINAL RAG)
# # ============================================================================

# class ConfidenceScorer:
#     """
#     Implements multiple confidence scoring methods:
#     1. Token-level log probabilities (Average, Perplexity)
#     2. Entropy-based scoring
#     3. Retrieval confidence (semantic similarity)
#     """
    
#     @staticmethod
#     def calculate_perplexity(log_probs: List[float]) -> float:
#         """Calculate perplexity from log probabilities."""
#         if not log_probs:
#             return float('inf')
        
#         avg_log_prob = np.mean(log_probs)
#         perplexity = np.exp(-avg_log_prob)
#         alpha = 0.1
#         return float(np.exp(-alpha * (perplexity - 1)))
    
#     @staticmethod
#     def calculate_average_token_confidence(log_probs: List[float]) -> float:
#         """Calculate average token log-probability."""
#         if not log_probs:
#             return 0.0
        
#         probs = [np.exp(lp) for lp in log_probs]
#         return np.mean(probs)
    
#     @staticmethod
#     def calculate_entropy_confidence(token_probs_distributions: List[np.ndarray]) -> float:
#         """Calculate entropy-based confidence."""
#         if not token_probs_distributions:
#             return float('inf')
        
#         entropies = []
#         for prob_dist in token_probs_distributions:
#             if len(prob_dist) > 0:
#                 ent = entropy(prob_dist)
#                 entropies.append(ent)
        
#         if not entropies:
#             return float('inf')
        
#         max_entropy = max(entropies)
#         confidence = 1 - min(max_entropy / 10.0, 1.0)
#         return confidence
    
#     @staticmethod
#     def calculate_top_k_weighted_confidence(log_probs: List[float], k: int = 5) -> float:
#         """Weighted confidence score using top k% tokens."""
#         if not log_probs or len(log_probs) < 5:
#             return np.mean([np.exp(lp) for lp in log_probs]) if log_probs else 0.0
        
#         probs = [np.exp(lp) for lp in log_probs]
#         sorted_probs = sorted(probs, reverse=True)
#         top_k_probs = sorted_probs[:k]
        
#         joint_top_k = np.prod(top_k_probs) ** (1/k)
#         joint_all = np.prod(probs) ** (1/len(probs))
        
#         weighted_score = 0.7 * joint_top_k + 0.3 * joint_all
#         return weighted_score
    
#     @staticmethod
#     def calculate_retrieval_confidence(
#         query_embedding: np.ndarray, 
#         doc_embeddings: List[np.ndarray], 
#         top_k: int = 5
#     ) -> float:
#         """Calculate confidence based on cosine similarity between query and documents."""
#         if not doc_embeddings:
#             return 0.0
        
#         cosine_similarities = []
#         for doc_emb in doc_embeddings[:top_k]:
#             query_reshaped = query_embedding.reshape(1, -1)
#             doc_reshaped = doc_emb.reshape(1, -1)
#             cos_sim = cosine_similarity(query_reshaped, doc_reshaped)[0][0]
#             cosine_similarities.append(cos_sim)
        
#         return float(np.mean(cosine_similarities))
    
#     @staticmethod
#     def calculate_combined_confidence(
#         log_probs: List[float],
#         retrieval_confidence: float,
#         token_probs_distributions: List[np.ndarray] = None,
#         selfrag_scores: Dict = None
#     ) -> Dict[str, float]:
#         """Calculate all confidence metrics and return a comprehensive score."""
#         scores = {}
#         scores['retrieval_confidence'] = retrieval_confidence 
        
#         if log_probs:
#             scores['avg_token_confidence'] = ConfidenceScorer.calculate_average_token_confidence(log_probs)
#             scores['perplexity'] = ConfidenceScorer.calculate_perplexity(log_probs)
#             scores['weighted_top_k'] = ConfidenceScorer.calculate_top_k_weighted_confidence(log_probs)
        
#         if token_probs_distributions:
#             scores['entropy_confidence'] = ConfidenceScorer.calculate_entropy_confidence(token_probs_distributions)

#         if selfrag_scores:
#             scores['selfrag_support'] = selfrag_scores.get('support_score', 0.0)
#             scores['selfrag_utility'] = selfrag_scores.get('utility_score', 0.0)
#             scores['selfrag_relevance'] = selfrag_scores.get('relevance_score', 0.0)
        
#         # Combined score (weighted average)
#         combined = 0.0
#         weight_sum = 0.0
        
#         if 'retrieval_confidence' in scores:
#             combined += 0.3 * scores['retrieval_confidence']
#             weight_sum += 0.3
        
#         if 'avg_token_confidence' in scores:
#             combined += 0.15 * scores['avg_token_confidence']
#             weight_sum += 0.15
        
#         if 'weighted_top_k' in scores:
#             combined += 0.15 * scores['weighted_top_k']
#             weight_sum += 0.15

#         if 'selfrag_support' in scores:
#             combined += 0.25 * scores['selfrag_support']
#             weight_sum += 0.25
        
#         if 'selfrag_utility' in scores:
#             combined += 0.15 * scores['selfrag_utility']
#             weight_sum += 0.15
        
#         if weight_sum > 0:
#             scores['combined_confidence'] = combined / weight_sum
#         else:
#             scores['combined_confidence'] = 0.0
        
#         return scores

# # ============ Supabase Client ============
# _SUPABASE_CLIENT = None

# def get_supabase_client() -> Client:
#     """Get or create Supabase client"""
#     global _SUPABASE_CLIENT
    
#     if _SUPABASE_CLIENT is None:
#         if not SUPABASE_URL or not SUPABASE_KEY:
#             raise ValueError(
#                 "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file\n"
#                 "Get these from: Supabase Dashboard -> Settings -> API"
#             )
        
#         print(f"Connecting to Supabase: {SUPABASE_URL}")
#         _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
    
#     return _SUPABASE_CLIENT

# def test_connection():
#     """Test Supabase connection"""
#     try:
#         supabase = get_supabase_client()
#         result = supabase.table("documents").select("doc_id").limit(1).execute()
        
#         print("✅ Connected to Supabase successfully!")
#         print(f"   URL: {SUPABASE_URL}")
        
#         try:
#             result = supabase.rpc('match_documents_simple', {
#                 'query_embedding': [0.0] * 768,
#                 'match_count': 1
#             }).execute()
#             print("✅ pgvector extension verified")
#         except Exception as e:
#             if 'does not exist' in str(e):
#                 print("⚠️  match_documents function not found - run the SQL setup")
#             else:
#                 print(f"⚠️  Could not verify pgvector: {e}")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Connection failed: {e}")
#         return False

# # ============ Brand Term Protection ============
# def protect_brand_terms(text: str) -> str:
#     """Protect brand terms during translation to prevent corruption."""
#     for term in BRAND_TERMS:
#         text = re.sub(rf"\b{re.escape(term)}\b", f"@@{term}@@", text, flags=re.IGNORECASE)
#     return text

# def restore_brand_terms(text: str) -> str:
#     """Restore brand terms after translation."""
#     return text.replace("@@", "")

# # ============ Enhanced Language Detection ============
# def is_urdu_script(text: str) -> bool:
#     """Check if text contains Urdu script (Arabic block)."""
#     return bool(re.search(r'[\u0600-\u06FF]', text))

# def looks_like_roman_urdu(text: str) -> bool:
#     """Heuristic detection of Roman Urdu text."""
#     if is_urdu_script(text):
#         return False
#     # If it contains many non-latin characters, skip
#     if re.search(r'[^\x00-\x7F]', text):
#         return False
#     tokens = re.findall(r"[a-zA-Z']+", text.lower())
#     if not tokens:
#         return False
#     hits = sum(1 for t in tokens if t in ROMAN_URDU_MARKERS)
#     # Heuristic: at least 2 marker tokens OR 1 marker token with short query
#     if hits >= 2:
#         return True
#     if hits >= 1 and len(tokens) <= 6:
#         return True
#     return False

# def translate_auto_to_english(text: str) -> str:
#     """Fallback translation for roman urdu if transliteration isn't available."""
#     try:
#         return GoogleTranslator(source='auto', target='en').translate(text)
#     except Exception:
#         return text

# class QueryLangProfile:
#     """
#     Keeps the pipeline consistent:
#     - original_query: what user typed
#     - input_lang: 'en' | 'ur' | 'roman_ur'
#     - query_en: English version used for embeddings/retrieval/classification/critic prompts
#     - output_lang: same as input_lang (answer must match)
#     - query_urdu_script: if roman_ur, we also produce Urdu-script version for better downstream translation prompts
#     """
#     def __init__(self, original_query: str, input_lang: str, query_en: str,
#                  output_lang: str, query_urdu_script: Optional[str] = None):
#         self.original_query = original_query
#         self.input_lang = input_lang
#         self.query_en = query_en
#         self.output_lang = output_lang
#         self.query_urdu_script = query_urdu_script

# def build_query_lang_profile(query: str) -> QueryLangProfile:
#     """Build language profile for a query."""
#     q = query.strip()

#     # 1) Urdu script
#     if is_urdu_script(q) or detect_language(q) == "ur":
#         q_en = translate_urdu_to_english(q)
#         return QueryLangProfile(original_query=q, input_lang="ur", query_en=q_en, output_lang="ur", query_urdu_script=q)

#     # 2) Roman Urdu (heuristic)
#     if looks_like_roman_urdu(q):
#         # Prefer direct auto->English translation (stable, no extra deps)
#         q_en = translate_auto_to_english(q)
#         return QueryLangProfile(
#             original_query=q,
#             input_lang="roman_ur",
#             query_en=q_en,
#             output_lang="roman_ur",
#             query_urdu_script=None
#         )

#     # 3) Default English
#     return QueryLangProfile(original_query=q, input_lang="en", query_en=q, output_lang="en", query_urdu_script=None)

# # ============ Language Detection ============
# def detect_language(text: str) -> str:
#     try:
#         return langdetect.detect(text)
#     except Exception:
#         return "en"

# def is_urdu(text: str) -> bool:
#     urdu_pattern = re.compile(r'[\u0600-\u06FF]')
#     has_urdu = bool(urdu_pattern.search(text))
#     return has_urdu or detect_language(text) == "ur"

# def translate_urdu_to_english(text: str) -> str:
#     try:
#         return GoogleTranslator(source='ur', target='en').translate(text)
#     except Exception:
#         return text

# def translate_english_to_urdu(text: str, timeout: int = 10) -> str:
#     """
#     Translate English text to Urdu with timeout protection.
#     Returns original text if translation fails or times out.
#     Uses concurrent.futures for cross-platform timeout support.
#     """
#     try:
#         from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
#         def _translate():
#             return GoogleTranslator(source='en', target='ur').translate(text)
        
#         with ThreadPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(_translate)
#             try:
#                 result = future.result(timeout=timeout)
#                 return result if result else text
#             except FutureTimeoutError:
#                 print(f"[WARNING] Translation timed out after {timeout}s. Returning original text.", flush=True)
#                 return text
#     except Exception as e:
#         print(f"[WARNING] Translation failed: {e}. Returning original text.", flush=True)
#         return text

# # ============ Query Analysis ============
# def analyze_query(query: str) -> Dict[str, Any]:
#     """Detect what kind of answer the user wants"""
#     q_lower = query.lower().strip()

#     # Treat "how to" questions as procedural (steps)
#     procedural_markers = ["how to", "how do i", "how can i", "steps", "step by step", "procedure"]
#     roman_markers = ["kaise", "kesy", "kese", "kaisay", "kya", "kyun", "kyu"]

#     list_keywords = ['list', 'points', 'bullet', 'enumerate', 'ways', 'methods']
#     summary_keywords = ['summarize', 'summary', 'briefly', 'overview', 'خلاصہ', 'khulasa']
#     detail_keywords = ['explain', 'detail', 'describe', 'why', 'تفصیل', 'tafseel']

#     wants_steps = any(m in q_lower for m in procedural_markers) or any(m in q_lower for m in roman_markers)
#     wants_list = wants_steps or any(kw in q_lower for kw in list_keywords)
#     wants_summary = any(kw in q_lower for kw in summary_keywords)
#     wants_detail = any(kw in q_lower for kw in detail_keywords)

#     return {
#         'wants_list': wants_list,
#         'wants_summary': wants_summary,
#         'wants_detail': wants_detail,
#         'is_urdu': is_urdu(query)
#     }

# def expand_query_for_retrieval(query_en: str, domain: str, query_info: Dict[str, Any]) -> str:
#     """
#     Small deterministic expansion for short/procedural queries to improve retrieval.
#     Keeps core RAG logic identical; only changes the string embedded.
#     """
#     q = query_en.strip()
#     if len(q.split()) >= 6 and not query_info.get("wants_list"):
#         return q

#     if domain == "donation":
#         extra = "donate donation methods how to donate steps JazzCash EasyPaisa bank transfer online donation international account"
#         return f"{q}. {extra}"
#     if domain == "healthcare":
#         extra = "Alkhidmat hospital clinic services eligibility locations how to get treatment"
#         return f"{q}. {extra}"

#     return q

# # ============ Document Processing ============
# def load_documents_from_zip(zip_path: str) -> Dict[str, List[Dict]]:
#     if not os.path.exists(zip_path):
#         raise FileNotFoundError(f"ZIP file not found: {zip_path}")
#     documents_by_category = {}
#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         for file_path in zip_ref.namelist():
#             if file_path.endswith("/") or not file_path.endswith(".txt"):
#                 continue
#             parts = Path(file_path).parts
#             if len(parts) < 3:
#                 continue
#             category = parts[-2]
#             filename = parts[-1]
#             try:
#                 with zip_ref.open(file_path) as f:
#                     content = f.read().decode("utf-8")
#                 if content.strip():
#                     documents_by_category.setdefault(category, []).append({
#                         "content": content,
#                         "filename": filename,
#                         "category": category,
#                         "file_path": file_path
#                     })
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
#     return documents_by_category

# def clean_text(text: str) -> str:
#     """Enhanced cleaning while preserving structure"""
#     text = re.sub(r'={3,}[\s\S]*?={3,}', '', text)
#     text = re.sub(r'URL:\s*https?://\S+', '', text)
#     text = re.sub(r'TITLE:.*?\n', '', text)
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     text = re.sub(r'[ \t]+', ' ', text)
#     return text.strip()

# def prepare_documents(zip_path: str) -> Tuple[List[str], List[Dict]]:
#     docs_by_cat = load_documents_from_zip(zip_path)
#     all_docs, metadata = [], []
#     for cat, docs in docs_by_cat.items():
#         for doc in docs:
#             c = clean_text(doc["content"])
#             if c:
#                 all_docs.append(c)
#                 metadata.append({
#                     "filename": doc["filename"],
#                     "category": doc["category"],
#                     "file_path": doc["file_path"]
#                 })
#     print(f"Prepared {len(all_docs)} documents")
#     return all_docs, metadata

# def split_documents(documents: List[str], metadata: List[Dict]):
#     """Improved semantic chunking"""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         length_function=len,
#         separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "؟ ", "۔ ", " ", ""]
#     )
#     chunks, metas = [], []
#     for doc, meta in zip(documents, metadata):
#         parts = splitter.split_text(doc)
#         for idx, p in enumerate(parts):
#             chunks.append(p)
#             chunk_meta = meta.copy()
#             chunk_meta['chunk_index'] = idx
#             metas.append(chunk_meta)
    
#     avg_len = int(np.mean([len(c) for c in chunks]))
#     print(f"Split into {len(chunks)} chunks (avg {avg_len} chars, overlap {CHUNK_OVERLAP})")
#     return chunks, metas

# # ============ Embeddings ============
# _EMBEDDER = None

# def get_embedder():
#     global _EMBEDDER
#     if _EMBEDDER is None:
#         print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
#         _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
#     return _EMBEDDER

# def create_embeddings(text_chunks: List[str]) -> np.ndarray:
#     embedder = get_embedder()
#     print(f"Creating embeddings for {len(text_chunks)} chunks...")
#     prefixed = [f"passage: {chunk}" for chunk in text_chunks]
#     embs = embedder.encode(prefixed, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
#     embs = np.array(embs).astype("float32")
#     print("Embeddings created:", embs.shape)
#     return embs

# # ============ Supabase Storage ============
# def save_chunks_to_supabase(chunks: List[str], metadata: List[Dict], embeddings: np.ndarray):
#     """Store chunks and embeddings in Supabase using client"""
#     supabase = get_supabase_client()
    
#     print("Saving to Supabase...")
    
#     rows = []
#     for i, chunk in enumerate(chunks):
#         meta = metadata[i]
#         emb_list = embeddings[i].tolist()
        
#         rows.append({
#             "doc_id": str(uuid.uuid4()),
#             "chunk_text": chunk,
#             "chunk_index": meta.get("chunk_index", 0),
#             "category": meta.get("category"),
#             "filename": meta.get("filename"),
#             "file_path": meta.get("file_path"),
#             "doc_domain": meta.get("category"),
#             "embedding": emb_list
#         })
    
#     batch_size = 100
#     total_inserted = 0
    
#     for i in range(0, len(rows), batch_size):
#         batch = rows[i:i + batch_size]
#         try:
#             result = supabase.table("documents").insert(batch).execute()
#             total_inserted += len(batch)
#             print(f"   Inserted batch {i//batch_size + 1}: {len(batch)} chunks")
#         except Exception as e:
#             print(f"   ⚠️  Error inserting batch {i//batch_size + 1}: {e}")
#             for row in batch:
#                 try:
#                     supabase.table("documents").insert(row).execute()
#                     total_inserted += 1
#                 except Exception as e2:
#                     print(f"   ⚠️  Failed to insert chunk: {e2}")
    
#     print(f"✅ Stored {total_inserted}/{len(rows)} chunks in Supabase")

# def clear_documents_table():
#     """Clear all documents (use with caution!)"""
#     supabase = get_supabase_client()
    
#     try:
#         result = supabase.table("documents").delete().neq("doc_id", "00000000-0000-0000-0000-000000000000").execute()
#         print("✅ Cleared all documents from Supabase")
#     except Exception as e:
#         print(f"⚠️  Error clearing documents: {e}")

# # ============ Build Pipeline ============
# def build_alkhidmat_rag(zip_path: str, clear_existing: bool = False):
#     """Build the RAG system and store in Supabase"""
#     print("\n" + "="*80)
#     print("BUILDING ALKHIDMAT RAG SYSTEM")
#     print("="*80)
    
#     if not test_connection():
#         print("❌ Cannot connect to Supabase. Aborting.")
#         return
    
#     if clear_existing:
#         print("\n⚠️  Clearing existing documents...")
#         clear_documents_table()
    
#     print("\nBUILD: Loading documents...")
#     docs, meta = prepare_documents(zip_path)
#     if not docs:
#         print("❌ No documents found. Aborting.")
#         return
    
#     print("\nBUILD: Splitting with improved chunking...")
#     chunks, chunk_meta = split_documents(docs, meta)
    
#     print("\nBUILD: Computing embeddings...")
#     embeddings = create_embeddings(chunks)
    
#     print("\nBUILD: Saving to Supabase...")
#     save_chunks_to_supabase(chunks, chunk_meta, embeddings)
    
#     print("\nBUILD: Initializing domain classification...")
#     DomainClassifier.initialize_domain_embeddings()
    
#     print("\n" + "="*80)
#     print("✅ BUILD COMPLETE!")
#     print("="*80)

# # ============ Retrieval (ENHANCED WITH EMBEDDINGS) ============
# def retrieve_from_supabase(query: str, top_k: int = 5, 
#                            filter_category: str = None) -> Tuple[List[Dict], np.ndarray, List[np.ndarray]]:
#     """
#     FIXED: Now properly extracts embeddings from Supabase for confidence scoring
#     """
#     embed_start = time.time()
#     embedder = get_embedder()
#     query_prefixed = f"query: {query}"
#     query_embedding = embedder.encode([query_prefixed], normalize_embeddings=True)[0]
#     print(f"[TIMING] Query embedding: {time.time() - embed_start:.2f}s", flush=True)
    
#     supabase = get_supabase_client()
    
#     try:
#         rpc_start = time.time()
#         params = {
#             'query_embedding': query_embedding.tolist(),
#             'match_threshold': RELEVANCE_THRESHOLD,
#             'match_count': top_k
#         }
        
#         if filter_category:
#             params['filter_category'] = filter_category
        
#         # First, get the matches using RPC
#         result = supabase.rpc('match_documents', params).execute()
#         rows = result.data
#         print(f"[TIMING] Supabase RPC call: {time.time() - rpc_start:.2f}s", flush=True)
        
#         # OPTIMIZED: Only fetch embeddings if we have results
#         # The RPC already returns similarity and text, so we only need embeddings for confidence scoring
#         if rows:
#             fetch_start = time.time()
#             doc_ids = [row['doc_id'] for row in rows]
            
#             # OPTIMIZATION: Only fetch embeddings, not full documents (we already have text from RPC)
#             # This reduces data transfer and speeds up the query
#             full_docs_result = supabase.table("documents").select(
#                 "doc_id, embedding"  # Only fetch what we need
#             ).in_("doc_id", doc_ids).execute()
            
#             # Create a mapping for easy lookup
#             doc_map = {doc['doc_id']: doc for doc in full_docs_result.data}
            
#             # Merge embedding data back into rows
#             for row in rows:
#                 if row['doc_id'] in doc_map:
#                     row['embedding'] = doc_map[row['doc_id']]['embedding']
#             print(f"[TIMING] Fetch embeddings: {time.time() - fetch_start:.2f}s", flush=True)
        
#     except Exception as e:
#         print(f"⚠️  RPC function not available, using fallback method: {e}")
        
#         # Fallback: manual similarity calculation
#         result = supabase.table("documents").select(
#             "doc_id, chunk_text, category, filename, file_path, chunk_index, embedding"
#         ).execute()
        
#         rows = []
#         for row in result.data:
#             if row['embedding']:
#                 doc_emb = np.array(row['embedding'])
#                 similarity = float(np.dot(query_embedding, doc_emb))
                
#                 if similarity > RELEVANCE_THRESHOLD:
#                     row['similarity'] = similarity
#                     rows.append(row)
        
#         rows = sorted(rows, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
#     results = []
#     doc_embeddings = []
    
#     for row in rows:
#         results.append({
#             "text": row['chunk_text'],
#             "category": row['category'],
#             "filename": row['filename'],
#             "file_path": row['file_path'],
#             "chunk_index": row.get('chunk_index', 0),
#             "similarity": float(row.get('similarity', 0))
#         })
        
#         # FIXED: Properly extract and convert embedding
#         if 'embedding' in row and row['embedding'] is not None:
#             try:
#                 # Handle different embedding formats
#                 if isinstance(row['embedding'], str):
#                     # If it's a string representation, try to parse it
#                     import ast
#                     embedding_data = ast.literal_eval(row['embedding'])
#                     doc_embeddings.append(np.array(embedding_data, dtype=np.float32))
#                 elif isinstance(row['embedding'], list):
#                     # If it's already a list
#                     doc_embeddings.append(np.array(row['embedding'], dtype=np.float32))
#                 elif isinstance(row['embedding'], np.ndarray):
#                     # If it's already a numpy array
#                     doc_embeddings.append(row['embedding'].astype(np.float32))
#                 else:
#                     print(f"⚠️  Unexpected embedding type: {type(row['embedding'])}")
#             except Exception as e:
#                 print(f"⚠️  Error parsing embedding: {e}")
#                 continue
    
#     print("\n" + "="*80, flush=True)
#     print("RETRIEVAL FROM SUPABASE", flush=True)
#     print(f"Retrieved {len(results)} relevant chunks (threshold: {RELEVANCE_THRESHOLD}):", flush=True)
#     for i, r in enumerate(results, 1):
#         print(f"{i}. [{r['category']}] {r['filename']} (similarity: {r['similarity']:.3f})", flush=True)
#     print(f"📊 Document embeddings extracted: {len(doc_embeddings)}/{len(results)}", flush=True)
#     print("="*80 + "\n", flush=True)
#     sys.stdout.flush()  # Force flush
    
#     return results, query_embedding, doc_embeddings

# def sanitize_chunk_text(text: str) -> str:
#     """Remove existing Q/A labels from chunks"""
#     text = re.sub(r'(?mi)^\s*(user\s+question|question|q:)\s*[:\-–]?\s*.*$', '', text)
#     text = re.sub(r'(?mi)^\s*(answer|a:)\s*[:\-–]?\s*.*$', '', text)
#     text = re.sub(r'(?is)(?:^|\n)\s*q[:\.\-\)]\s*.*?\n\s*a[:\.\-\)]\s*.*?(?:\n|$)', '', text)
#     text = re.sub(r'\[insert .*?\]', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'click here', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     return text.strip()

# # ============ LLM (Llama CPP) ============
# _LLM_MODEL = None

# def detect_apple_silicon():
#     """Detect if running on Apple Silicon (M1/M2/M3/etc)"""
#     import platform
#     try:
#         # Check if running on macOS
#         if platform.system() != "Darwin":
#             return False
        
#         # Check processor architecture
#         machine = platform.machine()
#         if machine == "arm64":
#             return True
        
#         # Alternative check using uname
#         import subprocess
#         result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
#         if result.returncode == 0 and 'arm64' in result.stdout:
#             return True
        
#         return False
#     except Exception:
#         return False

# def load_llm(model_filename: str = LLM_MODEL_FILENAME):
#     global _LLM_MODEL
#     if _LLM_MODEL is None:
#         print("Loading local LLM via llama-cpp:", model_filename)
        
#         if not os.path.exists(model_filename):
#             print(f"❌ Error: Model file not found at {model_filename}")
#             raise FileNotFoundError(f"Model file missing: {model_filename}")

#         # Detect Apple Silicon for Metal GPU acceleration
#         is_apple_silicon = detect_apple_silicon()
#         gpu_layers = -1 if is_apple_silicon else 0  # -1 = use all GPU layers on Metal
        
#         if is_apple_silicon:
#             print("🍎 Apple Silicon detected - enabling Metal GPU acceleration")
#         else:
#             print("💻 Using CPU mode (no GPU acceleration)")

#         _LLM_MODEL = Llama(
#             model_path=model_filename,
#             n_ctx=4096,           
#             n_gpu_layers=gpu_layers,  # -1 for Apple Silicon (Metal), 0 for CPU
#             verbose=False,
#             logits_all=True  # FIXED: Enable logits for log probability extraction
#         )
        
#         if is_apple_silicon:
#             print("✅ LLM loaded with Metal GPU acceleration")
#         else:
#             print("✅ LLM loaded in CPU mode")
#     return _LLM_MODEL

# def llm_generate(prompt: str, max_tokens: int = 400, stop_tokens: list = None) -> Tuple[str, List[float], List[np.ndarray]]:
#     """
#     MEMORY-OPTIMIZED: Generation with garbage collection
#     """
#     model = load_llm()
    
#     # Force garbage collection before generation
#     gc.collect()

#     try:
#         output = model(
#             prompt,
#             max_tokens=max_tokens,
#             temperature=0.2,
#             top_p=0.9,
#             repeat_penalty=1.2,
#             stop=stop_tokens or [],
#             echo=False,
#             logprobs=5
#         )
        
#         text = output['choices'][0]['text'].strip()
        
#         # Extract log probabilities if available
#         log_probs = []
#         token_probs_distributions = []
        
#         if 'logprobs' in output['choices'][0] and output['choices'][0]['logprobs']:
#             logprobs_data = output['choices'][0]['logprobs']
            
#             if 'token_logprobs' in logprobs_data and logprobs_data['token_logprobs']:
#                 log_probs = [lp for lp in logprobs_data['token_logprobs'] if lp is not None]
            
#             if 'top_logprobs' in logprobs_data and logprobs_data['top_logprobs']:
#                 for token_dict in logprobs_data['top_logprobs']:
#                     if token_dict:
#                         logprobs_list = list(token_dict.values())
#                         probs = np.exp(logprobs_list)
#                         probs = probs / np.sum(probs)
#                         token_probs_distributions.append(probs)
        
#         # Cleanup after generation
#         gc.collect()
        
#         return text, log_probs, token_probs_distributions
        
#     except Exception as e:
#         print(f"LLM Generation Error: {e}")
#         gc.collect()  # Cleanup even on error
#         return "Error generating response.", [], []
    
# # ============ Answer Generation (ENHANCED) ============
# def generate_answer(query: str, top_k: int = 5, max_tokens: int = 400, 
#                     filter_category: str = None):
#     """
#     ENHANCED: Now includes domain classification and confidence scoring
#     """
#     start_time = time.time()
#     print(f"[RAG] Processing query: {query[:50]}...", flush=True)
#     sys.stdout.flush()
    
#     query_info = analyze_query(query)
#     is_urdu = query_info['is_urdu']
#     original_query = query
    
#     # CLASSIFY DOMAIN (OPTIMIZED: Reuses embedding model)
#     domain_start = time.time()
#     print(f"[RAG] Classifying domain...", flush=True)
#     sys.stdout.flush()
#     domain_classification = DomainClassifier.classify_domain(query)
#     print(f"[TIMING] Domain classification: {time.time() - domain_start:.2f}s", flush=True)
    
#     # Retrieve with embeddings (ENHANCED)
#     retrieval_start = time.time()
#     print(f"[RAG] Retrieving from Supabase (top_k={top_k})...", flush=True)
#     sys.stdout.flush()
#     results, query_embedding, doc_embeddings = retrieve_from_supabase(
#         query, top_k=top_k, filter_category=filter_category
#     )
#     print(f"[TIMING] Retrieval: {time.time() - retrieval_start:.2f}s", flush=True)
    
#     if not results:
#         no_answer = "معاف کیجیے، میں اس سوال کا جواب دینے کے لیے متعلقہ معلومات نہیں ڈھونڈ سکا۔" if is_urdu else "I apologize, but I couldn't find relevant information to answer this question."
#         return no_answer, query, is_urdu, [], {}, domain_classification
    
#     # Build context
#     context_parts = []
#     for r in results:
#         chunk_text = sanitize_chunk_text(r["text"])
        
#         if is_urdu:
#             try:
#                 chunk_text = translate_english_to_urdu(chunk_text)
#             except Exception:
#                 pass
        
#         if chunk_text:
#             MAX_CHUNK_CHARS = 1200
#             if len(chunk_text) > MAX_CHUNK_CHARS:
#                 chunk_text = chunk_text[:MAX_CHUNK_CHARS].rsplit('\n', 1)[0] + "\n\n[truncated]"
#             context_parts.append(chunk_text)
    
#     context = "\n\n".join(context_parts)
    
#     # Build prompt
#     if is_urdu:
#         base_instruction = "آپ الخدمت فاؤنڈیشن پاکستان کے لیے ایک مددگار اسسٹنٹ ہیں۔"
        
#         if query_info['wants_list']:
#             format_instruction = "براہ کرم نقاط کی شکل میں واضح جواب دیں۔"
#         elif query_info['wants_summary']:
#             format_instruction = "براہ کرم 2-3 جملوں میں مختصر خلاصہ دیں۔"
#         else:
#             format_instruction = "براہ کرم مکمل اور واضح جواب دیں۔ اگر سیاق و سباق میں تفصیلات ہیں تو سب شامل کریں۔"
        
#         prompt = f"""{base_instruction}
# {format_instruction}

# صرف دیے گئے سیاق و سباق کی معلومات استعمال کریں۔ اگر جواب سیاق و سباق میں نہیں تو "مجھے معلوم نہیں" کہیں۔

# سیاق و سباق:
# {context}

# سوال: {query}

# جواب (صرف اردو میں، ماخذ شامل نہ کریں):
# """
#     else:
#         base_instruction = "You are a helpful customer support agent for Alkhidmat Foundation Pakistan."
        
#         if query_info['wants_list']:
#             format_instruction = "Provide a clear answer in bullet point format."
#         elif query_info['wants_summary']:
#             format_instruction = "Provide a brief summary in 2-3 sentences."
#         elif query_info['wants_detail']:
#             format_instruction = "Provide a detailed, comprehensive answer covering all relevant information from the context."
#         else:
#             format_instruction = "Provide a clear, complete answer. Include all relevant details from the context."
        
#         prompt = f"""{base_instruction}

# {format_instruction}

# Important instructions:
# - Use ONLY the information provided below.
# - Return ONLY the direct answer to the user's question.
# - DO NOT include any additional 'User question:', 'Question:', 'Answer:' lines, headings, or extra Q/A pairs.
# - DO NOT copy the context verbatim or print verbatim Q/A excerpts.
# - If the information is not present, reply exactly: "I don't have that information."
# - Output exactly one answer and nothing else (no extra labels, no summaries).

# Information:
# {context}

# User question: {query}

# Answer (ONLY the final answer, no labels):
# """
    
#     # Generate with confidence data (ENHANCED)
#     llm_start = time.time()
#     print(f"[RAG] Generating answer with LLM...", flush=True)
#     sys.stdout.flush()
#     answer, log_probs, token_probs_distributions = llm_generate(
#         prompt, max_tokens=max_tokens, stop_tokens=["\nUser question:", "\nQuestion:", "\nUser question"]
#     )
#     print(f"[TIMING] LLM generation: {time.time() - llm_start:.2f}s", flush=True)
    
#     # Clean response
#     print(f"[RAG] Cleaning response...", flush=True)
#     sys.stdout.flush()
#     answer = clean_llm_response(answer)
    
#     # CALCULATE CONFIDENCE SCORES
#     print(f"[RAG] Calculating confidence scores...", flush=True)
#     sys.stdout.flush()
#     retrieval_conf = ConfidenceScorer.calculate_retrieval_confidence(
#         query_embedding=query_embedding,
#         doc_embeddings=doc_embeddings,
#         top_k=top_k
#     )
    
#     confidence_scores = ConfidenceScorer.calculate_combined_confidence(
#         log_probs=log_probs,
#         retrieval_confidence=retrieval_conf,
#         token_probs_distributions=token_probs_distributions
#     )
    
#     # Print confidence scores
#     print("\n" + "="*80, flush=True)
#     print("CONFIDENCE SCORES:", flush=True)
#     print("="*80, flush=True)
#     combined_conf = confidence_scores.get('combined_confidence', 0) if isinstance(confidence_scores, dict) else 0
#     retrieval_conf_val = confidence_scores.get('retrieval_confidence', 0) if isinstance(confidence_scores, dict) else 0
#     avg_token_conf = confidence_scores.get('avg_token_confidence', 0) if isinstance(confidence_scores, dict) else 0
#     weighted_top_k = confidence_scores.get('weighted_top_k', 0) if isinstance(confidence_scores, dict) else 0
#     perplexity = confidence_scores.get('perplexity', 0) if isinstance(confidence_scores, dict) else 0
#     entropy_conf = confidence_scores.get('entropy_confidence', 0) if isinstance(confidence_scores, dict) else 0
    
#     print(f"  Combined Confidence:        {combined_conf:.4f} ⭐", flush=True)
#     print(f"  ├─ Retrieval Confidence:    {retrieval_conf_val:.4f}", flush=True)
#     print(f"  ├─ Avg Token Confidence:    {avg_token_conf:.4f}", flush=True)
#     print(f"  ├─ Weighted Top-K:          {weighted_top_k:.4f}", flush=True)
#     print(f"  ├─ Perplexity:              {perplexity:.4f}", flush=True)
#     print(f"  └─ Entropy Confidence:      {entropy_conf:.4f}", flush=True)
#     print("="*80, flush=True)
    
#     # Interpretation
#     if combined_conf >= 0.7:
#         print("✅ High confidence - Answer is likely reliable", flush=True)
#     elif combined_conf >= 0.5:
#         print("⚠️  Moderate confidence - Answer may need verification", flush=True)
#     else:
#         print("❌ Low confidence - Answer should be verified from sources", flush=True)
#     print("="*80 + "\n", flush=True)
#     sys.stdout.flush()
    
#     # Translate answer back to Urdu if needed
#     if is_urdu:
#         translation_start = time.time()
#         print(f"[RAG] Translating answer to Urdu...", flush=True)
#         sys.stdout.flush()
#         try:
#             answer = translate_english_to_urdu(answer, timeout=15)  # 15 second timeout
#             print(f"[TIMING] Translation: {time.time() - translation_start:.2f}s", flush=True)
#         except Exception as e:
#             print(f"[WARNING] Translation failed: {e}. Using English answer.", flush=True)
#             # Keep the English answer if translation fails
#             print(f"[TIMING] Translation (failed): {time.time() - translation_start:.2f}s", flush=True)
    
#     # Create sources list
#     sources = [{
#         "category": r['category'],
#         "filename": r['filename'],
#         "file_path": r['file_path'],
#         "similarity": r['similarity']
#     } for r in results]
    
#     total_time = time.time() - start_time
#     print(f"[RAG] Answer generated successfully (length: {len(answer)} chars)", flush=True)
#     print(f"[TIMING] Total time: {total_time:.2f}s", flush=True)
#     sys.stdout.flush()
    
#     return answer, original_query, is_urdu, sources, confidence_scores, domain_classification

# LATIN_ONLY_RE = re.compile(r'^[\x00-\x7F\s]+$')  # only ASCII + whitespace

# def romanize_to_roman_urdu_with_llm(urdu_text: str, max_tokens: int = 260) -> str:
#     """Convert Urdu script to Roman Urdu using LLM."""
#     if not urdu_text.strip():
#         return urdu_text

#     # Attempt 1-2: direct Urdu -> Roman Urdu
#     for attempt in range(2):
#         prompt = f"""Task: Convert Urdu (Arabic script) to Roman Urdu (Latin letters).

# STRICT RULES:
# - Output MUST be in Latin letters only (a-z). No Urdu/Arabic characters at all.
# - If you output any Urdu/Arabic characters, your answer is INVALID.
# - Keep meaning exactly the same.
# - Keep proper names like Alkhidmat as "Alkhidmat".
# - Keep phone numbers/URLs unchanged.
# - Output ONLY the Roman Urdu text (no labels).

# Urdu:
# {urdu_text}

# Roman Urdu (Latin-only):"""

#         model = load_llm()
#         out = model(prompt, max_tokens=max_tokens, temperature=0.0, stop=["\n\n", "\nUrdu:", "\nRoman Urdu:"], echo=False)
#         out = out['choices'][0]['text'].strip()

#         # Accept only if it's truly Latin-only AND not Urdu-script
#         if out and (not is_urdu_script(out)) and LATIN_ONLY_RE.match(out):
#             return out

#     # Fallback: Urdu -> English via GoogleTranslator, then English -> Roman Urdu via LLM
#     en = translate_urdu_to_english(urdu_text)
#     prompt2 = f"""Task: Translate English into Roman Urdu (Latin letters).

# STRICT RULES:
# - Output MUST be in Latin letters only (a-z). No Urdu/Arabic characters.
# - Keep meaning exactly the same.
# - Keep proper names like Alkhidmat as "Alkhidmat".
# - Keep phone numbers/URLs unchanged.
# - Output ONLY Roman Urdu text (no labels).

# English:
# {en}

# Roman Urdu (Latin-only):"""

#     model = load_llm()
#     out2 = model(prompt2, max_tokens=max_tokens, temperature=0.0, echo=False)
#     out2 = out2['choices'][0]['text'].strip()
#     if out2 and (not is_urdu_script(out2)) and LATIN_ONLY_RE.match(out2):
#         return out2

#     # Last resort: return an English version rather than Urdu (better than violating roman_ur contract)
#     return en

# def clean_llm_response(text: str) -> str:
#     """Clean up LLM output to remove unwanted artifacts"""
#     if not text:
#         return text

#     text = text.replace('\r\n', '\n')
#     text = re.sub(r'\[?Context \d+\]?', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'based on Context \d+', '', text, flags=re.IGNORECASE)
#     text = re.sub(r'as per (?:the )?context', '', text, flags=re.IGNORECASE)

#     cutoff_patterns = [r'\nUser question\s*:', r'\nQuestion\s*:', r'\nUser question', r'\nQuestion', r'\nAnswer\s*:']
#     earliest = len(text)
#     for pat in cutoff_patterns:
#         m = re.search(pat, text, flags=re.IGNORECASE)
#         if m:
#             earliest = min(earliest, m.start())

#     if earliest < len(text):
#         text = text[:earliest].strip()

#     text = re.sub(r'(?mi)^(user question|question|answer)\s*[:\-–]\s*', '', text)

#     lines = text.split('\n')
#     seen = set()
#     out_lines = []
#     for line in lines:
#         s = line.strip()
#         if not s:
#             out_lines.append('')
#             continue
#         if s.lower() in seen:
#             continue
#         seen.add(s.lower())
#         out_lines.append(line)
#     text = '\n'.join(out_lines)

#     text = re.sub(r'\n{3,}', '\n\n', text).strip()
#     return text

# # ============ SELF-RAG ANSWER GENERATION ============
# def generate_answer_selfrag(query: str, top_k: int = 5, max_tokens: int = 400, filter_category: str = None):
#     """
#     Multilingual Self-RAG implementation with enhanced verification.
#     Returns: (answer, original_query, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics)
#     """
#     start_time = time.time()
#     print(f"\n{'='*80}", flush=True)
#     print(f"SELF-RAG QUERY PROCESSING", flush=True)
#     print(f"{'='*80}", flush=True)
#     print(f"[SELF-RAG] Processing query: {query[:80]}...", flush=True)
#     sys.stdout.flush()

#     profile = build_query_lang_profile(query)
#     query_info = analyze_query(profile.original_query)
#     query_for_rag = profile.query_en
#     original_query = profile.original_query

#     llm_model = load_llm()
#     critic = SelfRAGCritic(llm_model)

#     selfrag_metrics = {
#         'domain_relevant': True,
#         'domain_confidence': 0.0,
#         'retrieve_needed': False,
#         'retrieve_confidence': 0.0,
#         'answer_in_context': False,
#         'answer_in_context_confidence': 0.0,
#         'relevance_score': 0.0,
#         'support_level': 'uncertain',
#         'support_score': 0.0,
#         'utility_score': 0.0,
#         'utility_rating': 0
#     }

#     # STEP 0: Check domain relevance
#     if SELFRAG_ENABLE:
#         print(f"\n[SELF-RAG] Step 0: Checking domain relevance...", flush=True)
#         is_domain_relevant, domain_conf = critic.is_domain_relevant(query_for_rag)
#         selfrag_metrics['domain_relevant'] = is_domain_relevant
#         selfrag_metrics['domain_confidence'] = domain_conf

#         if (not is_domain_relevant) and (domain_conf >= 0.75):
#             print(f"  ✗ Question is IRRELEVANT to Alkhidmat Foundation (confidence: {domain_conf:.2f})", flush=True)
#             dummy_domain_classification = {
#                 'domain': 'irrelevant',
#                 'confidence': 0.0,
#                 'all_scores': {'donation': 0.0, 'healthcare': 0.0, 'general': 0.0}
#             }
#             irrelevant_response = "That is an irrelevant question."
#             return (irrelevant_response, original_query, profile.input_lang, [], {'combined_confidence': 0.0},
#                     dummy_domain_classification, selfrag_metrics)
#         else:
#             print(f"  ✓ Question is RELEVANT to domain (confidence: {domain_conf:.2f})", flush=True)

#     # STEP 1: Should we retrieve?
#     if SELFRAG_ENABLE:
#         print(f"\n[SELF-RAG] Step 1: Checking retrieval necessity...", flush=True)
#         retrieve_needed, retrieve_conf = critic.should_retrieve(query_for_rag)
#         selfrag_metrics['retrieve_needed'] = retrieve_needed
#         selfrag_metrics['retrieve_confidence'] = retrieve_conf

#         if retrieve_needed:
#             print(f"  ✓ Retrieval NEEDED (confidence: {retrieve_conf:.2f})", flush=True)
#         else:
#             print(f"  ✗ Retrieval NOT needed (confidence: {retrieve_conf:.2f})", flush=True)
#             no_retrieval_answer = "I can help with that, but I need to access my knowledge base for specific information about Alkhidmat Foundation."
#             return (no_retrieval_answer, original_query, profile.input_lang, [],
#                     {'combined_confidence': 0.5}, {}, selfrag_metrics)
#     else:
#         retrieve_needed = True
#         selfrag_metrics['retrieve_needed'] = True

#     # Classify domain
#     domain_start = time.time()
#     print(f"\n[SELF-RAG] Classifying domain...", flush=True)
#     sys.stdout.flush()
#     domain_classification = DomainClassifier.classify_domain(query_for_rag)
#     print(f"[TIMING] Domain classification: {time.time() - domain_start:.2f}s", flush=True)
#     winning_domain = domain_classification.get("domain", "general")

#     # Expand query for retrieval
#     retrieval_query_en = expand_query_for_retrieval(query_for_rag, winning_domain, query_info)

#     # STEP 2: Retrieve documents
#     retrieval_start = time.time()
#     print(f"\n[SELF-RAG] Step 2: Retrieving documents...", flush=True)
#     sys.stdout.flush()
#     results, query_embedding, doc_embeddings = retrieve_from_supabase(
#         retrieval_query_en, top_k=top_k, filter_category=filter_category
#     )
#     print(f"[TIMING] Retrieval: {time.time() - retrieval_start:.2f}s", flush=True)

#     if not results:
#         if profile.output_lang == "ur":
#             no_answer = "معاف کیجیے، میں اس سوال کا جواب دینے کے لیے متعلقہ معلومات نہیں ڈھونڈ سکا۔"
#         else:
#             no_answer = "I apologize, but I couldn't find relevant information to answer this question."
#         return no_answer, original_query, profile.input_lang, [], {}, domain_classification, selfrag_metrics

#     # STEP 3: Assess relevance
#     if SELFRAG_ENABLE:
#         print(f"\n[SELF-RAG] Step 3: Assessing document relevance...", flush=True)
#         relevant_results = []
#         relevance_scores = []

#         for i, result in enumerate(results):
#             is_relevant, rel_conf = critic.assess_relevance(query_for_rag, result['text'])
#             relevance_scores.append(rel_conf if is_relevant else 0.0)

#             if is_relevant and rel_conf >= SELFRAG_RELEVANCE_THRESHOLD:
#                 relevant_results.append(result)
#                 print(f"  ✓ Doc {i+1}: RELEVANT (confidence: {rel_conf:.2f})", flush=True)
#             else:
#                 print(f"  ✗ Doc {i+1}: Not relevant (confidence: {rel_conf:.2f})", flush=True)

#         if not relevant_results:
#             print(f"  ⚠️  No relevant documents found after filtering!", flush=True)
#             no_answer = "I found some documents but they don't seem relevant enough to answer your question accurately."
#             return no_answer, original_query, profile.input_lang, results, {}, domain_classification, selfrag_metrics

#         results = relevant_results
#         avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
#         selfrag_metrics['relevance_score'] = float(avg_relevance)
#         print(f"  → Average relevance: {avg_relevance:.2f}", flush=True)

#     # Build context
#     context_parts = []
#     for r in results:
#         chunk_text = sanitize_chunk_text(r["text"])
#         if not chunk_text:
#             continue

#         if profile.output_lang == "ur" and TRANSLATE_CONTEXT_FOR_URDU_OUTPUT:
#             try:
#                 chunk_text = translate_english_to_urdu(chunk_text)
#             except Exception:
#                 pass

#         MAX_CHUNK_CHARS = 1200
#         if len(chunk_text) > MAX_CHUNK_CHARS:
#             chunk_text = chunk_text[:MAX_CHUNK_CHARS].rsplit('\n', 1)[0] + "\n\n[truncated]"
#         context_parts.append(chunk_text)

#     context = "\n\n".join(context_parts)

#     # STEP 3.5: Check if answer exists in context
#     if SELFRAG_ENABLE:
#         print(f"\n[SELF-RAG] Step 3.5: Checking if answer exists in context...", flush=True)
#         can_answer, answer_conf = critic.check_answer_in_context(query_for_rag, context)
#         selfrag_metrics['answer_in_context'] = can_answer
#         selfrag_metrics['answer_in_context_confidence'] = answer_conf

#         if not can_answer and answer_conf >= 0.7:
#             print(f"  ✗ Answer CANNOT be found in context (confidence: {answer_conf:.2f})", flush=True)
#             no_info_response = "I don't have that information."
#             return (no_info_response, original_query, profile.input_lang, results,
#                     {'combined_confidence': 0.0}, domain_classification, selfrag_metrics)
#         else:
#             print(f"  ✓ Answer CAN be found in context (confidence: {answer_conf:.2f})", flush=True)

#     # Build prompt
#     wants_list = query_info['wants_list']
#     wants_summary = query_info['wants_summary']
#     wants_detail = query_info['wants_detail']

#     if profile.output_lang == "ur":
#         base_instruction = "آپ الخدمت فاؤنڈیشن پاکستان کے لیے ایک مددگار اسسٹنٹ ہیں۔"
#         if wants_list:
#             format_instruction = "براہ کرم نقاط کی شکل میں واضح جواب دیں۔"
#         elif wants_summary:
#             format_instruction = "براہ کرم 2-3 جملوں میں مختصر خلاصہ دیں۔"
#         else:
#             format_instruction = "براہ کرم مکمل اور واضح جواب دیں۔"

#         if profile.input_lang == "roman_ur" and profile.query_urdu_script:
#             display_question = profile.query_urdu_script
#         else:
#             display_question = original_query

#         prompt = f"""{base_instruction}
# {format_instruction}

# اہم ہدایات:
# - صرف نیچے دیے گئے سیاق و سباق کی معلومات استعمال کریں
# - عمومی علم سے جواب نہ دیں
# - اگر جواب سیاق و سباق میں نہیں تو صرف یہ کہیں: "مجھے معلوم نہیں"
# - کوئی لیبل شامل نہ کریں
# - صرف براہ راست جواب دیں

# سیاق و سباق:
# {context}

# سوال: {display_question}

# جواب (صرف اردو میں، ماخذ شامل نہ کریں):
# """
#     else:
#         base_instruction = "You are a helpful customer support agent for Alkhidmat Foundation Pakistan."
#         if wants_list:
#             format_instruction = "Provide a clear answer in bullet point format."
#         elif wants_summary:
#             format_instruction = "Provide a brief summary in 2-3 sentences."
#         elif wants_detail:
#             format_instruction = "Provide a detailed, comprehensive answer covering all relevant information from the context."
#         else:
#             format_instruction = "Provide a clear, complete answer."

#         prompt = f"""{base_instruction}

# {format_instruction}

# CRITICAL INSTRUCTIONS:
# - Use ONLY the information provided in the context below
# - DO NOT answer from general knowledge
# - If the answer is not in the context, respond EXACTLY: "I don't know"
# - DO NOT include any labels like 'Answer:', 'Question:', etc.
# - Return ONLY the direct answer
# - DO NOT copy context verbatim

# Information:
# {context}

# User question: {original_query}

# Answer (ONLY the final answer, no labels):
# """

#     # STEP 4: Generate
#     llm_start = time.time()
#     print(f"\n[SELF-RAG] Step 4: Generating answer...", flush=True)
#     sys.stdout.flush()
#     answer, log_probs, token_probs_distributions = llm_generate(
#         prompt, max_tokens=max_tokens, stop_tokens=["\nUser question:", "\nQuestion:", "\nسوال:"]
#     )
#     print(f"[TIMING] LLM generation: {time.time() - llm_start:.2f}s", flush=True)
#     answer = clean_llm_response(answer)

#     # STEP 5: Verify support
#     if SELFRAG_ENABLE:
#         print(f"\n[SELF-RAG] Step 5: Verifying answer support...", flush=True)
#         support_level, support_conf = critic.verify_support(query_for_rag, answer, context)
#         selfrag_metrics['support_level'] = support_level
#         selfrag_metrics['support_score'] = support_conf
#         print(f"  → Support level: {support_level.upper()} (confidence: {support_conf:.2f})", flush=True)

#         if support_level == "no_support" and support_conf >= SELFRAG_SUPPORT_THRESHOLD:
#             print(f"  ⚠️  Answer NOT supported by context - rejecting!", flush=True)
#             no_answer = "I cannot provide a reliable answer based on the available information."
#             return no_answer, original_query, profile.input_lang, results, {}, domain_classification, selfrag_metrics

#     # STEP 6: Utility
#     if SELFRAG_ENABLE:
#         print(f"\n[SELF-RAG] Step 6: Evaluating answer utility...", flush=True)
#         utility_rating, utility_conf = critic.evaluate_utility(query_for_rag, answer)
#         selfrag_metrics['utility_rating'] = utility_rating
#         selfrag_metrics['utility_score'] = utility_rating / 5.0
#         print(f"  → Utility rating: {utility_rating}/5 (confidence: {utility_conf:.2f})", flush=True)

#         if utility_rating <= 2:
#             print(f"  ⚠️  Answer utility too low - rejecting!", flush=True)
#             no_answer = "I cannot provide a sufficiently useful answer to your question based on the available information."
#             return no_answer, original_query, profile.input_lang, results, {}, domain_classification, selfrag_metrics

#     # Confidence scores
#     print(f"\n[SELF-RAG] Calculating final confidence scores...", flush=True)
#     sys.stdout.flush()
#     retrieval_conf = ConfidenceScorer.calculate_retrieval_confidence(query_embedding, doc_embeddings, top_k=top_k)
    
#     # Prepare Self-RAG scores for confidence calculation
#     selfrag_scores = {
#         'support_score': selfrag_metrics.get('support_score', 0.0),
#         'utility_score': selfrag_metrics.get('utility_score', 0.0),
#         'relevance_score': selfrag_metrics.get('relevance_score', 0.0)
#     }
    
#     confidence_scores = ConfidenceScorer.calculate_combined_confidence(
#         log_probs=log_probs,
#         retrieval_confidence=retrieval_conf,
#         token_probs_distributions=token_probs_distributions,
#         selfrag_scores=selfrag_scores
#     )

#     combined_conf = confidence_scores.get('combined_confidence', 0)
#     if SELFRAG_ENABLE and combined_conf < SELFRAG_MIN_CONFIDENCE:
#         print(f"\n⚠️  SELF-RAG: Combined confidence ({combined_conf:.2f}) below threshold ({SELFRAG_MIN_CONFIDENCE})", flush=True)
#         no_answer = "I don't have enough confidence in my answer to provide it. Please rephrase your question or contact support."
#         return no_answer, original_query, profile.input_lang, results, confidence_scores, domain_classification, selfrag_metrics

#     # OUTPUT LANGUAGE POST-PROCESSING
#     if profile.output_lang == "ur":
#         if not is_urdu_script(answer):
#             answer = translate_english_to_urdu(answer, timeout=15)
#     elif profile.output_lang == "roman_ur":
#         protected = protect_brand_terms(answer)
#         answer_ur = translate_english_to_urdu(protected, timeout=15)
#         answer_ur = restore_brand_terms(answer_ur)
#         answer = romanize_to_roman_urdu_with_llm(answer_ur)

#     # Create sources list
#     sources = [{
#         "category": r['category'],
#         "filename": r['filename'],
#         "file_path": r['file_path'],
#         "similarity": r['similarity']
#     } for r in results]

#     total_time = time.time() - start_time
#     print(f"\n[SELF-RAG] Answer ACCEPTED and generated successfully", flush=True)
#     print(f"[TIMING] Total time: {total_time:.2f}s", flush=True)
#     print(f"{'='*80}\n", flush=True)
#     sys.stdout.flush()

#     return answer, original_query, profile.input_lang, sources, confidence_scores, domain_classification, selfrag_metrics

# # ============ CLI Functions (ENHANCED) ============
# def query_alkhidmat_rag(query: str, category: str = None):
#     """ENHANCED: Now displays domain classification and confidence scores"""
#     answer, original_query, is_urdu, sources, confidence_scores, domain_classification = generate_answer(
#         query, filter_category=category
#     )
    
#     print("\n" + "="*80)
#     print("QUESTION:", original_query)
#     if is_urdu:
#         print("(Urdu query detected)")
#     if category:
#         print("CATEGORY FILTER:", category)
#     print("="*80)
    
#     # Display domain classification (NEW)
#     domain_emoji = DomainClassifier.get_domain_emoji(domain_classification['domain'])
#     print(f"\n{domain_emoji} DOMAIN CLASSIFICATION:")
#     print(f"{'-'*80}")
#     print(f"  Classified Domain: {domain_classification['domain'].upper()}")
#     print(f"  Classification Confidence: {domain_classification['confidence']:.2%}")
#     print(f"\n  Similarity Scores:")
#     for domain, score in domain_classification['all_scores'].items():
#         emoji = DomainClassifier.get_domain_emoji(domain)
#         bar = '█' * int(score * 50)
#         print(f"    {emoji} {domain:12s}: {score:.4f} {bar}")
#     print(f"{'-'*80}")
    
#     print("\nANSWER:")
#     print(answer)
    
#     # Display confidence scores (NEW)
#     print("\n" + "="*80)
#     print("CONFIDENCE SCORES:")
#     print("="*80)
#     print(f"  Combined Confidence:        {confidence_scores.get('combined_confidence', 0):.4f} ⭐")
#     print(f"  ├─ Retrieval Confidence:    {confidence_scores.get('retrieval_confidence', 0):.4f}")
#     print(f"  ├─ Avg Token Confidence:    {confidence_scores.get('avg_token_confidence', 0):.4f}")
#     print(f"  ├─ Weighted Top-K:          {confidence_scores.get('weighted_top_k', 0):.4f}")
#     print(f"  ├─ Perplexity:              {confidence_scores.get('perplexity', 0):.4f}")
#     print(f"  └─ Entropy Confidence:      {confidence_scores.get('entropy_confidence', 0):.4f}")
#     print("="*80)
    
#     # Interpretation (NEW)
#     combined = confidence_scores.get('combined_confidence', 0)
#     if combined >= 0.7:
#         print("✅ High confidence - Answer is likely reliable")
#     elif combined >= 0.5:
#         print("⚠️  Moderate confidence - Answer may need verification")
#     else:
#         print("❌ Low confidence - Answer should be verified from sources")
    
#     print("\n" + "="*80 + "\n")
#     return answer

# def show_statistics():
#     """Show database statistics"""
#     supabase = get_supabase_client()
    
#     print("\n" + "="*80)
#     print("DATABASE STATISTICS")
#     print("="*80)
    
#     try:
#         result = supabase.table("documents").select("doc_id", count="exact").execute()
#         print(f"\nTotal chunks: {result.count}")
        
#         result = supabase.table("documents").select("category").execute()
#         categories = {}
#         for row in result.data:
#             cat = row.get('category', 'Unknown')
#             categories[cat] = categories.get(cat, 0) + 1
        
#         print("\nDocuments by Category:")
#         for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
#             print(f"  {cat}: {count} chunks")
            
#     except Exception as e:
#         print(f"Error getting statistics: {e}")

# # ============ Batch Processing (ENHANCED) ============
# def batch_query_file(input_file: str, output_file: str):
#     """ENHANCED: Now includes domain classification and confidence scores in output"""
#     os.environ['BATCH_MODE'] = 'True'
    
#     print(f"\n{'='*80}")
#     print(f"BATCH QUERY MODE: Processing {input_file} -> {output_file}")
#     print(f"{'='*80}")

#     try:
#         with open(input_file, 'r', encoding='utf-8') as f:
#             queries = [line.strip() for line in f if line.strip()]
        
#         print(f"Found {len(queries)} queries to process.")
        
#         with open(output_file, 'w', encoding='utf-8') as f_out:
#             f_out.write("ALKHIDMAT RAG (SUPABASE) - BATCH QUERY RESULTS\n")
#             f_out.write("WITH DOMAIN CLASSIFICATION & CONFIDENCE SCORING\n")
#             f_out.write(f"Source file: {input_file}\n")
#             f_out.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

#             for i, query in enumerate(queries):
#                 print(f"Processing {i+1}/{len(queries)}: '{query[:40]}...'")
                
#                 # Generate with enhanced features
#                 answer, original_query, is_urdu_query, sources, confidence_scores, domain_classification = generate_answer(query)
                
#                 f_out.write(f"{'='*80}\n")
#                 f_out.write(f"QUERY {i+1}/{len(queries)}\n")
#                 f_out.write(f"{'='*80}\n")
#                 f_out.write(f"QUERY: {original_query}\n")
#                 f_out.write(f"LANGUAGE: {'Urdu' if is_urdu_query else 'English'}\n\n")
                
#                 # Write domain classification (NEW)
#                 domain_emoji = DomainClassifier.get_domain_emoji(domain_classification['domain'])
#                 f_out.write(f"{domain_emoji} DOMAIN CLASSIFICATION:\n")
#                 f_out.write(f"{'-'*80}\n")
#                 f_out.write(f"Classified Domain: {domain_classification['domain'].upper()}\n")
#                 f_out.write(f"Classification Confidence: {domain_classification['confidence']:.2%}\n")
#                 f_out.write(f"Similarity Scores:\n")
#                 for domain, score in domain_classification['all_scores'].items():
#                     f_out.write(f"  - {domain}: {score:.4f}\n")
#                 f_out.write(f"\n")
                
#                 f_out.write(f"ANSWER:\n{answer}\n\n")
                
#                 # Write confidence scores (NEW)
#                 f_out.write(f"CONFIDENCE SCORES:\n")
#                 f_out.write(f"{'-'*80}\n")
#                 f_out.write(f"Combined Confidence:        {confidence_scores.get('combined_confidence', 0):.4f} ⭐\n")
#                 f_out.write(f"├─ Retrieval Confidence:    {confidence_scores.get('retrieval_confidence', 0):.4f}\n")
#                 f_out.write(f"├─ Avg Token Confidence:    {confidence_scores.get('avg_token_confidence', 0):.4f}\n")
#                 f_out.write(f"├─ Weighted Top-K:          {confidence_scores.get('weighted_top_k', 0):.4f}\n")
#                 f_out.write(f"├─ Perplexity:              {confidence_scores.get('perplexity', 0):.4f}\n")
#                 f_out.write(f"└─ Entropy Confidence:      {confidence_scores.get('entropy_confidence', 0):.4f}\n\n")
                
#                 combined = confidence_scores.get('combined_confidence', 0)
#                 if combined >= 0.7:
#                     f_out.write("✅ High confidence - Answer is likely reliable\n")
#                 elif combined >= 0.5:
#                     f_out.write("⚠️  Moderate confidence - Answer may need verification\n")
#                 else:
#                     f_out.write("❌ Low confidence - Answer should be verified from sources\n")
                
#                 f_out.write(f"\nSOURCES USED:\n")
#                 if sources:
#                     for s in sources:
#                         f_out.write(f" - [{s['category']}] {s['filename']} (Sim: {s['similarity']:.3f})\n")
#                 else:
#                     f_out.write(" - No relevant documents found.\n")
                
#                 f_out.write(f"\n{'='*40}\n\n")
                
#                 print(f"  ✓ Domain: {domain_classification['domain']} | Confidence: {confidence_scores.get('combined_confidence', 0):.2f}")

#         print(f"\n{'='*80}")
#         print(f"✅ BATCH PROCESSING COMPLETE. Results written to {output_file}")
#         print(f"{'='*80}\n")
        
#     except FileNotFoundError:
#         print(f"❌ Error: Input file not found at {input_file}")
#     except Exception as e:
#         print(f"❌ An unexpected error occurred during batch processing: {e}")
#     finally:
#         os.environ['BATCH_MODE'] = 'False'

# # ============ Main ============
# if __name__ == "__main__":
#     import sys
#     import atexit
    
#     # Add cleanup handler for LLM
#     def cleanup_llm():
#         global _LLM_MODEL
#         if _LLM_MODEL is not None:
#             try:
#                 _LLM_MODEL = None
#             except:
#                 pass
    
#     atexit.register(cleanup_llm)
    
#     DEFAULT_ZIP = "Al Khidmat Knowledge Base.zip"
    
#     if len(sys.argv) > 1 and sys.argv[1] == "test":
#         test_connection()
        
#     elif len(sys.argv) > 1 and sys.argv[1] == "build":
#         zip_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZIP
#         clear = "--clear" in sys.argv
#         build_alkhidmat_rag(zip_path, clear_existing=clear)
        
#     elif len(sys.argv) > 1 and sys.argv[1] == "query":
#         q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What donation methods does Alkhidmat accept?"
#         query_alkhidmat_rag(q)

#     elif len(sys.argv) > 1 and sys.argv[1] == "file_query":
#         input_file = sys.argv[2] if len(sys.argv) > 2 else "input_queries.txt"
#         output_file = sys.argv[3] if len(sys.argv) > 3 else "output_answers.txt"
#         batch_query_file(input_file, output_file)
        
#     elif len(sys.argv) > 1 and sys.argv[1] == "stats":
#         show_statistics()
        
#     else:
#         print("\n" + "="*80)
#         print("ALKHIDMAT RAG SYSTEM (SUPABASE + LLAMA-CPP)")
#         print("WITH DOMAIN CLASSIFICATION & CONFIDENCE SCORING")
#         print("="*80)
#         print("\nUSAGE:")
#         print("1. Test Connection:")
#         print("   python RAG_supabase_enhanced.py test")
        
#         print("\n2. Build Index (Upload to Supabase):")
#         print("   python RAG_supabase_enhanced.py build [zip_path] [--clear]")
        
#         print("\n3. Single Query (Terminal):")
#         print("   python RAG_supabase_enhanced.py query 'your question'")
        
#         print("\n4. Batch Query (File I/O):")
#         print("   python RAG_supabase_enhanced.py file_query input.txt output.txt")
        
#         print("\n5. View Stats:")
#         print("   python RAG_supabase_enhanced.py stats")
#         print("\n" + "="*80)

#!/usr/bin/env python3
"""
ALKHIDMAT RAG SYSTEM - SUPABASE CLIENT EDITION
WITH DOMAIN CLASSIFICATION & CONFIDENCE SCORING
Uses Supabase Python client (REST API) and Llama-cpp-python
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import uuid

import gc

import numpy as np
import zipfile
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings
from sentence_transformers import SentenceTransformer

# LLM (Llama CPP)
from llama_cpp import Llama

# Supabase Client
from supabase import create_client, Client

# Urdu helpers
from deep_translator import GoogleTranslator
import langdetect

# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import domain anchors
from domain_anchors import DOMAIN_ANCHOR_QUERIES

# ============ SUPABASE CONFIG ============
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
LLM_MODEL_FILENAME = os.environ.get("GPT4ALL_MODEL", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_DIM = 768

# Retrieval parameters
RELEVANCE_THRESHOLD = 0.7

# Self-RAG parameters
SELFRAG_ENABLE = True
SELFRAG_RETRIEVE_THRESHOLD = 0.5
SELFRAG_RELEVANCE_THRESHOLD = 0.6
SELFRAG_SUPPORT_THRESHOLD = 0.7
SELFRAG_MIN_CONFIDENCE = 0.6

# Brand terms protection
BRAND_TERMS = ["EasyPaisa", "JazzCash", "Alkhidmat", "Alkhidmat Foundation", "Bank of Punjab", "Taqwa Islamic Banking"]

# Roman Urdu detection markers
ROMAN_URDU_MARKERS = {
    "kya", "kyu", "kyun", "kaise", "kesy", "kese", "kis", "kon", "ka", "ki", "ko",
    "mein", "main", "mera", "meri", "mere", "hum", "ham", "aap", "ap", "tum", "yeh",
    "nahi", "nai", "han", "haan", "hai", "hain", "tha", "thi", "thay",
    "kr", "kar", "karo", "kren", "karein", "krna", "hona", "hogya", "ho", "hoga",
    "please", "plz"
}

# If True, translate context into Urdu before generation for Urdu/Roman Urdu outputs
TRANSLATE_CONTEXT_FOR_URDU_OUTPUT = True

# ============================================================================
# SELF-RAG REFLECTION TOKENS & CRITIC
# ============================================================================

class SelfRAGReflectionTokens:
    RETRIEVE = "[Retrieve]"
    NO_RETRIEVE = "[No Retrieval]"
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"
    FULLY_SUPPORTED = "[Fully supported]"
    PARTIALLY_SUPPORTED = "[Partially supported]"
    NO_SUPPORT = "[No support]"
    UTILITY_5 = "[Utility:5]"
    UTILITY_4 = "[Utility:4]"
    UTILITY_3 = "[Utility:3]"
    UTILITY_2 = "[Utility:2]"
    UTILITY_1 = "[Utility:1]"

class SelfRAGCritic:
    def __init__(self, llm_model=None):
        self.llm = llm_model

    def _get_llm(self):
        """Get LLM model, loading if necessary."""
        if self.llm is None:
            self.llm = load_llm()
        return self.llm

    def is_domain_relevant(self, query: str) -> Tuple[bool, float]:
        prompt = f"""Determine if this question is about Alkhidmat Foundation Pakistan.

Question: {query}

You are a strict binary classifier.

Task:
Decide whether the user's query is SPECIFICALLY about the NGO
"Alkhidmat Foundation Pakistan".

Definition of RELEVANT:
A query is RELEVANT ONLY IF it clearly and explicitly refers to:
- Alkhidmat Foundation Pakistan by name or by obvious NGO context
AND
- Its services, operations, donations, healthcare, education, relief work,
  offices, leadership, volunteers, or official programs.

Examples of RELEVANT queries:
- "What is Alkhidmat Foundation?"
- "How can I donate to Alkhidmat?"
- "Does Alkhidmat provide free medical tests?"
- "Where is the Alkhidmat hospital in Lahore?"
- "Who is the CEO of Alkhidmat Foundation?"

Definition of IRRELEVANT:
A query is IRRELEVANT if:
- Alkhidmat Foundation is NOT clearly mentioned or strongly implied
- The query is about general topics, even if related to charity, Islam, health, or Pakistan
- The query is about people, celebrities, politics, sports, self-help, fitness, news, or current affairs
- The query could apply to ANY NGO, not specifically Alkhidmat

Examples of IRRELEVANT queries:
- "Best charity in Pakistan"
- "What is zakat?"
- "Free hospitals in Karachi"
- "How to stay healthy"
- "Imran Khan latest news"
- "Flood situation in Pakistan"

Important rules:
- Do NOT assume the query is about Alkhidmat unless it is clearly stated.
- If the query is ambiguous or generic, mark it IRRELEVANT.
- When in doubt, choose IRRELEVANT.

Output:
Respond with ONLY ONE of the following labels:
[RELEVANT]
[IRRELEVANT]

Answer:"""

        try:
            model = self._get_llm()
            output = model(prompt, max_tokens=10, temperature=0.1, stop=["\n"], echo=False)
            response = output['choices'][0]['text'].strip().upper()
            if "[RELEVANT]" in response:
                return True, 0.85
            elif "[IRRELEVANT]" in response:
                return False, 0.85
            return True, 0.4
        except Exception as e:
            print(f"[SelfRAG] Domain relevance check error: {e}")
            return True, 0.5

    def should_retrieve(self, query: str) -> Tuple[bool, float]:
        prompt = f"""Determine if external knowledge retrieval is needed to answer this question.

Question: {query}

Consider:
- Does this require specific factual information?
- Is this about a specific organization, service, or policy?
- Can this be answered with general knowledge alone?

Respond with ONLY one of:
{SelfRAGReflectionTokens.RETRIEVE} - if retrieval is needed
{SelfRAGReflectionTokens.NO_RETRIEVE} - if general knowledge is sufficient

Answer:"""
        try:
            model = self._get_llm()
            output = model(prompt, max_tokens=10, temperature=0.1, stop=["\n"], echo=False)
            response = output['choices'][0]['text'].strip()
            if SelfRAGReflectionTokens.RETRIEVE in response:
                return True, 0.8
            elif SelfRAGReflectionTokens.NO_RETRIEVE in response:
                return False, 0.8
            return True, 0.5
        except Exception as e:
            print(f"[SelfRAG] Retrieval prediction error: {e}")
            return True, 0.5

    def assess_relevance(self, query: str, document: str) -> Tuple[bool, float]:
        doc_preview = document[:500] + "..." if len(document) > 500 else document
        prompt = f"""Assess if this document is relevant to answering the question.

Question: {query}

Document excerpt:
{doc_preview}

Is this document relevant for answering the question?

Respond with ONLY one of:
{SelfRAGReflectionTokens.RELEVANT} - if document is relevant
{SelfRAGReflectionTokens.IRRELEVANT} - if document is not relevant

Answer:"""
        try:
            model = self._get_llm()
            output = model(prompt, max_tokens=10, temperature=0.1, stop=["\n"], echo=False)
            response = output['choices'][0]['text'].strip()
            if SelfRAGReflectionTokens.RELEVANT in response:
                return True, 0.8
            elif SelfRAGReflectionTokens.IRRELEVANT in response:
                return False, 0.8
            return True, 0.5
        except Exception as e:
            print(f"[SelfRAG] Relevance assessment error: {e}")
            return True, 0.5

    def check_answer_in_context(self, query: str, context: str) -> Tuple[bool, float]:
        context_preview = context[:1000] + "..." if len(context) > 1000 else context
        prompt = f"""Check if the provided context contains information to answer the question.

Question: {query}

Context:
{context_preview}

Can this question be answered using ONLY the information in the context above?

Respond with ONLY one of:
[CAN_ANSWER] - context contains the answer
[CANNOT_ANSWER] - context does NOT contain the answer

Answer:"""
        try:
            model = self._get_llm()
            output = model(prompt, max_tokens=10, temperature=0.1, stop=["\n"], echo=False)
            response = output['choices'][0]['text'].strip().upper()
            if "[CAN_ANSWER]" in response:
                return True, 0.8
            elif "[CANNOT_ANSWER]" in response:
                return False, 0.8
            return True, 0.4
        except Exception as e:
            print(f"[SelfRAG] Answer presence check error: {e}")
            return True, 0.5

    def verify_support(self, query: str, answer: str, context: str) -> Tuple[str, float]:
        prompt = f"""Verify if the answer is supported by the provided context.

Question: {query}

Context:
{context[:800]}...

Answer:
{answer}

IMPORTANT: Check if the answer facts come from the context, or if the answer is making up information.

Respond with ONLY one of:
{SelfRAGReflectionTokens.FULLY_SUPPORTED} - answer is fully supported by context
{SelfRAGReflectionTokens.PARTIALLY_SUPPORTED} - answer is partially supported
{SelfRAGReflectionTokens.NO_SUPPORT} - answer is NOT supported or makes up information

Answer:"""
        try:
            model = self._get_llm()
            output = model(prompt, max_tokens=20, temperature=0.1, stop=["\n"], echo=False)
            response = output['choices'][0]['text'].strip()
            if SelfRAGReflectionTokens.FULLY_SUPPORTED in response:
                return "fully_supported", 0.9
            elif SelfRAGReflectionTokens.PARTIALLY_SUPPORTED in response:
                return "partially_supported", 0.6
            elif SelfRAGReflectionTokens.NO_SUPPORT in response:
                return "no_support", 0.9
            return "uncertain", 0.4
        except Exception as e:
            print(f"[SelfRAG] Support verification error: {e}")
            return "uncertain", 0.4

    def evaluate_utility(self, query: str, answer: str) -> Tuple[int, float]:
        prompt = f"""Evaluate how useful this answer is for the question.

Question: {query}

Answer:
{answer}

Rate the utility on a scale of 1-5:
5 = Excellent, complete, and directly answers the question
4 = Good, mostly answers the question
3 = Acceptable, provides some useful information
2 = Poor, barely addresses the question
1 = Very poor, does not answer the question

Respond with ONLY one of:
{SelfRAGReflectionTokens.UTILITY_5}
{SelfRAGReflectionTokens.UTILITY_4}
{SelfRAGReflectionTokens.UTILITY_3}
{SelfRAGReflectionTokens.UTILITY_2}
{SelfRAGReflectionTokens.UTILITY_1}

Answer:"""
        try:
            model = self._get_llm()
            output = model(prompt, max_tokens=15, temperature=0.1, stop=["\n"], echo=False)
            response = output['choices'][0]['text'].strip()
            if SelfRAGReflectionTokens.UTILITY_5 in response:
                return 5, 0.8
            elif SelfRAGReflectionTokens.UTILITY_4 in response:
                return 4, 0.8
            elif SelfRAGReflectionTokens.UTILITY_3 in response:
                return 3, 0.8
            elif SelfRAGReflectionTokens.UTILITY_2 in response:
                return 2, 0.8
            elif SelfRAGReflectionTokens.UTILITY_1 in response:
                return 1, 0.8
            return 3, 0.4
        except Exception as e:
            print(f"[SelfRAG] Utility evaluation error: {e}")
            return 3, 0.4

# ============================================================================
# DOMAIN CLASSIFICATION CLASS (FROM ORIGINAL RAG)
# ============================================================================

class DomainClassifier:
    """
    Classifies queries into domains using embedding similarity with anchor queries.
    Uses pre-computed domain centroids from representative queries.
    """
   
    _domain_embeddings_cache = None
    _embedding_model = None
   
    @staticmethod
    def initialize_domain_embeddings(model_name: str = None):
        """
        Pre-compute embeddings for all anchor queries and create domain centroids.
        OPTIMIZED: Reuses main embedding model instead of loading a separate one.
        """
        if DomainClassifier._domain_embeddings_cache is not None:
            return
       
        if os.environ.get('BATCH_MODE') != 'True':
            print("\n🔄 Initializing domain embeddings from anchor queries...")
       
        # OPTIMIZATION: Reuse the main embedding model instead of loading a new one
        # This saves ~1-2 seconds on first query
        # Get the embedder function from the module namespace (avoid circular import)
        import sys
        current_module = sys.modules[__name__]
        model = current_module.get_embedder() # Reuse main embedding model
        DomainClassifier._embedding_model = model
       
        domain_embeddings = {}
       
        for domain, queries in DOMAIN_ANCHOR_QUERIES.items():
            # Use the same prefix format as query encoding for consistency
            prefixed_queries = [f"query: {q}" for q in queries]
            embeddings = model.encode(prefixed_queries, show_progress_bar=False, normalize_embeddings=True)
            centroid = np.mean(embeddings, axis=0)
            domain_embeddings[domain] = centroid
           
            if os.environ.get('BATCH_MODE') != 'True':
                print(f" ✓ {domain}: {len(queries)} anchor queries → centroid computed")
       
        DomainClassifier._domain_embeddings_cache = domain_embeddings
       
        if os.environ.get('BATCH_MODE') != 'True':
            print("✅ Domain embeddings initialized!\n")
   
    @staticmethod
    def classify_domain(query: str) -> Dict[str, any]:
        """
        Classify query using embedding similarity to domain centroids.
        OPTIMIZED: Reuses main embedding model.
        """
        if DomainClassifier._domain_embeddings_cache is None:
            DomainClassifier.initialize_domain_embeddings()
       
        # Use same prefix format as query encoding for consistency
        query_prefixed = f"query: {query}"
        query_embedding = DomainClassifier._embedding_model.encode([query_prefixed], normalize_embeddings=True)[0]
       
        similarities = {}
        for domain, centroid in DomainClassifier._domain_embeddings_cache.items():
            query_reshaped = query_embedding.reshape(1, -1)
            centroid_reshaped = centroid.reshape(1, -1)
            similarity = cosine_similarity(query_reshaped, centroid_reshaped)[0][0]
            similarities[domain] = float(similarity)
       
        winning_domain = max(similarities, key=similarities.get)
        confidence = similarities[winning_domain]
       
        return {
            'domain': winning_domain,
            'confidence': confidence,
            'all_scores': similarities
        }
   
    @staticmethod
    def get_domain_emoji(domain: str) -> str:
        """Get emoji representation for domain."""
        emoji_map = {
            'donation': '💰',
            'healthcare': '🏥',
            'general': '📋'
        }
        return emoji_map.get(domain, '📋')

# ============================================================================
# CONFIDENCE SCORING CLASS (FROM ORIGINAL RAG)
# ============================================================================

class ConfidenceScorer:
    """
    Implements multiple confidence scoring methods:
    1. Token-level log probabilities (Average, Perplexity)
    2. Entropy-based scoring
    3. Retrieval confidence (semantic similarity)
    """
   
    @staticmethod
    def calculate_perplexity(log_probs: List[float]) -> float:
        """Calculate perplexity from log probabilities."""
        if not log_probs:
            return float('inf')
       
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        alpha = 0.1
        return float(np.exp(-alpha * (perplexity - 1)))
   
    @staticmethod
    def calculate_average_token_confidence(log_probs: List[float]) -> float:
        """Calculate average token log-probability."""
        if not log_probs:
            return 0.0
       
        probs = [np.exp(lp) for lp in log_probs]
        return np.mean(probs)
   
    @staticmethod
    def calculate_entropy_confidence(token_probs_distributions: List[np.ndarray]) -> float:
        """Calculate entropy-based confidence."""
        if not token_probs_distributions:
            return float('inf')
       
        entropies = []
        for prob_dist in token_probs_distributions:
            if len(prob_dist) > 0:
                ent = entropy(prob_dist)
                entropies.append(ent)
       
        if not entropies:
            return float('inf')
       
        max_entropy = max(entropies)
        confidence = 1 - min(max_entropy / 10.0, 1.0)
        return confidence
   
    @staticmethod
    def calculate_top_k_weighted_confidence(log_probs: List[float], k: int = 5) -> float:
        """Weighted confidence score using top k% tokens."""
        if not log_probs or len(log_probs) < 5:
            return np.mean([np.exp(lp) for lp in log_probs]) if log_probs else 0.0
       
        probs = [np.exp(lp) for lp in log_probs]
        sorted_probs = sorted(probs, reverse=True)
        top_k_probs = sorted_probs[:k]
       
        joint_top_k = np.prod(top_k_probs) ** (1/k)
        joint_all = np.prod(probs) ** (1/len(probs))
       
        weighted_score = 0.7 * joint_top_k + 0.3 * joint_all
        return weighted_score
   
    @staticmethod
    def calculate_retrieval_confidence(
        query_embedding: np.ndarray,
        doc_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> float:
        """Calculate confidence based on cosine similarity between query and documents."""
        if not doc_embeddings:
            return 0.0
       
        cosine_similarities = []
        for doc_emb in doc_embeddings[:top_k]:
            query_reshaped = query_embedding.reshape(1, -1)
            doc_reshaped = doc_emb.reshape(1, -1)
            cos_sim = cosine_similarity(query_reshaped, doc_reshaped)[0][0]
            cosine_similarities.append(cos_sim)
       
        return float(np.mean(cosine_similarities))
   
    @staticmethod
    def calculate_combined_confidence(
        log_probs: List[float],
        retrieval_confidence: float,
        token_probs_distributions: List[np.ndarray] = None,
        selfrag_scores: Dict = None
    ) -> Dict[str, float]:
        """Calculate all confidence metrics and return a comprehensive score."""
        scores = {}
        scores['retrieval_confidence'] = retrieval_confidence
       
        if log_probs:
            scores['avg_token_confidence'] = ConfidenceScorer.calculate_average_token_confidence(log_probs)
            scores['perplexity'] = ConfidenceScorer.calculate_perplexity(log_probs)
            scores['weighted_top_k'] = ConfidenceScorer.calculate_top_k_weighted_confidence(log_probs)
       
        if token_probs_distributions:
            scores['entropy_confidence'] = ConfidenceScorer.calculate_entropy_confidence(token_probs_distributions)

        if selfrag_scores:
            scores['selfrag_support'] = selfrag_scores.get('support_score', 0.0)
            scores['selfrag_utility'] = selfrag_scores.get('utility_score', 0.0)
            scores['selfrag_relevance'] = selfrag_scores.get('relevance_score', 0.0)
       
        # Combined score (weighted average)
        combined = 0.0
        weight_sum = 0.0
       
        if 'retrieval_confidence' in scores:
            combined += 0.3 * scores['retrieval_confidence']
            weight_sum += 0.3
       
        if 'avg_token_confidence' in scores:
            combined += 0.15 * scores['avg_token_confidence']
            weight_sum += 0.15
       
        if 'weighted_top_k' in scores:
            combined += 0.15 * scores['weighted_top_k']
            weight_sum += 0.15

        if 'selfrag_support' in scores:
            combined += 0.25 * scores['selfrag_support']
            weight_sum += 0.25
       
        if 'selfrag_utility' in scores:
            combined += 0.15 * scores['selfrag_utility']
            weight_sum += 0.15
       
        if weight_sum > 0:
            scores['combined_confidence'] = combined / weight_sum
        else:
            scores['combined_confidence'] = 0.0
       
        return scores

# ============ Supabase Client ============
_SUPABASE_CLIENT = None

def get_supabase_client() -> Client:
    """Get or create Supabase client"""
    global _SUPABASE_CLIENT
   
    if _SUPABASE_CLIENT is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file\n"
                "Get these from: Supabase Dashboard -> Settings -> API"
            )
       
        print(f"Connecting to Supabase: {SUPABASE_URL}")
        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
   
    return _SUPABASE_CLIENT

def test_connection():
    """Test Supabase connection"""
    try:
        supabase = get_supabase_client()
        result = supabase.table("documents").select("doc_id").limit(1).execute()
       
        print("✅ Connected to Supabase successfully!")
        print(f" URL: {SUPABASE_URL}")
       
        try:
            result = supabase.rpc('match_documents_simple', {
                'query_embedding': [0.0] * 768,
                'match_count': 1
            }).execute()
            print("✅ pgvector extension verified")
        except Exception as e:
            if 'does not exist' in str(e):
                print("⚠️ match_documents function not found - run the SQL setup")
            else:
                print(f"⚠️ Could not verify pgvector: {e}")
       
        return True
       
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

# ============ Brand Term Protection ============
def protect_brand_terms(text: str) -> str:
    """Protect brand terms during translation to prevent corruption."""
    for term in BRAND_TERMS:
        text = re.sub(rf"\b{re.escape(term)}\b", f"@@{term}@@", text, flags=re.IGNORECASE)
    return text

def restore_brand_terms(text: str) -> str:
    """Restore brand terms after translation."""
    return text.replace("@@", "")

# ============ Enhanced Language Detection ============
def is_urdu_script(text: str) -> bool:
    """Check if text contains Urdu script (Arabic block)."""
    return bool(re.search(r'[\u0600-\u06FF]', text))

def looks_like_roman_urdu(text: str) -> bool:
    """Heuristic detection of Roman Urdu text."""
    if is_urdu_script(text):
        return False
    # If it contains many non-latin characters, skip
    if re.search(r'[^\x00-\x7F]', text):
        return False
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in ROMAN_URDU_MARKERS)
    # Heuristic: at least 2 marker tokens OR 1 marker token with short query
    if hits >= 2:
        return True
    if hits >= 1 and len(tokens) <= 6:
        return True
    return False

def translate_auto_to_english(text: str) -> str:
    """Fallback translation for roman urdu if transliteration isn't available."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

class QueryLangProfile:
    """
    Keeps the pipeline consistent:
    - original_query: what user typed
    - input_lang: 'en' | 'ur' | 'roman_ur'
    - query_en: English version used for embeddings/retrieval/classification/critic prompts
    - output_lang: same as input_lang (answer must match)
    - query_urdu_script: if roman_ur, we also produce Urdu-script version for better downstream translation prompts
    """
    def __init__(self, original_query: str, input_lang: str, query_en: str,
                 output_lang: str, query_urdu_script: Optional[str] = None):
        self.original_query = original_query
        self.input_lang = input_lang
        self.query_en = query_en
        self.output_lang = output_lang
        self.query_urdu_script = query_urdu_script

def build_query_lang_profile(query: str) -> QueryLangProfile:
    """Build language profile for a query."""
    q = query.strip()

    # 1) Urdu script
    if is_urdu_script(q) or detect_language(q) == "ur":
        q_en = translate_urdu_to_english(q)
        return QueryLangProfile(original_query=q, input_lang="ur", query_en=q_en, output_lang="ur", query_urdu_script=q)

    # 2) Roman Urdu (heuristic)
    if looks_like_roman_urdu(q):
        # Prefer direct auto->English translation (stable, no extra deps)
        q_en = translate_auto_to_english(q)
        return QueryLangProfile(
            original_query=q,
            input_lang="roman_ur",
            query_en=q_en,
            output_lang="roman_ur",
            query_urdu_script=None
        )

    # 3) Default English
    return QueryLangProfile(original_query=q, input_lang="en", query_en=q, output_lang="en", query_urdu_script=None)

# ============ Language Detection ============
def detect_language(text: str) -> str:
    try:
        return langdetect.detect(text)
    except Exception:
        return "en"

def is_urdu(text: str) -> bool:
    urdu_pattern = re.compile(r'[\u0600-\u06FF]')
    has_urdu = bool(urdu_pattern.search(text))
    return has_urdu or detect_language(text) == "ur"

def translate_urdu_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='ur', target='en').translate(text)
    except Exception:
        return text

def translate_english_to_urdu(text: str, timeout: int = 10) -> str:
    """
    Translate English text to Urdu with timeout protection.
    Returns original text if translation fails or times out.
    Uses concurrent.futures for cross-platform timeout support.
    """
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
       
        def _translate():
            return GoogleTranslator(source='en', target='ur').translate(text)
       
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_translate)
            try:
                result = future.result(timeout=timeout)
                return result if result else text
            except FutureTimeoutError:
                print(f"[WARNING] Translation timed out after {timeout}s. Returning original text.", flush=True)
                return text
    except Exception as e:
        print(f"[WARNING] Translation failed: {e}. Returning original text.", flush=True)
        return text

# ============ Query Analysis ============
def analyze_query(query: str) -> Dict[str, Any]:
    """Detect what kind of answer the user wants"""
    q_lower = query.lower().strip()

    # Treat "how to" questions as procedural (steps)
    procedural_markers = ["how to", "how do i", "how can i", "steps", "step by step", "procedure"]
    roman_markers = ["kaise", "kesy", "kese", "kaisay", "kya", "kyun", "kyu"]

    list_keywords = ['list', 'points', 'bullet', 'enumerate', 'ways', 'methods']
    summary_keywords = ['summarize', 'summary', 'briefly', 'overview', 'خلاصہ', 'khulasa']
    detail_keywords = ['explain', 'detail', 'describe', 'why', 'تفصیل', 'tafseel']

    wants_steps = any(m in q_lower for m in procedural_markers) or any(m in q_lower for m in roman_markers)
    wants_list = wants_steps or any(kw in q_lower for kw in list_keywords)
    wants_summary = any(kw in q_lower for kw in summary_keywords)
    wants_detail = any(kw in q_lower for kw in detail_keywords)

    return {
        'wants_list': wants_list,
        'wants_summary': wants_summary,
        'wants_detail': wants_detail,
        'is_urdu': is_urdu(query)
    }

def expand_query_for_retrieval(query_en: str, domain: str, query_info: Dict[str, Any]) -> str:
    """
    Small deterministic expansion for short/procedural queries to improve retrieval.
    Keeps core RAG logic identical; only changes the string embedded.
    """
    q = query_en.strip()
    if len(q.split()) >= 6 and not query_info.get("wants_list"):
        return q

    if domain == "donation":
        extra = "donate donation methods how to donate steps JazzCash EasyPaisa bank transfer online donation international account"
        return f"{q}. {extra}"
    if domain == "healthcare":
        extra = "Alkhidmat hospital clinic services eligibility locations how to get treatment"
        return f"{q}. {extra}"

    return q

# ============ Document Processing ============
def load_documents_from_zip(zip_path: str) -> Dict[str, List[Dict]]:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    documents_by_category = {}
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_path in zip_ref.namelist():
            if file_path.endswith("/") or not file_path.endswith(".txt"):
                continue
            parts = Path(file_path).parts
            if len(parts) < 3:
                continue
            category = parts[-2]
            filename = parts[-1]
            try:
                with zip_ref.open(file_path) as f:
                    content = f.read().decode("utf-8")
                if content.strip():
                    documents_by_category.setdefault(category, []).append({
                        "content": content,
                        "filename": filename,
                        "category": category,
                        "file_path": file_path
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return documents_by_category

def clean_text(text: str) -> str:
    """Enhanced cleaning while preserving structure"""
    text = re.sub(r'={3,}[\s\S]*?={3,}', '', text)
    text = re.sub(r'URL:\s*https?://\S+', '', text)
    text = re.sub(r'TITLE:.*?\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def prepare_documents(zip_path: str) -> Tuple[List[str], List[Dict]]:
    docs_by_cat = load_documents_from_zip(zip_path)
    all_docs, metadata = [], []
    for cat, docs in docs_by_cat.items():
        for doc in docs:
            c = clean_text(doc["content"])
            if c:
                all_docs.append(c)
                metadata.append({
                    "filename": doc["filename"],
                    "category": doc["category"],
                    "file_path": doc["file_path"]
                })
    print(f"Prepared {len(all_docs)} documents")
    return all_docs, metadata

def split_documents(documents: List[str], metadata: List[Dict]):
    """Improved semantic chunking"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "؟ ", "۔ ", " ", ""]
    )
    chunks, metas = [], []
    for doc, meta in zip(documents, metadata):
        parts = splitter.split_text(doc)
        for idx, p in enumerate(parts):
            chunks.append(p)
            chunk_meta = meta.copy()
            chunk_meta['chunk_index'] = idx
            metas.append(chunk_meta)
   
    avg_len = int(np.mean([len(c) for c in chunks]))
    print(f"Split into {len(chunks)} chunks (avg {avg_len} chars, overlap {CHUNK_OVERLAP})")
    return chunks, metas

# ============ Embeddings ============
_EMBEDDER = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDER

def create_embeddings(text_chunks: List[str]) -> np.ndarray:
    embedder = get_embedder()
    print(f"Creating embeddings for {len(text_chunks)} chunks...")
    prefixed = [f"passage: {chunk}" for chunk in text_chunks]
    embs = embedder.encode(prefixed, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    embs = np.array(embs).astype("float32")
    print("Embeddings created:", embs.shape)
    return embs

# ============ Supabase Storage ============
def save_chunks_to_supabase(chunks: List[str], metadata: List[Dict], embeddings: np.ndarray):
    """Store chunks and embeddings in Supabase using client"""
    supabase = get_supabase_client()
   
    print("Saving to Supabase...")
   
    rows = []
    for i, chunk in enumerate(chunks):
        meta = metadata[i]
        emb_list = embeddings[i].tolist()
       
        rows.append({
            "doc_id": str(uuid.uuid4()),
            "chunk_text": chunk,
            "chunk_index": meta.get("chunk_index", 0),
            "category": meta.get("category"),
            "filename": meta.get("filename"),
            "file_path": meta.get("file_path"),
            "doc_domain": meta.get("category"),
            "embedding": emb_list
        })
   
    batch_size = 100
    total_inserted = 0
   
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            result = supabase.table("documents").insert(batch).execute()
            total_inserted += len(batch)
            print(f" Inserted batch {i//batch_size + 1}: {len(batch)} chunks")
        except Exception as e:
            print(f" ⚠️ Error inserting batch {i//batch_size + 1}: {e}")
            for row in batch:
                try:
                    supabase.table("documents").insert(row).execute()
                    total_inserted += 1
                except Exception as e2:
                    print(f" ⚠️ Failed to insert chunk: {e2}")
   
    print(f"✅ Stored {total_inserted}/{len(rows)} chunks in Supabase")

def clear_documents_table():
    """Clear all documents (use with caution!)"""
    supabase = get_supabase_client()
   
    try:
        result = supabase.table("documents").delete().neq("doc_id", "00000000-0000-0000-0000-000000000000").execute()
        print("✅ Cleared all documents from Supabase")
    except Exception as e:
        print(f"⚠️ Error clearing documents: {e}")

# ============ Build Pipeline ============
def build_alkhidmat_rag(zip_path: str, clear_existing: bool = False):
    """Build the RAG system and store in Supabase"""
    print("\n" + "="*80)
    print("BUILDING ALKHIDMAT RAG SYSTEM")
    print("="*80)
   
    if not test_connection():
        print("❌ Cannot connect to Supabase. Aborting.")
        return
   
    if clear_existing:
        print("\n⚠️ Clearing existing documents...")
        clear_documents_table()
   
    print("\nBUILD: Loading documents...")
    docs, meta = prepare_documents(zip_path)
    if not docs:
        print("❌ No documents found. Aborting.")
        return
   
    print("\nBUILD: Splitting with improved chunking...")
    chunks, chunk_meta = split_documents(docs, meta)
   
    print("\nBUILD: Computing embeddings...")
    embeddings = create_embeddings(chunks)
   
    print("\nBUILD: Saving to Supabase...")
    save_chunks_to_supabase(chunks, chunk_meta, embeddings)
   
    print("\nBUILD: Initializing domain classification...")
    DomainClassifier.initialize_domain_embeddings()
   
    print("\n" + "="*80)
    print("✅ BUILD COMPLETE!")
    print("="*80)

# ============ Retrieval (ENHANCED WITH EMBEDDINGS) ============
def retrieve_from_supabase(query: str, top_k: int = 5,
                           filter_category: str = None) -> Tuple[List[Dict], np.ndarray, List[np.ndarray]]:
    """
    FIXED: Now properly extracts embeddings from Supabase for confidence scoring
    """
    embed_start = time.time()
    embedder = get_embedder()
    query_prefixed = f"query: {query}"
    query_embedding = embedder.encode([query_prefixed], normalize_embeddings=True)[0]
    print(f"[TIMING] Query embedding: {time.time() - embed_start:.2f}s", flush=True)
   
    supabase = get_supabase_client()
   
    try:
        rpc_start = time.time()
        params = {
            'query_embedding': query_embedding.tolist(),
            'match_threshold': RELEVANCE_THRESHOLD,
            'match_count': top_k
        }
       
        if filter_category:
            params['filter_category'] = filter_category
       
        # First, get the matches using RPC
        result = supabase.rpc('match_documents', params).execute()
        rows = result.data
        print(f"[TIMING] Supabase RPC call: {time.time() - rpc_start:.2f}s", flush=True)
       
        # OPTIMIZED: Only fetch embeddings if we have results
        # The RPC already returns similarity and text, so we only need embeddings for confidence scoring
        if rows:
            fetch_start = time.time()
            doc_ids = [row['doc_id'] for row in rows]
           
            # OPTIMIZATION: Only fetch embeddings, not full documents (we already have text from RPC)
            # This reduces data transfer and speeds up the query
            full_docs_result = supabase.table("documents").select(
                "doc_id, embedding" # Only fetch what we need
            ).in_("doc_id", doc_ids).execute()
           
            # Create a mapping for easy lookup
            doc_map = {doc['doc_id']: doc for doc in full_docs_result.data}
           
            # Merge embedding data back into rows
            for row in rows:
                if row['doc_id'] in doc_map:
                    row['embedding'] = doc_map[row['doc_id']]['embedding']
            print(f"[TIMING] Fetch embeddings: {time.time() - fetch_start:.2f}s", flush=True)
       
    except Exception as e:
        print(f"⚠️ RPC function not available, using fallback method: {e}")
       
        # Fallback: manual similarity calculation
        result = supabase.table("documents").select(
            "doc_id, chunk_text, category, filename, file_path, chunk_index, embedding"
        ).execute()
       
        rows = []
        for row in result.data:
            if row['embedding']:
                doc_emb = np.array(row['embedding'])
                similarity = float(np.dot(query_embedding, doc_emb))
               
                if similarity > RELEVANCE_THRESHOLD:
                    row['similarity'] = similarity
                    rows.append(row)
       
        rows = sorted(rows, key=lambda x: x['similarity'], reverse=True)[:top_k]
   
    results = []
    doc_embeddings = []
   
    for row in rows:
        results.append({
            "text": row['chunk_text'],
            "category": row['category'],
            "filename": row['filename'],
            "file_path": row['file_path'],
            "chunk_index": row.get('chunk_index', 0),
            "similarity": float(row.get('similarity', 0))
        })
       
        # FIXED: Properly extract and convert embedding
        if 'embedding' in row and row['embedding'] is not None:
            try:
                # Handle different embedding formats
                if isinstance(row['embedding'], str):
                    # If it's a string representation, try to parse it
                    import ast
                    embedding_data = ast.literal_eval(row['embedding'])
                    doc_embeddings.append(np.array(embedding_data, dtype=np.float32))
                elif isinstance(row['embedding'], list):
                    # If it's already a list
                    doc_embeddings.append(np.array(row['embedding'], dtype=np.float32))
                elif isinstance(row['embedding'], np.ndarray):
                    # If it's already a numpy array
                    doc_embeddings.append(row['embedding'].astype(np.float32))
                else:
                    print(f"⚠️ Unexpected embedding type: {type(row['embedding'])}")
            except Exception as e:
                print(f"⚠️ Error parsing embedding: {e}")
                continue
   
    print("\n" + "="*80, flush=True)
    print("RETRIEVAL FROM SUPABASE", flush=True)
    print(f"Retrieved {len(results)} relevant chunks (threshold: {RELEVANCE_THRESHOLD}):", flush=True)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['category']}] {r['filename']} (similarity: {r['similarity']:.3f})", flush=True)
    print(f"📊 Document embeddings extracted: {len(doc_embeddings)}/{len(results)}", flush=True)
    print("="*80 + "\n", flush=True)
    sys.stdout.flush() # Force flush
   
    return results, query_embedding, doc_embeddings

def sanitize_chunk_text(text: str) -> str:
    """Remove existing Q/A labels from chunks"""
    text = re.sub(r'(?mi)^\s*(user\s+question|question|q:)\s*[:\-–]?\s*.*$', '', text)
    text = re.sub(r'(?mi)^\s*(answer|a:)\s*[:\-–]?\s*.*$', '', text)
    text = re.sub(r'(?is)(?:^|\n)\s*q[:\.\-\)]\s*.*?\n\s*a[:\.\-\)]\s*.*?(?:\n|$)', '', text)
    text = re.sub(r'\[insert .*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'click here', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ============ LLM (Llama CPP) ============
_LLM_MODEL = None

def detect_apple_silicon():
    """Detect if running on Apple Silicon (M1/M2/M3/etc)"""
    import platform
    try:
        # Check if running on macOS
        if platform.system() != "Darwin":
            return False
       
        # Check processor architecture
        machine = platform.machine()
        if machine == "arm64":
            return True
       
        # Alternative check using uname
        import subprocess
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
        if result.returncode == 0 and 'arm64' in result.stdout:
            return True
       
        return False
    except Exception:
        return False

def load_llm(model_filename: str = LLM_MODEL_FILENAME):
    global _LLM_MODEL
    if _LLM_MODEL is None:
        print("Loading local LLM via llama-cpp:", model_filename)
       
        if not os.path.exists(model_filename):
            print(f"❌ Error: Model file not found at {model_filename}")
            raise FileNotFoundError(f"Model file missing: {model_filename}")

        # Detect Apple Silicon for Metal GPU acceleration
        is_apple_silicon = detect_apple_silicon()
        gpu_layers = -1 if is_apple_silicon else 0 # -1 = use all GPU layers on Metal
       
        if is_apple_silicon:
            print("🍎 Apple Silicon detected - enabling Metal GPU acceleration")
        else:
            print("💻 Using CPU mode (no GPU acceleration)")

        _LLM_MODEL = Llama(
            model_path=model_filename,
            n_ctx=4096,
            n_gpu_layers=gpu_layers, # -1 for Apple Silicon (Metal), 0 for CPU
            verbose=False,
            logits_all=True # FIXED: Enable logits for log probability extraction
        )
       
        if is_apple_silicon:
            print("✅ LLM loaded with Metal GPU acceleration")
        else:
            print("✅ LLM loaded in CPU mode")
    return _LLM_MODEL

def llm_generate(prompt: str, max_tokens: int = 400, stop_tokens: list = None) -> Tuple[str, List[float], List[np.ndarray]]:
    """
    MEMORY-OPTIMIZED: Generation with garbage collection
    """
    model = load_llm()
   
    # Force garbage collection before generation
    gc.collect()

    try:
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=stop_tokens or [],
            echo=False,
            logprobs=5
        )
       
        text = output['choices'][0]['text'].strip()
       
        # Extract log probabilities if available
        log_probs = []
        token_probs_distributions = []
       
        if 'logprobs' in output['choices'][0] and output['choices'][0]['logprobs']:
            logprobs_data = output['choices'][0]['logprobs']
           
            if 'token_logprobs' in logprobs_data and logprobs_data['token_logprobs']:
                log_probs = [lp for lp in logprobs_data['token_logprobs'] if lp is not None]
           
            if 'top_logprobs' in logprobs_data and logprobs_data['top_logprobs']:
                for token_dict in logprobs_data['top_logprobs']:
                    if token_dict:
                        logprobs_list = list(token_dict.values())
                        probs = np.exp(logprobs_list)
                        probs = probs / np.sum(probs)
                        token_probs_distributions.append(probs)
       
        # Cleanup after generation
        gc.collect()
       
        return text, log_probs, token_probs_distributions
       
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        gc.collect() # Cleanup even on error
        return "Error generating response.", [], []
   
# ============ Answer Generation (ENHANCED) ============
def generate_answer(query: str, top_k: int = 5, max_tokens: int = 400,
                    filter_category: str = None):
    """
    ENHANCED: Now includes domain classification and confidence scoring
    FIXED: Added multilingual profile handling like in Self-RAG path for consistent Urdu query processing.
    """
    start_time = time.time()
    print(f"[RAG] Processing query: {query[:50]}...", flush=True)
    sys.stdout.flush()
   
    # FIXED: Use profile for multilingual handling
    profile = build_query_lang_profile(query)
    query_info = analyze_query(profile.original_query)
    query_for_rag = profile.query_en  # English for retrieval/classification
    original_query = profile.original_query
   
    # CLASSIFY DOMAIN (using English query)
    domain_start = time.time()
    print(f"[RAG] Classifying domain...", flush=True)
    sys.stdout.flush()
    domain_classification = DomainClassifier.classify_domain(query_for_rag)
    print(f"[TIMING] Domain classification: {time.time() - domain_start:.2f}s", flush=True)
    winning_domain = domain_classification.get("domain", "general")
   
    # FIXED: Expand English query for retrieval
    retrieval_query_en = expand_query_for_retrieval(query_for_rag, winning_domain, query_info)
   
    # Retrieve with embeddings (using English query)
    retrieval_start = time.time()
    print(f"[RAG] Retrieving from Supabase (top_k={top_k})...", flush=True)
    sys.stdout.flush()
    results, query_embedding, doc_embeddings = retrieve_from_supabase(
        retrieval_query_en, top_k=top_k, filter_category=filter_category
    )
    print(f"[TIMING] Retrieval: {time.time() - retrieval_start:.2f}s", flush=True)
   
    if not results:
        no_answer = "معاف کیجیے، میں اس سوال کا جواب دینے کے لیے متعلقہ معلومات نہیں ڈھونڈ سکا۔" if profile.output_lang == "ur" else "I apologize, but I couldn't find relevant information to answer this question."
        return no_answer, original_query, profile.input_lang, [], {}, domain_classification
   
    # Build context (translate if needed for output lang)
    context_parts = []
    for r in results:
        chunk_text = sanitize_chunk_text(r["text"])
       
        if profile.output_lang in ["ur", "roman_ur"] and TRANSLATE_CONTEXT_FOR_URDU_OUTPUT:
            try:
                chunk_text = translate_english_to_urdu(chunk_text)
            except Exception:
                pass
       
        if chunk_text:
            MAX_CHUNK_CHARS = 1200
            if len(chunk_text) > MAX_CHUNK_CHARS:
                chunk_text = chunk_text[:MAX_CHUNK_CHARS].rsplit('\n', 1)[0] + "\n\n[truncated]"
            context_parts.append(chunk_text)
   
    context = "\n\n".join(context_parts)
   
    # Build prompt based on output lang
    wants_list = query_info['wants_list']
    wants_summary = query_info['wants_summary']
    wants_detail = query_info['wants_detail']

    if profile.output_lang == "ur":
        base_instruction = "آپ الخدمت فاؤنڈیشن پاکستان کے لیے ایک مددگار اسسٹنٹ ہیں。"
        if wants_list:
            format_instruction = "براہ کرم نقاط کی شکل میں واضح جواب دیں۔"
        elif wants_summary:
            format_instruction = "براہ کرم 2-3 جملوں میں مختصر خلاصہ دیں۔"
        else:
            format_instruction = "براہ کرم مکمل اور واضح جواب دیں۔ اگر سیاق و سباق میں تفصیلات ہیں تو سب شامل کریں۔"
       
        # Use Urdu-script query for display if available
        display_question = profile.query_urdu_script or original_query

        prompt = f"""{base_instruction}
{format_instruction}

اہم ہدایات:
- صرف نیچے دیے گئے سیاق و سباق کی معلومات استعمال کریں
- عمومی علم سے جواب نہ دیں
- اگر جواب سیاق و سباق میں نہیں تو صرف یہ کہیں: "مجھے معلوم نہیں"
- کوئی لیبل شامل نہ کریں
- صرف براہ راست جواب دیں

سیاق و سباق:
{context}

سوال: {display_question}

جواب (صرف اردو میں، ماخذ شامل نہ کریں):
"""
    else:
        base_instruction = "You are a helpful customer support agent for Alkhidmat Foundation Pakistan."
       
        if wants_list:
            format_instruction = "Provide a clear answer in bullet point format."
        elif wants_summary:
            format_instruction = "Provide a brief summary in 2-3 sentences."
        elif wants_detail:
            format_instruction = "Provide a detailed, comprehensive answer covering all relevant information from the context."
        else:
            format_instruction = "Provide a clear, complete answer. Include all relevant details from the context."
       
        prompt = f"""{base_instruction}

{format_instruction}

CRITICAL INSTRUCTIONS:
- Use ONLY the information provided in the context below
- DO NOT answer from general knowledge
- If the answer is not in the context, respond EXACTLY: "I don't know"
- DO NOT include any labels like 'Answer:', 'Question:', etc.
- Return ONLY the direct answer
- DO NOT copy context verbatim
- This is ONLY for Alkhidmat Foundation questions

Information:
{context}

User question: {original_query}

Answer (ONLY the final answer, no labels):
"""
   
    # Generate with confidence data
    llm_start = time.time()
    print(f"[RAG] Generating answer with LLM...", flush=True)
    sys.stdout.flush()
    answer, log_probs, token_probs_distributions = llm_generate(
        prompt, max_tokens=max_tokens, stop_tokens=["\nUser question:", "\nQuestion:", "\nسوال:"]
    )
    print(f"[TIMING] LLM generation: {time.time() - llm_start:.2f}s", flush=True)
   
    # Clean response
    print(f"[RAG] Cleaning response...", flush=True)
    sys.stdout.flush()
    answer = clean_llm_response(answer)
   
    # CALCULATE CONFIDENCE SCORES
    print(f"[RAG] Calculating confidence scores...", flush=True)
    sys.stdout.flush()
    retrieval_conf = ConfidenceScorer.calculate_retrieval_confidence(
        query_embedding=query_embedding,
        doc_embeddings=doc_embeddings,
        top_k=top_k
    )
   
    confidence_scores = ConfidenceScorer.calculate_combined_confidence(
        log_probs=log_probs,
        retrieval_confidence=retrieval_conf,
        token_probs_distributions=token_probs_distributions
    )
   
    # FIXED: Post-process answer based on output_lang
    if profile.output_lang == "ur":
        if not is_urdu_script(answer):
            answer = translate_english_to_urdu(answer, timeout=15)
    elif profile.output_lang == "roman_ur":
        protected = protect_brand_terms(answer)
        answer_ur = translate_english_to_urdu(protected, timeout=15)
        answer_ur = restore_brand_terms(answer_ur)
        answer = romanize_to_roman_urdu_with_llm(answer_ur)

    # Print confidence scores
    print("\n" + "="*80, flush=True)
    print("CONFIDENCE SCORES:", flush=True)
    print("="*80, flush=True)
    combined_conf = confidence_scores.get('combined_confidence', 0) if isinstance(confidence_scores, dict) else 0
    retrieval_conf_val = confidence_scores.get('retrieval_confidence', 0) if isinstance(confidence_scores, dict) else 0
    avg_token_conf = confidence_scores.get('avg_token_confidence', 0) if isinstance(confidence_scores, dict) else 0
    weighted_top_k = confidence_scores.get('weighted_top_k', 0) if isinstance(confidence_scores, dict) else 0
    perplexity = confidence_scores.get('perplexity', 0) if isinstance(confidence_scores, dict) else 0
    entropy_conf = confidence_scores.get('entropy_confidence', 0) if isinstance(confidence_scores, dict) else 0
   
    print(f" Combined Confidence: {combined_conf:.4f} ⭐", flush=True)
    print(f" ├─ Retrieval Confidence: {retrieval_conf_val:.4f}", flush=True)
    print(f" ├─ Avg Token Confidence: {avg_token_conf:.4f}", flush=True)
    print(f" ├─ Weighted Top-K: {weighted_top_k:.4f}", flush=True)
    print(f" ├─ Perplexity: {perplexity:.4f}", flush=True)
    print(f" └─ Entropy Confidence: {entropy_conf:.4f}", flush=True)
    print("="*80, flush=True)
   
    # Interpretation
    if combined_conf >= 0.7:
        print("✅ High confidence - Answer is likely reliable", flush=True)
    elif combined_conf >= 0.5:
        print("⚠️ Moderate confidence - Answer may need verification", flush=True)
    else:
        print("❌ Low confidence - Answer should be verified from sources", flush=True)
    print("="*80 + "\n", flush=True)
    sys.stdout.flush()
   
    # Create sources list
    sources = [{
        "category": r['category'],
        "filename": r['filename'],
        "file_path": r['file_path'],
        "similarity": r['similarity']
    } for r in results]
   
    total_time = time.time() - start_time
    print(f"[RAG] Answer generated successfully (length: {len(answer)} chars)", flush=True)
    print(f"[TIMING] Total time: {total_time:.2f}s", flush=True)
    sys.stdout.flush()
   
    return answer, original_query, profile.input_lang, sources, confidence_scores, domain_classification

LATIN_ONLY_RE = re.compile(r'^[\x00-\x7F\s]+$') # only ASCII + whitespace

def romanize_to_roman_urdu_with_llm(urdu_text: str, max_tokens: int = 260) -> str:
    """Convert Urdu script to Roman Urdu using LLM."""
    if not urdu_text.strip():
        return urdu_text

    # Attempt 1-2: direct Urdu -> Roman Urdu
    for attempt in range(2):
        prompt = f"""Task: Convert Urdu (Arabic script) to Roman Urdu (Latin letters).

STRICT RULES:
- Output MUST be in Latin letters only (a-z). No Urdu/Arabic characters at all.
- If you output any Urdu/Arabic characters, your answer is INVALID.
- Keep meaning exactly the same.
- Keep proper names like Alkhidmat as "Alkhidmat".
- Keep phone numbers/URLs unchanged.
- Output ONLY the Roman Urdu text (no labels).

Urdu:
{urdu_text}

Roman Urdu (Latin-only):"""

        model = load_llm()
        out = model(prompt, max_tokens=max_tokens, temperature=0.0, stop=["\n\n", "\nUrdu:", "\nRoman Urdu:"], echo=False)
        out = out['choices'][0]['text'].strip()

        # Accept only if it's truly Latin-only AND not Urdu-script
        if out and (not is_urdu_script(out)) and LATIN_ONLY_RE.match(out):
            return out

    # Fallback: Urdu -> English via GoogleTranslator, then English -> Roman Urdu via LLM
    en = translate_urdu_to_english(urdu_text)
    prompt2 = f"""Task: Translate English into Roman Urdu (Latin letters).

STRICT RULES:
- Output MUST be in Latin letters only (a-z). No Urdu/Arabic characters.
- Keep meaning exactly the same.
- Keep proper names like Alkhidmat as "Alkhidmat".
- Keep phone numbers/URLs unchanged.
- Output ONLY Roman Urdu text (no labels).

English:
{en}

Roman Urdu (Latin-only):"""

    model = load_llm()
    out2 = model(prompt2, max_tokens=max_tokens, temperature=0.0, echo=False)
    out2 = out2['choices'][0]['text'].strip()
    if out2 and (not is_urdu_script(out2)) and LATIN_ONLY_RE.match(out2):
        return out2

    # Last resort: return an English version rather than Urdu (better than violating roman_ur contract)
    return en

def clean_llm_response(text: str) -> str:
    """Clean up LLM output to remove unwanted artifacts"""
    if not text:
        return text

    text = text.replace('\r\n', '\n')
    text = re.sub(r'\[?Context \d+\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'based on Context \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'as per (?:the )?context', '', text, flags=re.IGNORECASE)

    cutoff_patterns = [r'\nUser question\s*:', r'\nQuestion\s*:', r'\nUser question', r'\nQuestion', r'\nAnswer\s*:']
    earliest = len(text)
    for pat in cutoff_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            earliest = min(earliest, m.start())

    if earliest < len(text):
        text = text[:earliest].strip()

    text = re.sub(r'(?mi)^(user question|question|answer)\s*[:\-–]\s*', '', text)

    lines = text.split('\n')
    seen = set()
    out_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            out_lines.append('')
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out_lines.append(line)
    text = '\n'.join(out_lines)

    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

# ============ SELF-RAG ANSWER GENERATION ============
def generate_answer_selfrag(query: str, top_k: int = 5, max_tokens: int = 400, filter_category: str = None):
    """
    Multilingual Self-RAG implementation with enhanced verification.
    Returns: (answer, original_query, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics)
    """
    start_time = time.time()
    print(f"\n{'='*80}", flush=True)
    print(f"SELF-RAG QUERY PROCESSING", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"[SELF-RAG] Processing query: {query[:80]}...", flush=True)
    sys.stdout.flush()

    profile = build_query_lang_profile(query)
    query_info = analyze_query(profile.original_query)
    query_for_rag = profile.query_en
    original_query = profile.original_query

    llm_model = load_llm()
    critic = SelfRAGCritic(llm_model)

    selfrag_metrics = {
        'domain_relevant': True,
        'domain_confidence': 0.0,
        'retrieve_needed': False,
        'retrieve_confidence': 0.0,
        'answer_in_context': False,
        'answer_in_context_confidence': 0.0,
        'relevance_score': 0.0,
        'support_level': 'uncertain',
        'support_score': 0.0,
        'utility_score': 0.0,
        'utility_rating': 0
    }

    # STEP 0: Check domain relevance
    if SELFRAG_ENABLE:
        print(f"\n[SELF-RAG] Step 0: Checking domain relevance...", flush=True)
        is_domain_relevant, domain_conf = critic.is_domain_relevant(query_for_rag)
        selfrag_metrics['domain_relevant'] = is_domain_relevant
        selfrag_metrics['domain_confidence'] = domain_conf

        if (not is_domain_relevant) and (domain_conf >= 0.75):
            print(f" ✗ Question is IRRELEVANT to Alkhidmat Foundation (confidence: {domain_conf:.2f})", flush=True)
            dummy_domain_classification = {
                'domain': 'irrelevant',
                'confidence': 0.0,
                'all_scores': {'donation': 0.0, 'healthcare': 0.0, 'general': 0.0}
            }
            irrelevant_response = "That is an irrelevant question."
            return (irrelevant_response, original_query, profile.input_lang, [], {'combined_confidence': 0.0},
                    dummy_domain_classification, selfrag_metrics)
        else:
            print(f" ✓ Question is RELEVANT to domain (confidence: {domain_conf:.2f})", flush=True)

    # STEP 1: Should we retrieve?
    if SELFRAG_ENABLE:
        print(f"\n[SELF-RAG] Step 1: Checking retrieval necessity...", flush=True)
        retrieve_needed, retrieve_conf = critic.should_retrieve(query_for_rag)
        selfrag_metrics['retrieve_needed'] = retrieve_needed
        selfrag_metrics['retrieve_confidence'] = retrieve_conf

        if retrieve_needed:
            print(f" ✓ Retrieval NEEDED (confidence: {retrieve_conf:.2f})", flush=True)
        else:
            print(f" ✗ Retrieval NOT needed (confidence: {retrieve_conf:.2f})", flush=True)
            no_retrieval_answer = "I can help with that, but I need to access my knowledge base for specific information about Alkhidmat Foundation."
            return (no_retrieval_answer, original_query, profile.input_lang, [],
                    {'combined_confidence': 0.5}, {}, selfrag_metrics)
    else:
        retrieve_needed = True
        selfrag_metrics['retrieve_needed'] = True

    # Classify domain
    domain_start = time.time()
    print(f"\n[SELF-RAG] Classifying domain...", flush=True)
    sys.stdout.flush()
    domain_classification = DomainClassifier.classify_domain(query_for_rag)
    print(f"[TIMING] Domain classification: {time.time() - domain_start:.2f}s", flush=True)
    winning_domain = domain_classification.get("domain", "general")

    # Expand query for retrieval
    retrieval_query_en = expand_query_for_retrieval(query_for_rag, winning_domain, query_info)

    # STEP 2: Retrieve documents
    retrieval_start = time.time()
    print(f"\n[SELF-RAG] Step 2: Retrieving documents...", flush=True)
    sys.stdout.flush()
    results, query_embedding, doc_embeddings = retrieve_from_supabase(
        retrieval_query_en, top_k=top_k, filter_category=filter_category
    )
    print(f"[TIMING] Retrieval: {time.time() - retrieval_start:.2f}s", flush=True)

    if not results:
        if profile.output_lang == "ur":
            no_answer = "معاف کیجیے، میں اس سوال کا جواب دینے کے لیے متعلقہ معلومات نہیں ڈھونڈ سکا۔"
        else:
            no_answer = "I apologize, but I couldn't find relevant information to answer this question."
        return no_answer, original_query, profile.input_lang, [], {}, domain_classification, selfrag_metrics

    # STEP 3: Assess relevance
    if SELFRAG_ENABLE:
        print(f"\n[SELF-RAG] Step 3: Assessing document relevance...", flush=True)
        relevant_results = []
        relevance_scores = []

        for i, result in enumerate(results):
            is_relevant, rel_conf = critic.assess_relevance(query_for_rag, result['text'])
            relevance_scores.append(rel_conf if is_relevant else 0.0)

            if is_relevant and rel_conf >= SELFRAG_RELEVANCE_THRESHOLD:
                relevant_results.append(result)
                print(f" ✓ Doc {i+1}: RELEVANT (confidence: {rel_conf:.2f})", flush=True)
            else:
                print(f" ✗ Doc {i+1}: Not relevant (confidence: {rel_conf:.2f})", flush=True)

        if not relevant_results:
            print(f" ⚠️ No relevant documents found after filtering!", flush=True)
            no_answer = "I found some documents but they don't seem relevant enough to answer your question accurately."
            return no_answer, original_query, profile.input_lang, results, {}, domain_classification, selfrag_metrics

        results = relevant_results
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        selfrag_metrics['relevance_score'] = float(avg_relevance)
        print(f" → Average relevance: {avg_relevance:.2f}", flush=True)

    # Build context
    context_parts = []
    for r in results:
        chunk_text = sanitize_chunk_text(r["text"])
        if not chunk_text:
            continue

        if profile.output_lang == "ur" and TRANSLATE_CONTEXT_FOR_URDU_OUTPUT:
            try:
                chunk_text = translate_english_to_urdu(chunk_text)
            except Exception:
                pass

        MAX_CHUNK_CHARS = 1200
        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS].rsplit('\n', 1)[0] + "\n\n[truncated]"
        context_parts.append(chunk_text)

    context = "\n\n".join(context_parts)

    # STEP 3.5: Check if answer exists in context
    if SELFRAG_ENABLE:
        print(f"\n[SELF-RAG] Step 3.5: Checking if answer exists in context...", flush=True)
        can_answer, answer_conf = critic.check_answer_in_context(query_for_rag, context)
        selfrag_metrics['answer_in_context'] = can_answer
        selfrag_metrics['answer_in_context_confidence'] = answer_conf

        if not can_answer and answer_conf >= 0.7:
            print(f" ✗ Answer CANNOT be found in context (confidence: {answer_conf:.2f})", flush=True)
            no_info_response = "I don't have that information."
            return (no_info_response, original_query, profile.input_lang, results,
                    {'combined_confidence': 0.0}, domain_classification, selfrag_metrics)
        else:
            print(f" ✓ Answer CAN be found in context (confidence: {answer_conf:.2f})", flush=True)

    # Build prompt
    wants_list = query_info['wants_list']
    wants_summary = query_info['wants_summary']
    wants_detail = query_info['wants_detail']

    if profile.output_lang == "ur":
        base_instruction = "آپ الخدمت فاؤنڈیشن پاکستان کے لیے ایک مددگار اسسٹنٹ ہیں。"
        if wants_list:
            format_instruction = "براہ کرم نقاط کی شکل میں واضح جواب دیں۔"
        elif wants_summary:
            format_instruction = "براہ کرم 2-3 جملوں میں مختصر خلاصہ دیں۔"
        else:
            format_instruction = "براہ کرم مکمل اور واضح جواب دیں۔"

        if profile.input_lang == "roman_ur" and profile.query_urdu_script:
            display_question = profile.query_urdu_script
        else:
            display_question = original_query

        prompt = f"""{base_instruction}
{format_instruction}

اہم ہدایات:
- صرف نیچے دیے گئے سیاق و سباق کی معلومات استعمال کریں
- عمومی علم سے جواب نہ دیں
- اگر جواب سیاق و سباق میں نہیں تو صرف یہ کہیں: "مجھے معلوم نہیں"
- کوئی لیبل شامل نہ کریں
- صرف براہ راست جواب دیں

سیاق و سباق:
{context}

سوال: {display_question}

جواب (صرف اردو میں، ماخذ شامل نہ کریں):
"""
    else:
        base_instruction = "You are a helpful customer support agent for Alkhidmat Foundation Pakistan."
        if wants_list:
            format_instruction = "Provide a clear answer in bullet point format."
        elif wants_summary:
            format_instruction = "Provide a brief summary in 2-3 sentences."
        elif wants_detail:
            format_instruction = "Provide a detailed, comprehensive answer covering all relevant information from the context."
        else:
            format_instruction = "Provide a clear, complete answer."

        prompt = f"""{base_instruction}

{format_instruction}

CRITICAL INSTRUCTIONS:
- Use ONLY the information provided in the context below
- DO NOT answer from general knowledge
- If the answer is not in the context, respond EXACTLY: "I don't know"
- DO NOT include any labels like 'Answer:', 'Question:', etc.
- Return ONLY the direct answer
- DO NOT copy context verbatim

Information:
{context}

User question: {original_query}

Answer (ONLY the final answer, no labels):
"""

    # STEP 4: Generate
    llm_start = time.time()
    print(f"\n[SELF-RAG] Step 4: Generating answer...", flush=True)
    sys.stdout.flush()
    answer, log_probs, token_probs_distributions = llm_generate(
        prompt, max_tokens=max_tokens, stop_tokens=["\nUser question:", "\nQuestion:", "\nسوال:"]
    )
    print(f"[TIMING] LLM generation: {time.time() - llm_start:.2f}s", flush=True)
    answer = clean_llm_response(answer)

    # STEP 5: Verify support
    if SELFRAG_ENABLE:
        print(f"\n[SELF-RAG] Step 5: Verifying answer support...", flush=True)
        support_level, support_conf = critic.verify_support(query_for_rag, answer, context)
        selfrag_metrics['support_level'] = support_level
        selfrag_metrics['support_score'] = support_conf
        print(f" → Support level: {support_level.upper()} (confidence: {support_conf:.2f})", flush=True)

        if support_level == "no_support" and support_conf >= SELFRAG_SUPPORT_THRESHOLD:
            print(f" ⚠️ Answer NOT supported by context - rejecting!", flush=True)
            no_answer = "I cannot provide a reliable answer based on the available information."
            return no_answer, original_query, profile.input_lang, results, {}, domain_classification, selfrag_metrics

    # STEP 6: Utility
    if SELFRAG_ENABLE:
        print(f"\n[SELF-RAG] Step 6: Evaluating answer utility...", flush=True)
        utility_rating, utility_conf = critic.evaluate_utility(query_for_rag, answer)
        selfrag_metrics['utility_rating'] = utility_rating
        selfrag_metrics['utility_score'] = utility_rating / 5.0
        print(f" → Utility rating: {utility_rating}/5 (confidence: {utility_conf:.2f})", flush=True)

        if utility_rating <= 2:
            print(f" ⚠️ Answer utility too low - rejecting!", flush=True)
            no_answer = "I cannot provide a sufficiently useful answer to your question based on the available information."
            return no_answer, original_query, profile.input_lang, results, {}, domain_classification, selfrag_metrics

    # Confidence scores
    print(f"\n[SELF-RAG] Calculating final confidence scores...", flush=True)
    sys.stdout.flush()
    retrieval_conf = ConfidenceScorer.calculate_retrieval_confidence(query_embedding, doc_embeddings, top_k=top_k)
   
    # Prepare Self-RAG scores for confidence calculation
    selfrag_scores = {
        'support_score': selfrag_metrics.get('support_score', 0.0),
        'utility_score': selfrag_metrics.get('utility_score', 0.0),
        'relevance_score': selfrag_metrics.get('relevance_score', 0.0)
    }
   
    confidence_scores = ConfidenceScorer.calculate_combined_confidence(
        log_probs=log_probs,
        retrieval_confidence=retrieval_conf,
        token_probs_distributions=token_probs_distributions,
        selfrag_scores=selfrag_scores
    )

    combined_conf = confidence_scores.get('combined_confidence', 0)
    if SELFRAG_ENABLE and combined_conf < SELFRAG_MIN_CONFIDENCE:
        print(f"\n⚠️ SELF-RAG: Combined confidence ({combined_conf:.2f}) below threshold ({SELFRAG_MIN_CONFIDENCE})", flush=True)
        no_answer = "I don't have enough confidence in my answer to provide it. Please rephrase your question or contact support."
        return no_answer, original_query, profile.input_lang, results, confidence_scores, domain_classification, selfrag_metrics

    # OUTPUT LANGUAGE POST-PROCESSING
    if profile.output_lang == "ur":
        if not is_urdu_script(answer):
            answer = translate_english_to_urdu(answer, timeout=15)
    elif profile.output_lang == "roman_ur":
        protected = protect_brand_terms(answer)
        answer_ur = translate_english_to_urdu(protected, timeout=15)
        answer_ur = restore_brand_terms(answer_ur)
        answer = romanize_to_roman_urdu_with_llm(answer_ur)

    # Create sources list
    sources = [{
        "category": r['category'],
        "filename": r['filename'],
        "file_path": r['file_path'],
        "similarity": r['similarity']
    } for r in results]

    total_time = time.time() - start_time
    print(f"\n[SELF-RAG] Answer ACCEPTED and generated successfully", flush=True)
    print(f"[TIMING] Total time: {total_time:.2f}s", flush=True)
    print(f"{'='*80}\n", flush=True)
    sys.stdout.flush()

    return answer, original_query, profile.input_lang, sources, confidence_scores, domain_classification, selfrag_metrics

# ============ CLI Functions (ENHANCED) ============
def query_alkhidmat_rag(query: str, category: str = None, use_selfrag: bool = True):
    """ENHANCED: Now displays domain classification and confidence scores"""
    if use_selfrag and SELFRAG_ENABLE:
        answer, original_query, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics = generate_answer_selfrag(
            query, filter_category=category
        )
    else:
        answer, original_query, input_lang, sources, confidence_scores, domain_classification = generate_answer(
            query, filter_category=category
        )
        selfrag_metrics = {}
   
    print("\n" + "="*80)
    print("QUESTION:", original_query)
    print(f"(Detected input language: {input_lang})")
    if category:
        print("CATEGORY FILTER:", category)
    print("="*80)
   
    # Display domain classification (NEW)
    domain_emoji = DomainClassifier.get_domain_emoji(domain_classification['domain'])
    print(f"\n{domain_emoji} DOMAIN CLASSIFICATION:")
    print(f"{'-'*80}")
    print(f" Classified Domain: {domain_classification['domain'].upper()}")
    print(f" Classification Confidence: {domain_classification['confidence']:.2%}")
    print(f"\n Similarity Scores:")
    for domain, score in domain_classification['all_scores'].items():
        emoji = DomainClassifier.get_domain_emoji(domain)
        bar = '█' * int(score * 50)
        print(f" {emoji} {domain:12s}: {score:.4f} {bar}")
    print(f"{'-'*80}")
   
    print("\nANSWER:")
    print(answer)
   
    # Display confidence scores (NEW)
    print("\n" + "="*80)
    print("CONFIDENCE SCORES:")
    print("="*80)
    print(f" Combined Confidence: {confidence_scores.get('combined_confidence', 0):.4f} ⭐")
    print(f" ├─ Retrieval Confidence: {confidence_scores.get('retrieval_confidence', 0):.4f}")
    print(f" ├─ Avg Token Confidence: {confidence_scores.get('avg_token_confidence', 0):.4f}")
    print(f" ├─ Weighted Top-K: {confidence_scores.get('weighted_top_k', 0):.4f}")
    print(f" ├─ Perplexity: {confidence_scores.get('perplexity', 0):.4f}")
    print(f" └─ Entropy Confidence: {confidence_scores.get('entropy_confidence', 0):.4f}")
    print("="*80)
   
    # Interpretation (NEW)
    combined = confidence_scores.get('combined_confidence', 0)
    if combined >= 0.7:
        print("✅ High confidence - Answer is likely reliable")
    elif combined >= 0.5:
        print("⚠️ Moderate confidence - Answer may need verification")
    else:
        print("❌ Low confidence - Answer should be verified from sources")
   
    print("\n" + "="*80 + "\n")
    return answer

def show_statistics():
    """Show database statistics"""
    supabase = get_supabase_client()
   
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
   
    try:
        result = supabase.table("documents").select("doc_id", count="exact").execute()
        print(f"\nTotal chunks: {result.count}")
       
        result = supabase.table("documents").select("category").execute()
        categories = {}
        for row in result.data:
            cat = row.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
       
        print("\nDocuments by Category:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f" {cat}: {count} chunks")
           
    except Exception as e:
        print(f"Error getting statistics: {e}")

# ============ Batch Processing (ENHANCED) ============
def batch_query_file(input_file: str, output_file: str, use_selfrag: bool = True):
    """ENHANCED: Now includes domain classification and confidence scores in output"""
    os.environ['BATCH_MODE'] = 'True'
   
    print(f"\n{'='*80}")
    print(f"BATCH QUERY MODE: Processing {input_file} -> {output_file}")
    print(f"{'='*80}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
       
        print(f"Found {len(queries)} queries to process.")
       
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("ALKHIDMAT RAG (SUPABASE) - BATCH QUERY RESULTS\n")
            f_out.write("WITH DOMAIN CLASSIFICATION & CONFIDENCE SCORING\n")
            f_out.write(f"Source file: {input_file}\n")
            f_out.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for i, query in enumerate(queries):
                print(f"Processing {i+1}/{len(queries)}: '{query[:40]}...'")
               
                if use_selfrag and SELFRAG_ENABLE:
                    answer, original_query, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics = generate_answer_selfrag(query)
                else:
                    answer, original_query, input_lang, sources, confidence_scores, domain_classification = generate_answer(query)
                    selfrag_metrics = {}
               
                f_out.write(f"{'='*80}\n")
                f_out.write(f"QUERY {i+1}/{len(queries)}\n")
                f_out.write(f"{'='*80}\n")
                f_out.write(f"QUERY: {original_query}\n")
                f_out.write(f"LANGUAGE: {input_lang}\n\n")
               
                # Write domain classification (NEW)
                domain_emoji = DomainClassifier.get_domain_emoji(domain_classification['domain'])
                f_out.write(f"{domain_emoji} DOMAIN CLASSIFICATION:\n")
                f_out.write(f"{'-'*80}\n")
                f_out.write(f"Classified Domain: {domain_classification['domain'].upper()}\n")
                f_out.write(f"Classification Confidence: {domain_classification['confidence']:.2%}\n")
                f_out.write(f"Similarity Scores:\n")
                for domain, score in domain_classification['all_scores'].items():
                    f_out.write(f" - {domain}: {score:.4f}\n")
                f_out.write(f"\n")
               
                f_out.write(f"ANSWER:\n{answer}\n\n")
               
                # Write confidence scores (NEW)
                f_out.write(f"CONFIDENCE SCORES:\n")
                f_out.write(f"{'-'*80}\n")
                f_out.write(f"Combined Confidence: {confidence_scores.get('combined_confidence', 0):.4f} ⭐\n")
                f_out.write(f"├─ Retrieval Confidence: {confidence_scores.get('retrieval_confidence', 0):.4f}\n")
                f_out.write(f"├─ Avg Token Confidence: {confidence_scores.get('avg_token_confidence', 0):.4f}\n")
                f_out.write(f"├─ Weighted Top-K: {confidence_scores.get('weighted_top_k', 0):.4f}\n")
                f_out.write(f"├─ Perplexity: {confidence_scores.get('perplexity', 0):.4f}\n")
                f_out.write(f"└─ Entropy Confidence: {confidence_scores.get('entropy_confidence', 0):.4f}\n\n")
               
                combined = confidence_scores.get('combined_confidence', 0)
                if combined >= 0.7:
                    f_out.write("✅ High confidence - Answer is likely reliable\n")
                elif combined >= 0.5:
                    f_out.write("⚠️ Moderate confidence - Answer may need verification\n")
                else:
                    f_out.write("❌ Low confidence - Answer should be verified from sources\n")
               
                f_out.write(f"\nSOURCES USED:\n")
                if sources:
                    for s in sources:
                        f_out.write(f" - [{s['category']}] {s['filename']} (Sim: {s['similarity']:.3f})\n")
                else:
                    f_out.write(" - No relevant documents found.\n")
               
                f_out.write(f"\n{'='*40}\n\n")
               
                print(f" ✓ Domain: {domain_classification['domain']} | Confidence: {confidence_scores.get('combined_confidence', 0):.2f}")

        print(f"\n{'='*80}")
        print(f"✅ BATCH PROCESSING COMPLETE. Results written to {output_file}")
        print(f"{'='*80}\n")
       
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during batch processing: {e}")
    finally:
        os.environ['BATCH_MODE'] = 'False'

# ============ Main ============
if __name__ == "__main__":
    import sys
    import atexit
   
    # Add cleanup handler for LLM
    def cleanup_llm():
        global _LLM_MODEL
        if _LLM_MODEL is not None:
            try:
                _LLM_MODEL = None
            except:
                pass
   
    atexit.register(cleanup_llm)
   
    DEFAULT_ZIP = "Al Khidmat Knowledge Base.zip"
   
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connection()
       
    elif len(sys.argv) > 1 and sys.argv[1] == "build":
        zip_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZIP
        clear = "--clear" in sys.argv
        build_alkhidmat_rag(zip_path, clear_existing=clear)
       
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What donation methods does Alkhidmat accept?"
        use_selfrag = "--no-selfrag" not in sys.argv
        query_alkhidmat_rag(q, use_selfrag=use_selfrag)

    elif len(sys.argv) > 1 and sys.argv[1] == "file_query":
        input_file = sys.argv[2] if len(sys.argv) > 2 else "input_queries.txt"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "output_answers.txt"
        use_selfrag = "--no-selfrag" not in sys.argv
        batch_query_file(input_file, output_file, use_selfrag=use_selfrag)
       
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        show_statistics()
       
    else:
        print("\n" + "="*80)
        print("ALKHIDMAT RAG SYSTEM (SUPABASE + LLAMA-CPP)")
        print("WITH DOMAIN CLASSIFICATION & CONFIDENCE SCORING")
        print("="*80)
        print("\nUSAGE:")
        print("1. Test Connection:")
        print(" python RAG_supabase_enhanced.py test")
       
        print("\n2. Build Index (Upload to Supabase):")
        print(" python RAG_supabase_enhanced.py build [zip_path] [--clear]")
       
        print("\n3. Single Query (Terminal):")
        print(" python RAG_supabase_enhanced.py query 'your question'")
       
        print("\n4. Batch Query (File I/O):")
        print(" python RAG_supabase_enhanced.py file_query input.txt output.txt")
       
        print("\n5. View Stats:")
        print(" python RAG_supabase_enhanced.py stats")
        print("\n" + "="*80)