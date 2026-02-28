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
import zipfile

import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

# Urdu helpers
from deep_translator import GoogleTranslator
import langdetect

# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import domain anchors
from domain_anchors import DOMAIN_ANCHOR_QUERIES
from rag_config import (
    BRAND_TERMS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONVERSATION_MEMORY_ENABLE,
    DOMAIN_CENTROID_REUSE_THRESHOLD,
    EMBEDDING_CACHE_ENABLE,
    EMBEDDING_CACHE_SIMILARITY_THRESHOLD,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    EVIDENCE_COVERAGE_ENABLE,
    QUERY_ROUTER_ENABLE,
    RELEVANCE_THRESHOLD,
    RETRIEVAL_RETRY_ENABLE,
    RETRIEVAL_RETRY_MAX_ATTEMPTS,
    RETRIEVAL_RETRY_RELEVANCE_THRESHOLD,
    ROMAN_URDU_MARKERS,
    SELFRAG_ENABLE,
    SELFRAG_MIN_CONFIDENCE,
    SELFRAG_RELEVANCE_THRESHOLD,
    SELFRAG_RETRIEVE_THRESHOLD,
    SELFRAG_SUPPORT_THRESHOLD,
    SUPABASE_KEY,
    SUPABASE_URL,
    TRANSLATE_CONTEXT_FOR_URDU_OUTPUT,
)
from rag_embeddings import (
    cache_query_embedding,
    create_embeddings,
    find_similar_cached_embedding,
    get_embedder,
    normalize_query,
)
from rag_llm import load_llm, llm_generate
from rag_supabase_client import get_supabase_client, test_connection

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

    def is_domain_relevant(self, query: str) -> bool:
        # Pre-check: Look for Alkhidmat-related keywords and program names
        query_lower = query.lower().strip()
        
        # Alkhidmat Foundation keywords
        alkhidmat_keywords = [
            "alkhidmat", "alkhdmt", "alkhidmat foundation", "alkhidmat foundation pakistan"
        ]
        
        # Known Alkhidmat programs and services
        alkhidmat_programs = [
            "bano qabil", "banoqabil", "banu qabil",
            "aghosh", "aghosh home", "orphan",
            "mawakhat", "mawakhat program",
            "zakat", "sadaqah", "donation", "donate",
            "qurbani", "fidya", "kafara",
            "healthcare", "hospital", "clinic", "medical",
            "education", "scholarship",
            "disaster", "relief", "flood", "emergency",
            "volunteer", "volunteering"
        ]
        
        # Check if query contains Alkhidmat keywords
        has_alkhidmat_keyword = any(keyword in query_lower for keyword in alkhidmat_keywords)
        
        # Check if query contains program/service keywords
        has_program_keyword = any(program in query_lower for program in alkhidmat_programs)
        
        # If query contains program/service keywords, it's likely relevant even without "Alkhidmat" explicitly
        if has_program_keyword and not has_alkhidmat_keyword:
            question_words = ["what", "who", "where", "when", "how", "why", "kya", "kaun", "kahan", "kab"]
            is_question = any(query_lower.startswith(q) for q in question_words) or "?" in query
            
            if is_question:
                print(f"[DOMAIN-RELEVANCE] Detected Alkhidmat program/service query: '{query}' - Likely relevant")
            elif len(query_lower.split()) <= 5:
                print(f"[DOMAIN-RELEVANCE] Short query with program keyword: '{query}' - Likely relevant")
        
        prompt = f"""Determine if this question is about Alkhidmat Foundation Pakistan.

Question: {query}

You are a binary classifier.

Task:
Decide whether the user's query is about "Alkhidmat Foundation Pakistan" or its programs/services.

Definition of RELEVANT:
A query is RELEVANT if it refers to:
- Alkhidmat Foundation Pakistan by name OR
- Alkhidmat's programs, services, or operations (even if Alkhidmat name not mentioned)
- Known Alkhidmat programs: Bano Qabil, Aghosh, Mawakhat, etc.
- Alkhidmat services: donations, healthcare, education, disaster relief, etc.

Examples of RELEVANT queries:
- "What is Alkhidmat Foundation?"
- "How can I donate to Alkhidmat?"
- "What is bano qabil?" (Alkhidmat program)
- "What is bano qabil program?" (Alkhidmat program)
- "Where is the Alkhidmat hospital?"
- "How does Aghosh work?" (Alkhidmat program)
- "Tell me about Mawakhat" (Alkhidmat program)

Definition of IRRELEVANT:
A query is IRRELEVANT if:
- It's about general topics unrelated to Alkhidmat (politics, sports, celebrities, etc.)
- It's a generic question that could apply to ANY organization
- It's clearly not about Alkhidmat Foundation or its programs

Examples of IRRELEVANT queries:
- "Best charity in Pakistan" (too generic)
- "What is zakat?" (general Islamic concept, not Alkhidmat-specific)
- "Free hospitals in Karachi" (generic, not Alkhidmat-specific)
- "Imran Khan latest news" (unrelated)
- "How to stay healthy" (general health advice)

Important rules:
- If query mentions Alkhidmat programs/services (Bano Qabil, Aghosh, etc.), mark RELEVANT
- If query is about donations, healthcare, education in Pakistan context, consider RELEVANT
- Be lenient for program/service names even if "Alkhidmat" is not explicitly mentioned
- When in doubt, choose RELEVANT (especially for program/service queries)

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
                return True
            elif "[IRRELEVANT]" in response:
                if has_program_keyword:
                    print(f"[DOMAIN-RELEVANCE] LLM marked irrelevant but contains program keyword - Overriding to relevant")
                    return True
                return False
            if has_program_keyword:
                return True
            return True
        except Exception as e:
            print(f"[SelfRAG] Domain relevance check error: {e}")
            if has_program_keyword:
                return True
            return True

    def should_retrieve(self, query: str) -> bool:
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
                return True
            elif SelfRAGReflectionTokens.NO_RETRIEVE in response:
                return False
            return True
        except Exception as e:
            print(f"[SelfRAG] Retrieval prediction error: {e}")
            return True

    def assess_relevance(self, query: str, document: str) -> bool:
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
                return True
            elif SelfRAGReflectionTokens.IRRELEVANT in response:
                return False
            return True
        except Exception as e:
            print(f"[SelfRAG] Relevance assessment error: {e}")
            return True

    def check_answer_in_context(self, query: str, context: str) -> bool:
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
                return True
            elif "[CANNOT_ANSWER]" in response:
                return False
            return True
        except Exception as e:
            print(f"[SelfRAG] Answer presence check error: {e}")
            return True

    def verify_support(self, query: str, answer: str, context: str) -> str:
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
                return "fully_supported"
            elif SelfRAGReflectionTokens.PARTIALLY_SUPPORTED in response:
                return "partially_supported"
            elif SelfRAGReflectionTokens.NO_SUPPORT in response:
                return "no_support"
            return "uncertain"
        except Exception as e:
            print(f"[SelfRAG] Support verification error: {e}")
            return "uncertain"

    def evaluate_utility(self, query: str, answer: str) -> int:
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
                return 5
            elif SelfRAGReflectionTokens.UTILITY_4 in response:
                return 4
            elif SelfRAGReflectionTokens.UTILITY_3 in response:
                return 3
            elif SelfRAGReflectionTokens.UTILITY_2 in response:
                return 2
            elif SelfRAGReflectionTokens.UTILITY_1 in response:
                return 1
            return 3
        except Exception as e:
            print(f"[SelfRAG] Utility evaluation error: {e}")
            return 3

# ============================================================================
# AGENTIC AI ARCHITECTURE - EMBEDDING REGULATION & HALLUCINATION REDUCTION
# ============================================================================

_conversation_state: Dict[str, Dict] = {}  # {session_id: {last_domain, last_docs, last_answer_confidence}}

# ============================================================================
# AGENTIC AGENTS
# ============================================================================

class RouterAgent:
    """Routes queries before embedding - decides if retrieval is needed."""
    
    def __init__(self, critic: SelfRAGCritic):
        self.critic = critic
    
    def route(self, query: str) -> Tuple[bool, float, Optional[str]]:
        """
        Route query: should we retrieve?
        Returns: (should_retrieve, confidence, cached_answer_or_none)
        """
        if not QUERY_ROUTER_ENABLE:
            return True, 1.0, None
        
        # Check for chit-chat / greetings
        greetings = ["hello", "hi", "hey", "greetings", "salam", "assalam", "assalamu"]
        if any(query.lower().strip().startswith(g) for g in greetings):
            return False, 0.9, "Hello! How can I help you with information about Alkhidmat Foundation?"
        
        query_lower = query.lower().strip()
        
        # Check if it's a question FIRST (questions are NOT niceties)
        question_words = ["what", "who", "where", "when", "how", "why", "which", "whose", "whom",
                         "kya", "kaun", "kahan", "kab", "kyun", "kis", "kisne"]
        question_patterns = ["are you", "is it", "do you", "can you", "will you", "would you", 
                            "have you", "has it", "did you", "does it", "was it", "were you"]
        
        has_question_mark = "?" in query_lower
        starts_with_question = any(query_lower.startswith(q + " ") for q in question_words)
        has_question_pattern = any(pattern in query_lower for pattern in question_patterns)
        
        # If it's a question, skip nicety check
        if not (has_question_mark or starts_with_question or has_question_pattern):
            # Check for niceties (thank you, goodbye, etc.)
            nicety_patterns = [
                "thank you", "thanks", "thank", "thx", "ty", "shukriya", "shukria",
                "goodbye", "bye", "bye bye", "see you", "farewell", "khuda hafiz", "allah hafiz",
                "i appreciate", "appreciate it", "much appreciated", "grateful",
                "that's great", "that's good", "good job", "well done"
            ]
            if any(query_lower == pattern or query_lower.startswith(pattern + " ") for pattern in nicety_patterns):
                if any(query_lower.startswith(kw) for kw in ["thank you", "thanks", "thank", "thx", "ty", "shukriya", "i appreciate", "appreciate"]):
                    return False, 0.9, "You're welcome! I'm glad I could help. Is there anything else you'd like to know about Alkhidmat Foundation?"
                elif any(query_lower.startswith(kw) for kw in ["bye", "goodbye", "see you", "farewell", "khuda hafiz"]):
                    return False, 0.9, "Goodbye! Feel free to come back if you have any questions about Alkhidmat Foundation. Have a great day!"
                else:
                    return False, 0.9, "You're welcome! Is there anything else I can help you with?"
        
        # Check retrieval necessity via Self-RAG critic
        retrieve_needed = self.critic.should_retrieve(query)
        
        if not retrieve_needed:
            # Retrieval not needed - return with low confidence so caller can decide
            return False, 0.7, None
        
        return True, 0.9, None

class RetrieverAgent:
    """Retrieves documents with retry logic and query reformulation."""
    
    def __init__(self, llm_model=None):
        self.llm = llm_model
    
    def _get_llm(self):
        if self.llm is None:
            self.llm = load_llm()
        return self.llm
    
    def reformulate_query(self, original_query: str, domain: str, reason: str = "low_relevance") -> str:
        """Reformulate query for better retrieval."""
        if not RETRIEVAL_RETRY_ENABLE:
            return original_query
        
        prompt = f"""Reformulate this query to improve document retrieval for Alkhidmat Foundation knowledge base.

Original Query: {original_query}
Domain: {domain}
Reason for reformulation: {reason}

Create a reformulated query that:
1. Includes key terms related to Alkhidmat Foundation
2. Adds domain-specific keywords (donation, healthcare, etc.)
3. Maintains the original intent
4. Is optimized for vector search

Respond with ONLY the reformulated query, nothing else.

Reformulated Query:"""
        
        try:
            model = self._get_llm()
            output = model(prompt, max_tokens=50, temperature=0.3, stop=["\n"], echo=False)
            reformulated = output['choices'][0]['text'].strip()
            print(f"[RETRIEVER-AGENT] Reformulated: '{original_query}' → '{reformulated}'")
            return reformulated if reformulated else original_query
        except Exception as e:
            print(f"[RETRIEVER-AGENT] Reformulation error: {e}")
            return original_query
    
    def retrieve_with_retry(self, query: str, domain: str, top_k: int = 5, 
                           filter_category: Optional[str] = None) -> Tuple[List[Dict], np.ndarray, List[np.ndarray], bool]:
        """Retrieve documents with automatic retry on low relevance."""
        MIN_RELEVANT_DOCS = 2
        MIN_AVG_RELEVANCE = 0.6
        
        results, query_embedding, doc_embeddings = retrieve_from_supabase(
            query, top_k=top_k, filter_category=filter_category
        )
        was_retried = False
        
        if results:
            avg_relevance = sum(r.get('similarity', 0) for r in results) / len(results)
            high_relevance_count = sum(1 for r in results if r.get('similarity', 0) >= MIN_AVG_RELEVANCE)
            
            if high_relevance_count >= MIN_RELEVANT_DOCS and avg_relevance >= MIN_AVG_RELEVANCE:
                if not RETRIEVAL_RETRY_ENABLE:
                    return results, query_embedding, doc_embeddings, was_retried
                
                if avg_relevance >= RETRIEVAL_RETRY_RELEVANCE_THRESHOLD:
                    return results, query_embedding, doc_embeddings, was_retried
                
                print(f"[RETRIEVER-AGENT] ⚠️ Low relevance ({avg_relevance:.3f}), attempting reformulation...")
                
                for attempt in range(RETRIEVAL_RETRY_MAX_ATTEMPTS):
                    reformulated = self.reformulate_query(query, domain, reason="low_relevance")
                    
                    cached_emb = find_similar_cached_embedding(reformulated)
                    if cached_emb is not None:
                        query_embedding = cached_emb
                    else:
                        embedder = get_embedder()
                        query_prefixed = f"query: {reformulated}"
                        query_embedding = embedder.encode([query_prefixed], normalize_embeddings=True)[0]
                        cache_query_embedding(reformulated, query_embedding)
                    
                    retry_results, _, retry_doc_embeddings = retrieve_from_supabase(
                        reformulated, top_k=top_k, filter_category=filter_category, query_embedding=query_embedding
                    )
                    
                    if retry_results:
                        retry_avg_relevance = sum(r.get('similarity', 0) for r in retry_results) / len(retry_results)
                        if retry_avg_relevance > avg_relevance:
                            print(f"[RETRIEVER-AGENT] ✅ Reformulation improved relevance: {avg_relevance:.3f} → {retry_avg_relevance:.3f}")
                            was_retried = True
                            return retry_results, query_embedding, retry_doc_embeddings, was_retried
                        else:
                            print(f"[RETRIEVER-AGENT] ⚠️ Reformulation did not improve relevance")
                
                return results, query_embedding, doc_embeddings, was_retried
            
            is_domain_specific = (filter_category and filter_category != "general") or (not filter_category and domain != "general")
            
            if is_domain_specific:
                print(f"[RETRIEVER-AGENT] ⚠️ Only found {high_relevance_count} relevant doc(s) in {domain} domain (avg relevance: {avg_relevance:.3f})")
                print(f"[RETRIEVER-AGENT] 🔍 Falling back to general domain...")
                
                general_results, _, general_doc_embeddings = retrieve_from_supabase(
                    query, top_k=top_k, filter_category="general", query_embedding=query_embedding
                )
                
                if general_results:
                    seen = set()
                    combined_results = []
                    
                    for r in results:
                        key = (r.get('file_path', ''), r.get('chunk_index', 0))
                        if key not in seen:
                            seen.add(key)
                            combined_results.append(r)
                    
                    for r in general_results:
                        key = (r.get('file_path', ''), r.get('chunk_index', 0))
                        if key not in seen:
                            seen.add(key)
                            combined_results.append(r)
                    
                    combined_results = sorted(combined_results, key=lambda x: x.get('similarity', 0), reverse=True)[:top_k]
                    
                    combined_embeddings = doc_embeddings.copy()
                    combined_embeddings.extend(general_doc_embeddings)
                    
                    print(f"[RETRIEVER-AGENT] ✅ Combined {len(results)} {domain} docs + {len(general_results)} general docs → {len(combined_results)} total")
                    was_retried = True
                    return combined_results, query_embedding, combined_embeddings, was_retried
                else:
                    print(f"[RETRIEVER-AGENT] ⚠️ No results found in general domain either")
        
        if not results:
            is_domain_specific = (filter_category and filter_category != "general") or (not filter_category and domain != "general")
            if is_domain_specific:
                print(f"[RETRIEVER-AGENT] ⚠️ No results found in {domain} domain")
                print(f"[RETRIEVER-AGENT] 🔍 Falling back to general domain...")
                
                general_results, _, general_doc_embeddings = retrieve_from_supabase(
                    query, top_k=top_k, filter_category="general", query_embedding=query_embedding
                )
                
                if general_results:
                    print(f"[RETRIEVER-AGENT] ✅ Found {len(general_results)} documents in general domain")
                    was_retried = True
                    return general_results, query_embedding, general_doc_embeddings, was_retried
        
        if not RETRIEVAL_RETRY_ENABLE or not results:
            return results, query_embedding, doc_embeddings, was_retried
        
        return results, query_embedding, doc_embeddings, was_retried

class EvidenceCoverageAgent:
    """Verifies that all claims in the answer are supported by evidence."""
    
    def __init__(self, critic: SelfRAGCritic):
        self.critic = critic
    
    def _split_compound_claims(self, sentence: str) -> List[str]:
        """Split compound sentences into atomic claims."""
        import re
        
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            return []
        
        subject_match = re.match(r'^([A-Z][^,\.!?]+?)(?:\s+(?:is|are|was|were|provides|provide|works|work|supports|support|offers|offer|operates|operate))', sentence)
        subject = subject_match.group(1).strip() if subject_match else None
        
        if re.search(r',\s+(?:and|or)\s+', sentence, re.IGNORECASE):
            parts = re.split(r',\s+(?:and|or)\s+', sentence, flags=re.IGNORECASE)
            if len(parts) > 1:
                claims = []
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    part = re.sub(r'^,\s*', '', part)
                    if i > 0 and subject and not part[0].isupper():
                        if not re.search(r'\b(?:provides|provide|works|work|supports|support|offers|offer|operates|operate|is|are)\b', part, re.IGNORECASE):
                            part = f"{subject} {part}"
                    if len(part.strip()) > 10:
                        claims.append(part.strip())
                if claims:
                    return claims
        
        if re.search(r'\s+and\s+', sentence, re.IGNORECASE) and not re.search(r',\s+and\s+', sentence, re.IGNORECASE):
            parts = re.split(r'\s+and\s+', sentence, flags=re.IGNORECASE)
            if len(parts) > 1:
                claims = []
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    if i > 0 and subject and not part[0].isupper():
                        if not re.search(r'\b(?:provides|provide|works|work|supports|support|offers|offer|operates|operate|is|are)\b', part, re.IGNORECASE):
                            part = f"{subject} {part}"
                    if len(part.strip()) > 10:
                        claims.append(part.strip())
                if claims:
                    return claims
        
        comma_parts = [p.strip() for p in sentence.split(',')]
        if len(comma_parts) > 2:
            avg_len = sum(len(p) for p in comma_parts) / len(comma_parts)
            if avg_len < 50:
                claims = []
                if comma_parts[0]:
                    claims.append(comma_parts[0])
                for part in comma_parts[1:]:
                    part = part.strip()
                    if not part:
                        continue
                    part = re.sub(r'\s+(?:and|or)\s*$', '', part, flags=re.IGNORECASE)
                    if subject and not part[0].isupper():
                        if not re.search(r'\b(?:provides|provide|works|work|supports|support|offers|offer|operates|operate|is|are)\b', part, re.IGNORECASE):
                            part = f"{subject} {part}"
                    if len(part.strip()) > 10:
                        claims.append(part.strip())
                if len(claims) > 1:
                    return claims
        
        relative_match = re.search(r'(.+?)\s+(?:that|which|who)\s+(.+)', sentence)
        if relative_match:
            main_clause = relative_match.group(1).strip()
            relative_clause = relative_match.group(2).strip()
            claims = []
            if main_clause and len(main_clause) > 10:
                claims.append(main_clause)
            if relative_clause and len(relative_clause) > 10:
                if subject and not relative_clause[0].isupper():
                    relative_clause = f"{subject} {relative_clause}"
                claims.append(relative_clause)
            if claims:
                return claims
        
        return [sentence] if len(sentence.strip()) > 10 else []
    
    def check_coverage(self, answer: str, context: str, query: str, domain: str = "general", 
                      results: Optional[List[Dict]] = None) -> Tuple[bool, List[str], float]:
        """Check if all claims in answer are supported by context."""
        if not EVIDENCE_COVERAGE_ENABLE:
            return True, [], 1.0
        
        allow_abstractive_summary = self._is_summary_allowed(query, domain)
        
        if allow_abstractive_summary:
            print(f"[EVIDENCE-COVERAGE] Summary-allowed mode: Relaxing strictness for {domain} domain")
        
        import re
        sentences = re.split(r'[.!?]\s+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return True, [], 1.0
        
        all_claims = []
        for sentence in sentences:
            atomic_claims = self._split_compound_claims(sentence)
            all_claims.extend(atomic_claims)
        
        if not all_claims:
            return True, [], 1.0
        
        print(f"[EVIDENCE-COVERAGE] Split into {len(all_claims)} atomic claims from {len(sentences)} sentences")
        
        unsupported = []
        supported_count = 0
        
        document_contexts = None
        if results and len(results) > 1:
            document_contexts = [r.get('text', '') for r in results]
            print(f"[EVIDENCE-COVERAGE] Using multi-document support ({len(document_contexts)} documents)")
        
        for claim in all_claims:
            is_supported, confidence = self._check_claim_support(
                claim, context, query, 
                allow_multi_document=True,
                document_contexts=document_contexts,
                allow_abstractive=allow_abstractive_summary
            )
            
            threshold = 0.3 if allow_abstractive_summary else 0.4
            
            if not is_supported or confidence < threshold:
                unsupported.append(claim)
            else:
                supported_count += 1
        
        coverage_score = supported_count / len(all_claims) if all_claims else 1.0
        
        if allow_abstractive_summary:
            all_covered = coverage_score >= 0.5
        else:
            all_covered = coverage_score >= 0.4
        
        if not all_covered:
            print(f"[EVIDENCE-COVERAGE] ⚠️ Found {len(unsupported)} unsupported claims out of {len(all_claims)} (coverage: {coverage_score:.2f})")
            for claim in unsupported[:3]:
                print(f"   - Unsupported: '{claim[:80]}...'")
        else:
            print(f"[EVIDENCE-COVERAGE] ✅ All claims supported (coverage: {coverage_score:.2f})")
        
        return all_covered, unsupported, coverage_score
    
    def _is_summary_allowed(self, query: str, domain: str) -> bool:
        """Determine if abstractive summary is allowed for this query."""
        query_lower = query.lower()
        summary_domains = ["general", "about", "introduction"]
        summary_patterns = [
            "what is", "who is", "tell me about", "describe",
            "about", "overview", "introduction", "summary",
            "kya hai", "kya hain", "kya hota hai"
        ]
        is_summary_query = any(pattern in query_lower for pattern in summary_patterns)
        is_summary_domain = domain.lower() in summary_domains
        return is_summary_query or is_summary_domain
    
    def _check_claim_support(self, claim: str, context: str, query: str,
                            allow_multi_document: bool = True,
                            document_contexts: Optional[List[str]] = None,
                            allow_abstractive: bool = False) -> Tuple[bool, float]:
        """Check if a single claim is supported by context."""
        if document_contexts and len(document_contexts) > 1:
            context_parts = []
            for i, doc_context in enumerate(document_contexts[:5]):
                context_parts.append(f"Document {i+1}:\n{doc_context[:500]}")
            full_context = "\n\n".join(context_parts)
        else:
            full_context = context[:2000]
        
        if allow_abstractive:
            support_instruction = """The claim may be supported by information spread across multiple documents.
            A claim is SUPPORTED if the information in the context (across one or more documents) substantiates it, 
            OR if the context contains related information that could reasonably support the claim.
            This is a summary/overview query, so abstractive synthesis is acceptable.
            Be lenient - if the context is related to the claim topic, consider it SUPPORTED."""
        else:
            support_instruction = """The claim should be supported by the context.
            A claim is SUPPORTED if:
            - The information in the context (across one or more documents) substantiates it, OR
            - The context contains related information that could reasonably support the claim.
            Be lenient - if the context is related to the claim topic, consider it SUPPORTED."""
        
        prompt = f"""Check if this claim is supported by the provided context.

Claim: {claim}

Context (may contain multiple documents):
{full_context}

{support_instruction}

Answer with ONLY:
[SUPPORTED] - if the claim is clearly supported by the context (may span multiple documents)
[NOT_SUPPORTED] - if the claim is not supported or contradicts the context

Answer:"""
        
        try:
            model = self.critic._get_llm()
            output = model(prompt, max_tokens=15, temperature=0.1, stop=["\n"], echo=False)
            response = output['choices'][0]['text'].strip().upper()
            
            if "[SUPPORTED]" in response:
                return True, 0.8
            elif "[NOT_SUPPORTED]" in response:
                claim_keywords = set(claim.lower().split())
                context_keywords = set(full_context.lower().split())
                overlap = len(claim_keywords & context_keywords) / len(claim_keywords) if claim_keywords else 0
                if overlap > 0.3:
                    return True, 0.4
                return False, 0.2
            return True, 0.6
        except Exception as e:
            print(f"[EVIDENCE-COVERAGE] Error checking claim: {e}")
            return True, 0.5

class ConversationMemoryAgent:
    """Manages conversation state for follow-up queries without re-embedding."""
    
    def get_state(self, session_id: str) -> Optional[Dict]:
        if not CONVERSATION_MEMORY_ENABLE:
            return None
        return _conversation_state.get(session_id)
    
    def update_state(self, session_id: str, domain: str, doc_ids: List[str], 
                    confidence: float, query: str):
        if not CONVERSATION_MEMORY_ENABLE:
            return
        
        _conversation_state[session_id] = {
            'last_domain': domain,
            'last_doc_ids': doc_ids,
            'last_answer_confidence': confidence,
            'last_query': query,
            'timestamp': time.time()
        }
        
        current_time = time.time()
        expired_sessions = [
            sid for sid, state in _conversation_state.items()
            if current_time - state.get('timestamp', 0) > 3600
        ]
        for sid in expired_sessions:
            del _conversation_state[sid]
    
    def is_followup(self, query: str, session_id: str) -> Tuple[bool, Optional[Dict]]:
        if not CONVERSATION_MEMORY_ENABLE:
            return False, None
        
        state = self.get_state(session_id)
        if not state:
            return False, None
        
        last_confidence = state.get('last_answer_confidence', 0)
        last_timestamp = state.get('timestamp', 0)
        time_since_last = time.time() - last_timestamp
        
        is_recent = time_since_last < 300
        
        if last_confidence < 0.5 and not is_recent:
            return False, None
        
        query_lower = query.lower().strip()
        
        followup_indicators = [
            "what about", "how about", "and", "also", "tell me more",
            "more", "details", "elaborate", "explain", "kya", "aur",
            "can i", "how can i", "where can", "when can", "why",
            "receipt", "payment", "donation", "get", "my", "me"
        ]
        
        has_indicator = any(indicator in query_lower for indicator in followup_indicators)
        is_short = len(query.split()) <= 5
        
        last_query = state.get('last_query', '').lower()
        last_domain = state.get('last_domain', '').lower()
        
        domain_keywords = {
            'donation': ['donation', 'donate', 'payment', 'receipt', 'zakat', 'sadaqah', 'money', 'pay'],
            'healthcare': ['hospital', 'clinic', 'health', 'treatment', 'doctor', 'medical', 'service'],
            'general': ['alkhidmat', 'foundation', 'about', 'information', 'service', 'work']
        }
        
        relevant_keywords = domain_keywords.get(last_domain, [])
        has_domain_keywords = any(keyword in query_lower for keyword in relevant_keywords)
        
        last_query_words = set(last_query.split())
        current_query_words = set(query_lower.split())
        word_overlap = len(last_query_words & current_query_words) / max(len(last_query_words), 1)
        has_overlap = word_overlap > 0.2
        
        is_followup = (
            has_indicator or 
            (is_short and is_recent) or 
            (has_domain_keywords and is_recent) or
            (has_overlap and is_recent) or
            (is_short and last_confidence >= 0.6)
        )
        
        return is_followup, state if is_followup else None

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
        if DomainClassifier._domain_embeddings_cache is not None:
            return
       
        if os.environ.get('BATCH_MODE') != 'True':
            print("\n🔄 Initializing domain embeddings from anchor queries...")
       
        import sys
        current_module = sys.modules[__name__]
        model = current_module.get_embedder()
        DomainClassifier._embedding_model = model
       
        domain_embeddings = {}
       
        for domain, queries in DOMAIN_ANCHOR_QUERIES.items():
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
        if DomainClassifier._domain_embeddings_cache is None:
            DomainClassifier.initialize_domain_embeddings()
       
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
        if not log_probs:
            return float('inf')
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        alpha = 0.1
        return float(np.exp(-alpha * (perplexity - 1)))
   
    @staticmethod
    def calculate_average_token_confidence(log_probs: List[float]) -> float:
        if not log_probs:
            return 0.0
        probs = [np.exp(lp) for lp in log_probs]
        return np.mean(probs)
   
    @staticmethod
    def calculate_entropy_confidence(token_probs_distributions: List[np.ndarray]) -> float:
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
        """Calculate all confidence metrics and return a comprehensive score.
        
        Note: selfrag_scores parameter is kept for API compatibility but
        selfrag_support and selfrag_utility are no longer used as they were
        hardcoded constants, not genuinely calculated values.
        """
        scores = {}
        scores['retrieval_confidence'] = retrieval_confidence
       
        if log_probs:
            scores['avg_token_confidence'] = ConfidenceScorer.calculate_average_token_confidence(log_probs)
            scores['perplexity'] = ConfidenceScorer.calculate_perplexity(log_probs)
            scores['weighted_top_k'] = ConfidenceScorer.calculate_top_k_weighted_confidence(log_probs)
       
        if token_probs_distributions:
            scores['entropy_confidence'] = ConfidenceScorer.calculate_entropy_confidence(token_probs_distributions)

        # Note: selfrag_support/utility removed - they were hardcoded constants (not real scores)
        # Only keep selfrag_relevance if provided, as it comes from actual document assessments
        if selfrag_scores and selfrag_scores.get('relevance_score', 0) > 0:
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
       
        if weight_sum > 0:
            scores['combined_confidence'] = combined / weight_sum
        else:
            scores['combined_confidence'] = 0.0
       
        return scores

# ============ Brand Term Protection ============
def protect_brand_terms(text: str) -> str:
    for term in BRAND_TERMS:
        text = re.sub(rf"\b{re.escape(term)}\b", f"@@{term}@@", text, flags=re.IGNORECASE)
    return text

def restore_brand_terms(text: str) -> str:
    return text.replace("@@", "")

# ============ Enhanced Language Detection ============
def is_urdu_script(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))

def looks_like_roman_urdu(text: str) -> bool:
    if is_urdu_script(text):
        return False
    if re.search(r'[^\x00-\x7F]', text):
        return False
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in ROMAN_URDU_MARKERS)
    if hits >= 2:
        return True
    if hits >= 1 and len(tokens) <= 6:
        return True
    return False

def translate_auto_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

class QueryLangProfile:
    def __init__(self, original_query: str, input_lang: str, query_en: str,
                 output_lang: str, query_urdu_script: Optional[str] = None):
        self.original_query = original_query
        self.input_lang = input_lang
        self.query_en = query_en
        self.output_lang = output_lang
        self.query_urdu_script = query_urdu_script

def build_query_lang_profile(query: str) -> QueryLangProfile:
    q = query.strip()

    if is_urdu_script(q) or detect_language(q) == "ur":
        q_en = translate_urdu_to_english(q)
        return QueryLangProfile(original_query=q, input_lang="ur", query_en=q_en, output_lang="ur", query_urdu_script=q)

    if looks_like_roman_urdu(q):
        q_en = translate_auto_to_english(q)
        return QueryLangProfile(
            original_query=q,
            input_lang="roman_ur",
            query_en=q_en,
            output_lang="roman_ur",
            query_urdu_script=None
        )

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
    q_lower = query.lower().strip()

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

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import PyPDF2
        from io import BytesIO
        
        pdf_file = BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
            except Exception as e:
                print(f"   ⚠️ Error extracting text from page {page_num + 1}: {e}")
                continue
        
        return "\n\n".join(text_parts)
    except ImportError:
        print("   ⚠️ PyPDF2 not installed. Install with: pip install PyPDF2")
        return ""
    except Exception as e:
        print(f"   ⚠️ Error extracting PDF text: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        from io import BytesIO
        
        docx_file = BytesIO(file_bytes)
        doc = Document(docx_file)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_parts.append(" | ".join(row_text))
        
        return "\n\n".join(text_parts)
    except ImportError:
        print("   ⚠️ python-docx not installed. Install with: pip install python-docx")
        return ""
    except Exception as e:
        print(f"   ⚠️ Error extracting DOCX text: {e}")
        return ""

def load_documents_from_zip(zip_path: str, file_paths_filter: Optional[set] = None) -> Dict[str, List[Dict]]:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    documents_by_category = {}
    skipped_files = []
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_path in zip_ref.namelist():
            if file_path.endswith("/"):
                continue
            if "__MACOSX" in file_path or file_path.startswith("._") or Path(file_path).name.startswith("._"):
                skipped_files.append(file_path)
                continue
            if file_paths_filter is not None and file_path not in file_paths_filter:
                continue
            filename = Path(file_path).name.lower()
            if filename in [".ds_store", "thumbs.db", ".gitignore", ".gitkeep"]:
                skipped_files.append(file_path)
                continue
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in [".txt", ".pdf", ".docx"]:
                skipped_files.append(file_path)
                continue
            parts = Path(file_path).parts
            if len(parts) < 3:
                skipped_files.append(file_path)
                continue
            category = parts[-2]
            filename = parts[-1]
            try:
                with zip_ref.open(file_path) as f:
                    file_bytes = f.read()
                if file_ext == ".txt":
                    try:
                        content = file_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            content = file_bytes.decode("latin-1")
                        except Exception as e:
                            print(f"   ⚠️ Error decoding {file_path}: {e}")
                            skipped_files.append(file_path)
                            continue
                elif file_ext == ".pdf":
                    print(f"   📄 Extracting text from PDF: {filename}")
                    content = extract_text_from_pdf(file_bytes)
                    if not content.strip():
                        print(f"   ⚠️ No text extracted from PDF: {filename}")
                        skipped_files.append(file_path)
                        continue
                elif file_ext == ".docx":
                    print(f"   📄 Extracting text from DOCX: {filename}")
                    content = extract_text_from_docx(file_bytes)
                    if not content.strip():
                        print(f"   ⚠️ No text extracted from DOCX: {filename}")
                        skipped_files.append(file_path)
                        continue
                else:
                    skipped_files.append(file_path)
                    continue
                if content.strip():
                    documents_by_category.setdefault(category, []).append({
                        "content": content,
                        "filename": filename,
                        "category": category,
                        "file_path": file_path
                    })
                else:
                    skipped_files.append(file_path)
            except Exception as e:
                print(f"   ⚠️ Error processing {file_path}: {e}")
                skipped_files.append(file_path)
    
    if skipped_files:
        print(f"\n⚠️ Skipped {len(skipped_files)} files:")
        metadata_files = [f for f in skipped_files if "__MACOSX" in f or Path(f).name.startswith("._")]
        unsupported = [f for f in skipped_files if f not in metadata_files and Path(f).suffix.lower() not in [".txt", ".pdf", ".docx"]]
        empty_or_error = [f for f in skipped_files if f not in metadata_files and Path(f).suffix.lower() in [".txt", ".pdf", ".docx"]]
        if metadata_files:
            print(f"   - {len(metadata_files)} system/metadata file(s) (macOS resource forks, etc.)")
        if unsupported:
            print(f"   - {len(unsupported)} unsupported file type(s)")
        if empty_or_error:
            print(f"   - {len(empty_or_error)} file(s) with extraction errors or empty content")
    
    total_loaded = sum(len(docs) for docs in documents_by_category.values())
    print(f"\n✅ Successfully loaded {total_loaded} documents from ZIP")
    return documents_by_category

def clean_text(text: str) -> str:
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "؟ ", "۔ ", " ", ""],
    )
    chunks, metas = [], []
    for doc, meta in zip(documents, metadata):
        parts = splitter.split_text(doc)
        for idx, p in enumerate(parts):
            chunks.append(p)
            chunk_meta = meta.copy()
            chunk_meta["chunk_index"] = idx
            metas.append(chunk_meta)

    avg_len = int(np.mean([len(c) for c in chunks]))
    print(f"Split into {len(chunks)} chunks (avg {avg_len} chars, overlap {CHUNK_OVERLAP})")
    return chunks, metas

# ============ Supabase Storage ============
def save_chunks_to_supabase(chunks: List[str], metadata: List[Dict], embeddings: np.ndarray):
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
    supabase = get_supabase_client()
    try:
        result = supabase.table("documents").delete().neq("doc_id", "00000000-0000-0000-0000-000000000000").execute()
        print("✅ Cleared all documents from Supabase")
    except Exception as e:
        print(f"⚠️ Error clearing documents: {e}")

def document_exists(file_path: str) -> bool:
    supabase = get_supabase_client()
    try:
        result = supabase.table("documents").select("doc_id").eq("file_path", file_path).limit(1).execute()
        return len(result.data) > 0 if result.data else False
    except Exception as e:
        print(f"⚠️ Error checking document existence: {e}")
        return False

def delete_document_by_path(file_path: str) -> bool:
    supabase = get_supabase_client()
    try:
        result = supabase.table("documents").delete().eq("file_path", file_path).execute()
        deleted_count = len(result.data) if result.data else 0
        print(f"✅ Deleted {deleted_count} chunks for document: {file_path}")
        return True
    except Exception as e:
        print(f"⚠️ Error deleting document: {e}")
        return False

def add_document_incremental(file_path: str, content: str, category: str, filename: str, 
                             reindex: bool = False) -> bool:
    supabase = get_supabase_client()
    if document_exists(file_path):
        if reindex:
            print(f"🔄 Document exists, re-indexing: {file_path}")
            delete_document_by_path(file_path)
        else:
            print(f"⏭️ Document already exists, skipping: {file_path}")
            return True
    
    print(f"\n📄 Processing document incrementally: {filename}")
    print(f"   Category: {category}")
    print(f"   Path: {file_path}")
    
    cleaned_content = clean_text(content)
    if not cleaned_content:
        print(f"⚠️ Document has no content after cleaning, skipping")
        return False
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "؟ ", "۔ ", " ", ""]
    )
    
    chunks = splitter.split_text(cleaned_content)
    print(f"   Split into {len(chunks)} chunks")
    
    if not chunks:
        print(f"⚠️ No chunks created, skipping")
        return False
    
    print(f"   Creating embeddings for {len(chunks)} chunks...")
    embedder = get_embedder()
    prefixed = [f"passage: {chunk}" for chunk in chunks]
    embeddings = embedder.encode(prefixed, show_progress_bar=False, batch_size=32, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    
    rows = []
    for idx, chunk in enumerate(chunks):
        rows.append({
            "doc_id": str(uuid.uuid4()),
            "chunk_text": chunk,
            "chunk_index": idx,
            "category": category,
            "filename": filename,
            "file_path": file_path,
            "doc_domain": category,
            "embedding": embeddings[idx].tolist()
        })
    
    batch_size = 100
    total_inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            result = supabase.table("documents").insert(batch).execute()
            total_inserted += len(batch)
            print(f"   Inserted batch {i//batch_size + 1}: {len(batch)} chunks")
        except Exception as e:
            print(f"   ⚠️ Error inserting batch {i//batch_size + 1}: {e}")
            for row in batch:
                try:
                    supabase.table("documents").insert(row).execute()
                    total_inserted += 1
                except Exception as e2:
                    print(f"   ⚠️ Failed to insert chunk: {e2}")
    
    print(f"✅ Successfully added document: {filename} ({total_inserted}/{len(rows)} chunks)")
    return total_inserted > 0

def add_single_file_incremental(file_path: str, content: str, category: str = "general", 
                                reindex: bool = False) -> bool:
    filename = Path(file_path).name
    return add_document_incremental(file_path, content, category, filename, reindex=reindex)

def add_documents_from_zip_incremental(zip_path: str, reindex_existing: bool = False) -> Dict[str, int]:
    print("\n" + "="*80)
    print("INCREMENTAL DOCUMENT ADDITION")
    print("="*80)
    
    if not test_connection():
        print("❌ Cannot connect to Supabase. Aborting.")
        return {"added": 0, "skipped": 0, "reindexed": 0, "failed": 0}
    
    print("\n📦 Scanning ZIP file for documents...")
    import zipfile
    file_paths_to_process = []
    skipped_paths = []
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_path in zip_ref.namelist():
            if file_path.endswith("/"):
                continue
            if "__MACOSX" in file_path or Path(file_path).name.startswith("._"):
                continue
            filename = Path(file_path).name.lower()
            if filename in [".ds_store", "thumbs.db", ".gitignore", ".gitkeep"]:
                continue
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in [".txt", ".pdf", ".docx"]:
                continue
            parts = Path(file_path).parts
            if len(parts) < 3:
                continue
            exists = document_exists(file_path)
            if exists and not reindex_existing:
                skipped_paths.append(file_path)
                continue
            file_paths_to_process.append(file_path)
    
    print(f"   Found {len(file_paths_to_process)} new documents to process")
    print(f"   Skipping {len(skipped_paths)} existing documents")
    
    if not file_paths_to_process and not reindex_existing:
        print("\n✅ All documents already exist in knowledge base. Nothing to add.")
        return {"added": 0, "skipped": len(skipped_paths), "reindexed": 0, "failed": 0}
    
    print("\n📦 Loading documents from ZIP (extracting text only for new documents)...")
    docs_by_cat = load_documents_from_zip(zip_path, file_paths_filter=set(file_paths_to_process))
    
    stats = {"added": 0, "skipped": len(skipped_paths), "reindexed": 0, "failed": 0}
    
    total_docs = sum(len(docs) for docs in docs_by_cat.values())
    print(f"Found {total_docs} documents to process across {len(docs_by_cat)} categories\n")
    
    for cat, docs in docs_by_cat.items():
        print(f"📁 Category: {cat} ({len(docs)} documents)")
        for doc in docs:
            file_path = doc["file_path"]
            filename = doc["filename"]
            content = doc["content"]
            category = doc["category"]
            
            exists = document_exists(file_path)
            
            if exists and not reindex_existing:
                print(f"   ⏭️ Skipping (exists): {filename} [{file_path}]")
                stats["skipped"] += 1
                continue
            
            if exists and reindex_existing:
                print(f"   🔄 Re-indexing: {filename} [{file_path}]")
                stats["reindexed"] += 1
            
            print(f"   ➕ Processing: {filename} [{file_path}]")
            success = add_document_incremental(
                file_path=file_path,
                content=content,
                category=category,
                filename=filename,
                reindex=(exists and reindex_existing)
            )
            
            if success:
                stats["added"] += 1
                print(f"   ✅ Successfully added: {filename}")
            else:
                stats["failed"] += 1
                print(f"   ❌ Failed to add: {filename}")
    
    print("\n" + "="*80)
    print("INCREMENTAL ADDITION COMPLETE")
    print("="*80)
    print(f"✅ Added: {stats['added']}")
    print(f"⏭️ Skipped: {stats['skipped']}")
    print(f"🔄 Re-indexed: {stats['reindexed']}")
    print(f"❌ Failed: {stats['failed']}")
    print("="*80 + "\n")
    return stats

# ============ Build Pipeline ============
def build_alkhidmat_rag(zip_path: str, clear_existing: bool = False, incremental: bool = False):
    print("\n" + "="*80)
    print("BUILDING ALKHIDMAT RAG SYSTEM")
    print("="*80)
   
    if not test_connection():
        print("❌ Cannot connect to Supabase. Aborting.")
        return
   
    if incremental:
        stats = add_documents_from_zip_incremental(zip_path, reindex_existing=False)
        print("\nBUILD: Initializing domain classification...")
        DomainClassifier.initialize_domain_embeddings()
        print("\n" + "="*80)
        print("✅ INCREMENTAL BUILD COMPLETE!")
        print("="*80)
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
                           filter_category: str = None,
                           query_embedding: Optional[np.ndarray] = None) -> Tuple[List[Dict], np.ndarray, List[np.ndarray]]:
    embed_start = time.time()
    
    if query_embedding is not None:
        print(f"[RETRIEVAL] Using provided embedding", flush=True)
    else:
        cached_emb = find_similar_cached_embedding(query)
        if cached_emb is not None:
            query_embedding = cached_emb
            print(f"[RETRIEVAL] Using cached embedding", flush=True)
        else:
            embedder = get_embedder()
            query_prefixed = f"query: {query}"
            query_embedding = embedder.encode([query_prefixed], normalize_embeddings=True)[0]
            cache_query_embedding(query, query_embedding)
    
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
       
        result = supabase.rpc('match_documents', params).execute()
        rows = result.data
        print(f"[TIMING] Supabase RPC call: {time.time() - rpc_start:.2f}s", flush=True)
       
        if rows:
            fetch_start = time.time()
            doc_ids = [row['doc_id'] for row in rows]
           
            full_docs_result = supabase.table("documents").select(
                "doc_id, embedding"
            ).in_("doc_id", doc_ids).execute()
           
            doc_map = {doc['doc_id']: doc for doc in full_docs_result.data}
           
            for row in rows:
                if row['doc_id'] in doc_map:
                    row['embedding'] = doc_map[row['doc_id']]['embedding']
            print(f"[TIMING] Fetch embeddings: {time.time() - fetch_start:.2f}s", flush=True)
       
    except Exception as e:
        print(f"⚠️ RPC function not available, using fallback method: {e}")
       
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
       
        if 'embedding' in row and row['embedding'] is not None:
            try:
                if isinstance(row['embedding'], str):
                    import ast
                    embedding_data = ast.literal_eval(row['embedding'])
                    doc_embeddings.append(np.array(embedding_data, dtype=np.float32))
                elif isinstance(row['embedding'], list):
                    doc_embeddings.append(np.array(row['embedding'], dtype=np.float32))
                elif isinstance(row['embedding'], np.ndarray):
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
    sys.stdout.flush()
   
    return results, query_embedding, doc_embeddings

def sanitize_chunk_text(text: str) -> str:
    text = re.sub(r'(?mi)^\s*(user\s+question|question|q:)\s*[:\-–]?\s*.*$', '', text)
    text = re.sub(r'(?mi)^\s*(answer|a:)\s*[:\-–]?\s*.*$', '', text)
    text = re.sub(r'(?is)(?:^|\n)\s*q[:\.\-\)]\s*.*?\n\s*a[:\.\-\)]\s*.*?(?:\n|$)', '', text)
    text = re.sub(r'\[insert .*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'click here', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ============ LLM (Llama CPP) ============
_LLM_MODEL = None

    # NOTE: LLM loading and generation have been moved to `rag_llm`.
   
# ============ Answer Generation (ENHANCED) ============
def generate_answer(query: str, top_k: int = 5, max_tokens: int = 400,
                    filter_category: str = None):
    """ENHANCED: Now includes domain classification and confidence scoring"""
    start_time = time.time()
    print(f"[RAG] Processing query: {query[:50]}...", flush=True)
    sys.stdout.flush()
   
    profile = build_query_lang_profile(query)
    query_info = analyze_query(profile.original_query)
    query_for_rag = profile.query_en
    original_query = profile.original_query
   
    domain_start = time.time()
    print(f"[RAG] Classifying domain...", flush=True)
    sys.stdout.flush()
    domain_classification = DomainClassifier.classify_domain(query_for_rag)
    print(f"[TIMING] Domain classification: {time.time() - domain_start:.2f}s", flush=True)
    winning_domain = domain_classification.get("domain", "general")
   
    retrieval_query_en = expand_query_for_retrieval(query_for_rag, winning_domain, query_info)
   
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
   
    llm_start = time.time()
    print(f"[RAG] Generating answer with LLM...", flush=True)
    sys.stdout.flush()
    answer, log_probs, token_probs_distributions = llm_generate(
        prompt, max_tokens=max_tokens, stop_tokens=["\nUser question:", "\nQuestion:", "\nسوال:"]
    )
    print(f"[TIMING] LLM generation: {time.time() - llm_start:.2f}s", flush=True)
   
    print(f"[RAG] Cleaning response...", flush=True)
    sys.stdout.flush()
    answer = clean_llm_response(answer)
   
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
   
    if profile.output_lang == "ur":
        if not is_urdu_script(answer):
            answer = translate_english_to_urdu(answer, timeout=15)
    elif profile.output_lang == "roman_ur":
        protected = protect_brand_terms(answer)
        answer_ur = translate_english_to_urdu(protected, timeout=15)
        answer_ur = restore_brand_terms(answer_ur)
        answer = romanize_to_roman_urdu_with_llm(answer_ur)

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
   
    if combined_conf >= 0.7:
        print("✅ High confidence - Answer is likely reliable", flush=True)
    elif combined_conf >= 0.5:
        print("⚠️ Moderate confidence - Answer may need verification", flush=True)
    else:
        print("❌ Low confidence - Answer should be verified from sources", flush=True)
    print("="*80 + "\n", flush=True)
    sys.stdout.flush()
   
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

LATIN_ONLY_RE = re.compile(r'^[\x00-\x7F\s]+$')

def romanize_to_roman_urdu_with_llm(urdu_text: str, max_tokens: int = 260) -> str:
    if not urdu_text.strip():
        return urdu_text

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

        if out and (not is_urdu_script(out)) and LATIN_ONLY_RE.match(out):
            return out

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

    return en

def clean_llm_response(text: str) -> str:
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

# ============ SELF-RAG ANSWER GENERATION (WITH AGENTIC AI) ============
def generate_answer_selfrag(query: str, top_k: int = 5, max_tokens: int = 400, 
                            filter_category: str = None, session_id: Optional[str] = None):
    """
    Multilingual Self-RAG implementation with Agentic AI for embedding regulation and hallucination reduction.
    Returns: (answer, original_query, input_lang, sources, confidence_scores, domain_classification, selfrag_metrics)
    """
    start_time = time.time()
    print(f"\n{'='*80}", flush=True)
    print(f"AGENTIC SELF-RAG QUERY PROCESSING", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"[AGENTIC-RAG] Processing query: {query[:80]}...", flush=True)
    sys.stdout.flush()

    profile = build_query_lang_profile(query)
    query_info = analyze_query(profile.original_query)
    query_for_rag = profile.query_en
    original_query = profile.original_query

    llm_model = load_llm()
    critic = SelfRAGCritic(llm_model)
    
    # Initialize agentic agents
    router_agent = RouterAgent(critic)
    retriever_agent = RetrieverAgent(llm_model)
    evidence_agent = EvidenceCoverageAgent(critic)
    memory_agent = ConversationMemoryAgent()

    # Note: removed fake confidence fields (domain_confidence, retrieve_confidence,
    # answer_in_context_confidence, support_score, utility_score) - these were
    # hardcoded constants, not genuinely calculated values.
    selfrag_metrics = {
        'domain_relevant': True,
        'retrieve_needed': False,
        'answer_in_context': False,
        'relevance_score': 0.0,
        'support_level': 'uncertain',
        'utility_rating': 0,
        'embedding_cached': False,
        'retrieval_retried': False,
        'evidence_coverage': 1.0,
        'followup_detected': False
    }

    # AGENTIC STEP 0: Check conversation memory for follow-ups (BEFORE domain relevance check)
    is_followup = False
    memory_state = None
    if session_id:
        is_followup, memory_state = memory_agent.is_followup(query, session_id)
        if is_followup and memory_state:
            print(f"[AGENTIC-RAG] 🔄 Follow-up detected! Previous context: {memory_state.get('last_domain', 'unknown')} domain", flush=True)
            selfrag_metrics['followup_detected'] = True
            if memory_state.get('last_domain'):
                domain_to_category = {
                    'donation': 'Donors',
                    'healthcare': 'Health',
                    'general': 'General'
                }
                prev_category = domain_to_category.get(memory_state.get('last_domain'), None)
                if prev_category and not filter_category:
                    filter_category = prev_category
                    print(f"[AGENTIC-RAG] Using previous domain context: {filter_category}", flush=True)

    # AGENTIC STEP 0.5: Query Router Agent (before embeddings)
    # This is the SINGLE retrieval check - handles greetings, niceties, and retrieval necessity.
    # No duplicate check in Self-RAG Step 1 (removed).
    print(f"\n[AGENTIC-RAG] Router Agent: Checking if retrieval needed...", flush=True)
    should_retrieve, router_confidence, cached_answer = router_agent.route(query_for_rag)
    selfrag_metrics['retrieve_needed'] = should_retrieve
    
    if not should_retrieve and cached_answer:
        print(f"[AGENTIC-RAG] ✅ Router: No retrieval needed, returning cached answer", flush=True)
        return (cached_answer, original_query, profile.input_lang, [], 
                {'combined_confidence': router_confidence}, {}, selfrag_metrics)
    
    if not should_retrieve:
        print(f"[AGENTIC-RAG] ⚠️ Router: Retrieval not needed but no cached answer - continuing anyway", flush=True)

    # STEP 0: Check domain relevance (SKIP or RELAX for follow-ups)
    if SELFRAG_ENABLE:
        if is_followup:
            print(f"\n[SELF-RAG] Step 0: Skipping strict domain relevance check for follow-up query", flush=True)
            if memory_state and memory_state.get('last_domain'):
                prev_domain = memory_state.get('last_domain')
                print(f" ✓ Using previous domain context: {prev_domain}", flush=True)
                selfrag_metrics['domain_relevant'] = True
            else:
                print(f"\n[SELF-RAG] Step 0: Checking domain relevance (lenient for follow-up)...", flush=True)
                is_domain_relevant = critic.is_domain_relevant(query_for_rag)
                selfrag_metrics['domain_relevant'] = is_domain_relevant
                if not is_domain_relevant:
                    print(f" ✗ Question is IRRELEVANT to Alkhidmat Foundation", flush=True)
                    dummy_domain_classification = {
                        'domain': 'irrelevant',
                        'confidence': 0.0,
                        'all_scores': {'donation': 0.0, 'healthcare': 0.0, 'general': 0.0}
                    }
                    irrelevant_response = "That is an irrelevant question."
                    return (irrelevant_response, original_query, profile.input_lang, [], {'combined_confidence': 0.0},
                            dummy_domain_classification, selfrag_metrics)
                else:
                    print(f" ✓ Question is RELEVANT to domain", flush=True)
        else:
            print(f"\n[SELF-RAG] Step 0: Checking domain relevance...", flush=True)
            is_domain_relevant = critic.is_domain_relevant(query_for_rag)
            selfrag_metrics['domain_relevant'] = is_domain_relevant

            if not is_domain_relevant:
                print(f" ✗ Question is IRRELEVANT to Alkhidmat Foundation", flush=True)
                dummy_domain_classification = {
                    'domain': 'irrelevant',
                    'confidence': 0.0,
                    'all_scores': {'donation': 0.0, 'healthcare': 0.0, 'general': 0.0}
                }
                irrelevant_response = "That is an irrelevant question."
                return (irrelevant_response, original_query, profile.input_lang, [], {'combined_confidence': 0.0},
                        dummy_domain_classification, selfrag_metrics)
            else:
                print(f" ✓ Question is RELEVANT to domain", flush=True)

    # Retrieval always proceeds from here - RouterAgent already handled the check above
    selfrag_metrics['retrieve_needed'] = True

    # Classify domain (use previous domain for follow-ups if available)
    domain_start = time.time()
    if is_followup and memory_state and memory_state.get('last_domain'):
        prev_domain = memory_state.get('last_domain')
        print(f"\n[SELF-RAG] Using previous domain for follow-up: {prev_domain}", flush=True)
        domain_classification = {
            'domain': prev_domain,
            'confidence': 0.85,
            'all_scores': {prev_domain: 0.85}
        }
        winning_domain = prev_domain
    else:
        print(f"\n[SELF-RAG] Classifying domain...", flush=True)
        sys.stdout.flush()
        domain_classification = DomainClassifier.classify_domain(query_for_rag)
        print(f"[TIMING] Domain classification: {time.time() - domain_start:.2f}s", flush=True)
        winning_domain = domain_classification.get("domain", "general")

    # Expand query for retrieval
    retrieval_query_en = expand_query_for_retrieval(query_for_rag, winning_domain, query_info)

    # AGENTIC STEP 2: Check embedding cache before generating
    print(f"\n[AGENTIC-RAG] Checking embedding cache...", flush=True)
    cached_embedding = find_similar_cached_embedding(retrieval_query_en)
    if cached_embedding is not None:
        query_embedding = cached_embedding
        selfrag_metrics['embedding_cached'] = True
        print(f"[AGENTIC-RAG] ✅ Using cached embedding!", flush=True)
    else:
        embed_start = time.time()
        embedder = get_embedder()
        query_prefixed = f"query: {retrieval_query_en}"
        query_embedding = embedder.encode([query_prefixed], normalize_embeddings=True)[0]
        cache_query_embedding(retrieval_query_en, query_embedding)
        print(f"[TIMING] Query embedding: {time.time() - embed_start:.2f}s", flush=True)

    # STEP 2: Retrieve documents (with agentic retry)
    retrieval_start = time.time()
    print(f"\n[SELF-RAG] Step 2: Retrieving documents (with agentic retry)...", flush=True)
    sys.stdout.flush()
    results, query_embedding, doc_embeddings, was_retried = retriever_agent.retrieve_with_retry(
        retrieval_query_en, winning_domain, top_k=top_k, filter_category=filter_category
    )
    selfrag_metrics['retrieval_retried'] = was_retried
    if was_retried:
        print(f"[AGENTIC-RAG] ✅ Retrieval retry improved results", flush=True)
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
            is_relevant = critic.assess_relevance(query_for_rag, result['text'])
            # Use 0.8 as proxy score for relevant (1.0 would be harsh, 0 for irrelevant)
            relevance_score = 0.8 if is_relevant else 0.0
            relevance_scores.append(relevance_score)

            if is_relevant:
                relevant_results.append(result)
                print(f" ✓ Doc {i+1}: RELEVANT", flush=True)
            else:
                print(f" ✗ Doc {i+1}: Not relevant", flush=True)

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
        can_answer = critic.check_answer_in_context(query_for_rag, context)
        selfrag_metrics['answer_in_context'] = can_answer

        if not can_answer:
            print(f" ✗ Answer CANNOT be found in context", flush=True)
            no_info_response = "I don't have that information."
            return (no_info_response, original_query, profile.input_lang, results,
                    {'combined_confidence': 0.0}, domain_classification, selfrag_metrics)
        else:
            print(f" ✓ Answer CAN be found in context", flush=True)

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

    # AGENTIC STEP 4.5: Evidence Coverage Agent (claim-by-claim verification)
    if EVIDENCE_COVERAGE_ENABLE:
        print(f"\n[AGENTIC-RAG] Evidence Coverage Agent: Checking claim-by-claim support...", flush=True)
        all_covered, unsupported_claims, coverage_score = evidence_agent.check_coverage(
            answer, context, query_for_rag, 
            domain=winning_domain,
            results=results
        )
        selfrag_metrics['evidence_coverage'] = coverage_score
        
        allow_summary = evidence_agent._is_summary_allowed(query_for_rag, winning_domain)
        coverage_threshold = 0.3 if allow_summary else 0.4
        
        if not all_covered and coverage_score < coverage_threshold:
            print(f"[AGENTIC-RAG] ⚠️ Low evidence coverage ({coverage_score:.2f}), removing unsupported claims...", flush=True)
            for claim in unsupported_claims:
                import re
                sentences = re.split(r'[.!?]\s+', answer)
                filtered_sentences = []
                for sentence in sentences:
                    contains_unsupported = any(claim.lower() in sentence.lower() for claim in unsupported_claims)
                    if not contains_unsupported:
                        filtered_sentences.append(sentence)
                
                if filtered_sentences:
                    answer = '. '.join(filtered_sentences).strip()
                    if answer and not answer.endswith(('.', '!', '?')):
                        answer += '.'
                else:
                    answer = ""
            
            answer = re.sub(r'\s+', ' ', answer).strip()
            if not answer:
                answer = "I cannot provide a reliable answer as some claims cannot be verified from the available information."

    # STEP 5: Verify support
    if SELFRAG_ENABLE:
        print(f"\n[SELF-RAG] Step 5: Verifying answer support...", flush=True)
        support_level = critic.verify_support(query_for_rag, answer, context)
        selfrag_metrics['support_level'] = support_level
        print(f" → Support level: {support_level.upper()}", flush=True)

        if support_level == "no_support":
            print(f" ⚠️ Answer NOT supported by context - rejecting!", flush=True)
            no_answer = "I cannot provide a reliable answer based on the available information."
            return no_answer, original_query, profile.input_lang, results, {}, domain_classification, selfrag_metrics

    # STEP 6: Utility
    if SELFRAG_ENABLE:
        print(f"\n[SELF-RAG] Step 6: Evaluating answer utility...", flush=True)
        utility_rating = critic.evaluate_utility(query_for_rag, answer)
        selfrag_metrics['utility_rating'] = utility_rating
        print(f" → Utility rating: {utility_rating}/5", flush=True)

        if utility_rating <= 2:
            print(f" ⚠️ Answer utility too low - rejecting!", flush=True)
            no_answer = "I cannot provide a sufficiently useful answer to your question based on the available information."
            return no_answer, original_query, profile.input_lang, results, {}, domain_classification, selfrag_metrics

    # Confidence scores (based on real signals only)
    print(f"\n[SELF-RAG] Calculating final confidence scores...", flush=True)
    sys.stdout.flush()
    retrieval_conf = ConfidenceScorer.calculate_retrieval_confidence(query_embedding, doc_embeddings, top_k=top_k)
   
    # Only pass relevance_score - the one genuine Self-RAG signal we have
    selfrag_scores = {
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

    # AGENTIC STEP: Update conversation memory
    if session_id:
        doc_ids = [r.get('doc_id', '') for r in results if r.get('doc_id')]
        memory_agent.update_state(
            session_id, 
            winning_domain, 
            doc_ids, 
            combined_conf,
            query_for_rag
        )

    total_time = time.time() - start_time
    print(f"\n[SELF-RAG] Answer ACCEPTED and generated successfully", flush=True)
    print(f"[TIMING] Total time: {total_time:.2f}s", flush=True)
    if selfrag_metrics.get('embedding_cached'):
        print(f"[AGENTIC-RAG] ✅ Embedding cache hit saved ~0.5-1.0s", flush=True)
    print(f"{'='*80}\n", flush=True)
    sys.stdout.flush()

    return answer, original_query, profile.input_lang, sources, confidence_scores, domain_classification, selfrag_metrics

# ============ CLI Functions (ENHANCED) ============
def query_alkhidmat_rag(query: str, category: str = None, use_selfrag: bool = True):
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