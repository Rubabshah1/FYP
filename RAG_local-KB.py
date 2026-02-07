# ============================================================================
# ALKHIDMAT FOUNDATION KNOWLEDGE BASE RAG SYSTEM - WITH CONFIDENCE SCORING
# with Urdu Language Support and Multi-Method Confidence Scores
# ============================================================================
from domain_anchors import DOMAIN_ANCHOR_QUERIES
import os
import re
import pickle
import numpy as np
import torch
import faiss
import zipfile
from typing import List, Dict, Tuple
from pathlib import Path
from scipy.stats import entropy
#new
from sklearn.metrics.pairwise import cosine_similarity

# NEW IMPORTS FOR URDU SUPPORT
from deep_translator import GoogleTranslator
import langdetect
# END NEW IMPORTS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# DOMAIN CLASSIFICATION CLASS (EMBEDDING-BASED ONLY)
# ============================================================================

class DomainClassifier:
    """
    Classifies queries into domains using embedding similarity with anchor queries.
    Uses pre-computed domain centroids from representative queries.
    """
    
    # Cache for domain embeddings (computed once)
    _domain_embeddings_cache = None
    _embedding_model = None
    
    @staticmethod
    def initialize_domain_embeddings(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Pre-compute embeddings for all anchor queries and create domain centroids.
        This is called once during initialization.
        """
        if DomainClassifier._domain_embeddings_cache is not None:
            return  # Already initialized
        
        if os.environ.get('BATCH_MODE') != 'True':
            print("\n🔄 Initializing domain embeddings from anchor queries...")
        
        # Load embedding model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        DomainClassifier._embedding_model = model
        
        domain_embeddings = {}
        
        for domain, queries in DOMAIN_ANCHOR_QUERIES.items():
            # Embed all anchor queries for this domain
            embeddings = model.encode(queries, show_progress_bar=False)
            
            # Compute centroid (average embedding)
            centroid = np.mean(embeddings, axis=0)
            domain_embeddings[domain] = centroid
            
            if os.environ.get('BATCH_MODE') != 'True':
                print(f"  ✓ {domain}: {len(queries)} anchor queries → centroid computed")
        
        DomainClassifier._domain_embeddings_cache = domain_embeddings
        
        if os.environ.get('BATCH_MODE') != 'True':
            print("✅ Domain embeddings initialized!\n")
    
    @staticmethod
    def classify_domain(query: str) -> Dict[str, any]:
        """
        Classify query using embedding similarity to domain centroids.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with domain, confidence, and similarity scores for all domains
        """
        # Initialize if not done yet
        if DomainClassifier._domain_embeddings_cache is None:
            DomainClassifier.initialize_domain_embeddings()
        
        # Embed the query
        query_embedding = DomainClassifier._embedding_model.encode([query])[0]
        
        # Compute cosine similarity with each domain centroid
        similarities = {}
        for domain, centroid in DomainClassifier._domain_embeddings_cache.items():
            # Reshape for cosine_similarity
            query_reshaped = query_embedding.reshape(1, -1)
            centroid_reshaped = centroid.reshape(1, -1)
            
            similarity = cosine_similarity(query_reshaped, centroid_reshaped)[0][0]
            similarities[domain] = float(similarity)
        
        # Get the winning domain
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
# CONFIDENCE SCORING METHODS
# ============================================================================

class ConfidenceScorer:
    """
    Implements multiple confidence scoring methods based on literature review:
    1. Token-level log probabilities (Average, Max, Perplexity)
    2. Entropy-based scoring
    3. Retrieval confidence (semantic similarity)
    """
    
    @staticmethod
    def calculate_perplexity(log_probs: List[float]) -> float:
        """
        Calculate perplexity from log probabilities.
        Lower perplexity = higher confidence
        """
        if not log_probs:
            return float('inf')
        
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        pp=perplexity
        alpha=0.1
        return float(np.exp(-alpha * (pp - 1)))
    
    @staticmethod
    def calculate_average_token_confidence(log_probs: List[float]) -> float:
        """
        Calculate average token log-probability.
        Higher value = higher confidence
        """
        if not log_probs:
            return 0.0
        
        # Convert log probs to probabilities
        probs = [np.exp(lp) for lp in log_probs]
        avg_prob = np.mean(probs)
        return avg_prob
    
    @staticmethod
    def calculate_entropy_confidence(token_probs_distributions: List[np.ndarray]) -> float:
        """
        Calculate entropy-based confidence.
        Uses maximum entropy across all token positions.
        Lower entropy = higher confidence
        """
        if not token_probs_distributions:
            return float('inf')
        
        entropies = []
        for prob_dist in token_probs_distributions:
            if len(prob_dist) > 0:
                ent = entropy(prob_dist)
                entropies.append(ent)
        
        if not entropies:
            return float('inf')
        
        # Return max entropy (most uncertain position)
        max_entropy = max(entropies)
        
        # Normalize to 0-1 scale (inverse, so higher = more confident)
        # Typical entropy range for vocabulary ~50k is 0-10
        confidence = 1 - min(max_entropy / 10.0, 1.0)
        return confidence
    
    @staticmethod
    def calculate_top_k_weighted_confidence(log_probs: List[float], k: int = 5) -> float:
        """
        Weighted confidence score using top k% tokens (from literature).
        Combines 70% from top 5 tokens and 30% from all tokens.
        """
        if not log_probs or len(log_probs) < 5:
            return np.mean([np.exp(lp) for lp in log_probs]) if log_probs else 0.0
        
        # Convert to probabilities
        probs = [np.exp(lp) for lp in log_probs]
        
        # Sort and get top k
        sorted_probs = sorted(probs, reverse=True)
        top_k_probs = sorted_probs[:k]
        
        # Joint probability of top k
        joint_top_k = np.prod(top_k_probs) ** (1/k)  # Geometric mean
        
        # Joint probability of all
        joint_all = np.prod(probs) ** (1/len(probs))  # Geometric mean
        
        # Weighted combination: 70% top k, 30% all
        weighted_score = 0.7 * joint_top_k + 0.3 * joint_all
        
        return weighted_score
    
    # @staticmethod
    # def calculate_retrieval_confidence(distances: List[float], top_k: int = 5) -> float:
    #     """
    #     Calculate confidence based on retrieval quality.
    #     Lower distances = higher confidence in retrieved context.
    #     """
    #     if not distances:
    #         return 0.0
        
    #     # Normalize distances (L2 distances from FAISS)
    #     # Lower distance = better match
    #     avg_distance = np.mean(distances[:top_k])
        
    #     # Convert to confidence score (exponential decay)
    #     # Typical L2 distances range from 0 to ~2 for normalized embeddings
    #     confidence = np.exp(-avg_distance)
        
    #     return confidence
    
    @staticmethod
    def calculate_retrieval_confidence(
        query_embedding: np.ndarray, 
        doc_embeddings: List[np.ndarray], 
        top_k: int = 5
    ) -> float:
        """
        Calculate confidence based on COSINE SIMILARITY between query and retrieved documents.
        Higher cosine similarity = higher confidence in retrieved context.
        
        Args:
            query_embedding: The query vector (shape: (embedding_dim,))
            doc_embeddings: List of retrieved document vectors
            top_k: Number of top results to consider
        
        Returns:
            Average cosine similarity score (0-1 range)
        """
        if not doc_embeddings:
            return 0.0
        
        # Calculate cosine similarity between query and each retrieved document
        cosine_similarities = []
        for doc_emb in doc_embeddings[:top_k]:
            # Reshape for sklearn's cosine_similarity
            query_reshaped = query_embedding.reshape(1, -1)
            doc_reshaped = doc_emb.reshape(1, -1)
            
            # Calculate cosine similarity
            cos_sim = cosine_similarity(query_reshaped, doc_reshaped)[0][0]
            cosine_similarities.append(cos_sim)
        
        # Return average cosine similarity as confidence
        avg_cosine_similarity = np.mean(cosine_similarities)
    
        return float(avg_cosine_similarity)
    
    @staticmethod
    def calculate_combined_confidence(
        log_probs: List[float],
        retrieval_confidence: float,  # <-- New: pre-calculated
        token_probs_distributions: List[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all confidence metrics and return a comprehensive score.
        
        Returns:
            Dictionary with individual scores and a combined score
        """
        scores = {}
        scores['retrieval_confidence'] = retrieval_confidence 
        
        # Token-level scores
        if log_probs:
            scores['avg_token_confidence'] = ConfidenceScorer.calculate_average_token_confidence(log_probs)
            scores['perplexity'] = ConfidenceScorer.calculate_perplexity(log_probs)
            scores['weighted_top_k'] = ConfidenceScorer.calculate_top_k_weighted_confidence(log_probs)
        
        # Entropy-based score
        if token_probs_distributions:
            scores['entropy_confidence'] = ConfidenceScorer.calculate_entropy_confidence(token_probs_distributions)
        
        
        # Combined score (weighted average of normalized metrics)
        # Weights: retrieval 40%, avg_token 30%, weighted_top_k 30%
        combined = 0.0
        weight_sum = 0.0
        
        if 'retrieval_confidence' in scores:
            combined += 0.5 * scores['retrieval_confidence']
            weight_sum += 0.5
        
        if 'avg_token_confidence' in scores:
            combined += 0.25 * scores['avg_token_confidence']
            weight_sum += 0.25
        
        if 'weighted_top_k' in scores:
            combined += 0.25 * scores['weighted_top_k']
            weight_sum += 0.25
        
        if weight_sum > 0:
            scores['combined_confidence'] = combined / weight_sum
        else:
            scores['combined_confidence'] = 0.0
        
        return scores


# ============================================================================
# URDU LANGUAGE SUPPORT FUNCTIONS
# ============================================================================

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns: 'ur' for Urdu, 'en' for English, or other language codes
    """
    try:
        lang = langdetect.detect(text)
        return lang
    except Exception as e:
        print(f"Language detection failed: {e}")
        return 'en'  # Default to English if detection fails
    
def is_urdu(text: str) -> bool:
    """
    Check if the text is in Urdu.
    Uses both language detection and Unicode range checking.
    """
    # Check for Urdu Unicode range (U+0600 to U+06FF for Arabic/Urdu script)
    urdu_pattern = re.compile(r'[\u0600-\u06FF]')
    has_urdu_chars = bool(urdu_pattern.search(text))
    
    # Also use language detection
    detected_lang = detect_language(text)
    
    return has_urdu_chars or detected_lang == 'ur'

def translate_urdu_to_english(text: str) -> str:
    """
    Translate Urdu text to English using Google Translator.
    """
    try:
        translator = GoogleTranslator(source='ur', target='en')
        translated = translator.translate(text)
        if os.environ.get('BATCH_MODE') != 'True':
            print(f"\n🔄 Translated Query (Urdu → English):")
            print(f"   Original (Urdu): {text}")
            print(f"   Translated (English): {translated}\n")
        return translated
    except Exception as e:
        if os.environ.get('BATCH_MODE') != 'True':
             print(f"Translation error (Urdu → English): {e}")
        return text
    
def translate_english_to_urdu(text: str) -> str:
    """
    Translate English text to Urdu using Google Translator.
    """
    try:
        translator = GoogleTranslator(source='en', target='ur')
        translated = translator.translate(text)
        if os.environ.get('BATCH_MODE') != 'True':
            print(f"\n🔄 Translated Answer (English → Urdu):")
            print(f"   Original (English): {text[:100]}...")
            print(f"   Translated (Urdu): {translated[:100]}...\n")
        return translated
    except Exception as e:
        if os.environ.get('BATCH_MODE') != 'True':
            print(f"Translation error (English → Urdu): {e}")
        return text

# ============================================================================
# 1. ZIP FILE LOADING
# ============================================================================

def load_documents_from_zip(zip_path: str) -> Dict[str, List[Dict]]:
    """
    Load all documents from ZIP file and organize by subfolder.
    """
    print("\n" + "="*80)
    print("LOADING FROM ZIP FILE")
    print("="*80)
    print(f"ZIP file: {zip_path}\n")
    
    if not os.path.exists(zip_path):
        print(f"❌ Error: ZIP file not found at {zip_path}")
        return {}
    
    documents_by_category = {}
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            
            print(f"Total files in ZIP: {len(all_files)}\n")
            
            for file_path in all_files:
                if file_path.endswith('/') or not file_path.endswith('.txt'):
                    continue
                
                path_parts = Path(file_path).parts
                
                if len(path_parts) < 3:
                    continue
                
                category = path_parts[-2]
                filename = path_parts[-1]
                
                if os.environ.get('BATCH_MODE') != 'True':
                    print(f"  📄 Reading: [{category}] {filename}...", end=" ")
                
                try:
                    with zip_ref.open(file_path) as f:
                        content = f.read().decode('utf-8')
                    
                    if content:
                        if category not in documents_by_category:
                            documents_by_category[category] = []
                        
                        documents_by_category[category].append({
                            'content': content,
                            'filename': filename,
                            'category': category,
                            'file_path': file_path
                        })
                        if os.environ.get('BATCH_MODE') != 'True':
                            print(f"✓ ({len(content)} characters)")
                    else:
                        if os.environ.get('BATCH_MODE') != 'True':
                            print("⚠️  Empty file")
                
                except Exception as e:
                    if os.environ.get('BATCH_MODE') != 'True':
                        print(f"✗ Error: {e}")
            
            if os.environ.get('BATCH_MODE') != 'True':
                print(f"\n{'='*80}")
                print("LOADING COMPLETE")
                print(f"{'='*80}")
                
                if documents_by_category:
                    for category, docs in documents_by_category.items():
                        print(f"  {category}: {len(docs)} documents")
                    print(f"{'='*80}\n")
                else:
                    print("⚠️  No documents found!")
            
            return documents_by_category
    
    except zipfile.BadZipFile:
        print(f"❌ Error: Invalid ZIP file: {zip_path}")
        return {}
    except Exception as e:
        print(f"❌ Error reading ZIP file: {e}")
        return {}


# ============================================================================
# 2. DATA CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    text = re.sub(r'={3,}[\s\S]*?={3,}', '', text)
    text = re.sub(r'URL:\s*https?://\S+', '', text)
    text = re.sub(r'TITLE:.*?\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def prepare_documents(zip_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Load and clean all documents from ZIP file.
    """
    docs_by_category = load_documents_from_zip(zip_path)
    
    all_docs = []
    metadata = []
    
    for category, docs in docs_by_category.items():
        for doc in docs:
            cleaned = clean_text(doc['content'])
            if cleaned:
                all_docs.append(cleaned)
                metadata.append({
                    'filename': doc['filename'],
                    'category': doc['category'],
                    'file_path': doc['file_path']
                })
    
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"Total documents prepared: {len(all_docs)}\n")
    return all_docs, metadata


# ============================================================================
# 3. TEXT CHUNKING
# ============================================================================

def split_documents(documents: List[str], metadata: List[Dict], 
                    chunk_size: int = 500, chunk_overlap: int = 100) -> Tuple[List[str], List[Dict]]:
    """Split documents into chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks = []
    all_metadata = []
    
    for doc, meta in zip(documents, metadata):
        chunks = splitter.split_text(doc)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append(meta.copy())
    
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Average chunk length: {np.mean([len(c) for c in all_chunks]):.0f} characters\n")
    
    return all_chunks, all_metadata


# ============================================================================
# 4. EMBEDDING CREATION
# ============================================================================

def create_embeddings(text_chunks: List[str], 
                      model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> np.ndarray:
    """Generate embeddings for text chunks."""
    if os.environ.get('BATCH_MODE') != 'True':
        print("Loading embedding model...")
    model = SentenceTransformer(model_name)
    
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"Creating embeddings for {len(text_chunks)} chunks...")
    embeddings = model.encode(text_chunks, show_progress_bar=False, batch_size=32)
    
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"Embeddings shape: {embeddings.shape}\n")
    return np.array(embeddings)


# ============================================================================
# 5. FAISS INDEX STORAGE
# ============================================================================

def build_and_save_index(embeddings: np.ndarray, 
                         text_chunks: List[str],
                         metadata: List[Dict],
                         save_dir: str = "alkhidmat_index"):
    """Build FAISS index and save with metadata."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    dim = embeddings.shape[1]
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"Building FAISS index with dimension: {dim}")
    
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    
    index_path = os.path.join(save_dir, "faiss_index.index")
    faiss.write_index(index, index_path)
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"✓ Saved FAISS index to {index_path}")
    
    chunks_path = os.path.join(save_dir, "text_chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(text_chunks, f)
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"✓ Saved text chunks to {chunks_path}")
    
    metadata_path = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"✓ Saved metadata to {metadata_path}\n")


# ============================================================================
# 6. RETRIEVAL FUNCTIONS
# ============================================================================

def load_index_and_data(index_dir: str = "alkhidmat_index"):
    """Load FAISS index, chunks, and metadata."""
    if os.environ.get('BATCH_MODE') != 'True':
        print("Loading RAG system components...")
    
    index_path = os.path.join(index_dir, "faiss_index.index")
    chunks_path = os.path.join(index_dir, "text_chunks.pkl")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    
    index = faiss.read_index(index_path)
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"✓ Loaded FAISS index")
    
    with open(chunks_path, "rb") as f:
        text_chunks = pickle.load(f)
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"✓ Loaded {len(text_chunks)} text chunks")
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"✓ Loaded metadata\n")
    
    return index, text_chunks, metadata


# def retrieve_context(query: str, index, text_chunks: List[str], 
#                      metadata: List[Dict], top_k: int = 5,
#                      model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
#     """Retrieve relevant context with source attribution."""
    
#     model = SentenceTransformer(model_name)
#     query_vector = model.encode([query]).astype('float32')
    
#     distances, indices = index.search(query_vector, top_k)
    
#     results = []
#     for idx, dist in zip(indices[0], distances[0]):
#         results.append({
#             'text': text_chunks[idx],
#             'category': metadata[idx]['category'],
#             'filename': metadata[idx]['filename'],
#             'distance': float(dist)
#         })
    
#     if os.environ.get('BATCH_MODE') != 'True':
#         print(f"\n{'='*80}")
#         print(f"Retrieved {len(results)} relevant chunks:")
#         for i, result in enumerate(results, 1):
#             print(f"{i}. [{result['category']}] {result['filename']} (distance: {result['distance']:.3f})")
#         print(f"{'='*80}\n")
    
#     return results, distances[0].tolist()

def retrieve_context(query: str, index, text_chunks: List[str], 
                     metadata: List[Dict], top_k: int = 5,
                     model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """Retrieve relevant context with source attribution."""
    model = SentenceTransformer(model_name)
    query_vector = model.encode([query]).astype('float32')
    
    distances, indices = index.search(query_vector, top_k)
    
    # Encode retrieved documents to get their embeddings
    doc_embeddings = []
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        # Get the document text and encode it
        doc_text = text_chunks[idx]
        doc_embedding = model.encode([doc_text]).astype('float32')[0]
        doc_embeddings.append(doc_embedding)
        
        results.append({
            'text': doc_text,
            'category': metadata[idx]['category'],
            'filename': metadata[idx]['filename'],
            'distance': float(dist)
        })
    
    # ... rest of code (print statements etc) ...
    
    # Return embeddings AND distances (in case distances are used elsewhere)
    return results, query_vector[0], doc_embeddings, distances[0].tolist()


# ============================================================================
# 7. ANSWER GENERATION WITH CONFIDENCE SCORING
# ============================================================================

def generate_answer_with_confidence(query: str, index_dir: str = "alkhidmat_index", 
                                   top_k: int = 5, max_tokens: int = 250):
    """
    Generate answer using retrieved context with comprehensive confidence scoring.
    """
    
    # Step 1: Detect language and translate if needed
    original_query = query
    query_is_urdu = is_urdu(query)
    
    if query_is_urdu:
        if os.environ.get('BATCH_MODE') != 'True':
            print(f"\n🇵🇰 Urdu query detected!")
        query = translate_urdu_to_english(query)
    
    # Step 2: Retrieve context
    index, text_chunks, metadata = load_index_and_data(index_dir)
    # results, retrieval_distances = retrieve_context(query, index, text_chunks, metadata, top_k)
    results, query_embedding, doc_embeddings, distances = retrieve_context(query, index, text_chunks, metadata, top_k)
    
    # Step 2.5: CLASSIFY DOMAIN using embedding-based method
    domain_classification = DomainClassifier.classify_domain(query)
    
    # Build context
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Source {i} - {result['category']}/{result['filename']}]\n{result['text']}"
        )
    context = "\n\n".join(context_parts)
    
    # Step 3: Load LLM
    if os.environ.get('BATCH_MODE') != 'True':
        print("Loading LLM for answer generation...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    prompt = f"""You are a helpful assistant answering questions about Alkhidmat Foundation Pakistan.

Context from Alkhidmat knowledge base:
{context}

Question: {query}

Answer based on the context above. Include relevant details about donation methods, programs, or services.

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    # Step 4: Generate with log probabilities
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    # Extract generated text
    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
    full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    answer = full_text.split("Answer:")[-1].strip() if "Answer:" in full_text else full_text[len(prompt):].strip()
    
    # Step 5: Calculate confidence scores
    log_probs = [] #this um contains only log probabilities
    token_probs_distributions = [] #This contains full probabiliyu distribution for each token for all vocab basically kaam aegee humein entropy mein
    
    
    if hasattr(outputs, 'scores') and outputs.scores:
        for t, score in enumerate(outputs.scores):
            selected_token_idx = generated_ids[t]

            if selected_token_idx in tokenizer.all_special_ids:
                continue

            probs = torch.softmax(score / 0.7, dim=-1)

            token_log_prob = torch.log(probs[0, selected_token_idx]).item()
            log_probs.append(token_log_prob)

            token_probs_distributions.append(probs[0].cpu().numpy())

    
    # Calculate all confidence metrics
    # confidence_scores = ConfidenceScorer.calculate_combined_confidence(
    #     log_probs=log_probs,
    #     retrieval_distances=retrieval_distances,
    #     token_probs_distributions=token_probs_distributions
    # )
    retrieval_conf = ConfidenceScorer.calculate_retrieval_confidence(
        query_embedding=query_embedding,
        doc_embeddings=doc_embeddings,
        top_k=top_k
    )
    confidence_scores = ConfidenceScorer.calculate_combined_confidence(
    log_probs=log_probs,
    retrieval_confidence=retrieval_conf,  # <-- Pass the calculated confidence
    token_probs_distributions=token_probs_distributions
    )
    
    # Step 6: Translate answer back to Urdu if needed
    if query_is_urdu:
        answer = translate_english_to_urdu(answer)
    
    # Step 7: Display results
    if os.environ.get('BATCH_MODE') != 'True':
        print(f"\n{'='*80}")
        print(f"QUESTION: {original_query}")
        if query_is_urdu:
            print(f"(Translated to English: {query})")
        print(f"{'='*80}")
        # ADD THIS DOMAIN SECTION HERE:
        domain_emoji = DomainClassifier.get_domain_emoji(domain_classification['domain'])
        print(f"\n{domain_emoji} DOMAIN CLASSIFICATION (Embedding-Based):")
        print(f"{'='*80}")
        print(f"  Classified Domain: {domain_classification['domain'].upper()}")
        print(f"  Classification Confidence: {domain_classification['confidence']:.2%}")
        print(f"\n  Similarity Scores:")
        for domain, score in domain_classification['all_scores'].items():
            emoji = DomainClassifier.get_domain_emoji(domain)
            bar = '█' * int(score * 50)  # Visual bar
            print(f"    {emoji} {domain:12s}: {score:.4f} {bar}")
        print(f"{'='*80}")
        print(f"\nANSWER:\n{answer}")
        print(f"\n{'='*80}")
        print("CONFIDENCE SCORES:")
        print(f"{'='*80}")
        print(f"  Combined Confidence:        {confidence_scores.get('combined_confidence', 0):.4f} ⭐")
        print(f"  ├─ Retrieval Confidence:    {confidence_scores.get('retrieval_confidence', 0):.4f}")
        print(f"  ├─ Avg Token Confidence:    {confidence_scores.get('avg_token_confidence', 0):.4f}")
        print(f"  ├─ Weighted Top-K:          {confidence_scores.get('weighted_top_k', 0):.4f}")
        print(f"  ├─ Perplexity:              {confidence_scores.get('perplexity', 0):.4f}")
        print(f"  └─ Entropy Confidence:      {confidence_scores.get('entropy_confidence', 0):.4f}")
        print(f"{'='*80}\n")
        
        # Interpretation
        combined = confidence_scores.get('combined_confidence', 0)
        if combined >= 0.7:
            print("✅ High confidence - Answer is likely reliable")
        elif combined >= 0.5:
            print("⚠️  Moderate confidence - Answer may need verification")
        else:
            print("❌ Low confidence - Answer should be verified from sources")
        print()
    
    
    return answer, confidence_scores, domain_classification, original_query, query_is_urdu


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def build_alkhidmat_rag(zip_path: str, index_dir: str = "alkhidmat_index"):
    """Complete pipeline to build RAG system from ZIP file."""
    
    print("\n" + "="*80)
    print("ALKHIDMAT FOUNDATION RAG SYSTEM - BUILD FROM ZIP FILE")
    print("with Urdu Language Support and Confidence Scoring")
    print("="*80 + "\n")
    
    print("STEP 1: Loading documents from ZIP file...")
    documents, metadata = prepare_documents(zip_path)
    
    if not documents:
        print("\n⚠️  No documents loaded! Please check ZIP file structure.")
        return
    
    print("\nSTEP 2: Splitting documents into chunks...")
    text_chunks, chunk_metadata = split_documents(documents, metadata, 500, 100)
    
    print("\nSTEP 3: Creating embeddings...")
    embeddings = create_embeddings(text_chunks)
    
    print("\nSTEP 4: Building and saving FAISS index...")
    build_and_save_index(embeddings, text_chunks, chunk_metadata, index_dir)
    
    print("\n" + "="*80)
    print("✓ RAG SYSTEM BUILD COMPLETE!")
    print("System supports: English & Urdu queries + Confidence Scoring")
    print("="*80 + "\n")
    print("Initializing domain classification...")
    DomainClassifier.initialize_domain_embeddings()


def query_alkhidmat_rag(query: str, index_dir: str = "alkhidmat_index"):
    """Query the Alkhidmat RAG system with confidence scores and domain classification."""
    answer, confidence_scores, domain_classification, _, _ = generate_answer_with_confidence(query, index_dir)
    return answer, confidence_scores, domain_classification


# ============================================================================
# 9. FILE I/O BATCH PROCESSING
# ============================================================================

def batch_query_file(input_file: str, output_file: str, index_dir: str = "alkhidmat_index"):
    """
    Reads queries from an input file, processes them with confidence scoring,
    and writes results to an output file.
    """
    os.environ['BATCH_MODE'] = 'True'
    
    print(f"\n{'='*80}")
    print(f"BATCH QUERY MODE: Processing {input_file} -> {output_file}")
    print(f"{'='*80}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(queries)} queries to process.")
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("ALKHIDMAT RAG SYSTEM - BATCH QUERY RESULTS WITH CONFIDENCE SCORES\n")
            f_out.write(f"Source file: {input_file}\n\n")

            for i, query in enumerate(queries):
                answer, confidence_scores, domain_classification, original_query, is_urdu_query = \
                    generate_answer_with_confidence(query, index_dir)
        
                f_out.write(f"{'='*80}\n")
                f_out.write(f"QUERY {i+1}/{len(queries)}\n")
                f_out.write(f"{'='*80}\n")
                f_out.write(f"QUERY: {original_query}\n")
                f_out.write(f"Language: {'Urdu' if is_urdu_query else 'English'}\n")
                # ADD DOMAIN CLASSIFICATION OUTPUT:
                domain_emoji = DomainClassifier.get_domain_emoji(domain_classification['domain'])
                f_out.write(f"{domain_emoji} DOMAIN CLASSIFICATION:\n")
                f_out.write(f"{'-'*80}\n")
                f_out.write(f"Classified Domain: {domain_classification['domain'].upper()}\n")
                f_out.write(f"Classification Confidence: {domain_classification['confidence']:.2%}\n")
                f_out.write(f"Similarity Scores:\n")
                for domain, score in domain_classification['all_scores'].items():
                    f_out.write(f"  - {domain}: {score:.4f}\n")
                f_out.write(f"\n")
                f_out.write(f"\nANSWER:\n{answer}\n\n")
                
                f_out.write(f"CONFIDENCE SCORES:\n")
                f_out.write(f"{'-'*80}\n")
                f_out.write(f"Combined Confidence:        {confidence_scores.get('combined_confidence', 0):.4f} ⭐\n")
                f_out.write(f"├─ Retrieval Confidence:    {confidence_scores.get('retrieval_confidence', 0):.4f}\n")
                f_out.write(f"├─ Avg Token Confidence:    {confidence_scores.get('avg_token_confidence', 0):.4f}\n")
                f_out.write(f"├─ Weighted Top-K:          {confidence_scores.get('weighted_top_k', 0):.4f}\n")
                f_out.write(f"├─ Min Token Confidence:    {confidence_scores.get('min_token_confidence', 0):.4f}\n")
                f_out.write(f"├─ Perplexity:              {confidence_scores.get('perplexity', 0):.4f}\n")
                f_out.write(f"└─ Entropy Confidence:      {confidence_scores.get('entropy_confidence', 0):.4f}\n\n")
                
                combined = confidence_scores.get('combined_confidence', 0)
                if combined >= 0.7:
                    f_out.write("✅ High confidence - Answer is likely reliable\n")
                elif combined >= 0.5:
                    f_out.write("⚠️  Moderate confidence - Answer may need verification\n")
                else:
                    f_out.write("❌ Low confidence - Answer should be verified from sources\n")
                
                f_out.write(f"\n")
                
                print(f"  Query {i+1}: '{original_query[:50]}...' -> Domain: {domain_classification['domain']} (conf: {domain_classification['confidence']:.2%})")

        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE. Results written to {output_file}")
        print(f"{'='*80}\n")
        
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    
    os.environ['BATCH_MODE'] = 'False'


# ============================================================================
# 10. USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    DEFAULT_ZIP_PATH = "Al Khidmat Knowledge Base.zip"
    
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        zip_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZIP_PATH
        build_alkhidmat_rag(zip_path, "alkhidmat_index")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What donation methods does Alkhidmat accept?"
        query_alkhidmat_rag(query)

    elif len(sys.argv) > 1 and sys.argv[1] == "file_query":
        input_file = sys.argv[2] if len(sys.argv) > 2 else "input_queries.txt"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "output_answers.txt"
        batch_query_file(input_file, output_file)
    
    else:
        print("\n" + "="*80)
        print("ALKHIDMAT RAG SYSTEM - USAGE (with Urdu & Confidence Scoring)")
        print("="*80)
        print("\n1. Build the index:")
        print(f"  python rag_alkhidmat_with_confidence.py build [path_to_zip]")
        print(f"  (Uses default: {DEFAULT_ZIP_PATH})")
        
        print("\n2. Query the system (Direct Terminal Output):")
        print("  python rag_alkhidmat_with_confidence.py query 'How can I donate?'")
        print("  python rag_alkhidmat_with_confidence.py query 'الخدمت کون سے پروگرام چلاتی ہے؟'")
        
        print("\n3. Query the system (File I/O for clean Urdu Output + Confidence):")
        print("   Create a file (e.g., `urdu_queries.txt`) with one query per line.")
        print("  python rag_alkhidmat_with_confidence.py file_query urdu_queries.txt output.txt")
        print("\n" + "="*80 + "\n")