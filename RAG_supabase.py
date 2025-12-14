#!/usr/bin/env python3
"""
ALKHIDMAT RAG SYSTEM - SUPABASE CLIENT EDITION
Uses Supabase Python client (REST API) instead of direct PostgreSQL connection
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import uuid

import numpy as np
import zipfile

# Embeddings
from sentence_transformers import SentenceTransformer

# LLM (GPT4All)
from gpt4all import GPT4All

# Supabase Client
from supabase import create_client, Client

# Urdu helpers
from deep_translator import GoogleTranslator
import langdetect

# text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============ SUPABASE CONFIG ============
# Get these from: Supabase Dashboard → Settings → API
SUPABASE_URL = os.environ.get("SUPABASE_URL")  # https://xxxxx.supabase.co
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")  # Your anon/service_role key

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
LLM_MODEL_FILENAME = os.environ.get("GPT4ALL_MODEL", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_DIM = 768

# Retrieval parameters
RELEVANCE_THRESHOLD = 0.7

# ============ Supabase Client ============
_SUPABASE_CLIENT = None

def get_supabase_client() -> Client:
    """Get or create Supabase client"""
    global _SUPABASE_CLIENT
    
    if _SUPABASE_CLIENT is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env file\n"
                "Get these from: Supabase Dashboard → Settings → API"
            )
        
        print(f"Connecting to Supabase: {SUPABASE_URL}")
        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    return _SUPABASE_CLIENT

def test_connection():
    """Test Supabase connection"""
    try:
        supabase = get_supabase_client()
        
        # Try to query documents table
        result = supabase.table("documents").select("doc_id").limit(1).execute()
        
        print("✅ Connected to Supabase successfully!")
        print(f"   URL: {SUPABASE_URL}")
        
        # Check if pgvector extension is enabled by trying to query
        try:
            # This will fail if pgvector is not enabled
            result = supabase.rpc('match_documents_simple', {
                'query_embedding': [0.0] * 768,
                'match_count': 1
            }).execute()
            print("✅ pgvector extension verified")
        except Exception as e:
            if 'does not exist' in str(e):
                print("⚠️  match_documents function not found - run the SQL setup")
            else:
                print(f"⚠️  Could not verify pgvector: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nMake sure you have:")
        print("1. Set SUPABASE_URL in .env")
        print("2. Set SUPABASE_ANON_KEY in .env")
        print("3. Run the SQL schema setup in Supabase SQL Editor")
        return False

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

def translate_english_to_urdu(text: str) -> str:
    try:
        return GoogleTranslator(source='en', target='ur').translate(text)
    except Exception:
        return text

# ============ Query Analysis ============
def analyze_query(query: str) -> Dict[str, Any]:
    """Detect what kind of answer the user wants"""
    q_lower = query.lower()
    
    list_keywords = ['list', 'points', 'bullet', 'enumerate', 'steps', 'ways']
    summary_keywords = ['summarize', 'summary', 'briefly', 'overview', 'خلاصہ']
    detail_keywords = ['explain', 'detail', 'describe', 'how does', 'why', 'تفصیل']
    
    wants_list = any(kw in q_lower for kw in list_keywords)
    wants_summary = any(kw in q_lower for kw in summary_keywords)
    wants_detail = any(kw in q_lower for kw in detail_keywords)
    
    return {
        'wants_list': wants_list,
        'wants_summary': wants_summary,
        'wants_detail': wants_detail,
        'is_urdu': is_urdu(query)
    }

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
    
    # Prepare data
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
    
    # Insert in batches (Supabase has limits on request size)
    batch_size = 100
    total_inserted = 0
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            result = supabase.table("documents").insert(batch).execute()
            total_inserted += len(batch)
            print(f"   Inserted batch {i//batch_size + 1}: {len(batch)} chunks")
        except Exception as e:
            print(f"   ⚠️  Error inserting batch {i//batch_size + 1}: {e}")
            # Try one by one if batch fails
            for row in batch:
                try:
                    supabase.table("documents").insert(row).execute()
                    total_inserted += 1
                except Exception as e2:
                    print(f"   ⚠️  Failed to insert chunk: {e2}")
    
    print(f"✅ Stored {total_inserted}/{len(rows)} chunks in Supabase")

def clear_documents_table():
    """Clear all documents (use with caution!)"""
    supabase = get_supabase_client()
    
    try:
        # Delete all documents
        result = supabase.table("documents").delete().neq("doc_id", "00000000-0000-0000-0000-000000000000").execute()
        print("✅ Cleared all documents from Supabase")
    except Exception as e:
        print(f"⚠️  Error clearing documents: {e}")

# ============ Build Pipeline ============
def build_alkhidmat_rag(zip_path: str, clear_existing: bool = False):
    """Build the RAG system and store in Supabase"""
    print("\n" + "="*80)
    print("BUILDING ALKHIDMAT RAG SYSTEM")
    print("="*80)
    
    # Test connection
    if not test_connection():
        print("❌ Cannot connect to Supabase. Aborting.")
        return
    
    if clear_existing:
        print("\n⚠️  Clearing existing documents...")
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
    
    print("\n" + "="*80)
    print("✅ BUILD COMPLETE!")
    print("="*80)

# ============ Retrieval ============
def retrieve_from_supabase(query: str, top_k: int = 5, 
                           filter_category: str = None) -> Tuple[List[Dict], List[Dict]]:
    """Retrieve similar documents from Supabase using RPC function"""
    
    # Create query embedding
    embedder = get_embedder()
    query_prefixed = f"query: {query}"
    query_embedding = embedder.encode([query_prefixed], normalize_embeddings=True)[0]
    
    supabase = get_supabase_client()
    
    try:
        # Call the RPC function for vector similarity search
        params = {
            'query_embedding': query_embedding.tolist(),
            'match_threshold': RELEVANCE_THRESHOLD,
            'match_count': top_k
        }
        
        if filter_category:
            params['filter_category'] = filter_category
        
        result = supabase.rpc('match_documents', params).execute()
        
        rows = result.data
        
    except Exception as e:
        # Fallback: manual similarity search if RPC function doesn't exist
        print(f"⚠️  RPC function not available, using fallback method: {e}")
        
        # Get all documents (not ideal for large datasets)
        result = supabase.table("documents").select(
            "doc_id, chunk_text, category, filename, file_path, chunk_index, embedding"
        ).execute()
        
        rows = []
        for row in result.data:
            if row['embedding']:
                # Calculate cosine similarity manually
                doc_emb = np.array(row['embedding'])
                similarity = float(np.dot(query_embedding, doc_emb))
                
                if similarity > RELEVANCE_THRESHOLD:
                    row['similarity'] = similarity
                    rows.append(row)
        
        # Sort by similarity and take top_k
        rows = sorted(rows, key=lambda x: x['similarity'], reverse=True)[:top_k]
    
    results = []
    sources = []
    
    for row in rows:
        results.append({
            "text": row['chunk_text'],
            "category": row['category'],
            "filename": row['filename'],
            "file_path": row['file_path'],
            "chunk_index": row.get('chunk_index', 0),
            "similarity": float(row.get('similarity', 0))
        })
        
        sources.append({
            "doc_id": str(row['doc_id']),
            "category": row['category'],
            "filename": row['filename'],
            "file_path": row['file_path'],
            "similarity": float(row.get('similarity', 0))
        })
    
    # Print retrieval info
    print("\n" + "="*80)
    print("RETRIEVAL FROM SUPABASE")
    print(f"Retrieved {len(results)} relevant chunks (threshold: {RELEVANCE_THRESHOLD}):")
    for i, s in enumerate(sources, 1):
        print(f"{i}. [{s['category']}] {s['filename']} (similarity: {s['similarity']:.3f})")
    print("="*80 + "\n")
    
    return results, sources

def sanitize_chunk_text(text: str) -> str:
    """
    Remove existing Q/A labels and short FAQ blocks from a chunk so the LLM
    doesn't copy them into the answer.
    """
    # Remove common question/answer labels
    text = re.sub(r'(?mi)^\s*(user\s+question|question|q:)\s*[:\-–]?\s*.*$', '', text)
    text = re.sub(r'(?mi)^\s*(answer|a:)\s*[:\-–]?\s*.*$', '', text)

    # Remove blocks that look like Q/A pairs (Q: ... A: ...), two or more occurrences
    text = re.sub(r'(?is)(?:^|\n)\s*q[:\.\-\)]\s*.*?\n\s*a[:\.\-\)]\s*.*?(?:\n|$)', '', text)

    # Remove short placeholders like "[insert email address]" or "click here"
    text = re.sub(r'\[insert .*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'click here', '', text, flags=re.IGNORECASE)

    # Collapse many newlines and trim
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ============ LLM ============
_GPT_MODEL = None

def load_llm(model_filename: str = LLM_MODEL_FILENAME):
    global _GPT_MODEL
    if _GPT_MODEL is None:
        print("Loading local LLM:", model_filename)
        _GPT_MODEL = GPT4All(model_filename, model_path=".", allow_download=False)
    return _GPT_MODEL

def llm_generate(prompt: str, max_tokens: int = 400, stop_tokens: list = None) -> str:
    """Generate with guarded params. If the underlying model supports stop tokens, pass them."""
    model = load_llm()

    # Conservative generation settings reduce the chance of extra appended Q/A
    generation_kwargs = dict(
        prompt=prompt,
        max_tokens=max_tokens,
        temp=0.2,         # lower temp for deterministic outputs
        top_p=0.9,
        repeat_penalty=1.2
    )

    # Many GPT4All wrappers accept 'stop' but if not, this will be ignored by the wrapper
    if stop_tokens:
        generation_kwargs['stop'] = stop_tokens

    try:
        resp = model.generate(**generation_kwargs)
    except TypeError:
        # fallback for wrapper that doesn't support kwargs
        resp = model.generate(prompt, max_tokens=max_tokens)

    if isinstance(resp, (list, tuple)) and len(resp) > 0:
        return str(resp[0]).strip()
    return str(resp).strip()


# ============ Answer Generation ============
def generate_answer(query: str, top_k: int = 5, max_tokens: int = 400, 
                   filter_category: str = None):
    """Query-aware answer generation"""
    query_info = analyze_query(query)
    is_urdu = query_info['is_urdu']
    
    # Retrieve from Supabase
    results, sources = retrieve_from_supabase(query, top_k=top_k, 
                                             filter_category=filter_category)
    
    if not results:
        no_answer = "معاف کیجیے، میں اس سوال کا جواب دینے کے لیے متعلقہ معلومات نہیں ڈھونڈ سکا۔" if is_urdu else "I apologize, but I couldn't find relevant information to answer this question."
        return no_answer, query, is_urdu, sources
    
    # Build context
        # Build context - sanitize each chunk to avoid copying Q/A style artifacts
    context_parts = []
    for r in results:
        chunk_text = r["text"]

        # Sanitize chunk to remove embedded Q/A or "User question" labels
        chunk_text = sanitize_chunk_text(chunk_text)

        if is_urdu:
            try:
                chunk_text = translate_english_to_urdu(chunk_text)
            except Exception:
                pass

        if chunk_text:
            # Optionally truncate very long chunks to keep prompt small
            MAX_CHUNK_CHARS = 1200
            if len(chunk_text) > MAX_CHUNK_CHARS:
                chunk_text = chunk_text[:MAX_CHUNK_CHARS].rsplit('\n', 1)[0] + "\n\n[truncated]"
            context_parts.append(chunk_text)

    context = "\n\n".join(context_parts)

    
    # Build query-aware prompt
    if is_urdu:
        base_instruction = "آپ الخدمت فاؤنڈیشن پاکستان کے لیے ایک مددگار اسسٹنٹ ہیں۔"
        
        if query_info['wants_list']:
            format_instruction = "براہ کرم نقاط کی شکل میں واضح جواب دیں۔"
        elif query_info['wants_summary']:
            format_instruction = "براہ کرم 2-3 جملوں میں مختصر خلاصہ دیں۔"
        else:
            format_instruction = "براہ کرم مکمل اور واضح جواب دیں۔ اگر سیاق و سباق میں تفصیلات ہیں تو سب شامل کریں۔"
        
        prompt = f"""{base_instruction}
{format_instruction}

صرف دیے گئے سیاق و سباق کی معلومات استعمال کریں۔ اگر جواب سیاق و سباق میں نہیں تو "مجھے معلوم نہیں" کہیں۔

سیاق و سباق:
{context}

سوال: {query}

جواب (صرف اردو میں، ماخذ شامل نہ کریں):
"""
    else:
        base_instruction = "You are a helpful customer support agent for Alkhidmat Foundation Pakistan."
        
        if query_info['wants_list']:
            format_instruction = "Provide a clear answer in bullet point format."
        elif query_info['wants_summary']:
            format_instruction = "Provide a brief summary in 2-3 sentences."
        elif query_info['wants_detail']:
            format_instruction = "Provide a detailed, comprehensive answer covering all relevant information from the context."
        else:
            format_instruction = "Provide a clear, complete answer. Include all relevant details from the context."
        
        prompt = f"""{base_instruction}

{format_instruction}

Important instructions:
- Use ONLY the information provided below.
- Return ONLY the direct answer to the user's question.
- DO NOT include any additional 'User question:', 'Question:', 'Answer:' lines, headings, or extra Q/A pairs.
- DO NOT copy the context verbatim or print verbatim Q/A excerpts.
- If the information is not present, reply exactly: "I don't have that information."
- Output exactly one answer and nothing else (no extra labels, no summaries).

Negative example (not allowed):
User question: What is X?
Answer: ...
User question: What is Y?
Answer: ...

Information:
{context}

User question: {query}

Answer (ONLY the final answer, no labels):
"""
    answer = llm_generate(prompt, max_tokens=max_tokens, stop_tokens=["\nUser question:", "\nQuestion:", "\nUser question"])
    # Post-process to remove any remaining artifacts 
    answer = clean_llm_response(answer)
    return answer, query, is_urdu, sources

def clean_llm_response(text: str) -> str:
    """Clean up LLM output to remove unwanted artifacts and cut off extra Q/A blocks."""
    if not text:
        return text

    # Normalize whitespace
    text = text.replace('\r\n', '\n')

    # Remove references to context numbers and meta lines
    text = re.sub(r'\[?Context \d+\]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'based on Context \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'as per (?:the )?context', '', text, flags=re.IGNORECASE)

    # Remove explicit "User question:" / "Question:" / "Answer:" lines and anything that follows:
    # If the model appended additional Q/A sections, truncate the answer at the first occurrence
    cutoff_patterns = [r'\nUser question\s*:', r'\nQuestion\s*:', r'\nUser question', r'\nQuestion', r'\nAnswer\s*:']
    earliest = len(text)
    for pat in cutoff_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            earliest = min(earliest, m.start())

    if earliest < len(text):
        text = text[:earliest].strip()

    # Remove leftover labels inline (rare)
    text = re.sub(r'(?mi)^(user question|question|answer)\s*[:\-–]\s*', '', text)

    # Remove repetitive identical lines
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

    # Final trim
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

# ============ CLI Functions ============
def query_alkhidmat_rag(query: str, category: str = None):
    answer, _, _, sources = generate_answer(query, filter_category=category)
    print("\n" + "="*80)
    print("QUESTION:", query)
    if category:
        print("CATEGORY FILTER:", category)
    print("="*80)
    print("ANSWER:\n", answer)
    print("\n" + "="*80)
    return answer

def show_statistics():
    """Show database statistics"""
    supabase = get_supabase_client()
    
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    
    try:
        # Get total count
        result = supabase.table("documents").select("doc_id", count="exact").execute()
        print(f"\nTotal chunks: {result.count}")
        
        # Get by category
        result = supabase.table("documents").select("category").execute()
        categories = {}
        for row in result.data:
            cat = row.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nDocuments by Category:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} chunks")
            
    except Exception as e:
        print(f"Error getting statistics: {e}")

# ============ Batch Processing ============
def batch_query_file(input_file: str, output_file: str):
    """
    Reads queries from an input file (one query per line), processes them
    via Supabase/LLM, and writes results to an output file.
    """
    print(f"\n{'='*80}")
    print(f"BATCH QUERY MODE: Processing {input_file} -> {output_file}")
    print(f"{'='*80}")

    try:
        # Read queries (using utf-8 encoding for Urdu)
        with open(input_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(queries)} queries to process.")
        
        # Write results
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("ALKHIDMAT RAG (SUPABASE) - BATCH QUERY RESULTS\n")
            f_out.write(f"Source file: {input_file}\n")
            f_out.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for i, query in enumerate(queries):
                print(f"Processing {i+1}/{len(queries)}: '{query[:40]}...'")
                
                # Run the generation pipeline
                # Note: your generate_answer returns 4 values
                answer, original_query, is_urdu_query, sources = generate_answer(query)
                
                # Format output to file
                f_out.write(f"--- QUERY {i+1}/{len(queries)} ---\n")
                f_out.write(f"QUERY: {original_query}\n")
                f_out.write(f"LANGUAGE: {'Urdu' if is_urdu_query else 'English'}\n")
                f_out.write("-" * 20 + "\n")
                f_out.write(f"ANSWER:\n{answer}\n\n")
                
                # Write sources used
                f_out.write("SOURCES USED:\n")
                if sources:
                    for s in sources:
                        f_out.write(f" - [{s['category']}] {s['filename']} (Sim: {s['similarity']:.3f})\n")
                else:
                    f_out.write(" - No relevant documents found.\n")
                
                f_out.write("\n" + "="*40 + "\n\n")

        print(f"\n{'='*80}")
        print(f"✅ BATCH PROCESSING COMPLETE. Results written to {output_file}")
        print(f"{'='*80}\n")
        
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during batch processing: {e}")

# ============ Main ============
if __name__ == "__main__":
    import sys
    DEFAULT_ZIP = "Al Khidmat Knowledge Base.zip"
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connection()
        
    elif len(sys.argv) > 1 and sys.argv[1] == "build":
        zip_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZIP
        clear = "--clear" in sys.argv
        build_alkhidmat_rag(zip_path, clear_existing=clear)
        
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What donation methods does Alkhidmat accept?"
        query_alkhidmat_rag(q)

    elif len(sys.argv) > 1 and sys.argv[1] == "file_query":
        # Batch Query mode
        input_file = sys.argv[2] if len(sys.argv) > 2 else "input_queries.txt"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "output_answers.txt"
        batch_query_file(input_file, output_file)
        
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        show_statistics()
        
    else:
        print("\n" + "="*60)
        print("ALKHIDMAT RAG SYSTEM (SUPABASE EDITION)")
        print("="*60)
        print("\nUSAGE:")
        print("1. Test Connection:")
        print("   python RAG_supabase_client.py test")
        
        print("\n2. Build Index (Upload to Supabase):")
        print("   python RAG_supabase_client.py build [zip_path] [--clear]")
        
        print("\n3. Single Query (Terminal):")
        print("   python RAG_supabase_client.py query 'your question'")
        
        print("\n4. Batch Query (File I/O for Urdu):")
        print("   python RAG_supabase_client.py file_query input.txt output.txt")
        
        print("\n5. View Stats:")
        print("   python RAG_supabase_client.py stats")
        print("\n" + "="*60)