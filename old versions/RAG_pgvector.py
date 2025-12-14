#!/usr/bin/env python3
"""
ALKHIDMAT RAG SYSTEM (Hybrid FAISS + Postgres) with Urdu-aware output
Modifications:
 - Retrieval printed before generation
 - Model answer does NOT include source attributions
 - Batch output JSON contains structured retrieved_sources and clean 'answer'
 - Urdu queries: retrieve with Urdu embedding, translate contexts to Urdu, prompt in Urdu
"""

import os
import re
import pickle
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import faiss
import zipfile

# Embeddings
from sentence_transformers import SentenceTransformer

# LLM (GPT4All)
from gpt4all import GPT4All

# Postgres
import psycopg2
from psycopg2.extras import execute_values

# Urdu helpers
from deep_translator import GoogleTranslator
import langdetect

# text splitter from langchain_text_splitters (your package)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============ Config (customize) ============
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_DB = os.environ.get("PG_DB", "alkhidmat")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "1234")
PG_PORT = int(os.environ.get("PG_PORT", 5432))

FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "alkhidmat_index")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
LLM_MODEL_FILENAME = os.environ.get("GPT4ALL_MODEL", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")  # set to your downloaded gguf name
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_DIM = 768  # multilingual-e5-base -> 768

# ============ Utilities: language detection & translation ============
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

# ============ Postgres helpers ============
def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST,
        database=PG_DB,
        user=PG_USER,
        password=PG_PASS,
        port=PG_PORT
    )

def ensure_pg_table():
    conn = get_pg_connection()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS kb_chunks (
              id SERIAL PRIMARY KEY,
              chunk TEXT,
              category TEXT,
              filename TEXT,
              file_path TEXT,
              embedding VECTOR({EMBEDDING_DIM})
            );
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS kb_chunks (
              id SERIAL PRIMARY KEY,
              chunk TEXT,
              category TEXT,
              filename TEXT,
              file_path TEXT,
              embedding FLOAT8[]
            );
        """)
        conn.commit()
    cur.close()
    conn.close()

def save_chunks_to_postgres(chunks: List[str], metadata: List[Dict], embeddings: np.ndarray):
    conn = get_pg_connection()
    cur = conn.cursor()
    rows = []
    for i, chunk in enumerate(chunks):
        meta = metadata[i]
        emb_list = embeddings[i].tolist()
        rows.append((chunk, meta.get("category"), meta.get("filename"), meta.get("file_path"), emb_list))
    execute_values(cur, """
        INSERT INTO kb_chunks (chunk, category, filename, file_path, embedding)
        VALUES %s
    """, rows)
    conn.commit()
    cur.close()
    conn.close()
    print(f"✓ Stored {len(rows)} chunks+embeddings in Postgres")

def fetch_metadata_by_chunk(chunk_text: str) -> Dict:
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("SELECT category, filename, file_path FROM kb_chunks WHERE chunk = %s LIMIT 1", (chunk_text,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {"category": row[0], "filename": row[1], "file_path": row[2]}
    return {"category": None, "filename": None, "file_path": None}

# ============ Zip loading / cleaning / chunking ============
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
    text = re.sub(r'={3,}[\s\S]*?={3,}', '', text)
    text = re.sub(r'URL:\s*https?://\S+', '', text)
    text = re.sub(r'TITLE:.*?\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
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

def split_documents(documents: List[str], metadata: List[Dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks, metas = [], []
    for doc, meta in zip(documents, metadata):
        parts = splitter.split_text(doc)
        for p in parts:
            chunks.append(p)
            metas.append(meta.copy())
    print(f"Split into {len(chunks)} chunks (avg len {int(np.mean([len(c) for c in chunks]))} chars)")
    return chunks, metas

# ============ Embeddings (Sentence-Transformers) ============
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
    embs = embedder.encode(text_chunks, show_progress_bar=True, batch_size=32)
    embs = np.array(embs).astype("float32")
    print("Embeddings created:", embs.shape)
    return embs

# ============ FAISS: build / load / search ============
def build_faiss_index(embeddings: np.ndarray, save_dir: str = FAISS_INDEX_DIR):
    os.makedirs(save_dir, exist_ok=True)
    dim = embeddings.shape[1]
    print(f"Building FAISS Index dim={dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(save_dir, "faiss.index"))
    print("Saved FAISS index to", save_dir)
    return index

def load_faiss_index(save_dir: str = FAISS_INDEX_DIR):
    idx_path = os.path.join(save_dir, "faiss.index")
    if not os.path.exists(idx_path):
        raise FileNotFoundError("FAISS index not found. Run build first.")
    index = faiss.read_index(idx_path)
    return index

# ============ Hybrid build: build FAISS + store metadata in Postgres ============
def build_alkhidmat_rag(zip_path: str, save_dir: str = FAISS_INDEX_DIR):
    print("BUILD: loading docs...")
    docs, meta = prepare_documents(zip_path)
    if not docs:
        print("No documents. abort.")
        return
    print("BUILD: splitting...")
    chunks, chunk_meta = split_documents(docs, meta)
    print("BUILD: computing embeddings...")
    embeddings = create_embeddings(chunks)
    print("BUILD: building FAISS...")
    build_faiss_index(embeddings, save_dir=save_dir)
    print("BUILD: ensuring Postgres table...")
    ensure_pg_table()
    print("BUILD: saving metadata+embeddings to Postgres...")
    save_chunks_to_postgres(chunks, chunk_meta, embeddings)
    pickle.dump(chunks, open(os.path.join(save_dir, "chunks.pkl"), "wb"))
    pickle.dump(chunk_meta, open(os.path.join(save_dir, "metadata.pkl"), "wb"))
    print("BUILD COMPLETE ✅")

# ============ Hybrid retrieval: FAISS -> Postgres metadata ============
def hybrid_retrieve(query: str, top_k: int = 5, save_dir: str = FAISS_INDEX_DIR) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Returns (results_list, retrieved_sources_list)
    results_list: [{'text':..., 'category':..., 'filename':..., 'file_path':...}, ...]
    retrieved_sources_list: [{'category':..., 'filename':..., 'file_path':..., 'distance':...}, ...]
    """
    index = load_faiss_index(save_dir)
    chunks = pickle.load(open(os.path.join(save_dir, "chunks.pkl"), "rb"))
    embedder = get_embedder()
    # encode query in original language (multilingual embedder)
    qv = embedder.encode([query]).astype("float32")
    distances, indices = index.search(qv, top_k)
    results = []
    sources = []
    for dist, idx in zip(distances[0], indices[0]):
        chunk_text = chunks[idx]
        meta = fetch_metadata_by_chunk(chunk_text)
        results.append({
            "text": chunk_text,
            "category": meta.get("category"),
            "filename": meta.get("filename"),
            "file_path": meta.get("file_path")
        })
        sources.append({
            "category": meta.get("category"),
            "filename": meta.get("filename"),
            "file_path": meta.get("file_path"),
            "distance": float(dist)
        })
    # Print retrieval summary (visible before generation)
    print("\n" + "="*80)
    print("LOADING RAG system components... (FAISS + Postgres)")
    print("✓ Loaded FAISS index")
    print(f"✓ Loaded {len(chunks)} text chunks")
    print("\nRetrieved {} relevant chunks:".format(len(results)))
    for i, s in enumerate(sources, 1):
        print(f"{i}. [{s['category']}] {s['filename']} (distance: {s['distance']:.3f})")
    print("="*80 + "\n")
    return results, sources

# ============ LLM (GPT4All) ============
_GPT_MODEL = None
def load_llm(model_filename: str = LLM_MODEL_FILENAME):
    global _GPT_MODEL
    if _GPT_MODEL is None:
        print("Loading local LLM via GPT4All:", model_filename)
        _GPT_MODEL = GPT4All(model_filename, model_path=".", allow_download=False)
    return _GPT_MODEL

def llm_generate(prompt: str, max_tokens: int = 350) -> str:
    model = load_llm()
    resp = model.generate(prompt=prompt, max_tokens=max_tokens)
    if isinstance(resp, (list, tuple)) and len(resp) > 0:
        return str(resp[0]).strip()
    return str(resp).strip()

# ============ Answer generation (Urdu aware, no source in answer) ============
def generate_answer(query: str, top_k: int = 5, max_tokens: int = 350):
    original_query = query
    was_urdu = is_urdu(query)

    # Do not force-translate query. Use multilingual embedder to retrieve.
    results, sources = hybrid_retrieve(query, top_k=top_k)

    # If the query is Urdu, translate context chunks to Urdu so model receives Urdu context
    context_parts = []
    for i, r in enumerate(results, 1):
        chunk_text = r["text"]
        if was_urdu:
            # translate context to Urdu for better fluent Urdu generation
            try:
                chunk_text_urdu = translate_english_to_urdu(chunk_text)
            except Exception:
                chunk_text_urdu = chunk_text
            context_parts.append(f"[ماخذ {i} - {r['category']}/{r['filename']}]\n{chunk_text_urdu}")
        else:
            context_parts.append(f"[Source {i} - {r['category']}/{r['filename']}]\n{chunk_text}")
    context = "\n\n".join(context_parts)

    # Build prompt. IMPORTANT: we do NOT request the model to print sources in the answer.
    if was_urdu:
        prompt = f"""آپ الخدمت فاؤنڈیشن پاکستان کے لیے ایک مددگار اسسٹنٹ ہیں۔
مندرجہ ذیل مواد کو استعمال کریں — صرف اسی مواد کی بنیاد پر جواب دیں۔ اگر جواب مواد میں موجود نہیں تو "مجھے معلوم نہیں" کہیں۔

سیاق و سباق:
{context}

سوال:
{query}

براہ کرم ایک مختصر جواب دیں (2-4 جملے) اور اگر ممکن ہو تو 2-3 مختصر نقاط میں خلاصہ کریں۔
جواب صرف اردو میں دیں، اور ماخذ (sources) متن میں شامل نہ کریں۔
جواب:
"""
    else:
        prompt = f"""You are a helpful assistant for Alkhidmat Foundation Pakistan.
Use only the context provided to answer the question. If the answer is not contained in the context, say "I don't know".

Context:
{context}

Question:
{query}

Please provide:
1) A short answer (2-4 sentences).
2) A concise 2-3 bullet summary if applicable.

Answer:
"""
    raw = llm_generate(prompt, max_tokens=max_tokens)
    answer = raw.strip()

    # Do NOT append source attributions to answer. Instead return structured sources separately.
    return answer, original_query, was_urdu, sources

# ============ CLI helpers (printing + batch JSON output) ============
def query_alkhidmat_rag(query: str):
    answer, _, was_urdu, _ = generate_answer(query)
    print("\n" + "="*80)
    print("QUESTION:", query)
    print("="*80)
    print("ANSWER:\n", answer)
    print("\n" + "="*80)
    return answer

def save_answer_to_txt(output_path: str, query: str, answer: str, retrieved_sources: List[Dict]):
    """
    Writes the answer in a clean TXT format (not JSON).
    """

    lines = []
    lines.append("Query:")
    lines.append(query)
    lines.append("\nAnswer:")
    lines.append(answer)
    lines.append("\nRetrieved Sources:")

    for i, src in enumerate(retrieved_sources, 1):
        lines.append(
            f"{i}. {src['category']} — {src['filename']} — {src['file_path']} — distance: {src['distance']:.3f}"
        )

    lines.append("\n" + "-" * 60 + "\n")

    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

def batch_query_file(infile: str, outfile: str):
    os.environ["BATCH_MODE"] = "True"

    # Clear old file content before writing fresh results
    open(outfile, "w", encoding="utf-8").close()

    with open(infile, "r", encoding="utf-8") as f:
        queries = [l.strip() for l in f if l.strip()]

    count = 0
    for q in queries:
        answer, orig, was_ur, sources = generate_answer(q)

        # Save formatted text for each query
        save_answer_to_txt(outfile, orig, answer, sources)

        count += 1

    os.environ["BATCH_MODE"] = "False"
    print(f"Wrote {count} formatted answers to {outfile}")

# ============ Main ============
if __name__ == "__main__":
    import sys
    DEFAULT_ZIP = "Al Khidmat Knowledge Base.zip"
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        zip_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZIP
        build_alkhidmat_rag(zip_path, save_dir=FAISS_INDEX_DIR)
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What donation methods does Alkhidmat accept?"
        query_alkhidmat_rag(q)
    elif len(sys.argv) > 1 and sys.argv[1] == "file_query":
        infile = sys.argv[2] if len(sys.argv) > 2 else "urdu_queries.txt"
        outfile = sys.argv[3] if len(sys.argv) > 3 else "output_answers.json"
        batch_query_file(infile, outfile)
    else:
        print("Usage:")
        print("  python RAG_pgvector.py build [zip_path]")
        print("  python RAG_pgvector.py query 'your question'")
        print("  python RAG_pgvector.py file_query input.txt output.json")
