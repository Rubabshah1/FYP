# ============================================================================
# ALKHIDMAT FOUNDATION KNOWLEDGE BASE RAG SYSTEM - ZIP FILE VERSION
# ============================================================================
# This RAG system reads from a ZIP file with the following structure:
# Al Khidmat Knowledge Base.zip
#   └── Al Khidmat Knowledge Base/
#       ├── Donors/
#       │   └── *.txt files
#       ├── General/
#       │   └── *.txt files
#       └── Health/
#           └── *.txt files
# ============================================================================

import os
import re
import pickle
import numpy as np
import torch
import faiss
import zipfile
from typing import List, Dict, Tuple
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 1. ZIP FILE LOADING
# ============================================================================

def load_documents_from_zip(zip_path: str) -> Dict[str, List[Dict]]:
    """
    Load all documents from ZIP file and organize by subfolder.
    
    Expected structure:
    Al Khidmat Knowledge Base.zip
      └── Al Khidmat Knowledge Base/
          ├── Donors/
          ├── General/
          └── Health/
    
    Args:
        zip_path: Path to the ZIP file
    
    Returns:
        Dictionary organized by category (subfolder name)
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
            # Get all file names in the ZIP
            all_files = zip_ref.namelist()
            
            print(f"Total files in ZIP: {len(all_files)}\n")
            
            # Process each file
            for file_path in all_files:
                # Skip directories and non-txt files
                if file_path.endswith('/') or not file_path.endswith('.txt'):
                    continue
                
                # Parse the path structure
                # Expected: "Al Khidmat Knowledge Base/Category/filename.txt"
                path_parts = Path(file_path).parts
                
                # Skip if not enough depth (need at least root/category/file.txt)
                if len(path_parts) < 3:
                    continue
                
                # Extract category (the folder name before the file)
                category = path_parts[-2]
                filename = path_parts[-1]
                
                # Read file content
                print(f"  📄 Reading: [{category}] {filename}...", end=" ")
                
                try:
                    with zip_ref.open(file_path) as f:
                        content = f.read().decode('utf-8')
                    
                    if content:
                        # Add to category
                        if category not in documents_by_category:
                            documents_by_category[category] = []
                        
                        documents_by_category[category].append({
                            'content': content,
                            'filename': filename,
                            'category': category,
                            'file_path': file_path
                        })
                        print(f"✓ ({len(content)} characters)")
                    else:
                        print("⚠️  Empty file")
                
                except Exception as e:
                    print(f"✗ Error: {e}")
            
            # Summary
            print(f"\n{'='*80}")
            print("LOADING COMPLETE")
            print(f"{'='*80}")
            
            if documents_by_category:
                for category, docs in documents_by_category.items():
                    print(f"  {category}: {len(docs)} documents")
                print(f"{'='*80}\n")
            else:
                print("⚠️  No documents found!")
                print("\nExpected ZIP structure:")
                print("  Al Khidmat Knowledge Base.zip")
                print("    └── Al Khidmat Knowledge Base/")
                print("        ├── Donors/")
                print("        │   └── *.txt")
                print("        ├── General/")
                print("        │   └── *.txt")
                print("        └── Health/")
                print("            └── *.txt")
                print(f"{'='*80}\n")
            
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
    # Remove URL metadata blocks
    text = re.sub(r'={3,}[\s\S]*?={3,}', '', text)
    text = re.sub(r'URL:\s*https?://\S+', '', text)
    text = re.sub(r'TITLE:.*?\n', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def prepare_documents(zip_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Load and clean all documents from ZIP file.
    
    Returns:
        Tuple of (cleaned_texts, metadata_list)
    """
    # Load all documents
    docs_by_category = load_documents_from_zip(zip_path)
    
    # Flatten and clean
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
    
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Average chunk length: {np.mean([len(c) for c in all_chunks]):.0f} characters\n")
    
    return all_chunks, all_metadata


# ============================================================================
# 4. EMBEDDING CREATION
# ============================================================================

def create_embeddings(text_chunks: List[str], 
                     model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> np.ndarray:
    """Generate embeddings for text chunks."""
    print("Loading embedding model...")
    model = SentenceTransformer(model_name)
    
    print(f"Creating embeddings for {len(text_chunks)} chunks...")
    embeddings = model.encode(text_chunks, show_progress_bar=True, batch_size=32)
    
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
    print(f"Building FAISS index with dimension: {dim}")
    
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    
    # Save components
    index_path = os.path.join(save_dir, "faiss_index.index")
    faiss.write_index(index, index_path)
    print(f"✓ Saved FAISS index to {index_path}")
    
    chunks_path = os.path.join(save_dir, "text_chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(text_chunks, f)
    print(f"✓ Saved text chunks to {chunks_path}")
    
    metadata_path = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata to {metadata_path}\n")


# ============================================================================
# 6. RETRIEVAL FUNCTIONS
# ============================================================================

def load_index_and_data(index_dir: str = "alkhidmat_index"):
    """Load FAISS index, chunks, and metadata."""
    print("Loading RAG system components...")
    
    index_path = os.path.join(index_dir, "faiss_index.index")
    chunks_path = os.path.join(index_dir, "text_chunks.pkl")
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    
    index = faiss.read_index(index_path)
    print(f"✓ Loaded FAISS index")
    
    with open(chunks_path, "rb") as f:
        text_chunks = pickle.load(f)
    print(f"✓ Loaded {len(text_chunks)} text chunks")
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    print(f"✓ Loaded metadata\n")
    
    return index, text_chunks, metadata


def retrieve_context(query: str, index, text_chunks: List[str], 
                    metadata: List[Dict], top_k: int = 5,
                    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """Retrieve relevant context with source attribution."""
    
    model = SentenceTransformer(model_name)
    query_vector = model.encode([query]).astype('float32')
    
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            'text': text_chunks[idx],
            'category': metadata[idx]['category'],
            'filename': metadata[idx]['filename'],
            'distance': float(dist)
        })
    
    print(f"\n{'='*80}")
    print(f"Retrieved {len(results)} relevant chunks:")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['category']}] {result['filename']} (distance: {result['distance']:.3f})")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# 7. ANSWER GENERATION
# ============================================================================

def generate_answer(query: str, index_dir: str = "alkhidmat_index", 
                   top_k: int = 5, max_tokens: int = 250):
    """Generate answer using retrieved context."""
    
    index, text_chunks, metadata = load_index_and_data(index_dir)
    results = retrieve_context(query, index, text_chunks, metadata, top_k)
    
    # Build context
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Source {i} - {result['category']}/{result['filename']}]\n{result['text']}"
        )
    context = "\n\n".join(context_parts)
    
    # Load LLM
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
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("Answer:")[-1].strip() if "Answer:" in full_text else full_text[len(prompt):].strip()
    
    print(f"\n{'='*80}")
    print(f"QUESTION: {query}")
    print(f"{'='*80}")
    print(f"\nANSWER:\n{answer}")
    print(f"\n{'='*80}\n")
    
    return answer, results


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def build_alkhidmat_rag(zip_path: str, index_dir: str = "alkhidmat_index"):
    """Complete pipeline to build RAG system from ZIP file."""
    
    print("\n" + "="*80)
    print("ALKHIDMAT FOUNDATION RAG SYSTEM - BUILD FROM ZIP FILE")
    print("="*80 + "\n")
    
    # Step 1: Load from ZIP
    print("STEP 1: Loading documents from ZIP file...")
    documents, metadata = prepare_documents(zip_path)
    
    if not documents:
        print("\n⚠️  No documents loaded! Please check:")
        print("  1. ZIP file exists at the specified path")
        print("  2. ZIP structure matches expected format")
        print("  3. Text files exist in category folders")
        return
    
    # Step 2: Split into chunks
    print("\nSTEP 2: Splitting documents into chunks...")
    text_chunks, chunk_metadata = split_documents(documents, metadata, 500, 100)
    
    # Step 3: Create embeddings
    print("\nSTEP 3: Creating embeddings...")
    embeddings = create_embeddings(text_chunks)
    
    # Step 4: Build and save index
    print("\nSTEP 4: Building and saving FAISS index...")
    build_and_save_index(embeddings, text_chunks, chunk_metadata, index_dir)
    
    print("\n" + "="*80)
    print("✓ RAG SYSTEM BUILD COMPLETE!")
    print("="*80 + "\n")


def query_alkhidmat_rag(query: str, index_dir: str = "alkhidmat_index"):
    """Query the Alkhidmat RAG system."""
    return generate_answer(query, index_dir)


# ============================================================================
# 9. USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    DEFAULT_ZIP_PATH = "Al Khidmat Knowledge Base.zip"
    
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # Build from provided ZIP path or use default
        zip_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZIP_PATH
        build_alkhidmat_rag(zip_path, "alkhidmat_index")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        # Query mode
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What donation methods does Alkhidmat accept?"
        query_alkhidmat_rag(query)
    
    else:
        print("\n" + "="*80)
        print("ALKHIDMAT RAG SYSTEM - USAGE")
        print("="*80)
        print("\nBuild from ZIP file:")
        print(f"  python rag_alkhidmat.py build")
        print(f"  (Uses default: {DEFAULT_ZIP_PATH})")
        print("\nOr specify custom ZIP path:")
        print("  python rag_alkhidmat.py build /path/to/your/knowledge_base.zip")
        print("\nQuery the system:")
        print("  python rag_alkhidmat.py query 'How can I donate?'")
        print("\n" + "="*80 + "\n")