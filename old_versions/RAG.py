# ============================================================================
# ALKHIDMAT FOUNDATION KNOWLEDGE BASE RAG SYSTEM - GOOGLE DRIVE FOLDER INTEGRATION
# ============================================================================
# This RAG system pulls all files from a Google Drive folder automatically
# ============================================================================

import os
import re
import pickle
import numpy as np
import torch
import faiss
import requests
from typing import List, Dict, Tuple
from io import BytesIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Google Drive API (no auth required for public folders)
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Translation libraries
from deep_translator import GoogleTranslator
import langdetect


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
        print(f"\n🔄 Translated Query (Urdu → English):")
        print(f"   Original (Urdu): {text}")
        print(f"   Translated (English): {translated}\n")
        return translated
    except Exception as e:
        print(f"Translation error (Urdu → English): {e}")
        return text  # Return original if translation fails
    
def translate_english_to_urdu(text: str) -> str:
    """
    Translate English text to Urdu using Google Translator.
    """
    try:
        translator = GoogleTranslator(source='en', target='ur')
        translated = translator.translate(text)
        print(f"\n🔄 Translated Answer (English → Urdu):")
        print(f"   Original (English): {text[:100]}...")
        print(f"   Translated (Urdu): {translated[:100]}...\n")
        return translated
    except Exception as e:
        print(f"Translation error (English → Urdu): {e}")
        return text  # Return original if translation fails
# ============================================================================
# 1. GOOGLE DRIVE FOLDER LOADING (Public Access)
# ============================================================================

def extract_folder_id(drive_url: str) -> str:
    """
    Extract folder ID from Google Drive URL.
    Example: https://drive.google.com/drive/folders/1S0lZyog7EsXrT3ol1rKp7_C2lQ3EEQMk
    """
    patterns = [
        r'/folders/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    
    # If no pattern matches, assume the URL itself is the ID
    if re.match(r'^[a-zA-Z0-9_-]+$', drive_url):
        return drive_url
    
    raise ValueError(f"Could not extract folder ID from: {drive_url}")


def get_drive_service():
    """
    Create Drive service without explicit authentication (for public files only).
    """
    try:
        # Use no credentials. The API relies on the folder being "Anyone with the link."
        # If this succeeds, it means the public access is working.
        service = build('drive', 'v3', cache_discovery=False)
        return service
    except Exception as e:
        # We will keep the print statement here just to confirm the failure path
        print(f"Note: Unable to create Drive service: {e}") 
        return None


def list_files_in_folder_api(folder_id: str, service=None) -> List[Dict]:
    """
    List all files in a Google Drive folder using Drive API.
    Works for public folders without authentication.
    """
    if service is None:
        service = get_drive_service()
    
    if service is None:
        return []
    
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        files = results.get('files', [])
        return files
    
    except Exception as e:
        print(f"Error listing files with API: {e}")
        return []


def list_files_in_folder_http(folder_id: str) -> List[Dict]:
    """
    Alternative method: Parse shared folder page HTML to get file list.
    This works when API access is restricted.
    """
    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Failed to access folder: HTTP {response.status_code}")
        
        # Parse the HTML to extract file information
        # Note: This is a simplified approach and may need adjustment
        import json
        
        # Look for data in the page
        pattern = r'\["([a-zA-Z0-9_-]{25,})".*?"([^"]+\.txt)"'
        matches = re.findall(pattern, response.text)
        
        files = []
        for file_id, filename in matches:
            files.append({
                'id': file_id,
                'name': filename,
                'mimeType': 'text/plain'
            })
        
        return files
    
    except Exception as e:
        print(f"Error parsing folder HTML: {e}")
        return []


def download_file_content(file_id: str) -> str:
    """
    Download file content from Google Drive.
    """
    try:
        # Try direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.text
        
        # Handle confirmation token for large files
        if 'download_warning' in response.text or 'virus scan' in response.text.lower():
            # Look for confirmation link
            confirm_pattern = r'confirm=([^&"]+)'
            match = re.search(confirm_pattern, response.text)
            if match:
                confirm_token = match.group(1)
                url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                response = requests.get(url)
                if response.status_code == 200:
                    return response.text
        
        print(f"Warning: Could not download file {file_id}")
        return ""
    
    except Exception as e:
        print(f"Error downloading file {file_id}: {e}")
        return ""


def load_folder_recursively(folder_id: str, category: str = "Root") -> List[Dict]:
    """
    Load all text files from a folder and its subfolders.
    """
    print(f"\n{'='*80}")
    print(f"Scanning folder: {category}")
    print(f"{'='*80}")
    
    all_documents = []
    
    # Try API method first
    service = get_drive_service()
    files = list_files_in_folder_api(folder_id, service)
    
    # Fallback to HTTP parsing if API fails
    if not files:
        print("API method failed, trying HTTP parsing...")
        files = list_files_in_folder_http(folder_id)
    
    if not files:
        print(f"⚠️  No files found in folder. Make sure folder is public.")
        return all_documents
    
    print(f"Found {len(files)} items in folder\n")
    
    for file in files:
        file_name = file.get('name', 'unknown')
        file_id = file.get('id')
        mime_type = file.get('mimeType', '')
        
        # If it's a subfolder, recurse
        if mime_type == 'application/vnd.google-apps.folder':
            print(f"  📁 Entering subfolder: {file_name}")
            subfolder_docs = load_folder_recursively(file_id, file_name)
            all_documents.extend(subfolder_docs)
        
        # If it's a text file, download it
        elif file_name.endswith('.txt') or mime_type == 'text/plain':
            print(f"  📄 Downloading: {file_name}...", end=" ")
            content = download_file_content(file_id)
            
            if content:
                all_documents.append({
                    'content': content,
                    'filename': file_name,
                    'category': category,
                    'file_id': file_id
                })
                print(f"✓ ({len(content)} characters)")
            else:
                print("✗ Failed")
    
    return all_documents


def load_all_documents_from_main_folder(main_folder_url: str) -> Dict[str, List[Dict]]:
    """
    Load all documents from main folder and organize by subfolder.
    
    Expected structure:
    Main Folder (AI Khidmat Knowledge Base)
    ├── Donors/
    │   ├── donate.txt
    │   ├── donate-summary.txt
    │   └── way-to-donate.txt
    ├── General/
    │   └── general-info.txt
    └── Health/
        └── health-programs.txt
    
    Args:
        main_folder_url: URL or ID of the main shared folder
    
    Returns:
        Dictionary organized by category (subfolder name)
    """
    print("\n" + "="*80)
    print("LOADING FROM GOOGLE DRIVE FOLDER")
    print("="*80)
    
    folder_id = extract_folder_id(main_folder_url)
    print(f"Folder ID: {folder_id}\n")
    
    # Get all subfolders
    service = get_drive_service()
    
    try:
        # List subfolders in main folder
        if service:
            results = service.files().list(
                q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            subfolders = results.get('files', [])
        else:
            # Fallback: Assume standard structure
            subfolders = []
            print("⚠️  Using manual subfolder configuration")
            print("Please provide subfolder IDs if automatic detection fails\n")
        
        documents_by_category = {}
        
        if subfolders:
            print(f"Found {len(subfolders)} subfolders:")
            for folder in subfolders:
                print(f"  - {folder['name']}")
            print()
            
            # Load from each subfolder
            for folder in subfolders:
                category_name = folder['name']
                subfolder_id = folder['id']
                
                docs = load_folder_recursively(subfolder_id, category_name)
                documents_by_category[category_name] = docs
        else:
            # If no subfolders, load directly from main folder
            print("Loading directly from main folder...")
            docs = load_folder_recursively(folder_id, "Main")
            documents_by_category["Main"] = docs
        
        # Summary
        print(f"\n{'='*80}")
        print("LOADING COMPLETE")
        print(f"{'='*80}")
        for category, docs in documents_by_category.items():
            print(f"  {category}: {len(docs)} documents")
        print(f"{'='*80}\n")
        
        return documents_by_category
    
    except Exception as e:
        print(f"Error loading from Drive: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure folder is set to 'Anyone with the link can view'")
        print("2. Check that the folder URL is correct")
        print("3. Verify that text files exist in the subfolders")
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


def prepare_documents(main_folder_url: str) -> Tuple[List[str], List[Dict]]:
    """
    Load and clean all documents from Google Drive main folder.
    
    Returns:
        Tuple of (cleaned_texts, metadata_list)
    """
    # Load all documents
    docs_by_category = load_all_documents_from_main_folder(main_folder_url)
    
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
                    'file_id': doc['file_id']
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
    """Generate answer using retrieved context with Urdu support."""
    
    # ========== NEW CODE STARTS HERE ==========
    # Step 1: Detect language and translate if needed
    original_query = query
    query_is_urdu = is_urdu(query)
    
    if query_is_urdu:
        print(f"\n🇵🇰 Urdu query detected!")
        query = translate_urdu_to_english(query)
    # ========== NEW CODE ENDS HERE ==========
    
    # Step 2: Process with English query (existing pipeline - NO CHANGES)
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
    
    # ========== NEW CODE STARTS HERE ==========
    # Step 3: Translate answer back to Urdu if original query was Urdu
    if query_is_urdu:
        answer = translate_english_to_urdu(answer)
    # ========== NEW CODE ENDS HERE ==========
    
    # ========== MODIFIED DISPLAY CODE ==========
    # Display results
    print(f"\n{'='*80}")
    print(f"QUESTION: {original_query}")
    if query_is_urdu:
        print(f"(Translated to English: {query})")
    print(f"{'='*80}")
    print(f"\nANSWER:\n{answer}")
    print(f"\n{'='*80}\n")
    # ========== END OF MODIFIED DISPLAY ==========
    
    return answer, results
# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def build_alkhidmat_rag(main_folder_url: str, index_dir: str = "alkhidmat_index"):
    """Complete pipeline to build RAG system from single Google Drive folder URL."""
    
    print("\n" + "="*80)
    print("ALKHIDMAT FOUNDATION RAG SYSTEM - BUILD FROM GOOGLE DRIVE")
    print("with Urdu Language Support")
    print("="*80 + "\n")
    
    # Step 1: Load from Drive
    print("STEP 1: Loading documents from Google Drive folder...")
    documents, metadata = prepare_documents(main_folder_url)
    
    if not documents:
        print("\n⚠️  No documents loaded! Please check:")
        print("  1. Folder is set to 'Anyone with the link can view'")
        print("  2. Folder URL is correct")
        print("  3. Text files exist in subfolders")
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
    print("System supports both English and Urdu queries")
    print("="*80 + "\n")


def query_alkhidmat_rag(query: str, index_dir: str = "alkhidmat_index"):
    """Query the Alkhidmat RAG system."""
    return generate_answer(query, index_dir)


# ============================================================================
# 9. USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    MAIN_FOLDER_URL = "https://drive.google.com/drive/folders/1S0lZyog7EsXrT3oI1rKp7_C2lQ3EEQMk?usp=sharing"
    
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # Build from provided URL or use default
        folder_url = sys.argv[2] if len(sys.argv) > 2 else MAIN_FOLDER_URL
        build_alkhidmat_rag(folder_url, "alkhidmat_index")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        # Query mode
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What donation methods does Alkhidmat accept?"
        query_alkhidmat_rag(query)
    
    else:
        print("\n" + "="*80)
        print("ALKHIDMAT RAG SYSTEM - USAGE (with urdu support)")
        print("="*80)
        print("\nBuild from your main Google Drive folder:")
        print(f"  python rag_alkhidmat.py build {MAIN_FOLDER_URL}")
        print("\nOr build from custom folder:")
        print("  python rag_alkhidmat.py build <your_folder_url>")
        print("\nQuery the system:")
        print("  python rag_alkhidmat.py query 'How can I donate?'")
        print("\n" + "="*80 + "\n")