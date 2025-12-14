import os

def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs

import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

from load_data import load_documents
from clean_data import clean_text

def prepare_docs(folder_path="data/"):
    """
    Loads and cleans all text documents from the given folder.
    """
    # Load Documents
    raw_docs = load_documents(folder_path)

    # Clean Documents
    cleaned_docs = [clean_text(doc) for doc in raw_docs]

    print(f"Prepared {len(cleaned_docs)} documents.")
    return cleaned_docs

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, chunk_size=500, chunk_overlap=100):
 
   # define the splitter
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=chunk_size,
       chunk_overlap=chunk_overlap
   )

   # use the splitter to split docs into chunks
   chunks = splitter.create_documents(documents)
   print(f"Total chunks created: {len(chunks)}")

   return chunks
 
 from sentence_transformers import SentenceTransformer
import numpy as np

def get_embeddings(text_chunks):
  
   # Load embedding model
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  
   print(f"Creating embeddings for {len(text_chunks)} chunks:")
   embeddings = model.encode(text_chunks, show_progress_bar=True)
  
   print(f"Embeddings shape: {embeddings.shape}")
   return np.array(embeddings)

import faiss
import numpy as np
import pickle

def build_faiss_index(embeddings, save_path="faiss_index"):
   """
   Builds FAISS index and saves it.
   """
   dim = embeddings.shape[1]
   print(f"Building FAISS index with dimension: {dim}")

   # Use a simple flat L2 index
   index = faiss.IndexFlatL2(dim)
   index.add(embeddings.astype('float32'))

   # Save FAISS index
   faiss.write_index(index, f"{save_path}.index")
   print(f"Saved FAISS index to {save_path}.index")

   return index

def save_metadata(text_chunks, path="faiss_metadata.pkl"):
   """
   Saves the mapping of vector positions to text chunks.
   """
   with open(path, "wb") as f:
       pickle.dump(text_chunks, f)
   print(f"Saved text metadata to {path}")
   
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path="faiss_index.index"):
    """
    Loads the saved FAISS index from disk.
    """
    print("Loading FAISS index.")
    return faiss.read_index(index_path)

def load_metadata(metadata_path="faiss_metadata.pkl"):
    """
    Loads text chunk metadata (the actual text pieces).
    """
    print("Loading text metadata.")
    with open(metadata_path, "rb") as f:
        return pickle.load(f)
    
def retrieve_similar_chunks(query, index, text_chunks, top_k=3):
    """
    Retrieves top_k most relevant chunks for a given query.
  
    Parameters:
        query (str): The user's input question.
        index (faiss.Index): FAISS index object.
        text_chunks (list): Original text chunks.
        top_k (int): Number of top results to return.
  
    Returns:
        list: Top matching text chunks.
    """
  
    # Embed the query
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Ensure query vector is float32 as required by FAISS
    query_vector = model.encode([query]).astype('float32')
  
    # Search FAISS for nearest vectors
    distances, indices = index.search(query_vector, top_k)
  
    print(f"Retrieved top {top_k} similar chunks.")
    return [text_chunks[i] for i in indices[0]]
 
 
context_chunks = retrieve_similar_chunks(query, index, text_chunks, top_k=3)
context = "\n\n".join(context_chunks)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from retrieve_faiss import load_faiss_index, load_metadata, retrieve_similar_chunks

def generate_answer(query, top_k=3):
    """
    Retrieves relevant chunks and generates a final answer.
    """
    # Load FAISS index and metadata
    index = load_faiss_index()
    text_chunks = load_metadata()

    # Retrieve top relevant chunks
    context_chunks = retrieve_similar_chunks(query, index, text_chunks, top_k=top_k)
    context = "\n\n".join(context_chunks)

    # Load open-source LLM
    print("Loading LLM...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # Load tokenizer and model, using a device map for efficient loading
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Build the prompt
    prompt = f"""
    Context:
    {context}
    Question:
    {query}
    Answer:
    """

    # Generate output
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Use the correct input for model generation
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and clean up the answer, removing the original prompt
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Simple way to remove the prompt part from the output
    answer = full_text.split("Answer:")[1].strip() if "Answer:" in full_text else full_text.strip()
    
    print("\nFinal Answer:")
    print(answer)
    
    
# Data preparation
from prepare_data import prepare_docs
from split_text import split_docs

# Embedding and storage
from create_embeddings import get_embeddings
from store_faiss import build_faiss_index, save_metadata

# Retrieval and answer generation
from generate_answer import generate_answer

def run_pipeline():
    """
    Runs the full end-to-end RAG workflow.
    """
    print("\nLoad and Clean Data:")
    documents = prepare_docs("data/")
    print(f"Loaded {len(documents)} clean documents.\n")

    print("Split Text into Chunks:")
    # documents is a list of strings, but split_docs expects a list of documents
    # For this simple example where documents are small, we pass them as strings
    chunks_as_text = split_docs(documents, chunk_size=500, chunk_overlap=100)
    # In this case, chunks_as_text is a list of LangChain Document objects

    # Extract text content from LangChain Document objects
    texts = [c.page_content for c in chunks_as_text]
    print(f"Created {len(texts)} text chunks.\n")

    print("Generate Embeddings:")
    embeddings = get_embeddings(texts)
  
    print("Store Embeddings in FAISS:")
    index = build_faiss_index(embeddings)
    save_metadata(texts)
    print("Stored embeddings and metadata successfully.\n")

    print("Retrieve & Generate Answer:")
    query = "Does unsupervised ML cover regression tasks?"
    generate_answer(query)
    
if __name__ == "__main__":
    run_pipeline()
