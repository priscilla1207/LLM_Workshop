import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Configuration
DATA_PATH = "../data/sample_curriculum.txt"
INDEX_PATH = "../data/curriculum_index.faiss"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKS_PATH = "../data/chunks.pkl"

def load_data(filepath):
    """Loads curriculum data from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks of specified size with overlap."""
    # Simple chunking by paragraphs representing individual curriculum topics.
    paragraphs = text.split('\n\n')
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks

def embed_chunks(chunks, model_name):
    """Generates embeddings for a list of text chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings, model

def build_faiss_index(embeddings):
    """Builds a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def main():
    print("Starting preprocessing...")
    
    # Ensure data directory exists based on paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, DATA_PATH)
    index_file = os.path.join(base_dir, INDEX_PATH)
    chunks_file = os.path.join(base_dir, CHUNKS_PATH)
    
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at {data_file}")
        return

    # 1. Load data
    text = load_data(data_file)
    
    # 2. Chunk data
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")
    
    # 3. Embed chunks
    print(f"Loading embedding model '{MODEL_NAME}'...")
    embeddings, _ = embed_chunks(chunks, MODEL_NAME)
    
    # 4. Build FAISS index
    index = build_faiss_index(embeddings)
    
    # 5. Save index and chunks
    faiss.write_index(index, index_file)
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunks, f)
        
    print(f"Preprocessing complete. Index saved to {index_file}")

if __name__ == "__main__":
    main()
