import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

class CurriculumRetriever:
    """Retrieves relevant syllabus chunks for a given query."""
    
    def __init__(self, index_path, chunks_path, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
            
    def retrieve(self, query, top_k=2):
        """Retrieves top_k most relevant chunks for the query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 if not enough results
                results.append({
                    "text": self.chunks[idx],
                    "distance": distances[0][i]  # Distance is smaller for more relevant chunks (L2)
                })
        return results

# Example usage/testing
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_file = os.path.join(base_dir, "../data/curriculum_index.faiss")
    chunks_file = os.path.join(base_dir, "../data/chunks.pkl")
    
    if os.path.exists(index_file) and os.path.exists(chunks_file):
        retriever = CurriculumRetriever(index_file, chunks_file)
        res = retriever.retrieve("What is thermodynamics?")
        print("Retrieved context:", res)
    else:
        print("Index or chunks not found. Run preprocessing first.")
