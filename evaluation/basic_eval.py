import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.retriever import CurriculumRetriever
from llm.generator import CurriculumLLM

def evaluate_system():
    """Runs a basic evaluation on correctness and domain enforcement."""
    print("Starting evaluation...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_file = os.path.join(base_dir, "data", "curriculum_index.faiss")
    chunks_file = os.path.join(base_dir, "data", "chunks.pkl")
    
    if not os.path.exists(index_file):
        print("Index not found. Cannot evaluate. Run preprocessing first.")
        return
        
    retriever = CurriculumRetriever(index_file, chunks_file)
    llm = CurriculumLLM()  # Uses Groq with GROQ_API_KEY env var
    
    test_cases = [
        {"query": "What is kinematics?", "expected_domain": True},
        {"query": "State Newton's second law of motion.", "expected_domain": True},
        {"query": "Explain Ohm's law.", "expected_domain": True},
        {"query": "How do you bake a chocolate cake?", "expected_domain": False},
        {"query": "Who won the World Cup in 2022?", "expected_domain": False},
    ]
    
    DISTANCE_THRESHOLD = 1.3
    correct_classification = 0
    
    print("\n--- Evaluation Results ---")
    for idx, case in enumerate(test_cases):
        query = case["query"]
        expected = case["expected_domain"]
        
        contexts = retriever.retrieve(query, top_k=3)
        valid_contexts = [c for c in contexts if c["distance"] < DISTANCE_THRESHOLD]
        is_in_domain = len(valid_contexts) > 0
        
        print(f"\nTest {idx+1}: {query}")
        print(f"  Expected In-Domain: {expected}")
        print(f"  Actual In-Domain:   {is_in_domain}")
        
        if is_in_domain:
            ans = llm.generate_answer(query, valid_contexts)
            print(f"  Answer Preview: {ans[:120]}...")
        else:
            print("  Answer: Outside of scope (correctly refused).")
            
        if is_in_domain == expected:
            correct_classification += 1
            print("  Result: [PASS]")
        else:
            print("  Result: [FAIL]")
        print("-" * 40)
            
    accuracy = correct_classification / len(test_cases)
    print(f"\nDomain Classification Accuracy: {accuracy * 100:.1f}%")

if __name__ == "__main__":
    evaluate_system()
