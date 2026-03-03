import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.retriever import CurriculumRetriever
from llm.generator import CurriculumLLM

app = FastAPI(
    title="Curriculum-Native LLM API",
    description="An API that answers questions strictly from curriculum-approved content using RAG + Groq LLM.",
    version="1.0.0"
)

# Serve frontend static files (CSS, JS)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Models for request/response
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    is_in_curriculum: bool
    retrieved_contexts: list

# Global instances
retriever = None
llm = None

# Distance threshold for domain classification
DISTANCE_THRESHOLD = 1.6

@app.on_event("startup")
async def startup_event():
    """Initializes retriever and Groq LLM on API startup."""
    global retriever, llm
    print("Initializing API services...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_file = os.path.join(base_dir, "data", "curriculum_index.faiss")
    chunks_file = os.path.join(base_dir, "data", "chunks.pkl")

    if not os.path.exists(index_file) or not os.path.exists(chunks_file):
        print("WARNING: Index or chunks not found. Run preprocessing first.")
    else:
        retriever = CurriculumRetriever(index_file, chunks_file)

    llm = CurriculumLLM(model_id="llama-3.3-70b-versatile")

@app.get("/")
async def serve_frontend():
    """Serves the frontend UI."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/topics")
async def list_topics():
    """Returns the list of curriculum topics/chapters available in the system."""
    if not retriever:
        raise HTTPException(status_code=500, detail="System not initialized.")

    topics = []
    for chunk in retriever.chunks:
        lines = chunk.strip().split("\n")
        first_line = lines[0].strip()
        if first_line.startswith("Chapter") or first_line.startswith("Section"):
            topics.append(first_line)

    return {
        "available_topics": topics,
        "total_chunks": len(retriever.chunks),
        "sample_questions": [
            "What is Newton's second law of motion?",
            "Explain the laws of thermodynamics",
            "What is Ohm's law?",
            "Describe projectile motion",
            "What is the photoelectric effect?",
            "Explain refraction of light",
            "What are semiconductors?",
            "Define kinetic energy and its formula",
        ]
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    User Question → Syllabus-aware Retriever (RAG) → Prompt Constructor → Groq LLM → Final Answer
    """
    if not retriever or not llm:
        raise HTTPException(status_code=500, detail="System not fully initialized. Run preprocessing.")

    question = request.question
    contexts = retriever.retrieve(question, top_k=3)
    valid_contexts = [c for c in contexts if c["distance"] < DISTANCE_THRESHOLD]
    is_in_curriculum = len(valid_contexts) > 0

    if is_in_curriculum:
        answer = llm.generate_answer(question, valid_contexts)
    else:
        answer = (
            "This question is outside the scope of the curriculum. "
            "I can only answer questions related to the syllabus."
        )

    return QueryResponse(
        question=question,
        answer=answer,
        is_in_curriculum=is_in_curriculum,
        retrieved_contexts=[c["text"] for c in valid_contexts]
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
