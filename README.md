# LLM_Workshop: Curriculum-Native LLM

This is **NOT** a generic RAG app. This project is an education-focused, syllabus-aligned LLM system.

## Project Goal
To design and implement a Curriculum-Native LLM system that:
- Answers questions strictly from curriculum-approved content.
- Produces exam-oriented, step-by-step answers.
- Minimizes hallucinations using strict domain locking.
- Uses RAG (Retrieval-Augmented Generation) ONLY as a knowledge retrieval layer.
- Uses a Base LLM for reasoning and answer generation.

## Architecture
```
User Query 
 → Syllabus / Domain Classifier (via Vector DB thresholds)
 → Curriculum Retriever (FAISS + SentenceTransformers)
 → Curriculum-Native LLM (Local/OSS Language Model)
 → Structured, Exam-Oriented Answer
```

## Repository Structure

- `data/`: Contains the curriculum text and generated Faiss index/chunks.
- `preprocessing/`: Scripts to chunk and embed curriculum data (`chunk_and_embed.py`).
- `retriever/`: Logic for vector similarity search and domain restriction (`retriever.py`).
- `llm/`: The generator that formulates answers using retrieved context (`generator.py`).
- `inference/`: FastAPI application exposing the `/ask` endpoint (`api.py`).
- `evaluation/`: Scripts for evaluating retrieval accuracy and domain strictness (`basic_eval.py`).

## Tech Stack
- **Language**: Python
- **Base Model**: Open-source LLM (`gpt2` used as a quick, small placeholder; replace with LLaMA/Mistral for production accuracy)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector DB**: `faiss-cpu`
- **Backend API**: `FastAPI` + `Uvicorn`

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess the Curriculum Data
This parses `data/sample_curriculum.txt`, generates embeddings via SentenceTransformers, and builds the FAISS index.
```bash
python preprocessing/chunk_and_embed.py
```

### 3. Run the API Server
Start the FastAPI endpoint.
```bash
python inference/api.py
```

### 4. Ask a Question
The API provides an endpoint at `http://localhost:8000/ask`.

Example request (using `curl`):
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is kinematics?"}'
```

### 5. Evaluate the System
Run the evaluation script to test if the system correctly rejects out-of-domain questions (like asking how to bake a cake) and successfully accepts in-domain physics questions.
```bash
python evaluation/basic_eval.py
```

## Important Notes
- **Placeholder Weights**: The system defaults to using `gpt2` for demonstration and speed. Production deployments should modify `model_id` in `llm/generator.py` and `inference/api.py` to use a strong reasoning model.
- **Domain Strictness**: Out-of-syllabus questions are rejected automatically by analyzing the vector distance of the retrieved chunks in FAISS.
