import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import openai
from functools import lru_cache

# Config
OPENAI_API_KEY = "sk-..."
openai.api_key = OPENAI_API_KEY

app = FastAPI(title="Semantic Search with Re-ranking")

# Mock 88 API documentation documents
doc_store = [
    {
        "id": idx,
        "content": f"API endpoint {idx}: Authentication and authorization methods. This document covers OAuth2, API keys, JWT tokens, and session management for securing API requests.",
        "metadata": {"source": f"https://api.example.com/docs/{idx}", "category": "authentication"}
    } if idx % 3 == 0 else {
        "id": idx,
        "content": f"API endpoint {idx}: Rate limiting and throttling policies. Details on request quotas, burst limits, and backoff strategies for API consumers.",
        "metadata": {"source": f"https://api.example.com/docs/{idx}", "category": "rate-limiting"}
    } if idx % 3 == 1 else {
        "id": idx,
        "content": f"API endpoint {idx}: Error handling and response codes. Comprehensive guide to HTTP status codes, error messages, and debugging techniques.",
        "metadata": {"source": f"https://api.example.com/docs/{idx}", "category": "errors"}
    }
    for idx in range(88)
]

embeddings_cache = {}

def get_embedding(text: str) -> np.ndarray:
    if text in embeddings_cache:
        return embeddings_cache[text]
    try:
        emb = openai.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )['data'][0]['embedding']
        emb = np.array(emb, dtype=np.float32)
        embeddings_cache[text] = emb
        return emb
    except Exception as e:
        raise HTTPException(500, f"Embedding error: {e}")

# Precompute document embeddings at startup
for doc in doc_store:
    doc['embedding'] = get_embedding(doc['content'])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = True
    rerankK: int = 3

class SearchResult(BaseModel):
    id: int
    score: float
    content: str
    metadata: Dict

class SearchResponse(BaseModel):
    results: List[SearchResult]
    reranked: bool
    metrics: Dict

def batch_llm_rerank(query: str, candidates: List[Dict], model="gpt-3.5-turbo") -> List[float]:
    scores = []
    for doc in candidates:
        prompt = f'''Query: "{query}"
Document: "{doc['content'][:500]}"
\nRate the relevance of this document to the query on a scale of 0-10.\nRespond with only the number.'''  
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4
            )
            score_text = response.choices[0].message.content.strip()
            s = float(score_text)
            s = max(0, min(10, s))
            scores.append(s / 10.0)
        except Exception as e:
            scores.append(0.0)
    return scores

@app.post("/semantic-search", response_model=SearchResponse)
def semantic_search(req: SearchRequest):
    t0 = time.time()
    
    try:
        query_emb = get_embedding(req.query)
    except Exception as e:
        raise HTTPException(500, f"Query embedding error: {e}")

    doc_scores = []
    for doc in doc_store:
        score = cosine_similarity(query_emb, doc['embedding'])
        doc_scores.append((doc['id'], score, doc))

    top_candidates = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:req.k]
    result_docs = [x[2] for x in top_candidates]
    sim_scores = [x[1] for x in top_candidates]

    reranked = False

    if req.rerank and len(result_docs) > 0:
        try:
            rerank_scores = batch_llm_rerank(req.query, result_docs)
            scored_results = []
            for idx, doc in enumerate(result_docs):
                scored_results.append({
                    "id": doc["id"],
                    "score": float(np.clip(rerank_scores[idx], 0, 1)),
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                })
            scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)[:req.rerankK]
            reranked = True
        except Exception as e:
            scored_results = []
            for idx, doc in enumerate(result_docs):
                score = float(np.clip((sim_scores[idx] + 1) / 2, 0, 1))
                scored_results.append({
                    "id": doc["id"],
                    "score": score,
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                })
    else:
        scored_results = []
        for idx, doc in enumerate(result_docs):
            score = float(np.clip((sim_scores[idx] + 1) / 2, 0, 1))
            scored_results.append({
                "id": doc["id"],
                "score": score,
                "content": doc["content"],
                "metadata": doc["metadata"]
            })

    scored_results = sorted(scored_results, key=lambda x: (x["score"], -x["id"]), reverse=True)
    latency_ms = int((time.time() - t0) * 1000)
    
    response = {
        "results": scored_results,
        "reranked": reranked,
        "metrics": {
            "latency": latency_ms,
            "totalDocs": len(doc_store)
        }
    }
    return response

@app.get("/health")
def health_check():
    return {"status": "healthy", "totalDocs": len(doc_store)}

@app.get("/")
def root():
    return {"message": "Semantic Search with Re-ranking API", "endpoint": "/semantic-search", "health": "/health"}