"""
CodeSage API.

Endpoints:
    POST /ingest   - paste a GitHub link (or a local repo path) to index it
    POST /query    - ask a question, get an answer grounded in the repo
    GET  /health   - liveness check for Railway/uptime monitors
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph import ask
from app.indexer import RepoIndexer

app = FastAPI(
    title="CodeSage",
    description="RAG over GitHub repositories with AST chunking, hybrid retrieval, and a LangGraph agent.",
    version="0.1.0",
)

# Wide open by default so a frontend (e.g. a demo UI or Aria) can call this
# from any origin. Tighten to specific origins before handling real traffic.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    # Paste a GitHub link here (https://github.com/user/repo, a .git URL,
    # or an ssh remote) — or, if you're running this locally, a path to a
    # folder already on disk. Auto-detected either way.
    source: str
    repo_name: str | None = None


class IngestResponse(BaseModel):
    repo: str
    files_seen: int
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    repo_filter: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]


@app.get("/health")
def health():
    # Kept intentionally cheap (no model/index loading) so Railway's health
    # check passes immediately during a cold start, even while the first
    # real request is still warming up the embedding model.
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    indexer = RepoIndexer()
    try:
        result = indexer.ingest(req.source, repo_name=req.repo_name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    state = ask(req.question, repo_filter=req.repo_filter)
    return QueryResponse(
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
    )


if __name__ == "__main__":
    # Lets you also run `python app/main.py` locally; Railway uses the
    # Dockerfile CMD / railway.json startCommand instead.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
