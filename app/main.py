"""
CodeSage API.

Endpoints:
    POST /ingest                 - kick off ingestion as a background job,
                                    returns a job_id immediately (202 Accepted)
    GET  /ingest/status/{job_id} - poll for that job's progress/result
    POST /query                  - ask a question, get an answer grounded in the repo
    GET  /health                 - liveness check for Railway/uptime monitors

Why /ingest is a background job:
    Cloning a repo, AST-chunking every file, and running CPU embeddings
    (sentence-transformers, no GPU on Railway) can take minutes for a
    real-sized repo. Doing that inline, inside a single HTTP request, means
    any proxy in front of the app (Railway's included) eventually gives up
    and returns 502 "Application failed to respond" — the request itself
    was still running fine, it just outlived the proxy's patience.

    So /ingest now does the minimum possible work synchronously (validate
    the request, create a job record) and hands the real work off to a
    background task, returning a job_id right away. Poll
    GET /ingest/status/{job_id} to find out when it's done.
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph import ask
from app.indexer import RepoIndexer

app = FastAPI(
    title="CodeSage",
    description="RAG over GitHub repositories with AST chunking, hybrid retrieval, and a LangGraph agent.",
    version="0.2.0",
)

# Wide open by default so a frontend (e.g. a demo UI or Aria) can call this
# from any origin. Tighten to specific origins before handling real traffic.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory job store. Fine for a single-instance deployment like this one —
# it does NOT survive a process restart/redeploy (an in-flight job is lost,
# same risk that existed before this change, just now visible as a 404 on
# /ingest/status instead of a dropped connection). If this ever needs to run
# with multiple replicas, swap this dict for Redis or a small database table
# keyed the same way.
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# RepoIndexer.__init__ loads the sentence-transformers embedding model,
# which is the slow part to construct. Build it once and reuse it across
# every ingest job — same singleton pattern app/graph.py already uses for
# the retriever/graph — instead of reloading the model on every request.
_indexer: RepoIndexer | None = None
_indexer_lock = threading.Lock()


def _get_indexer() -> RepoIndexer:
    global _indexer
    if _indexer is None:
        with _indexer_lock:
            if _indexer is None:  # re-check inside the lock (another thread may have built it first)
                _indexer = RepoIndexer()
    return _indexer


def _run_ingest_job(job_id: str, source: str, repo_name: str | None) -> None:
    """Runs on a background thread (FastAPI/Starlette executes sync
    background tasks via a threadpool, so this does not block the event
    loop or new incoming requests, including status polls for this job)."""
    with _jobs_lock:
        _jobs[job_id]["status"] = JobStatus.RUNNING
        _jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        result = _get_indexer().ingest(source, repo_name=repo_name)
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.DONE
            _jobs[job_id]["result"] = result
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:  # noqa: BLE001 - surface any failure via job status rather than losing it
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.ERROR
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


class IngestRequest(BaseModel):
    # Paste a GitHub link here (https://github.com/user/repo, a .git URL,
    # or an ssh remote) — or, if you're running this locally, a path to a
    # folder already on disk. Auto-detected either way.
    source: str
    repo_name: str | None = None


class IngestAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus


class IngestStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    source: str
    repo_name: str | None = None
    result: dict | None = None   # {"repo", "files_seen", "chunks_indexed"} once status == DONE
    error: str | None = None     # populated only if status == ERROR
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


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


@app.post("/ingest", response_model=IngestAcceptedResponse, status_code=202)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    source = req.source.strip()
    if not source:
        raise HTTPException(400, "source must not be empty")

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": JobStatus.PENDING,
            "source": source,
            "repo_name": req.repo_name,
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "finished_at": None,
        }

    background_tasks.add_task(_run_ingest_job, job_id, source, req.repo_name)
    return IngestAcceptedResponse(job_id=job_id, status=JobStatus.PENDING)


@app.get("/ingest/status/{job_id}", response_model=IngestStatusResponse)
def ingest_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No job found with id '{job_id}'")
        job = dict(job)  # shallow copy so the response reflects a stable snapshot

    return IngestStatusResponse(
        job_id=job_id,
        status=job["status"],
        source=job["source"],
        repo_name=job["repo_name"],
        result=job["result"],
        error=job["error"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        finished_at=job["finished_at"],
    )


class DeleteRepoResponse(BaseModel):
    repo: str
    chunks_deleted: int


@app.delete("/repos/{repo_name}", response_model=DeleteRepoResponse)
def delete_repo(repo_name: str):
    """Remove a previously-ingested repo from both the vector store and the
    BM25 index. Fast (no embedding work involved), so this runs inline
    rather than as a background job — unlike /ingest."""
    deleted_count = _get_indexer().delete_repo(repo_name)
    if deleted_count == 0:
        raise HTTPException(404, f"No indexed chunks found for repo '{repo_name}'")
    return DeleteRepoResponse(repo=repo_name, chunks_deleted=deleted_count)


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
