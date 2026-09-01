"""
Ingestion pipeline: walk a repo -> chunk files -> embed + store in ChromaDB
-> build a parallel BM25 keyword index over the same chunks.

Both indexes are keyed by the same chunk `id` so results can be merged
downstream with reciprocal rank fusion (see retriever.py).
"""
from __future__ import annotations

import os
import pickle
import re
import shutil
import subprocess
import tempfile
import uuid

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from app.chunker import chunk_file
from app.config import settings


def is_github_url(source: str) -> bool:
    """True for anything that looks like a git remote rather than a local
    path — github.com/gitlab.com links, any .git URL, or an ssh remote."""
    source = source.strip()
    return bool(
        re.match(r"^(https?://|git@)", source) or source.endswith(".git")
    )


def _clone_repo(repo_url: str) -> str:
    """Shallow-clones repo_url into a fresh temp dir and returns that path.
    Caller is responsible for cleaning it up afterwards."""
    tmp_dir = os.path.join(tempfile.gettempdir(), f"codesage_{uuid.uuid4().hex[:8]}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, tmp_dir],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"git clone failed for '{repo_url}': {e.stderr.strip()}")
    return tmp_dir


def _default_repo_name(source: str) -> str:
    cleaned = source.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    return os.path.basename(cleaned) or cleaned


def _iter_repo_files(repo_path: str):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in settings.ignored_dirs and not d.startswith(".")]
        for fname in files:
            if fname.endswith(settings.supported_extensions) or fname in ("README.md", "readme.md"):
                yield os.path.join(root, fname)


def _tokenize(text: str) -> list[str]:
    """Lightweight tokenizer for BM25 — splits on non-alphanumerics and
    lowercases, which is enough for code identifiers and prose alike."""
    import re
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


class RepoIndexer:
    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        os.makedirs(os.path.dirname(settings.bm25_index_path) or ".", exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=self.embedding_fn,
        )

    def ingest(self, source: str, repo_name: str | None = None) -> dict:
        """Single entry point for ingestion: pass either a pasted GitHub
        link (https://github.com/user/repo, a .git URL, or an ssh remote)
        or a local folder path — it's auto-detected and handled either way.

        This is what both the CLI script and the /ingest API route call, so
        pasting a link behaves identically everywhere in the project."""
        source = source.strip()
        if not source:
            raise ValueError("Please paste a GitHub link or a local repo path.")

        if is_github_url(source):
            repo_name = repo_name or _default_repo_name(source)
            tmp_dir = _clone_repo(source)
            try:
                return self.index_repo(tmp_dir, repo_name=repo_name)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if not os.path.isdir(source):
            raise ValueError(
                f"'{source}' doesn't look like a GitHub link or an existing local folder."
            )
        return self.index_repo(source, repo_name=repo_name)

    def index_repo(self, repo_path: str, repo_name: str | None = None) -> dict:
        """Chunk every supported file in repo_path and write it into both
        the vector store and the BM25 index. Returns a small ingestion report."""
        repo_name = repo_name or os.path.basename(os.path.normpath(repo_path))
        all_docs = []
        files_seen = 0

        for file_path in _iter_repo_files(repo_path):
            files_seen += 1
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except (UnicodeDecodeError, OSError):
                continue

            rel_path = os.path.relpath(file_path, repo_path)
            for chunk in chunk_file(rel_path, source):
                doc = chunk.to_document()
                doc["metadata"]["repo"] = repo_name
                all_docs.append(doc)

        if not all_docs:
            return {"repo": repo_name, "files_seen": files_seen, "chunks_indexed": 0}

        # Delete this repo's previously-indexed chunks first. upsert() only
        # adds/updates by id — it never removes chunks a re-ingest no longer
        # produces, so without this, a stale chunk from an older chunking
        # strategy (e.g. plain-text blocks from before AST chunking was
        # added) lingers forever alongside the new ones and pollutes results.
        self._delete_repo(repo_name)

        self._upsert_in_batches(all_docs)

        self._update_bm25_index(all_docs, repo_name=repo_name)

        return {
            "repo": repo_name,
            "files_seen": files_seen,
            "chunks_indexed": len(all_docs),
        }

    def _delete_repo(self, repo_name: str) -> None:
        """Remove every chunk currently indexed under this repo name from
        ChromaDB, so a re-ingest starts clean rather than accumulating stale
        chunks from a previous chunking strategy or a deleted/renamed file."""
        try:
            self.collection.delete(where={"repo": repo_name})
        except Exception:
            pass  # nothing indexed yet for this repo — fine, nothing to delete

    def _upsert_in_batches(self, docs: list[dict]) -> None:
        """ChromaDB rejects a single upsert call above its configured max
        batch size (hit this at 438 chunks on a repo with many small
        TypeScript/JS files — Python repos rarely produce enough chunks per
        file to trip this). Query the client's actual limit when available
        and fall back to a conservative constant otherwise."""
        try:
            max_batch_size = self.chroma_client.get_max_batch_size()
        except Exception:
            max_batch_size = 100

        for start in range(0, len(docs), max_batch_size):
            batch = docs[start:start + max_batch_size]
            self.collection.upsert(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
            )

    def _update_bm25_index(self, new_docs: list[dict], repo_name: str) -> None:
        """BM25 (rank_bm25) has no incremental-update API, so we merge new
        chunks with whatever was previously persisted and rebuild in memory.
        Fine for repo-scale corpora; swap for a proper inverted-index store
        (e.g. Whoosh/Elasticsearch) if this needs to scale past ~100k chunks.

        Old chunks belonging to this same repo_name are dropped before
        merging — same reasoning as _delete_repo() above, so a re-ingest
        with a changed chunking strategy doesn't leave stale BM25 entries."""
        existing = self._load_bm25_state()
        kept = [d for d in existing.get("docs", []) if d["metadata"].get("repo") != repo_name]
        docs = kept + new_docs

        tokenized_corpus = [_tokenize(d["text"]) for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(settings.bm25_index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "docs": docs}, f)

    @staticmethod
    def _load_bm25_state() -> dict:
        if not os.path.exists(settings.bm25_index_path):
            return {"docs": []}
        with open(settings.bm25_index_path, "rb") as f:
            return pickle.load(f)
