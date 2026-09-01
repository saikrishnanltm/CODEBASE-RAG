"""
Central configuration for CodeSage.

All values can be overridden via environment variables (see .env.example).
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env into the process environment BEFORE any os.getenv() calls below
# run — without this, .env is never actually read and every setting silently
# falls back to its default (this is why GROQ_API_KEY was coming back empty).
load_dotenv()

# Disable ChromaDB's anonymous telemetry. Harmless when it fails (just a
# noisy "Failed to send telemetry event" line), but distracting in a live
# demo. Set here in config.py — not indexer.py — because both the ingest
# path (via indexer.py) and the query path (via retriever.py) import this
# module first, so this is the one place that covers both.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@dataclass
class Settings:
    # Embeddings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Vector store
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION", "codesage_chunks")

    # BM25 index location (pickled on disk so it survives restarts)
    bm25_index_path: str = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")

    # Chunking
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "1500"))
    min_chunk_chars: int = int(os.getenv("MIN_CHUNK_CHARS", "40"))

    # Retrieval
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "10"))
    top_k_sparse: int = int(os.getenv("TOP_K_SPARSE", "10"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "6"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))  # standard RRF damping constant

    # Generation (Groq, matching kk's existing stack)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Repo ingestion
    # AST chunking applies to: .py (Python's own `ast` module), .ipynb
    # (parsed cell-by-cell, with code cells re-using the .py AST chunker —
    # see chunk_notebook_file() in chunker.py), and
    # .js/.jsx/.ts/.tsx/.java/.html/.htm/.css/.scss (tree-sitter grammars,
    # see ts_chunker.py). Everything else here still gets walked and
    # indexed via the plain-text sliding-window fallback in chunker.py.
    supported_extensions: tuple = (
        ".py",
        ".ipynb",
        ".ts", ".tsx",
        ".js", ".jsx",
        ".java",
        ".html", ".htm",
        ".css", ".scss",
        ".json", ".md",
    )
    ignored_dirs: tuple = (
        ".git", "__pycache__", "node_modules", "venv", ".venv",
        "dist", "build", ".next", "coverage",
    )


settings = Settings()
