"""
Hybrid retrieval: dense (ChromaDB embeddings) + sparse (BM25 keywords),
combined with Reciprocal Rank Fusion.

Why hybrid: dense retrieval is great at "what does this code *mean*"
(paraphrased queries, conceptual questions) but weak on exact identifiers
(a variable name, an error string, a decorator). BM25 is the opposite. RRF
combines the two rank lists without needing to normalize incomparable
similarity scores (cosine distance vs. BM25 score) — it just rewards chunks
that rank highly in *either* list, and especially both.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings
from app.indexer import _tokenize


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def citation(self) -> str:
        m = self.metadata
        return f"{m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')} ({m.get('qualified_name')})"


class HybridRetriever:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=self.embedding_fn,
        )

    def _dense_search(self, query: str, k: int, repo_filter: str | None) -> list[str]:
        """Returns chunk ids in dense-similarity rank order."""
        where = {"repo": repo_filter} if repo_filter else None
        results = self.collection.query(query_texts=[query], n_results=k, where=where)
        return results["ids"][0] if results["ids"] else []

    def _sparse_search(self, query: str, k: int, repo_filter: str | None) -> list[str]:
        """Returns chunk ids in BM25 rank order."""
        try:
            with open(settings.bm25_index_path, "rb") as f:
                state = pickle.load(f)
        except FileNotFoundError:
            return []

        bm25 = state["bm25"]
        docs = state["docs"]
        if repo_filter:
            keep_idx = [i for i, d in enumerate(docs) if d["metadata"].get("repo") == repo_filter]
        else:
            keep_idx = list(range(len(docs)))
        if not keep_idx:
            return []

        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(keep_idx, key=lambda i: scores[i], reverse=True)[:k]
        return [docs[i]["id"] for i in ranked]

    def retrieve(self, query: str, repo_filter: str | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_final

        dense_ids = self._dense_search(query, settings.top_k_dense, repo_filter)
        sparse_ids = self._sparse_search(query, settings.top_k_sparse, repo_filter)

        fused_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}

        for rank, doc_id in enumerate(dense_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            dense_ranks[doc_id] = rank + 1

        for rank, doc_id in enumerate(sparse_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            sparse_ranks[doc_id] = rank + 1

        top_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)[:top_k]
        if not top_ids:
            return []

        fetched = self.collection.get(ids=top_ids)
        by_id = {
            fetched["ids"][i]: (fetched["documents"][i], fetched["metadatas"][i])
            for i in range(len(fetched["ids"]))
        }

        results = []
        for doc_id in top_ids:
            if doc_id not in by_id:
                continue
            text, metadata = by_id[doc_id]
            results.append(RetrievedChunk(
                id=doc_id,
                text=text,
                metadata=metadata,
                dense_rank=dense_ranks.get(doc_id),
                sparse_rank=sparse_ranks.get(doc_id),
                rrf_score=fused_scores[doc_id],
            ))
        return results
