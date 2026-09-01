"""
LangGraph agentic pipeline for CodeSage.

Four nodes, same shape as the OrbitDesk pipeline this reuses patterns from:

    triage -> retrieve -> generate -> verify

- triage:   classify the query (needs code lookup vs. answerable from
            conversation alone) and lightly rewrite it for retrieval.
- retrieve: hybrid dense+sparse search via HybridRetriever (RRF fusion).
- generate: call the LLM (Groq) grounded on retrieved chunks, with citations.
- verify:   check the answer actually cites real retrieved chunks; if it
            hallucinated a citation or came back empty, loop back to
            retrieve once with a broadened query before giving up.
"""
from __future__ import annotations

import os
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.config import settings
from app.retriever import HybridRetriever, RetrievedChunk


class AgentState(TypedDict, total=False):
    query: str
    repo_filter: str | None
    search_query: str
    needs_retrieval: bool
    retrieved: list[RetrievedChunk]
    answer: str
    citations: list[str]
    verified: bool
    retry_count: int


_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def triage_node(state: AgentState) -> AgentState:
    query = state["query"].strip()
    # Simple heuristic triage: greetings/meta questions skip retrieval.
    trivial_patterns = ("hi", "hello", "thanks", "who are you", "what can you do")
    needs_retrieval = not any(query.lower().startswith(p) for p in trivial_patterns)
    return {
        **state,
        "needs_retrieval": needs_retrieval,
        "search_query": query,
        "retry_count": state.get("retry_count", 0),
    }


def retrieve_node(state: AgentState) -> AgentState:
    if not state.get("needs_retrieval", True):
        return {**state, "retrieved": []}

    retriever = _get_retriever()
    chunks = retriever.retrieve(
        query=state["search_query"],
        repo_filter=state.get("repo_filter"),
        top_k=settings.top_k_final,
    )
    return {**state, "retrieved": chunks}


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] {c.citation}\n{c.text}")
    return "\n\n".join(blocks)


def generate_node(state: AgentState) -> AgentState:
    chunks = state.get("retrieved", [])

    if not chunks and state.get("needs_retrieval", True):
        return {
            **state,
            "answer": "I couldn't find anything relevant in the indexed repo for that question.",
            "citations": [],
        }

    if not state.get("needs_retrieval", True):
        answer = _call_llm(
            system="You are CodeSage, an assistant for exploring a codebase.",
            user=state["query"],
        )
        return {**state, "answer": answer, "citations": []}

    context = _build_context_block(chunks)
    system = (
        "You are CodeSage, a code Q&A assistant. Answer ONLY using the numbered "
        "code excerpts below. Cite the excerpt number(s) you used like [1], [2]. "
        "If the excerpts don't contain the answer, say so plainly instead of guessing."
    )
    user = f"Question: {state['query']}\n\nCode excerpts:\n{context}"
    answer = _call_llm(system=system, user=user)

    citations = [c.citation for c in chunks]
    return {**state, "answer": answer, "citations": citations}


def verify_node(state: AgentState) -> AgentState:
    """Guard against hallucinated citations: if the model cited a bracket
    number that doesn't exist in the retrieved set, or produced an empty
    answer while chunks *were* available, retry retrieval once with a
    broadened query before surfacing the (unverified) answer as a fallback."""
    answer = state.get("answer", "")
    chunks = state.get("retrieved", [])
    retry_count = state.get("retry_count", 0)

    cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    valid_range = set(range(1, len(chunks) + 1))
    hallucinated = bool(cited_numbers - valid_range) if chunks else False
    empty = not answer.strip()

    if (hallucinated or empty) and chunks and retry_count < 1:
        broadened = " ".join(state["search_query"].split()[:3]) or state["search_query"]
        return {**state, "search_query": broadened, "retry_count": retry_count + 1, "verified": False}

    return {**state, "verified": True}


def _route_after_verify(state: AgentState) -> str:
    return "retrieve" if not state.get("verified", True) else END


def _call_llm(system: str, user: str) -> str:
    """Thin wrapper around Groq's OpenAI-compatible chat completion API.
    Kept isolated so it's trivial to swap providers or mock in tests."""
    if not settings.groq_api_key:
        return (
            "[No GROQ_API_KEY set — returning retrieved context only]\n\n"
            f"{user[:2000]}"
        )
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", _route_after_verify, {"retrieve": "retrieve", END: END})

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(query: str, repo_filter: str | None = None) -> AgentState:
    graph = get_graph()
    return graph.invoke({"query": query, "repo_filter": repo_filter})
