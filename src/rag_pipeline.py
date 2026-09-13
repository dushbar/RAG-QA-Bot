"""Orchestrates retrieval + grounded generation. This is the one class the
CLI and the Streamlit app both call."""
from typing import Any, Dict, List

from .embeddings import LocalEmbedder
from .llm import get_llm_provider
from .vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are a precise, honest assistant answering questions using ONLY the "
    "context excerpts provided below, which come from the user's own documents.\n"
    "Rules:\n"
    "- Answer strictly from the given context. Never use outside knowledge.\n"
    "- If the context doesn't contain the answer, say so plainly instead of guessing.\n"
    "- Be concise and directly answer the question.\n"
    "- Do not repeat the context verbatim; synthesize it into a clear answer."
)


def _build_user_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        label = c["metadata"].get("source", "unknown")
        blocks.append(f"[{label}]\n{c['text']}")
    context = "\n\n---\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


class RAGPipeline:
    def __init__(self, persist_dir: str, collection_name: str, top_k: int = 5, embedder=None, llm=None):
        # `embedder`/`llm` params exist mainly so tests can inject fakes.
        self.embedder = embedder or LocalEmbedder()
        self.store = VectorStore(persist_dir, collection_name)
        self.llm = llm or get_llm_provider()
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_query(question)
        results = self.store.query(query_embedding, top_k or self.top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [{"text": t, "metadata": m, "distance": d} for t, m, d in zip(docs, metas, dists)]

    def answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        chunks = self.retrieve(question, top_k)
        if not chunks:
            return {
                "answer": "This collection is empty — ingest some documents first.",
                "sources": [],
                "chunks": [],
            }
        prompt = _build_user_prompt(question, chunks)
        answer_text = self.llm.generate(SYSTEM_PROMPT, prompt)
        sources = sorted({c["metadata"].get("source", "unknown") for c in chunks})
        return {"answer": answer_text, "sources": sources, "chunks": chunks}
