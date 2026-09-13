"""Local embeddings via sentence-transformers.

Deliberately NOT calling an embeddings API: ingesting a client's whole doc
set costs $0 and no data leaves the machine at embedding time (only the
final question + retrieved chunks go to the LLM). First run downloads the
model (~80MB) from Hugging Face and caches it locally.
"""
from typing import List

_DEFAULT_MODEL = "all-MiniLM-L6-v2"  # fast, 384-dim, good default for most doc Q&A


class LocalEmbedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # imported lazily: slow import, big dep

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, show_progress_bar=len(texts) > 20, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode([text], normalize_embeddings=True)
        return vector[0].tolist()
