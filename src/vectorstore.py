"""Thin wrapper around a local, persistent ChromaDB collection.

Tracks a per-collection manifest (source file -> content hash -> chunk ids)
so re-running ingest on an unchanged file is a no-op, and a changed file has
its stale chunks removed before the new ones are added. This is what makes
re-running ingestion on a client's doc folder cheap and safe.
"""
import hashlib
import json
import os
from typing import Any, Dict, List

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.manifest_path = os.path.join(persist_dir, f"{collection_name}.manifest.json")
        self.manifest: Dict[str, Any] = self._load_manifest()

    # ---- manifest (change tracking) ----------------------------------
    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    @staticmethod
    def file_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def needs_reindex(self, source_path: str) -> bool:
        return self.manifest.get(source_path, {}).get("hash") != self.file_hash(source_path)

    # ---- writes ---------------------------------------------------------
    def remove_source(self, source_path: str):
        existing_ids = self.manifest.get(source_path, {}).get("chunk_ids", [])
        if existing_ids:
            self.collection.delete(ids=existing_ids)
        self.manifest.pop(source_path, None)

    def add_chunks(
        self,
        source_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ):
        self.remove_source(source_path)  # clear any stale chunks from a prior version first
        path_hash = hashlib.md5(source_path.encode()).hexdigest()[:10]
        ids = [f"{path_hash}::{i}" for i in range(len(chunks))]
        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        self.manifest[source_path] = {"hash": self.file_hash(source_path), "chunk_ids": ids}
        self._save_manifest()

    # ---- reads ------------------------------------------------------------
    def query(self, query_embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def count(self) -> int:
        return self.collection.count()

    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]
