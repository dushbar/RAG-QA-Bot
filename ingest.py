#!/usr/bin/env python3
"""Ingest a folder of documents into a persistent vector store collection.

Examples:
    python ingest.py --docs_dir docs/sample --collection demo
    python ingest.py --docs_dir ./client_docs --collection acme_corp --force
"""
import argparse
import os
import sys

import config
from src.chunking import chunk_text
from src.embeddings import LocalEmbedder
from src.loaders import load_documents
from src.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("--docs_dir", required=True, help="Folder with .pdf/.docx/.txt/.md/.csv files")
    parser.add_argument("--collection", default=config.DEFAULT_COLLECTION, help="One collection per client/project")
    parser.add_argument("--persist_dir", default=config.DEFAULT_PERSIST_DIR)
    parser.add_argument("--chunk_size", type=int, default=config.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=config.DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--force", action="store_true", help="Re-embed every file, even unchanged ones")
    args = parser.parse_args()

    if not os.path.isdir(args.docs_dir):
        print(f"Docs directory not found: {args.docs_dir}")
        sys.exit(1)

    print(f"Scanning {args.docs_dir} ...")
    documents = load_documents(args.docs_dir)
    if not documents:
        print("No supported documents found (.pdf, .docx, .txt, .md, .csv).")
        sys.exit(1)
    print(f"Found {len(documents)} document(s).")

    store = VectorStore(args.persist_dir, args.collection)
    embedder = None  # loaded lazily so a run with nothing new to do stays fast

    added, skipped = 0, 0
    for doc in documents:
        path = doc["path"]
        if not args.force and not store.needs_reindex(path):
            skipped += 1
            continue

        if embedder is None:
            print("Loading local embedding model (first run downloads ~80MB, cached after that)...")
            embedder = LocalEmbedder()

        chunks = chunk_text(doc["text"], args.chunk_size, args.chunk_overlap)
        if not chunks:
            continue
        metadatas = [{"source": os.path.basename(path), "chunk_index": i} for i in range(len(chunks))]
        embeddings = embedder.embed_documents(chunks)
        store.add_chunks(path, chunks, embeddings, metadatas)
        print(f"  [ingested] {os.path.basename(path)} -> {len(chunks)} chunks")
        added += 1

    print(f"\nDone. {added} file(s) (re)indexed, {skipped} unchanged file(s) skipped.")
    print(f"Collection '{args.collection}' now has {store.count()} chunks total.")
    print(f"Persisted at: {os.path.abspath(args.persist_dir)}")


if __name__ == "__main__":
    main()
