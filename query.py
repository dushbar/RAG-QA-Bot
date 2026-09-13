#!/usr/bin/env python3
"""Ask questions against an ingested document collection.

Examples:
    python query.py --collection demo "What is the refund policy?"
    python query.py --collection demo        # interactive chat mode
"""
import argparse

import config
from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Query the RAG Q&A bot.")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("--collection", default=config.DEFAULT_COLLECTION)
    parser.add_argument("--persist_dir", default=config.DEFAULT_PERSIST_DIR)
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    print(f"Loading collection '{args.collection}' ...")
    pipeline = RAGPipeline(args.persist_dir, args.collection, top_k=args.top_k)

    if args.question:
        _ask(pipeline, args.question)
        return

    print("Interactive mode. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        _ask(pipeline, question)


def _ask(pipeline: RAGPipeline, question: str):
    result = pipeline.answer(question)
    print(f"\nBot: {result['answer']}")
    if result["sources"]:
        print(f"Sources: {', '.join(result['sources'])}")
    print()


if __name__ == "__main__":
    main()
