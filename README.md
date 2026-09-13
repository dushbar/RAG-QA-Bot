# Docs Q&A Bot — a reusable RAG pipeline for "answer questions from my documents" projects

A self-contained retrieval-augmented generation (RAG) system that ingests a
folder of documents and answers questions about them, with every answer
grounded in — and cited to — the source files. Built to be handed a new
client's docs folder and be answering questions within minutes, not to be a
one-off demo tied to a specific dataset.

## Why it's built this way

- **Embeddings are local and free.** Ingestion uses `sentence-transformers`
  on your own machine — no per-document API cost, and client documents never
  leave the machine at ingestion time. The only thing that leaves the
  machine is the final question + a handful of retrieved snippets, sent to
  whichever LLM you configure.
- **No vendor lock-in on generation.** Swap between Anthropic and OpenAI with
  one line in `.env`. Adding a third provider (Ollama, Gemini, etc.) means
  implementing one `LLMProvider` subclass in `src/llm.py`.
- **Re-running ingestion is cheap.** Each collection keeps a manifest of
  file → content hash. Unchanged files are skipped automatically; only new
  or edited files get re-embedded. Point it at a client's live docs folder
  and re-run ingest on a schedule without re-processing everything.
- **One collection per client.** ChromaDB collections are namespaced by
  name, so the same installation can serve multiple clients/projects — just
  use a different `--collection` value.
- **No black-box framework.** No LangChain/LlamaIndex abstraction layer —
  every step (chunking, embedding, retrieval, prompting) is ~50 lines of
  plain Python in `src/`, so it's easy to read, explain, and customize per
  engagement.

## Architecture

```
docs/ (pdf, docx, txt, md, csv)
      │
      ▼
 loaders.py  ──►  chunking.py  ──►  embeddings.py (local, free)
                                          │
                                          ▼
                                  vectorstore.py (ChromaDB, on disk)
                                          ▲
                                          │  (top-k similarity search)
question ──► embeddings.py.embed_query ──┘
                                          │
                                          ▼
                              rag_pipeline.py builds a grounded prompt
                                          │
                                          ▼
                          llm.py  (Anthropic or OpenAI)  ──► answer + sources
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching API key

# 1. Ingest the included sample docs
python ingest.py --docs_dir docs/sample --collection demo

# 2. Ask questions
python query.py --collection demo "What's the return window?"
python query.py --collection demo          # interactive chat mode

# or launch the web UI
python -m streamlit run app.py
```

First ingest run downloads the local embedding model (~80MB, one time,
cached afterward). If a client's network blocks Hugging Face, run ingestion
once from an unrestricted machine — the `data/` folder is fully portable.

## Using this on an actual gig

1. **Point it at their docs.** Drop the client's PDFs/Word docs/CSVs/markdown
   into a folder and run `ingest.py --docs_dir <folder> --collection <client_name>`.
2. **Pick a delivery shape.** The CLI is enough for a quick turnaround; run
   `streamlit run app.py` (or deploy it — see below) if they want a chat UI
   their team can use directly.
3. **Tune retrieval, not the prompt, first.** If answers feel off, the
   usual fix is `--chunk_size`/`--chunk_overlap` or `top_k`, not a longer
   system prompt. Smaller chunk sizes help precise fact lookups (pricing,
   policies); larger ones help "explain this process" questions.
4. **Add a loader if they use a format you don't support yet.** Extend
   `SUPPORTED_EXTENSIONS` and add a `_load_xxx()` function in
   `src/loaders.py` — everything downstream (chunking, embedding, storage)
   works unchanged. HTML, JSON, Notion/Confluence exports, and email
   archives are common asks.
5. **Multi-tenant by default.** Different `--collection` names keep clients'
   data separate within the same `data/vectorstore/` folder.

## Deployment options

- **Streamlit Community Cloud** — free, fastest path to a shareable link;
  fine for internal tools and demos.
- **Docker on a small VM** — wrap `streamlit run app.py` in a container for
  anything client-facing or needing auth in front of it.
- **CLI only** — for scripted/batch use (e.g. answering a queue of support
  tickets), skip the UI entirely and call `RAGPipeline` directly from
  another script.

## Project layout

```
config.py            # defaults, loaded from .env
ingest.py             # CLI: folder of docs -> vector store collection
query.py               # CLI: ask questions (single-shot or interactive)
app.py                  # Streamlit chat UI + drag-and-drop ingestion
src/
  loaders.py            # pdf, docx, txt, md, csv -> plain text
  chunking.py            # recursive text splitter (no external dep)
  embeddings.py            # local sentence-transformers wrapper
  vectorstore.py            # ChromaDB wrapper + incremental re-index tracking
  llm.py                     # Anthropic / OpenAI provider abstraction
  rag_pipeline.py              # retrieval + grounded-prompt orchestration
docs/sample/                    # small sample knowledge base to try immediately
```

## Known limitations / natural next steps

- Retrieval is single-vector top-k similarity — no re-ranking or hybrid
  keyword+vector search. Worth adding for larger doc sets (thousands of
  pages) where precision matters more.
- No auth on the Streamlit app — add a login layer before exposing it
  outside a trusted network.
- No answer caching — repeated identical questions re-call the LLM. Cheap
  to add with a hash-of-question cache if a client's usage is spiky.
- Chunking is generic; a client with heavily tabular or structured docs
  (e.g. a large pricing catalog) may do better with a format-specific
  chunking strategy instead of the default recursive splitter.
