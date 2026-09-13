"""Streamlit chat UI for the RAG Q&A bot.

Run with:  streamlit run app.py
"""
import os

import streamlit as st

import config
from src.chunking import chunk_text
from src.embeddings import LocalEmbedder
from src.loaders import load_documents
from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore

st.set_page_config(page_title="Docs Q&A Bot", page_icon="💬", layout="wide")
st.title("💬 Docs Q&A Bot")
st.caption("Ask questions about your documents. Answers are grounded only in what's been ingested.")

with st.sidebar:
    st.header("Collection")
    collection = st.text_input(
        "Collection name", value=config.DEFAULT_COLLECTION, help="Use a separate collection per client/project."
    )
    persist_dir = st.text_input("Storage folder", value=config.DEFAULT_PERSIST_DIR)
    top_k = st.slider("Chunks to retrieve", 1, 10, config.DEFAULT_TOP_K)

    st.divider()
    st.subheader("Add documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, MD, or CSV",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Ingest uploaded files", type="primary"):
        os.makedirs(persist_dir, exist_ok=True)
        upload_dir = os.path.join(persist_dir, "_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        with st.spinner("Reading and embedding documents..."):
            for f in uploaded_files:
                with open(os.path.join(upload_dir, f.name), "wb") as out:
                    out.write(f.getbuffer())

            documents = load_documents(upload_dir)
            store = VectorStore(persist_dir, collection)
            embedder = LocalEmbedder()
            ingested = 0
            for doc in documents:
                if not store.needs_reindex(doc["path"]):
                    continue
                chunks = chunk_text(doc["text"], config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP)
                if not chunks:
                    continue
                metadatas = [
                    {"source": os.path.basename(doc["path"]), "chunk_index": i} for i in range(len(chunks))
                ]
                embeddings = embedder.embed_documents(chunks)
                store.add_chunks(doc["path"], chunks, embeddings, metadatas)
                ingested += 1
        st.success(f"Ingested {ingested} file(s) into '{collection}'.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content, sources in st.session_state.history:
    with st.chat_message(role):
        st.write(content)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.history.append(("user", question, None))
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                pipeline = RAGPipeline(persist_dir, collection, top_k=top_k)
                result = pipeline.answer(question)
                st.write(result["answer"])
                if result["sources"]:
                    st.caption(f"Sources: {', '.join(result['sources'])}")
                st.session_state.history.append(("assistant", result["answer"], result["sources"]))
            except Exception as e:  # noqa: BLE001 - surface config errors (e.g. missing API key) in the UI
                st.error(f"Error: {e}")
