# ---- RAG Q&A Bot: Streamlit container ----
FROM python:3.11-slim

# build-essential: some sentence-transformers/torch deps need to compile
# curl: used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .

# CPU-only torch — sentence-transformers otherwise pulls the full CUDA build,
# which balloons the image to 5GB+. This keeps it closer to ~1.5GB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the container doesn't need to hit
# the Hugging Face Hub on first run (faster cold start, works even if outbound
# HF access is blocked). Adjust the model name if your ingest.py uses a
# different sentence-transformers model.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# chroma_db is meant to be volume-mounted at runtime (see docker-compose.yml),
# not baked in — this just ensures the path exists if no volume is attached.
RUN mkdir -p /app/chroma_db

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=true"]
