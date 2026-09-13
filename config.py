"""Central place for defaults, all overridable via .env or CLI flags."""
import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PERSIST_DIR = os.environ.get("PERSIST_DIR", "./data/vectorstore")
DEFAULT_COLLECTION = os.environ.get("DEFAULT_COLLECTION", "default")
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 900))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))
DEFAULT_TOP_K = int(os.environ.get("TOP_K", 5))
