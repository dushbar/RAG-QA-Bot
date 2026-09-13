"""Document loaders. Add a new file type by adding an extension + a
`_load_xxx(path) -> str` function — everything else (chunking, embedding,
storage) works unchanged.
"""
import csv
import os
from typing import Any, Dict, List

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def load_documents(docs_dir: str) -> List[Dict[str, Any]]:
    """Walk docs_dir and return [{"path": ..., "text": ...}, ...] for every
    supported file found (recursively). Unreadable/unsupported files are
    skipped with a printed warning rather than crashing the whole ingest run.
    """
    documents = []
    for root, _, files in os.walk(docs_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            path = os.path.join(root, fname)
            try:
                text = _load_single(path, ext)
            except Exception as e:  # noqa: BLE001 - keep ingesting other files
                print(f"  [skip] {path}: {e}")
                continue
            if text and text.strip():
                documents.append({"path": path, "text": text})
    return documents


def _load_single(path: str, ext: str) -> str:
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext == ".csv":
        return _load_csv(path)
    return _load_text(path)  # .txt, .md


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[page {i + 1}]\n{page_text}")
    return "\n\n".join(pages)


def _load_docx(path: str) -> str:
    import docx

    doc = docx.Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _load_csv(path: str) -> str:
    """Turns each row into a 'col: value; col: value' line so the meaning of
    each field survives chunking (raw CSV rows are meaningless without headers).
    """
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line = "; ".join(f"{k}: {v}" for k, v in row.items() if v)
            if line:
                lines.append(line)
    return "\n".join(lines)
