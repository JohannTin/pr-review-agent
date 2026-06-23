"""
RAG pipeline: function/class-level chunking → ChromaDB → semantic retrieval.

Python files are chunked by AST boundaries (functions and classes).
All other code files fall back to a fixed 60-line sliding window.
"""

import ast
import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, CHROMA_COLLECTION, TOP_K_CHUNKS

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(CHROMA_COLLECTION)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_python(path: str, content: str) -> list[dict]:
    """Split a Python file into function- and class-level chunks via AST."""
    chunks: list[dict] = []
    try:
        tree = ast.parse(content)
        lines = content.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno or len(lines)
                chunks.append({
                    "text": "\n".join(lines[start:end]),
                    "path": path,
                    "name": node.name,
                    "kind": type(node).__name__,
                    "line": node.lineno,
                })
    except SyntaxError:
        # Whole file as one chunk if parsing fails
        chunks.append({"text": content, "path": path, "name": "module", "kind": "Module", "line": 1})
    return chunks


def _chunk_generic(path: str, content: str, window: int = 60) -> list[dict]:
    """Split any file into fixed-size line windows."""
    lines = content.splitlines()
    return [
        {
            "text": "\n".join(lines[i : i + window]),
            "path": path,
            "name": f"lines_{i + 1}_{min(i + window, len(lines))}",
            "kind": "chunk",
            "line": i + 1,
        }
        for i in range(0, len(lines), window)
        if lines[i : i + window]
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def index_files(files: list) -> int:
    """
    Chunk and embed a list of files into ChromaDB.
    Accepts RepoFile objects or plain dicts with 'path' and 'content' keys.
    Returns the number of chunks indexed.
    """
    collection = _get_collection()
    model = _get_model()
    all_chunks: list[dict] = []

    for f in files:
        path = f["path"] if isinstance(f, dict) else f.path
        content = f["content"] if isinstance(f, dict) else f.content
        if not content.strip():
            continue
        if path.endswith(".py"):
            all_chunks.extend(_chunk_python(path, content))
        else:
            all_chunks.extend(_chunk_generic(path, content))

    if not all_chunks:
        return 0

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    ids = [f"{c['path']}::{c['name']}::{c['line']}" for c in all_chunks]
    metadatas = [
        {"path": c["path"], "name": c["name"], "kind": c["kind"], "line": c["line"]}
        for c in all_chunks
    ]

    collection.upsert(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(all_chunks)


def retrieve_context(query: str, top_k: int = TOP_K_CHUNKS) -> list[str]:
    """Return the top-k most semantically similar code chunks for the query."""
    collection = _get_collection()
    model = _get_model()

    try:
        count = collection.count()
    except Exception:
        return []

    if count == 0:
        return []

    embedding = model.encode([query], show_progress_bar=False).tolist()
    n = min(top_k, count)
    results = collection.query(query_embeddings=embedding, n_results=n)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        header = f"# {meta['path']} — {meta['name']} (line {meta['line']})"
        chunks.append(f"{header}\n{doc}")
    return chunks


def clear_collection() -> None:
    """Drop the ChromaDB collection so each PR review starts from a clean index."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
