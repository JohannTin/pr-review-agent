"""
Measures how much smaller the LLM's context gets when using RAG retrieval
instead of sending every adjacent file.

Indexes this repo's own src/ files (standing in for the "adjacent files"
fetched for a real PR) and compares:
  - baseline: tokens across ALL indexed files, i.e. what you'd send the LLM
    if you just dumped every adjacent file into the prompt
  - RAG: tokens across the top-k chunks actually retrieved for a query

Usage: python3 -m eval.context_size
"""
from pathlib import Path

import tiktoken

from src import rag
from src.config import TOP_K_CHUNKS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def main():
    src_dir = PROJECT_ROOT / "src"
    files = []
    for py_file in sorted(src_dir.glob("*.py")):
        content = py_file.read_text()
        if content.strip():
            files.append({"path": f"src/{py_file.name}", "content": content})

    baseline_text = "\n\n".join(f["content"] for f in files)
    baseline_tokens = count_tokens(baseline_text)

    rag.clear_collection()
    num_chunks = rag.index_files(files)

    # Mirrors the query agent.py's retrieve_context_node builds on its first pass
    query = "code context for: agent.py"
    retrieved = rag.retrieve_context(query, top_k=TOP_K_CHUNKS)
    rag_text = "\n\n---\n\n".join(retrieved)
    rag_tokens = count_tokens(rag_text)

    reduction_pct = (1 - rag_tokens / baseline_tokens) * 100

    print(f"Files indexed:        {len(files)}")
    print(f"Total chunks indexed: {num_chunks}")
    print(f"Chunks retrieved:     {len(retrieved)} (top_k={TOP_K_CHUNKS})")
    print()
    print(f"Baseline tokens (all adjacent files): {baseline_tokens:,}")
    print(f"RAG tokens (top-{TOP_K_CHUNKS} retrieved):     {rag_tokens:,}")
    print(f"Reduction:                             {reduction_pct:.1f}%")
    print()
    print("--- Retrieved chunk headers ---")
    for c in retrieved:
        print(c.splitlines()[0])


if __name__ == "__main__":
    main()
