from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- GitHub ---
GITHUB_TOKEN: str = os.environ["GITHUB_TOKEN"]

# --- Claude API (primary) ---
CLAUDE_MODEL: str = "claude-opus-4-8"  # valid — verified via client.models.list() on 2026-06-16

# --- Local LLM (Ollama fallback) ---
LLM_MODEL: str = "qwen2.5-coder:7b"

# --- ChromaDB ---
CHROMA_DIR: Path = Path(__file__).parent.parent / ".chroma"
CHROMA_COLLECTION: str = "pr_context"

# --- Agent ---
MAX_REFLECTION_LOOPS: int = 2
TOP_K_CHUNKS: int = 8
