# PR Review AI Agent

An autonomous PR review agent that fetches a GitHub PR diff, retrieves relevant code context via RAG, reflects on its own analysis, and posts a structured review back to GitHub — all from a single CLI command.

Built with **LangGraph** · **ChromaDB** · **FastMCP** · **Claude API** (Ollama fallback)

---

## How It Works

```
pr-review yourname/yourrepo 99 --verbose
```

1. Fetches the PR diff and description from GitHub
2. Indexes source files from the same directories as the changed files into a local vector database
3. Runs semantic search to retrieve the most relevant code context
4. Sends everything to Claude (or Ollama locally) for structured analysis
5. If the LLM needs more context, it loops back and searches again (reflection loop)
6. Posts the final review directly to the GitHub PR, with the verdict visible in the comment body

---

## File Structure

```
pr-review-agent/
├── .env                  ← your secrets (GITHUB_TOKEN, ANTHROPIC_API_KEY)
├── .env.example          ← template
├── pyproject.toml        ← dependencies + CLI entry point
└── src/
    ├── config.py         ← env vars and constants
    ├── github.py         ← GitHub API wrapper
    ├── mcp_server.py     ← FastMCP stdio server exposing GitHub tools
    ├── rag.py            ← ChromaDB indexing and retrieval
    ├── agent.py          ← LangGraph graph and reflection loop
    └── main.py           ← Typer CLI entrypoint
```

---

## Architecture

```
CLI (main.py)
  └── LangGraph agent (agent.py)
        ├── GitHub tools via FastMCP subprocess (mcp_server.py → github.py)
        └── RAG via ChromaDB + local embeddings (rag.py)
```

**LLM backends:** The agent tries Claude API first (`claude-opus-4-8`). If the API is unavailable or out of credits, it falls back automatically to a local Ollama model (`qwen2.5-coder:7b`). You can also force either backend with `--llm`.

**RAG chunking:** Python files are chunked at AST boundaries (function/class level); other files use a fixed sliding window. This keeps retrieved context tight and relevant rather than returning entire files.

**Reflection loop:** After the initial analysis, the LLM can request additional context by specifying a new search query. The graph routes back to retrieval and re-runs the analysis with the expanded context. This repeats up to a configurable maximum.

**MCP server:** GitHub API calls are wrapped in a FastMCP stdio server that the agent launches as a subprocess. This also lets you register the GitHub tools in Claude Desktop (see below).

---

## Setup

```bash
git clone <repo-url>
cd pr-review-agent
pip install -e .

# Add your tokens
cp .env.example .env
# edit .env:
#   GITHUB_TOKEN=ghp_...
#   ANTHROPIC_API_KEY=sk-ant-...   ← required for Claude (primary LLM)

# Optional: set up Ollama as a fallback
ollama pull qwen2.5-coder:7b
ollama serve
```

**GitHub token scope required:** `repo`

---

## Usage

```bash
pr-review <owner/repo> <pr_number> [options]
```

| Option | Description |
|---|---|
| `--verbose` / `-v` | Show a per-file issues table |
| `--action comment` | Post review only, never merge (default) |
| `--action merge` | Post review and merge the PR if verdict is APPROVE |
| `--llm auto` | Try Claude first, fall back to Ollama (default) |
| `--llm claude` | Force Claude API — error if unavailable |
| `--llm local` | Force local Ollama — skip Claude entirely |
| `--force` | Merge regardless of verdict (requires `--action merge`) |

### Examples

```bash
# Standard review, auto LLM selection
pr-review torvalds/linux 99 --verbose

# Review and merge if approved
pr-review yourname/yourrepo 42 --action merge

# Force local model (no API cost, no network)
pr-review yourname/yourrepo 42 --llm local

# Force Claude with verbose output
pr-review yourname/yourrepo 42 --llm claude --verbose

# Override false positives and merge anyway
pr-review yourname/yourrepo 42 --action merge --force
```

The verdict always appears at the top of the GitHub comment — so if you run with `--action comment` and see `**APPROVE**`, you know it would have merged.

### Review comment format

Issues are labelled by severity:
- `[critical]` — security vulnerabilities, data loss, crashes
- `[issue]` — bugs, wrong logic, missing error handling
- `[note]` — style, naming, non-blocking suggestions

Verdict follows from labels: any `[critical]` or `[issue]` → `REQUEST_CHANGES`; only `[note]` → `COMMENT`; nothing → `APPROVE`.

### Example GitHub comment

```
**REQUEST_CHANGES** · 2 files · 1 critical · 1 issue · 1 note

Missing error handling in the auth handler and a timing attack risk in middleware.

**`src/auth/handler.py`**
- [critical] Use hmac.compare_digest instead of == (timing attack)
- [issue] Missing try/except around db.save()

**`tests/test_auth.py`**
- [note] No test for expired token case

---
*🤖 PR Review AI Agent · claude-opus-4-8 (or whichever backend ran).*
```

### Example terminal output

```
Verdict: ✗ REQUEST_CHANGES   (1 reflection iteration)

╭─── Review Summary ───────────────────────────────────────────────╮
│ Missing error handling in the auth handler and a timing attack   │
│ risk in middleware.                                              │
╰──────────────────────────────────────────────────────────────────╯

Review posted to GitHub.
```

---

## MCP Server (Claude Desktop)

Register the GitHub tools for use in any Claude conversation:

```json
{
  "mcpServers": {
    "github-pr": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/pr-review-agent"
    }
  }
}
```

---

## Tech Stack

| Library | Role |
|---|---|
| `anthropic` | Claude API client (primary LLM) |
| `langgraph` | Stateful graph with conditional edges and reflection loop |
| `langchain-ollama` | LangChain wrapper for local Ollama fallback |
| `langchain-mcp-adapters` | Converts MCP tools into LangChain-compatible tools |
| `fastmcp` | FastMCP framework for the stdio MCP server |
| `chromadb` | Local persistent vector database |
| `sentence-transformers` | Local embedding model (all-MiniLM-L6-v2) |
| `PyGithub` | GitHub REST API client |
| `typer` + `rich` | CLI and terminal display |
| `pydantic` | Structured output schema for LLM responses |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `KeyError: GITHUB_TOKEN` | `.env` not filled in | Add your token to `.env` |
| `AuthenticationError` | Missing or invalid `ANTHROPIC_API_KEY` | Add key to `.env`, or use `--llm local` |
| `Connection refused` | Ollama not running | Run `ollama serve` in a new terminal |
| `404 Not Found` | Wrong repo or PR number | Double-check `owner/repo` and PR number |
| `403 Forbidden` | Token missing `repo` scope | Regenerate token with `repo` checked |
| Both LLMs failed | Claude down + Ollama not running | Check API key and run `ollama serve` |
| `PR head SHA changed` | PR was updated between review and merge | Re-run the review before merging |
