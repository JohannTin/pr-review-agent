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

**LLM backends:** The agent tries Claude API first (`claude-opus-4-8`). If the API is unavailable or out of credits, it falls back automatically to a local Ollama model (`qwen3.5:9b`). You can also force either backend with `--llm`.

**RAG chunking:** Python files are chunked at AST boundaries (function/class level); other files use a fixed sliding window. This keeps retrieved context tight and relevant rather than returning entire files.

**Retrieval query:** The first-pass query embeds the actual diff content (not just the changed filenames), since a filename alone rarely shares vocabulary with the code that uses a changed constant, function, or exception elsewhere. Retrieval also does a hybrid pass: any indexed chunk containing an exact identifier from the diff (`rag.retrieve_context`'s `boost_text`) is force-included even if it didn't rank in the top-k, since a renamed/removed symbol's usage elsewhere often isn't semantically similar to the diff even though it's exactly the code that breaks.

**Reflection loop:** After the initial analysis, the LLM can request additional context by specifying a new search query. The graph routes back to retrieval and re-runs the analysis with the expanded context. This repeats up to a configurable maximum. The review prompt (`agent.build_review_system_prompt`) also explicitly instructs the model to check whether a changed signature, constant, or exception type is still referenced elsewhere in the retrieved context in a way that's now inconsistent.

**MCP server:** GitHub API calls are wrapped in a FastMCP stdio server that the agent launches as a subprocess. This also lets you register the GitHub tools in Claude Desktop (see below).

---

## Evaluation

Two scripts in `eval/` measure whether retrieval is actually earning its place, rather than just assumed to help.

```bash
pip install -e ".[eval]"
python3 -m eval.context_size
python3 -m eval.retrieval_quality
```

**`eval/context_size.py`** indexes this repo's own `src/` files and compares the token count of every indexed file against the top-k chunks actually retrieved for a query.

> **73.2% fewer tokens** sent to the LLM (7,624 → 2,040) across 6 files / 38 chunks.

**`eval/retrieval_quality.py`** plants 5 bugs in the same domain as this project (GitHub API wrappers, PR fetching/merging, MCP-style tool functions). Each one looks fine in the diff alone and only breaks because of unrelated code elsewhere, buried among ~16 distractor functions. Every scenario runs twice, once with no retrieved context and once with it, on both the Claude and local Ollama backends.

| Scenario | Diff changes... | ...but breaks | Without RAG | With RAG |
|---|---|---|---|---|
| Removed `dedupe` parameter | `post_comment()` drops its `dedupe` param | `post_review_summary()` still calls it with `dedupe=True` → `TypeError` | ✗ both | ✓ both |
| Rate limit window unit change | `RATE_LIMIT_WINDOW` changed from minutes to seconds | `wait_for_rate_limit_reset()` still multiplies by 60 assuming minutes → sleeps ~60x too long | ✗ both | ✓ both |
| Removed `head_sha` validation | `validate_pr_payload()` no longer checks for `head_sha` | `merge_pr()` assumes that guarantee → `KeyError` | ✗ both | ✓ Claude / ✗ Ollama |
| Exception type changed | `fetch_pr_diff()` now raises `GithubApiError` instead of `RateLimitExceeded` | `run_review()`'s `except RateLimitExceeded` no longer catches it → unhandled crash | ✗ both | ✓ both |
| Return shape changed | `get_adjacent_files()` returns a single file instead of a list | `index_context()` still passes it to `index_files()` expecting a list → silently breaks | ✗ both | ✓ Claude / ✗ Ollama |

> **Claude caught 0/5 without retrieved context, 5/5 with it. The local `qwen3.5:9b` fallback caught 0/5 without context, 3/5 with it.** Retrieval surfaced the correct chunk in 5/5 cases. Small, hand-built sample (five scenarios), not a large benchmark — see `eval/retrieval_quality.py` for the exact code and full model output per run.

<details>
<summary><strong>Full code for each scenario</strong></summary>

**1. Removed `dedupe` parameter** — diff (`github/comments.py`):
```python
def post_comment(repo, pr_number, body):
    """Post a comment on a pull request."""
    return _api_post(f"/repos/{repo}/issues/{pr_number}/comments", {"body": body})
```
Unseen unless retrieved (`github/review.py`):
```python
from .comments import post_comment

def post_review_summary(repo, pr_number, summary):
    post_comment(repo, pr_number, summary, dedupe=True)
```
`post_comment` dropped its `dedupe` parameter, but `post_review_summary` still calls it with `dedupe=True` → `TypeError: unexpected keyword argument`.

---

**2. Rate limit window unit change** — diff (`github/constants.py`):
```python
# Rate limit window, now expressed in seconds instead of minutes
RATE_LIMIT_WINDOW = 3600
```
Unseen unless retrieved (`github/throttle.py`):
```python
from .constants import RATE_LIMIT_WINDOW
import time

def wait_for_rate_limit_reset():
    time.sleep(RATE_LIMIT_WINDOW * 60)
```
`RATE_LIMIT_WINDOW` changed from minutes (`60`) to seconds (`3600`), but `wait_for_rate_limit_reset` still multiplies by 60 assuming minutes, so it now sleeps ~60x too long (60 hours instead of 1).

---

**3. Removed `head_sha` validation** — diff (`github/validate.py`):
```python
def validate_pr_payload(payload):
    if "diffs" not in payload:
        raise ValueError("missing diffs")
    return True
```
Unseen unless retrieved (`github/merge.py`):
```python
from .validate import validate_pr_payload

def merge_pr(payload):
    validate_pr_payload(payload)
    return _do_merge(payload["head_sha"])
```
`validate_pr_payload` no longer checks for `head_sha`, but `merge_pr` assumes that guarantee and will raise `KeyError` on a payload missing it.

---

**4. Exception type changed** — diff (`github/fetch.py`):
```python
def fetch_pr_diff(owner, repo, pr_number):
    try:
        return _api_get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
    except HTTPError:
        raise GithubApiError("failed to fetch PR diff")
```
Unseen unless retrieved (`agent/run.py`):
```python
from github.fetch import fetch_pr_diff

def run_review(owner, repo, pr_number):
    try:
        return fetch_pr_diff(owner, repo, pr_number)
    except RateLimitExceeded:
        wait_and_retry()
```
`fetch_pr_diff` now raises `GithubApiError` instead of `RateLimitExceeded` on HTTP failures, but `run_review`'s `except RateLimitExceeded` no longer catches it, so the error now propagates uncaught instead of triggering the retry path.

---

**5. Return shape changed** — diff (`github/files.py`):
```python
def get_adjacent_files(owner, repo, filenames):
    results = _fetch_all(owner, repo, filenames)
    return results[0] if results else None
```
Unseen unless retrieved (`rag/indexer.py`):
```python
from github.files import get_adjacent_files

def index_context(owner, repo, filenames):
    files = get_adjacent_files(owner, repo, filenames)
    return index_files(files)
```
`get_adjacent_files` now returns a single file dict (or `None`) instead of a list, but `index_context` still passes it straight to `index_files`, which expects a list, so it silently breaks.

Each scenario's relevant file is indexed alongside ~16 unrelated distractor functions (string/date/cache/logging/math helpers, see `DISTRACTORS` in `eval/retrieval_quality.py`), so retrieval has to actually outrank real noise, not just find the only other file in the index.

</details>

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
ollama pull qwen3.5:9b
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
