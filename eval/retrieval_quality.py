"""
Ablation test: does retrieved context actually change whether the reviewer
catches a bug, or is RAG just adding noise?

Each scenario plants a bug that looks fine in the diff alone and only breaks
because of unrelated code elsewhere (e.g. a caller still passing a parameter
that was just removed). The relevant file is buried in a pool of distractor
functions so retrieval has to actually rank it above the noise.

For each scenario, runs both LLM backends (Claude and local Ollama) twice:
  - "no_rag": context_chunks empty, exactly like agent.py when nothing is retrieved
  - "rag":    context_chunks = the actual top-k retrieval result for the query

This does not auto-grade correctness — keyword matching on free-text LLM
output is unreliable. Read the printed issues/summary per run and judge
whether the specific planted bug was named, not just "an issue" was found.

Usage: python3 -m eval.retrieval_quality
Last measured result (5 scenarios): Claude caught 0/5 without retrieved
context and 5/5 with it. The local Ollama fallback caught 0/5 without
context and 3/5 with it. Retrieval hit its target in 5/5 cases.
See README.md's Evaluation section for the before/after story.
"""
import asyncio
import json
import sys

from src import rag
from src.config import TOP_K_CHUNKS
from src.agent import _analyze_with_claude, _analyze_with_ollama, build_review_system_prompt

DISTRACTORS = [
    {"path": "utils/strings.py", "content": '''
def slugify(text):
    return text.lower().strip().replace(" ", "-")

def truncate(text, length=100):
    return text if len(text) <= length else text[:length] + "..."

def camel_to_snake(name):
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
'''},
    {"path": "utils/dates.py", "content": '''
from datetime import datetime, timedelta

def days_ago(n):
    return datetime.now() - timedelta(days=n)

def format_date(dt):
    return dt.strftime("%Y-%m-%d")

def is_weekend(dt):
    return dt.weekday() >= 5
'''},
    {"path": "utils/cache.py", "content": '''
_cache = {}

def cache_get(key):
    return _cache.get(key)

def cache_set(key, value, ttl=300):
    _cache[key] = value

def cache_clear():
    _cache.clear()
'''},
    {"path": "utils/logging_helpers.py", "content": '''
import logging

def get_logger(name):
    return logging.getLogger(name)

def log_duration(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def redact(text):
    return text.replace("password", "****")
'''},
    {"path": "utils/math_helpers.py", "content": '''
def clamp(value, lo, hi):
    return max(lo, min(value, hi))

def percentage(part, whole):
    return (part / whole) * 100 if whole else 0

def round_to(value, step):
    return round(value / step) * step
'''},
]

SCENARIOS = [
    {
        "name": "removed dedupe parameter",
        "diff_filename": "github/comments.py",
        "diff_content": '''def post_comment(repo, pr_number, body):
    """Post a comment on a pull request."""
    return _api_post(f"/repos/{repo}/issues/{pr_number}/comments", {"body": body})''',
        "context_filename": "github/review.py",
        "context_content": '''from .comments import post_comment

def post_review_summary(repo, pr_number, summary):
    post_comment(repo, pr_number, summary, dedupe=True)''',
        "ground_truth": "post_comment dropped its 'dedupe' parameter, but post_review_summary in review.py still calls it with dedupe=True, which will raise a TypeError for an unexpected keyword argument.",
    },
    {
        "name": "rate limit window unit change",
        "diff_filename": "github/constants.py",
        "diff_content": '''# Rate limit window, now expressed in seconds instead of minutes
RATE_LIMIT_WINDOW = 3600''',
        "context_filename": "github/throttle.py",
        "context_content": '''from .constants import RATE_LIMIT_WINDOW
import time

def wait_for_rate_limit_reset():
    time.sleep(RATE_LIMIT_WINDOW * 60)''',
        "ground_truth": "RATE_LIMIT_WINDOW changed from minutes (60) to seconds (3600), but throttle.py's wait_for_rate_limit_reset still multiplies it by 60 assuming minutes, so it now sleeps ~60x too long (60 hours instead of 1 hour).",
    },
    {
        "name": "removed head_sha validation another function relies on",
        "diff_filename": "github/validate.py",
        "diff_content": '''def validate_pr_payload(payload):
    if "diffs" not in payload:
        raise ValueError("missing diffs")
    return True''',
        "context_filename": "github/merge.py",
        "context_content": '''from .validate import validate_pr_payload

def merge_pr(payload):
    validate_pr_payload(payload)
    return _do_merge(payload["head_sha"])''',
        "ground_truth": "validate_pr_payload no longer checks for 'head_sha', but merge_pr in merge.py assumes that guarantee and will raise KeyError on payloads missing head_sha.",
    },
    {
        "name": "exception type changed breaks caller's except clause",
        "diff_filename": "github/fetch.py",
        "diff_content": '''def fetch_pr_diff(owner, repo, pr_number):
    try:
        return _api_get(f"/repos/{owner}/{repo}/pulls/{pr_number}")
    except HTTPError:
        raise GithubApiError("failed to fetch PR diff")''',
        "context_filename": "agent/run.py",
        "context_content": '''from github.fetch import fetch_pr_diff

def run_review(owner, repo, pr_number):
    try:
        return fetch_pr_diff(owner, repo, pr_number)
    except RateLimitExceeded:
        wait_and_retry()''',
        "ground_truth": "fetch_pr_diff now raises GithubApiError instead of RateLimitExceeded on HTTP failures, but run_review in run.py only catches RateLimitExceeded, so GithubApiError now propagates uncaught instead of triggering the retry path.",
    },
    {
        "name": "adjacent files return shape changed from list to single file",
        "diff_filename": "github/files.py",
        "diff_content": '''def get_adjacent_files(owner, repo, filenames):
    results = _fetch_all(owner, repo, filenames)
    return results[0] if results else None''',
        "context_filename": "rag/indexer.py",
        "context_content": '''from github.files import get_adjacent_files

def index_context(owner, repo, filenames):
    files = get_adjacent_files(owner, repo, filenames)
    return index_files(files)''',
        "ground_truth": "get_adjacent_files now returns a single file dict (or None) instead of a list, but index_context in indexer.py still passes it straight to index_files, which expects a list of files, so it silently breaks.",
    },
]

# Imported from src.agent so this eval always tests the same prompt production uses,
# instead of a hand-copied string that can silently drift out of sync.
SYSTEM_TEMPLATE = build_review_system_prompt()


def build_user_content(diff_filename, diff_content, context_text):
    return f"""**PR Description:**
(no description provided)

**Changed Files:**
### {diff_filename} (modified)
```python
{diff_content}
```

**Repository Context (semantic search results):**
{context_text}

Iteration 1/3."""


async def run_scenario(scenario):
    files_to_index = DISTRACTORS + [
        {"path": scenario["context_filename"], "content": scenario["context_content"]}
    ]
    rag.clear_collection()
    total_chunks = rag.index_files(files_to_index)

    # Mirrors agent.py's retrieve_context_node: embed the actual diff content
    # (not just the filename) and boost in any exact identifier match.
    query = f"{scenario['diff_filename']}\n{scenario['diff_content']}"
    retrieved = rag.retrieve_context(query, top_k=TOP_K_CHUNKS, boost_text=scenario["diff_content"])
    retrieved_hit = any(scenario["context_filename"] in c.splitlines()[0] for c in retrieved)

    conditions = {
        "no_rag": "No additional context retrieved.",
        "rag": "\n\n---\n\n".join(retrieved),
    }

    result = {
        "scenario": scenario["name"],
        "ground_truth": scenario["ground_truth"],
        "retrieved_hit": retrieved_hit,
        "total_chunks_indexed": total_chunks,
        "runs": {},
    }

    for cond_name, context_text in conditions.items():
        user_content = build_user_content(
            scenario["diff_filename"], scenario["diff_content"], context_text
        )

        for backend_name, fn in [("claude", _analyze_with_claude), ("ollama", _analyze_with_ollama)]:
            key = f"{backend_name}_{cond_name}"
            try:
                analysis = await fn(SYSTEM_TEMPLATE, user_content)
                result["runs"][key] = {
                    "verdict": analysis.verdict,
                    "summary": analysis.summary,
                    "issues": [
                        f"{fr.path}: {i}" for fr in analysis.file_reviews for i in fr.issues
                    ],
                }
            except Exception as exc:
                result["runs"][key] = {"error": f"{type(exc).__name__}: {exc}"}

    return result


async def main():
    for scenario in SCENARIOS:
        print(f"Running scenario: {scenario['name']}...", file=sys.stderr)
        r = await run_scenario(scenario)
        print(json.dumps(r, indent=2))
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
