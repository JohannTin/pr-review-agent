"""
Standalone FastMCP stdio server that exposes GitHub PR tooling.

Run directly:   python -m src.mcp_server
Or via agent:   StdioServerParameters(command="python", args=["-m", "src.mcp_server"])
"""

import json
import sys
from pathlib import Path

# Allow import of src.* whether run as script or module
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastmcp import FastMCP

from src.github import fetch_pr, fetch_adjacent_files, post_review

mcp = FastMCP(
    name="GitHubPRServer",
    instructions=(
        "Tools for interacting with GitHub pull requests: "
        "fetch diffs, retrieve file contents, post reviews, and assign labels."
    ),
)


@mcp.tool
def get_pr_diff(owner: str, repo: str, pr_number: int) -> str:
    """
    Fetch the unified diff for a GitHub pull request.
    Returns JSON: {diffs: [{filename, patch, status, additions, deletions}], pr_description, head_sha}
    """
    diffs, description, head_sha = fetch_pr(owner, repo, pr_number)
    return json.dumps({
        "diffs": [d.to_dict() for d in diffs],
        "pr_description": description,
        "head_sha": head_sha,
    })


@mcp.tool
def get_adjacent_files(owner: str, repo: str, filenames_json: str) -> str:
    """
    Fetch source files in the same directories as the changed files.
    filenames_json: JSON array of file paths from the PR diff.
    Returns JSON array of {path, content} objects.
    """
    filenames: list[str] = json.loads(filenames_json)
    files = fetch_adjacent_files(owner, repo, filenames)
    return json.dumps([{"path": f.path, "content": f.content} for f in files])


@mcp.tool
def post_pr_review(
    owner: str,
    repo: str,
    pr_number: int,
    summary: str,
    verdict: str,
    file_reviews_json: str = "[]",
    llm: str = "auto",
) -> str:
    """
    Post a code review on a GitHub pull request.
    verdict: one of APPROVE, REQUEST_CHANGES, COMMENT.
    file_reviews_json: JSON array of {path, issues: [str]} objects.
    """
    file_reviews: list[dict] = json.loads(file_reviews_json)
    post_review(owner, repo, pr_number, file_reviews, verdict, summary, llm)
    return f"Review posted on PR #{pr_number} with verdict: {verdict}"


@mcp.tool
def assign_label(owner: str, repo: str, pr_number: int, label: str) -> str:
    """Assign a label to a GitHub pull request."""
    from github import Github
    from src.config import GITHUB_TOKEN

    g = Github(GITHUB_TOKEN)
    repository = g.get_repo(f"{owner}/{repo}")
    pr = repository.get_pull(pr_number)
    pr.add_to_labels(label)
    return f"Label '{label}' assigned to PR #{pr_number}"


if __name__ == "__main__":
    mcp.run()
