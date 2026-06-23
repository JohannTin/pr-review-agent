from dataclasses import dataclass, asdict
from pathlib import Path

from github import Github, GithubException

from .config import GITHUB_TOKEN

_CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".java", ".rb", ".rs", ".c", ".cpp", ".h",
}


@dataclass
class FileDiff:
    filename: str
    patch: str
    status: str
    additions: int
    deletions: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepoFile:
    path: str
    content: str


def _client() -> Github:
    return Github(GITHUB_TOKEN)


def fetch_pr(owner: str, repo: str, pr_number: int) -> tuple[list[FileDiff], str, str]:
    """Return (diffs, pr_description, head_sha) for the given pull request."""
    g = _client()
    repository = g.get_repo(f"{owner}/{repo}")
    pr = repository.get_pull(pr_number)

    diffs = [
        FileDiff(
            filename=f.filename,
            patch=f.patch or "",
            status=f.status,
            additions=f.additions,
            deletions=f.deletions,
        )
        for f in pr.get_files()
        if f.patch
    ]

    return diffs, pr.body or "", pr.head.sha


def fetch_adjacent_files(owner: str, repo: str, filenames: list[str]) -> list[RepoFile]:
    """Fetch source files in the same directories as the changed files."""
    g = _client()
    repository = g.get_repo(f"{owner}/{repo}")

    dirs = {str(Path(f).parent) for f in filenames}
    files: list[RepoFile] = []
    seen: set[str] = set()

    for dir_path in dirs:
        try:
            items = repository.get_contents(dir_path)
            for item in items:
                if (
                    item.type == "file"
                    and item.path not in seen
                    and Path(item.path).suffix in _CODE_EXTENSIONS
                ):
                    try:
                        text = item.decoded_content.decode("utf-8", errors="ignore")
                        files.append(RepoFile(path=item.path, content=text))
                        seen.add(item.path)
                    except Exception:
                        pass
        except GithubException:
            pass

    return files


def merge_pr(owner: str, repo: str, pr_number: int, head_sha: str = "") -> None:
    """Merge the PR if it is mergeable. Verifies head SHA if provided."""
    g = _client()
    pr = g.get_repo(f"{owner}/{repo}").get_pull(pr_number)
    if head_sha and pr.head.sha != head_sha:
        raise ValueError(
            f"PR head SHA changed since review ({head_sha[:7]} → {pr.head.sha[:7]}). "
            "Re-run the review before merging."
        )
    if pr.mergeable:
        pr.merge(merge_method="squash")


def post_review(
    owner: str,
    repo: str,
    pr_number: int,
    file_reviews: list[dict],
    verdict: str,
    summary: str,
    llm: str = "auto",
) -> None:
    """Post a batched review — one block per file — as a single GitHub PR review."""
    g = _client()
    repository = g.get_repo(f"{owner}/{repo}")
    pr = repository.get_pull(pr_number)

    files_with_issues = [fr for fr in file_reviews if fr.get("issues")]
    n_files = len(files_with_issues)
    all_issues = [i for fr in files_with_issues for i in fr["issues"]]
    counts = {"critical": 0, "issue": 0, "note": 0}
    unlabeled = 0
    for i in all_issues:
        normalized = i.strip().lower()
        matched = False
        for label in counts:
            # match both [label] (Claude) and "label: " (Ollama coerced dict format)
            if normalized.startswith(f"[{label}]") or normalized.startswith(f"{label}:"):
                counts[label] += 1
                matched = True
                break
        if not matched:
            unlabeled += 1

    def _fmt(label, n):
        return f"{n} {label}{'s' if n != 1 else ''}"

    parts_breakdown = [_fmt(k, v) for k, v in counts.items() if v > 0]
    if unlabeled:
        parts_breakdown.append(_fmt("unlabeled", unlabeled))
    breakdown = " · ".join(parts_breakdown) if all_issues else "no issues"
    header = f"**{verdict}** · {n_files} file{'s' if n_files != 1 else ''} · {breakdown}"

    parts = [header] + (["", summary] if summary else [])
    for fr in files_with_issues:
        parts.append(f"\n**`{fr['path']}`**")
        parts.extend(f"- {issue}" for issue in fr["issues"])
    parts.append(f"\n---\n*🤖 PR Review AI Agent · {llm}*")
    body = "\n".join(parts)
    event = verdict if verdict in {"APPROVE", "REQUEST_CHANGES", "COMMENT"} else "COMMENT"
    try:
        pr.create_review(body=body, event=event)
    except GithubException as e:
        if e.status == 422 and "own pull request" in str(e.data):
            # GitHub doesn't allow REQUEST_CHANGES on your own PR — fall back to COMMENT
            pr.create_review(body=body, event="COMMENT")
        else:
            raise
