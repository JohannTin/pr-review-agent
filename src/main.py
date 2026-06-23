import asyncio

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent import run_review
from .github import merge_pr

app = typer.Typer(
    name="pr-review",
    help="Autonomous PR review agent powered by LangGraph, ChromaDB, and Claude.",
    add_completion=False,
)
console = Console()

_VERDICT_COLOR = {
    "APPROVE": "green",
    "REQUEST_CHANGES": "red",
    "COMMENT": "yellow",
}

_VERDICT_ICON = {
    "APPROVE": "✓",
    "REQUEST_CHANGES": "✗",
    "COMMENT": "◎",
}


@app.command()
def review(
    repo: str = typer.Argument(..., help="Repository in owner/repo format"),
    pr_number: int = typer.Argument(..., help="Pull request number"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-file issue table"),
    action: str = typer.Option("comment", "--action", help="comment (default) or merge (merge if APPROVE)"),
    llm: str = typer.Option("auto", "--llm", help="auto (Claude→Ollama fallback), claude, or local"),
    force: bool = typer.Option(False, "--force", help="merge regardless of verdict (overrides --action merge)"),
):
    """Run an AI-powered code review on a GitHub pull request and post it to GitHub."""
    try:
        owner, repo_name = repo.split("/", 1)
    except ValueError:
        console.print("[red]Error:[/red] repo must be owner/repo (e.g. octocat/Hello-World)")
        raise typer.Exit(1)

    if force and action != "merge":
        console.print("[red]Error:[/red] --force requires --action merge")
        raise typer.Exit(1)
    if action not in {"comment", "merge"}:
        console.print(f"[red]Error:[/red] --action must be 'comment' or 'merge', got '{action}'")
        raise typer.Exit(1)
    if llm not in {"auto", "claude", "local"}:
        console.print(f"[red]Error:[/red] --llm must be 'auto', 'claude', or 'local', got '{llm}'")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]PR Review AI Agent[/bold cyan]\n"
        f"[dim]Repository:[/dim] [yellow]{repo}[/yellow]   "
        f"[dim]PR:[/dim] [yellow]#{pr_number}[/yellow]",
        expand=False,
    ))

    with console.status("[bold green]Fetching PR and building context...[/bold green]", spinner="dots"):
        try:
            result = asyncio.run(run_review(owner, repo_name, pr_number, llm=llm))
        except ExceptionGroup as eg:
            def _flatten(group):
                for exc in group.exceptions:
                    if isinstance(exc, ExceptionGroup):
                        yield from _flatten(exc)
                    else:
                        yield exc
            for exc in _flatten(eg):
                console.print(f"\n[red]Error:[/red] {type(exc).__name__}: {exc}")
            raise typer.Exit(1)
        except Exception as exc:
            import traceback
            console.print(f"\n[red]Error during review:[/red] {type(exc).__name__}: {exc}")
            console.print(traceback.format_exc())
            raise typer.Exit(1)

    verdict = result.get("verdict", "COMMENT")
    head_sha = result.get("head_sha", "")
    color = _VERDICT_COLOR.get(verdict, "yellow")
    icon = _VERDICT_ICON.get(verdict, "◎")
    iterations = result.get("reflection_count", 1)

    console.print(
        f"\n[bold]Verdict:[/bold] [{color}]{icon} {verdict}[/{color}]   "
        f"[dim]({iterations} reflection iteration{'s' if iterations != 1 else ''})[/dim]\n"
    )

    console.print(Panel(
        Markdown(result.get("summary", "")),
        title="[bold]Review Summary[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))

    if verbose:
        file_reviews = [fr for fr in result.get("file_reviews", []) if fr.get("issues")]
        if file_reviews:
            table = Table(title="File Issues", box=box.ROUNDED, show_lines=True)
            table.add_column("File", style="cyan", no_wrap=True)
            table.add_column("Issues")
            for fr in file_reviews:
                table.add_row(fr["path"], "\n".join(f"• {i}" for i in fr["issues"]))
            console.print(table)

    should_merge = (action == "merge" and verdict == "APPROVE") or force
    if should_merge:
        try:
            merge_pr(owner, repo_name, pr_number, head_sha=head_sha)
            note = " (forced)" if force and verdict != "APPROVE" else ""
            console.print(f"\n[green]Review posted and PR merged{note}.[/green]")
        except Exception as exc:
            console.print(f"\n[green]Review posted.[/green] [red]Merge failed:[/red] {exc}")
    elif action == "merge" and verdict != "APPROVE":
        console.print(f"\n[green]Review posted.[/green] [yellow]Not merged — verdict was {verdict}. Use --force to override.[/yellow]")
    else:
        console.print("\n[green]Review posted to GitHub.[/green]")
