"""
LangGraph agent with reflection loop.

Graph shape:
  fetch_pr → index_context → retrieve_context → analyze_diff
                                    ↑                  │
                                    └── (reflection) ──┘ needs_more_context=True
                                                       │
                                                       └──→ post_review → END
"""

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import anthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, field_validator

from . import rag
from .config import CLAUDE_MODEL, LLM_MODEL, MAX_REFLECTION_LOOPS


# ── State ──────────────────────────────────────────────────────────────────────

class PRReviewState(TypedDict):
    owner: str
    repo: str
    pr_number: int
    # fetched from GitHub via MCP
    diffs: list[dict]
    pr_description: str
    head_sha: str
    # RAG
    context_chunks: list[str]
    context_query: str
    # LLM output
    file_reviews: list[dict]   # [{path, issues: [str]}]
    verdict: str               # APPROVE | REQUEST_CHANGES | COMMENT
    summary: str
    # reflection control
    needs_more_context: bool
    reflection_count: int


# ── Structured output schema ───────────────────────────────────────────────────

class FileIssue(BaseModel):
    path: str
    issues: list[str]

    @field_validator("issues", mode="before")
    @classmethod
    def coerce_issues(cls, v):
        result = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                desc = item.get("description") or item.get("issue") or str(item)
                issue_type = item.get("issue_type") or item.get("type")
                result.append(f"{issue_type}: {desc}" if issue_type else desc)
            else:
                result.append(str(item))
        return result


class PRAnalysis(BaseModel):
    file_reviews: list[FileIssue] = []
    verdict: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"] = "COMMENT"
    summary: str = ""
    needs_more_context: bool = False
    context_query: str = ""


# ── LLM backends ──────────────────────────────────────────────────────────────

async def _analyze_with_claude(system_prompt: str, user_content: str) -> PRAnalysis:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is not set — add it to .env to use Claude")
    client = anthropic.AsyncAnthropic()
    # messages.parse, output_format, and parsed_output are valid SDK APIs —
    # verified present in anthropic>=0.50.0 via inspect.signature and ParsedMessage
    response = await client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        output_format=PRAnalysis,
    )
    return response.parsed_output


async def _analyze_with_ollama(system_prompt: str, user_content: str) -> PRAnalysis:
    llm = ChatOllama(model=LLM_MODEL, format=PRAnalysis.model_json_schema())
    raw = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])
    try:
        data = json.loads(raw.content)
        return PRAnalysis.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"Local LLM returned unparseable output: {exc}\nRaw: {raw.content[:300]}"
        ) from exc


def _format_diffs(diffs: list[dict]) -> str:
    parts = []
    for d in diffs:
        parts.append(
            f"### {d['filename']} ({d['status']}, +{d['additions']} -{d['deletions']})\n"
            f"```diff\n{d['patch']}\n```"
        )
    return "\n\n".join(parts)


def _unwrap_tool_result(result) -> str:
    """Normalise MCP tool return values to a valid JSON string."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result)
    if isinstance(result, list) and result:
        # MCP returns [TextContent(type='text', text='...')] — extract the text
        item = result[0]
        if hasattr(item, "text"):
            return item.text
        if isinstance(item, dict) and "text" in item:
            return item["text"]
        if isinstance(item, str):
            return item
        return json.dumps(item)
    if hasattr(result, "text"):
        return result.text
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)


# ── Nodes ──────────────────────────────────────────────────────────────────────

async def fetch_pr_node(state: PRReviewState, config: RunnableConfig) -> dict:
    tool_map = config["configurable"]["tool_map"]
    raw = await tool_map["get_pr_diff"].ainvoke({
        "owner": state["owner"],
        "repo": state["repo"],
        "pr_number": state["pr_number"],
    })
    data = json.loads(_unwrap_tool_result(raw))
    return {
        "diffs": data["diffs"],
        "pr_description": data.get("pr_description", ""),
        "head_sha": data.get("head_sha", ""),
    }


async def index_context_node(state: PRReviewState, config: RunnableConfig) -> dict:
    """Fetch files adjacent to the diff and index them into ChromaDB."""
    tool_map = config["configurable"]["tool_map"]
    filenames = [d["filename"] for d in state["diffs"]]

    raw = await tool_map["get_adjacent_files"].ainvoke({
        "owner": state["owner"],
        "repo": state["repo"],
        "filenames_json": json.dumps(filenames),
    })
    files = json.loads(_unwrap_tool_result(raw))

    rag.clear_collection()
    rag.index_files(files)

    return {"context_chunks": [], "context_query": ""}


def retrieve_context_node(state: PRReviewState) -> dict:
    """Semantic search over ChromaDB using the current context query."""
    query = state.get("context_query") or (
        "code context for: " + ", ".join(d["filename"] for d in state["diffs"])
    )
    new_chunks = rag.retrieve_context(query)
    existing = state.get("context_chunks", [])
    # accumulate unique chunks across reflection iterations
    combined = existing + [c for c in new_chunks if c not in existing]
    return {"context_chunks": combined}


async def analyze_diff_node(state: PRReviewState, config: RunnableConfig) -> dict:
    """Ask the LLM to review the diff given the retrieved context."""
    iteration = state["reflection_count"] + 1
    max_iter = MAX_REFLECTION_LOOPS + 1
    at_limit = iteration >= max_iter

    context_text = (
        "\n\n---\n\n".join(state["context_chunks"])
        if state["context_chunks"]
        else "No additional context retrieved."
    )

    ignore_file = Path(__file__).parent.parent / ".pr-review-ignore"
    ignore_entries = ""
    if ignore_file.exists():
        lines = [l.strip() for l in ignore_file.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        if lines:
            ignore_entries = "\n\nThe following have been explicitly verified — do not flag them:\n" + \
                             "\n".join(f"- {l}" for l in lines)

    system = SystemMessage(content=f"""You are a senior software engineer performing a thorough code review.

For each changed file, list only the most important issues (max 3 per file, one short sentence each).
Prefix every issue with one of these labels:
- [critical] — security vulnerabilities, data loss, crashes
- [issue] — bugs, wrong logic, missing error handling
- [note] — style, naming, minor suggestions (non-blocking)

summary: 1-2 sentences max. State the verdict reason and the single most important issue.

verdict:
- APPROVE: no [critical] or [issue] labels found
- REQUEST_CHANGES: any [critical] or [issue] label present
- COMMENT: only [note] labels, no blocking problems

Set needs_more_context=true ONLY if seeing specific related code would meaningfully change your verdict.
On the final iteration always set needs_more_context=false.

You MUST respond with valid JSON matching this exact structure:
{{
  "file_reviews": [
    {{"path": "filename.py", "issues": ["issue 1", "issue 2"]}}
  ],
  "verdict": "APPROVE" | "REQUEST_CHANGES" | "COMMENT",
  "summary": "one or two sentence summary",
  "needs_more_context": false,
  "context_query": ""
}}
{ignore_entries}""")

    human = HumanMessage(content=f"""**PR Description:**
{state['pr_description'] or '(no description provided)'}

**Changed Files:**
{_format_diffs(state['diffs'])}

**Repository Context (semantic search results):**
{context_text}

Iteration {iteration}/{max_iter}.{"This is the final iteration — do not request more context." if at_limit else ""}
""")

    system_prompt = system.content
    user_content = human.content
    llm_choice = config["configurable"].get("llm", "auto")

    if llm_choice == "local":
        analysis = await _analyze_with_ollama(system_prompt, user_content)
    elif llm_choice == "claude":
        analysis = await _analyze_with_claude(system_prompt, user_content)
    else:  # auto: Claude with Ollama fallback on API/auth errors only
        try:
            analysis = await _analyze_with_claude(system_prompt, user_content)
        # anthropic.AuthenticationError subclasses APIError — all auth/network/credit
        # failures are caught here; verified via issubclass(AuthenticationError, APIError)
        except (anthropic.APIError, ValueError) as exc:
            print(f"\n[Claude unavailable: {type(exc).__name__} — falling back to local LLM]")
            try:
                analysis = await _analyze_with_ollama(system_prompt, user_content)
            except Exception as inner_exc:
                raise ValueError(
                    f"Both Claude API and local LLM failed.\n"
                    f"Claude: {exc}\nOllama: {inner_exc}"
                ) from inner_exc

    if not analysis.summary and not any(r.issues for r in analysis.file_reviews):
        raise ValueError("LLM returned an empty analysis (no summary, no file issues).")

    return {
        "file_reviews": [{"path": r.path, "issues": r.issues} for r in analysis.file_reviews],
        "verdict": analysis.verdict,
        "summary": analysis.summary,
        "needs_more_context": analysis.needs_more_context and not at_limit,
        "context_query": analysis.context_query,
        "reflection_count": state["reflection_count"] + 1,
    }


async def post_review_node(state: PRReviewState, config: RunnableConfig) -> dict:
    """Post the final review to GitHub via MCP, then merge if approved."""
    if not state.get("summary") and not state.get("file_reviews"):
        raise ValueError("Refusing to post an empty review to GitHub.")

    tool_map = config["configurable"]["tool_map"]
    await tool_map["post_pr_review"].ainvoke({
        "owner": state["owner"],
        "repo": state["repo"],
        "pr_number": state["pr_number"],
        "summary": state["summary"],
        "verdict": state["verdict"],
        "file_reviews_json": json.dumps(state["file_reviews"]),
        "llm": config["configurable"].get("llm", "auto"),
    })
    return {}


# ── Router ─────────────────────────────────────────────────────────────────────

def should_continue(state: PRReviewState) -> str:
    """Route back to retrieve_context for another pass, or proceed to post_review."""
    if state.get("needs_more_context"):
        return "retrieve_context"
    return "post_review"


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(PRReviewState)

    g.add_node("fetch_pr", fetch_pr_node)
    g.add_node("index_context", index_context_node)
    g.add_node("retrieve_context", retrieve_context_node)
    g.add_node("analyze_diff", analyze_diff_node)
    g.add_node("post_review", post_review_node)

    g.set_entry_point("fetch_pr")
    g.add_edge("fetch_pr", "index_context")
    g.add_edge("index_context", "retrieve_context")
    g.add_edge("retrieve_context", "analyze_diff")
    g.add_conditional_edges(
        "analyze_diff",
        should_continue,
        {"retrieve_context": "retrieve_context", "post_review": "post_review"},
    )
    g.add_edge("post_review", END)

    return g.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────

async def run_review(owner: str, repo: str, pr_number: int, llm: str = "auto") -> dict:
    """
    Launch the MCP server as a subprocess, wire its tools into the LangGraph
    agent, and run the full PR review workflow end-to-end.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_server"],
        cwd=str(Path(__file__).parent.parent),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            tool_map = {t.name: t for t in tools}

            graph = build_graph()
            initial_state: PRReviewState = {
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "diffs": [],
                "pr_description": "",
                "head_sha": "",
                "context_chunks": [],
                "context_query": "",
                "file_reviews": [],
                "verdict": "",
                "summary": "",
                "needs_more_context": False,
                "reflection_count": 0,
            }

            return await graph.ainvoke(
                initial_state,
                config={"configurable": {"tool_map": tool_map, "llm": llm}},
            )
