"""Small deterministic child agent used by the FGA agent-to-agent demo.

The child graph intentionally has no tools and no dynamic code execution.  It
extracts a few useful sentences and then composes a short summary.  The same
compiled graph can be called directly or through the parent agent's delegation
tool.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph


MAX_INPUT_CHARS = 4_000
MAX_POINTS = 3


class SummaryState(TypedDict, total=False):
    source_text: str
    key_points: list[str]
    summary: str


def _extract_key_points(state: SummaryState) -> dict[str, list[str]]:
    """Keep the first few non-empty sentences as the child's evidence."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", state["source_text"])
        if sentence.strip()
    ]
    return {"key_points": sentences[:MAX_POINTS]}


def _compose_summary(state: SummaryState) -> dict[str, str]:
    """Compose a stable, inspectable result without making another model call."""
    points = state.get("key_points", [])
    if not points:
        summary = "The summary agent found no substantive points to report."
    else:
        summary = "Summary: " + " ".join(points)
    return {"summary": summary}


def build_summary_agent():
    """Compile the two-node child graph."""
    builder = StateGraph(SummaryState)
    builder.add_node("extract_key_points", _extract_key_points)
    builder.add_node("compose_summary", _compose_summary)
    builder.set_entry_point("extract_key_points")
    builder.add_edge("extract_key_points", "compose_summary")
    builder.add_edge("compose_summary", END)
    return builder.compile()


SUMMARY_AGENT = build_summary_agent()


async def invoke_summary_agent(text: str) -> dict[str, Any]:
    """Invoke the child graph with bounded, validated input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("summary input must be a non-empty string")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"summary input exceeds {MAX_INPUT_CHARS} characters")

    result = await SUMMARY_AGENT.ainvoke({"source_text": text.strip()})
    return {
        "summary": result.get("summary", ""),
        "key_points": result.get("key_points", []),
        "agent_id": "agent:summary-bot",
    }


@tool
async def delegate_to_summary_agent(text: str) -> dict[str, Any]:
    """Delegate text to the governed summary child agent.

    This tool is intentionally narrow: authorization is performed by the
    parent graph middleware before this function can execute.
    """
    return await invoke_summary_agent(text)
