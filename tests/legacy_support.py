"""Deterministic parent-agent model for graph-level tests and demos."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class ScriptedParentModel:
    """Emit one safe delegation call, then summarize the child result.

    This implements only the small LangChain model surface required by
    ``build_fga_graph``.  It is deliberately deterministic and never executes
    code or selects tools other than the registered summary delegation tool.
    """

    def bind_tools(self, tools: list[Any]) -> "ScriptedParentModel":
        tool_names = {getattr(candidate, "name", "") for candidate in tools}
        if "delegate_to_summary_agent" not in tool_names:
            raise ValueError("scripted model requires the summary delegation tool")
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        last = messages[-1]

        if isinstance(last, HumanMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate_to_summary_agent",
                        "args": {"text": last.content[:4_000]},
                        "id": f"scripted-delegate-{uuid.uuid4().hex[:12]}",
                    }
                ],
            )

        if isinstance(last, ToolMessage):
            if getattr(last, "status", None) == "error":
                return AIMessage(
                    content=f"The parent agent could not complete delegation: {last.content}"
                )
            try:
                result = json.loads(str(last.content))
            except (TypeError, json.JSONDecodeError):
                return AIMessage(content=f"Child agent result: {last.content}")
            return AIMessage(content=str(result.get("summary", last.content)))

        return AIMessage(content="The parent agent completed without delegation.")
