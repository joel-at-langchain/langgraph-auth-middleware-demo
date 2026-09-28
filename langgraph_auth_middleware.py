"""
LangGraph Auth Failure Middleware Demo
======================================

Demonstrates three strategies for handling tool-call authentication failures
with custom middleware, each producing rich LangSmith trace annotations.

Strategies
----------
1. Hard Error      - log to trace, re-raise -> graph crashes
2. Soft Error      - log to trace, return error ToolMessage -> agent responds gracefully
3. Retry+Fallback  - retry N times with backoff, then fallback ToolMessage -> agent responds

Run
---
    pip install -r requirements.txt
    # fill in .env with real keys
    python langgraph_auth_middleware.py
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Callable

import langsmith as ls
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langsmith.run_helpers import get_current_run_tree

load_dotenv()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class AuthenticationError(Exception):
    """Raised when an external API rejects credentials."""

    def __init__(self, service: str, status_code: int):
        self.service = service
        self.status_code = status_code
        super().__init__(f"Authentication failed for {service} (HTTP {status_code})")


# ---------------------------------------------------------------------------
# Trace enrichment helper
# ---------------------------------------------------------------------------


def _enrich_trace(
    event_name: str,
    metadata: dict[str, Any],
    tags: list[str],
) -> None:
    """Add an event, metadata, and tags to the current LangSmith span."""
    rt = get_current_run_tree()
    if rt is None:
        return

    # Metadata
    if rt.extra is None:
        rt.extra = {}
    rt.extra.setdefault("metadata", {}).update(metadata)

    # Tags
    if rt.tags is None:
        rt.tags = []
    for t in tags:
        if t not in rt.tags:
            rt.tags.append(t)

    # Event
    if rt.events is None:
        rt.events = []
    rt.events.append(
        {
            "name": event_name,
            "time": datetime.now(timezone.utc).isoformat(),
            "kwargs": metadata,
        }
    )


# ---------------------------------------------------------------------------
# Simulated tool -- always fails auth
# ---------------------------------------------------------------------------


@tool
def fetch_user_profile(user_id: str) -> dict:
    """Fetch a user profile from the UserProfileAPI.

    Args:
        user_id: The ID of the user to fetch.
    """
    raise AuthenticationError(service="UserProfileAPI", status_code=403)


# ---------------------------------------------------------------------------
# Middleware wrappers
# ---------------------------------------------------------------------------


async def hard_error_wrapper(tool_call: dict, handler: Callable) -> ToolMessage:
    """Log auth failure to trace and re-raise so the graph crashes."""
    try:
        return await handler(tool_call)
    except AuthenticationError as exc:
        _enrich_trace(
            "auth_failure",
            {
                "error_type": type(exc).__name__,
                "error_service": exc.service,
                "error_status_code": exc.status_code,
                "strategy": "hard_error",
            },
            ["auth-failure", "hard-error"],
        )
        raise


async def soft_error_wrapper(tool_call: dict, handler: Callable) -> ToolMessage:
    """Log auth failure to trace and return an error ToolMessage."""
    try:
        return await handler(tool_call)
    except AuthenticationError as exc:
        _enrich_trace(
            "auth_failure_handled",
            {
                "error_type": type(exc).__name__,
                "error_service": exc.service,
                "error_status_code": exc.status_code,
                "strategy": "soft_error",
                "handled": True,
            },
            ["auth-failure", "soft-error", "handled"],
        )
        return ToolMessage(
            content=(
                f"Error: Authentication failed for {exc.service} "
                f"(HTTP {exc.status_code}). The service rejected the request."
            ),
            tool_call_id=tool_call["id"],
            status="error",
        )


def make_retry_wrapper(
    max_retries: int = 3,
    base_delay: float = 0.3,
) -> Callable:
    """Return a retry-with-backoff middleware closure."""

    async def retry_wrapper(tool_call: dict, handler: Callable) -> ToolMessage:
        last_exc: AuthenticationError | None = None

        for attempt in range(1, max_retries + 1):
            try:
                print(f"  [RETRY] Attempt {attempt}/{max_retries}...")
                return await handler(tool_call)
            except AuthenticationError as exc:
                last_exc = exc
                _enrich_trace(
                    "retry_attempt",
                    {
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "error_type": type(exc).__name__,
                        "service": exc.service,
                        "status_code": exc.status_code,
                    },
                    [],
                )
                _enrich_trace(
                    "retry_failed",
                    {"attempt": attempt, "error": str(exc)},
                    [],
                )
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * 2 ** (attempt - 1))

        # All retries exhausted
        _enrich_trace(
            "retry_exhausted",
            {
                "error_type": type(last_exc).__name__,
                "error_service": last_exc.service,
                "error_status_code": last_exc.status_code,
                "strategy": "retry",
                "total_attempts": max_retries,
                "all_retries_failed": True,
            },
            ["auth-failure", "retry-exhausted", "fallback"],
        )
        return ToolMessage(
            content=(
                f"Error: Authentication failed for {last_exc.service} "
                f"(HTTP {last_exc.status_code}) after {max_retries} attempts. "
                f"All retries exhausted."
            ),
            tool_call_id=tool_call["id"],
            status="error",
        )

    return retry_wrapper


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(
    tools: list,
    llm: ChatAnthropic,
    middleware: Callable,
):
    """Build a LangGraph agent whose tool node uses the given middleware.

    The middleware receives ``(tool_call, handler)`` and must return a
    ``ToolMessage`` or raise an exception (hard-error strategy).
    """
    model = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def agent(state: MessagesState):
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    async def tool_node(state: MessagesState):
        outputs: list[ToolMessage] = []
        for tc in state["messages"][-1].tool_calls:
            tool_fn = tools_by_name[tc["name"]]

            async def handler(call: dict, _tool=tool_fn) -> ToolMessage:
                result = await _tool.ainvoke(call["args"])
                return ToolMessage(content=str(result), tool_call_id=call["id"])

            result = await middleware(tc, handler)
            outputs.append(result)
        return {"messages": outputs}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")
    return builder.compile()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

STRATEGIES: list[tuple[str, Callable, list[str]]] = [
    ("Hard Error", hard_error_wrapper, ["hard-error"]),
    ("Soft Error", soft_error_wrapper, ["soft-error"]),
    (
        "Retry with Fallback",
        make_retry_wrapper(max_retries=3, base_delay=0.3),
        ["retry"],
    ),
]


# ---------------------------------------------------------------------------
# Main -- one parent trace wrapping all three strategy runs
# ---------------------------------------------------------------------------


@ls.traceable(name="auth_middleware_demo")
async def main():
    base_url = os.environ.get("BASE_URL")
    # The Anthropic SDK appends /v1/messages, so strip a trailing /v1
    # to avoid a doubled path through the gateway.
    if base_url and base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/")[:-3]
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        **({"base_url": base_url} if base_url else {}),
    )
    tools = [fetch_user_profile]
    user_msg = "Please fetch the profile for user u-1234."

    for name, wrapper, tags in STRATEGIES:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {name}")
        print("=" * 60)

        graph = build_graph(tools, llm, middleware=wrapper)
        config = {
            "run_name": f"auth_{name.lower().replace(' ', '_')}",
            "tags": tags,
        }

        try:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=user_msg)]},
                config=config,
            )
            final = result["messages"][-1].content
            print(f"\nAgent response:\n{final}")
        except AuthenticationError as exc:
            print(f"\n[HARD ERROR] Graph crashed as expected!")
            print(f"  Exception: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
