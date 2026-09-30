"""
OpenFGA Authorization Governance for LangGraph Agents
=====================================================

Demonstrates fine-grained authorization (FGA) as LangGraph tool middleware,
with structured LangSmith trace events, HITL escalation via ``interrupt()``,
and a scenario runner that exercises all four auth paths.

Run
---
    pip install -r requirements.txt
    python -m examples.legacy.langgraph_fga_governance
"""

import asyncio
import hashlib
import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import langsmith as ls
from dotenv import load_dotenv
from demo.paths import REPO_ROOT
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from langsmith.run_helpers import get_current_run_tree

from examples.legacy.agent_to_agent import delegate_to_summary_agent, invoke_summary_agent
from demo.fga import TOOL_RESOURCE_MAP, FGAStore, create_demo_store

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic

load_dotenv(REPO_ROOT / ".env")

# Module-level store: incident_id -> incident dict
# Populated when a conditional HITL decision is saved.
PENDING_CONDITIONAL_REQUESTS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Trace enrichment helpers
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
    rt.add_metadata(metadata)
    rt.add_tags(tags)
    rt.add_event(
        {
            "name": event_name,
            "time": datetime.now(timezone.utc).isoformat(),
            "kwargs": metadata,
        }
    )


# ---------------------------------------------------------------------------
# Demo tools
# ---------------------------------------------------------------------------


@tool
def fetch_user_profile(user_id: str) -> dict:
    """Fetch a user profile from the UserProfileAPI.

    Args:
        user_id: The ID of the user to fetch.
    """
    return {
        "user_id": user_id,
        "name": "Jane Doe",
        "email": "jane@example.com",
        "role": "engineer",
    }


@tool
def search_documents(query: str) -> list[dict]:
    """Search internal documents by keyword.

    Args:
        query: The search query string.
    """
    return [
        {"id": "doc-001", "title": "Q3 Report", "snippet": f"...{query}..."},
        {"id": "doc-002", "title": "Architecture RFC", "snippet": f"...{query}..."},
    ]


@tool
def write_document(doc_id: str, content: str) -> dict:
    """Write content to a document (requires elevated permissions).

    Args:
        doc_id: The document identifier.
        content: The content to write.
    """
    return {
        "status": "written",
        "doc_id": doc_id,
        "bytes_written": len(content),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@tool
def delete_document(doc_id: str) -> dict:
    """Delete a document (requires admin permissions).

    Args:
        doc_id: The document identifier to delete.
    """
    return {
        "status": "deleted",
        "doc_id": doc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


ALL_TOOLS = [
    fetch_user_profile,
    search_documents,
    write_document,
    delete_document,
    delegate_to_summary_agent,
]

ALLOWED_INVOCATION_MODES = {"tool", "handoff"}


# ---------------------------------------------------------------------------
# FGA middleware (awrap_tool_call signature)
# ---------------------------------------------------------------------------


def make_fga_middleware(
    fga_store: FGAStore,
) -> Callable:
    """Return an async middleware closure bound to the given FGA store."""

    async def fga_middleware(
        request: Any,  # ToolCallRequest
        execute: Callable[..., Awaitable[Any]],
    ) -> Any:  # ToolMessage | Command
        """FGA authorization middleware for ToolNode.

        1. Extract identity from config
        2. Resolve tool -> FGA resource + relation
        3. fga_store.check(user, relation, object)
        4. Log fga_check + fga_decision to trace
        5. If elevated: interrupt() for HITL
        6. Execute tool or return denial
        """
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        config = request.runtime.config
        configurable = config.get("configurable", {})

        # Both user and agent identities may be present; we check the user
        # first and fall back to the agent (delegated access).
        user_id = configurable.get("user_id")
        agent_id = configurable.get("agent_id")
        identity = user_id or agent_id or "unknown"
        # secondary_identity is checked if the primary is denied
        secondary_identity = agent_id if user_id and agent_id else None

        # Look up tool in resource map
        resource_info = TOOL_RESOURCE_MAP.get(tool_name)
        if resource_info is None:
            _enrich_trace(
                "fga_decision",
                {
                    "decision": "deny",
                    "user": identity,
                    "tool_name": tool_name,
                    "reasoning": f"Tool '{tool_name}' not registered in TOOL_RESOURCE_MAP",
                },
                ["fga-deny"],
            )
            return ToolMessage(
                content=f"Access denied: tool '{tool_name}' is not registered.",
                tool_call_id=tool_call_id,
                status="error",
            )

        fga_object = resource_info["object"]
        fga_relation = resource_info["relation"]
        sensitivity = resource_info["sensitivity"]
        resource_type = resource_info.get("resource_type", "tool")

        # Log the FGA check event
        _enrich_trace(
            "fga_check",
            {
                "user": identity,
                "relation": fga_relation,
                "object": fga_object,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "sensitivity": sensitivity,
                "resource_type": resource_type,
            },
            [],
        )

        # Perform the authorization check (try primary, then secondary identity)
        check_result = fga_store.check(identity, fga_relation, fga_object)
        effective_identity = identity
        if not check_result.allowed and secondary_identity:
            alt_result = fga_store.check(secondary_identity, fga_relation, fga_object)
            if alt_result.allowed:
                check_result = alt_result
                effective_identity = secondary_identity

        # Build decision tags
        decision_tags: list[str] = []
        if check_result.allowed:
            decision_tags.append("fga-allow")
        else:
            decision_tags.append("fga-deny")
        if sensitivity == "elevated":
            decision_tags.append("elevated-access")
            decision_tags.append("sensitive-resource")

        # Log the FGA decision event
        _enrich_trace(
            "fga_decision",
            {
                "decision": "allow" if check_result.allowed else "deny",
                "user": effective_identity,
                "relation": fga_relation,
                "object": fga_object,
                "tool_name": tool_name,
                "resource_type": resource_type,
                "reasoning": check_result.reasoning,
                "requires_elevation": check_result.requires_elevation,
            },
            decision_tags,
        )

        # Denied
        if not check_result.allowed:
            if resource_type == "agent":
                _enrich_trace(
                    "agent_call_denied",
                    {
                        "parent_agent_id": agent_id or "none",
                        "child_agent_id": fga_object,
                        "user": effective_identity,
                        "relation": fga_relation,
                        "reasoning": check_result.reasoning,
                    },
                    ["agent-to-agent", "agent-call-denied"],
                )
            print(f"  [FGA] DENIED: {effective_identity} -> {fga_relation} on {fga_object}")
            return ToolMessage(
                content=(
                    f"Access denied: {check_result.reasoning}. "
                    f"Contact your administrator to request '{fga_relation}' "
                    f"access on '{fga_object}'."
                ),
                tool_call_id=tool_call_id,
                status="error",
            )

        # Allowed but requires elevation -> HITL
        if check_result.requires_elevation:
            print(
                f"  [FGA] ELEVATION REQUIRED: {effective_identity} -> "
                f"{fga_relation} on {fga_object} (HITL interrupt)"
            )
            _enrich_trace(
                "hitl_request",
                {
                    "user": effective_identity,
                    "relation": fga_relation,
                    "object": fga_object,
                    "tool_name": tool_name,
                    "elevation_level": check_result.elevation_level,
                    "agent_id": agent_id or "none",
                },
                ["hitl-requested"],
            )

            # Pause the graph for human review
            human_decision = interrupt(
                {
                    "type": "elevation_request",
                    "tool_name": tool_name,
                    "resource": fga_object,
                    "relation": fga_relation,
                    "user": effective_identity,
                    "agent_id": agent_id or "none",
                    "elevation_level": check_result.elevation_level,
                    "message": (
                        f"Tool '{tool_name}' requires elevated '{fga_relation}' "
                        f"access on '{fga_object}' for identity '{effective_identity}'. Approve?"
                    ),
                }
            )

            # Process the human decision (returned when graph is resumed)
            decision     = human_decision.get("decision", "deny")
            reviewer     = human_decision.get("reviewer", "unknown")
            conditions   = human_decision.get("conditions", None)
            hitl_comment = human_decision.get("hitl_comment")

            if decision == "approve":
                hitl_tag = "hitl-approved"
            elif decision == "conditional":
                hitl_tag = "hitl-conditional"
            else:
                hitl_tag = "hitl-denied"

            hitl_response_meta = {
                "human_decision": decision,
                "conditions":     conditions,
                "reviewer":       reviewer,
                "user":           effective_identity,
                "tool_name":      tool_name,
            }
            if hitl_comment:
                hitl_response_meta["hitl_comment"] = hitl_comment

            _enrich_trace("hitl_response", hitl_response_meta, [hitl_tag])

            if decision == "approve":
                print(f"  [HITL] APPROVED by {reviewer}")
                result = await execute(request)
                return result

            if decision == "conditional":
                # Generate a stable, short incident ID from the request context + timestamp
                incident_id = hashlib.sha256(
                    f"{tool_name}:{fga_object}:{effective_identity}:"
                    f"{datetime.now(timezone.utc).isoformat()}".encode()
                ).hexdigest()[:12]

                incident = {
                    "incident_id":  incident_id,
                    "tool_name":    tool_name,
                    "resource":     fga_object,
                    "relation":     fga_relation,
                    "user":         effective_identity,
                    "agent_id":     agent_id or "none",
                    "reviewer":     reviewer,
                    "conditions":   conditions,
                    "hitl_comment": hitl_comment,
                    "status":       "pending_secondary_approval",
                    "created_at":   datetime.now(timezone.utc).isoformat(),
                    "thread_id":    configurable.get("thread_id"),
                }
                PENDING_CONDITIONAL_REQUESTS[incident_id] = incident
                print(f"  [HITL] CONDITIONAL by {reviewer} — saved as incident #{incident_id}")

                _enrich_trace(
                    "hitl_conditional_saved",
                    {**incident},
                    ["hitl-conditional", "incident-saved"],
                )

                return ToolMessage(
                    content=(
                        f"Your request to perform '{tool_name}' on '{fga_object}' has been "
                        f"saved as a pending conditional approval.\n\n"
                        f"Incident ID: {incident_id}\n"
                        f"Reviewer: {reviewer}\n"
                        f"Conditions noted: {conditions or 'none'}\n"
                        f"Status: Awaiting secondary approval\n\n"
                        f"Please reference incident #{incident_id} when following up."
                    ),
                    tool_call_id=tool_call_id,
                )

            else:
                print(f"  [HITL] DENIED by {reviewer}")
                return ToolMessage(
                    content=(
                        f"Access denied by reviewer '{reviewer}': "
                        f"elevated '{fga_relation}' on '{fga_object}' was not approved."
                    ),
                    tool_call_id=tool_call_id,
                    status="error",
                )

        # Allowed, no elevation needed
        print(f"  [FGA] ALLOWED: {effective_identity} -> {fga_relation} on {fga_object}")
        if resource_type == "agent":
            _enrich_trace(
                "agent_call_started",
                {
                    "parent_agent_id": agent_id or "none",
                    "child_agent_id": fga_object,
                    "user": effective_identity,
                    "relation": fga_relation,
                    "invocation_mode": "tool",
                },
                ["agent-to-agent", "agent-call-started"],
            )
        result = await execute(request)
        if resource_type == "agent":
            _enrich_trace(
                "agent_call_completed",
                {
                    "parent_agent_id": agent_id or "none",
                    "child_agent_id": fga_object,
                    "user": effective_identity,
                    "relation": fga_relation,
                    "invocation_mode": "tool",
                },
                ["agent-to-agent", "agent-call-completed"],
            )
        return result

    return fga_middleware


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _authorize_agent_handoff(
    fga_store: FGAStore,
    config: RunnableConfig,
) -> tuple[bool, str, dict[str, str], Any]:
    """Authorize an explicit parent -> child graph handoff.

    Handoffs do not pass through ToolNode, so they perform the equivalent FGA
    check here and emit the same audit vocabulary as the tool path.
    """
    configurable = config.get("configurable", {})
    user_id = configurable.get("user_id")
    parent_agent_id = configurable.get("agent_id")
    primary_identity = parent_agent_id or user_id or "unknown"
    secondary_identity = user_id if parent_agent_id and user_id else None
    resource_info = TOOL_RESOURCE_MAP["delegate_to_summary_agent"]
    fga_object = resource_info["object"]
    fga_relation = resource_info["relation"]

    _enrich_trace(
        "fga_check",
        {
            "user": primary_identity,
            "relation": fga_relation,
            "object": fga_object,
            "tool_name": "delegate_to_summary_agent",
            "resource_type": "agent",
            "invocation_mode": "handoff",
        },
        [],
    )

    check_result = fga_store.check(primary_identity, fga_relation, fga_object)
    effective_identity = primary_identity
    if not check_result.allowed and secondary_identity:
        alt_result = fga_store.check(secondary_identity, fga_relation, fga_object)
        if alt_result.allowed:
            check_result = alt_result
            effective_identity = secondary_identity

    _enrich_trace(
        "fga_decision",
        {
            "decision": "allow" if check_result.allowed else "deny",
            "user": effective_identity,
            "relation": fga_relation,
            "object": fga_object,
            "tool_name": "delegate_to_summary_agent",
            "resource_type": "agent",
            "invocation_mode": "handoff",
            "reasoning": check_result.reasoning,
        },
        ["fga-allow" if check_result.allowed else "fga-deny"],
    )

    if not check_result.allowed:
        _enrich_trace(
            "agent_call_denied",
            {
                "parent_agent_id": parent_agent_id or "none",
                "child_agent_id": fga_object,
                "user": effective_identity,
                "relation": fga_relation,
                "invocation_mode": "handoff",
                "reasoning": check_result.reasoning,
            },
            ["agent-to-agent", "agent-call-denied"],
        )
        return False, effective_identity, resource_info, check_result

    return True, effective_identity, resource_info, check_result


def build_fga_graph(
    tools: list,
    llm: "ChatAnthropic",
    fga_store: FGAStore,
):
    """Build a LangGraph agent with FGA middleware and HITL support."""
    model = llm.bind_tools(tools)
    middleware = make_fga_middleware(fga_store)
    tool_node = ToolNode(tools, awrap_tool_call=middleware)

    def route_invocation(state: MessagesState, config: RunnableConfig) -> str:
        mode = config.get("configurable", {}).get("invocation_mode", "tool")
        if mode not in ALLOWED_INVOCATION_MODES:
            raise ValueError(
                f"Unsupported invocation_mode '{mode}'. "
                f"Expected one of {sorted(ALLOWED_INVOCATION_MODES)}."
            )
        return mode

    def router(state: MessagesState) -> dict:
        return {}

    async def handoff(state: MessagesState, config: RunnableConfig):
        source_text = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage) and isinstance(message.content, str)
            ),
            "",
        )
        allowed, effective_identity, resource_info, check_result = _authorize_agent_handoff(
            fga_store, config
        )
        child_agent_id = resource_info["object"]
        if not allowed:
            content = f"Access denied: {check_result.reasoning}."
            event = {
                "event": "agent_call_denied",
                "parent_agent_id": config.get("configurable", {}).get("agent_id", "none"),
                "child_agent_id": child_agent_id,
                "invocation_mode": "handoff",
            }
            return {
                "messages": [
                    AIMessage(content=content, additional_kwargs={"agent_call": event})
                ]
            }

        _enrich_trace(
            "agent_call_started",
            {
                "parent_agent_id": config.get("configurable", {}).get("agent_id", "none"),
                "child_agent_id": child_agent_id,
                "user": effective_identity,
                "relation": resource_info["relation"],
                "invocation_mode": "handoff",
            },
            ["agent-to-agent", "agent-call-started"],
        )
        result = await invoke_summary_agent(source_text)
        _enrich_trace(
            "agent_call_completed",
            {
                "parent_agent_id": config.get("configurable", {}).get("agent_id", "none"),
                "child_agent_id": child_agent_id,
                "user": effective_identity,
                "relation": resource_info["relation"],
                "invocation_mode": "handoff",
                "key_point_count": len(result.get("key_points", [])),
            },
            ["agent-to-agent", "agent-call-completed"],
        )
        event = {
            "event": "agent_call_completed",
            "parent_agent_id": config.get("configurable", {}).get("agent_id", "none"),
            "child_agent_id": child_agent_id,
            "invocation_mode": "handoff",
            "key_point_count": len(result.get("key_points", [])),
        }
        return {
            "messages": [
                AIMessage(
                    content=result["summary"],
                    additional_kwargs={"agent_call": event},
                )
            ]
        }

    async def agent(state: MessagesState):
        # Filter escalation notification messages out of the LLM context.
        # The Anthropic API requires every tool_use block to be immediately
        # followed by a tool_result block; notification AIMessages between them
        # violate that constraint.
        messages = [
            m for m in state["messages"]
            if getattr(m, "name", None) != "escalation_notification"
        ]
        response = await model.ainvoke(messages)

        # If the model wants to call elevated tools, emit a user-visible
        # notification message BEFORE the tool-calling AIMessage in state.
        # ToolNode searches messages in reverse for the last AIMessage; placing
        # the notification first ensures ToolNode still sees the tool-calling
        # AIMessage as "latest" and correctly executes its tool_calls.
        tool_calls = getattr(response, "tool_calls", None) or []
        elevated = [
            tc for tc in tool_calls
            if TOOL_RESOURCE_MAP.get(tc["name"], {}).get("sensitivity") == "elevated"
        ]
        if elevated:
            names = ", ".join(f"'{tc['name']}'" for tc in elevated)
            resources = ", ".join(
                f"'{TOOL_RESOURCE_MAP[tc['name']]['object']}'" for tc in elevated
            )
            notification = AIMessage(
                content=(
                    f"Your request to perform {names} on {resources} requires elevated "
                    f"access and has been escalated for human review. "
                    f"The operation is now pending approval — you will be notified once "
                    f"a decision has been made."
                ),
                name="escalation_notification",
            )
            # notification first, tool-calling message last — preserves ToolNode lookup
            return {"messages": [notification, response]}

        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("router", router)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_node("handoff", handoff)
    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        route_invocation,
        {"tool": "agent", "handoff": "handoff"},
    )
    builder.add_conditional_edges("agent", should_continue)
    builder.add_edge("tools", "agent")

    # InMemorySaver is required for interrupt() to work
    return builder.compile(checkpointer=InMemorySaver())


async def invoke_main_agent(
    graph: Any,
    user_msg: str,
    *,
    user_id: str,
    agent_id: str,
    invocation_mode: str = "tool",
) -> dict[str, Any]:
    """Invoke a compiled parent agent through its public graph boundary.

    This helper is intentionally provider-neutral so tests and local demos can
    use a scripted model while production callers pass the real LLM-backed
    graph.  It returns sanitized observable events rather than raw LangGraph
    state or message objects.
    """
    if invocation_mode not in ALLOWED_INVOCATION_MODES:
        raise ValueError(
            f"Unsupported invocation_mode '{invocation_mode}'. "
            f"Expected one of {sorted(ALLOWED_INVOCATION_MODES)}."
        )
    if not isinstance(user_msg, str) or not user_msg.strip():
        raise ValueError("user_msg must be a non-empty string")

    thread_id = f"agent-{uuid.uuid4().hex}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "invocation_mode": invocation_mode,
        },
        "run_name": "generic_agent_invocation",
        "tags": ["fga-governance-demo", "generic-agent-invocation"],
    }

    events: list[dict[str, Any]] = []
    final_response = ""
    interrupted = False

    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=user_msg.strip())]},
        config,
        stream_mode="updates",
    ):
        if "__interrupt__" in chunk:
            interrupted = True
            events.append({"event": "hitl_request"})
            continue

        if "agent" in chunk:
            for message in chunk["agent"].get("messages", []):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    resource = TOOL_RESOURCE_MAP.get(tool_call["name"], {})
                    events.append(
                        {
                            "event": "tool_intent",
                            "tool_name": tool_call["name"],
                            "resource": resource.get("object", ""),
                            "resource_type": resource.get("resource_type", "tool"),
                        }
                    )
                if getattr(message, "content", None) and not getattr(
                    message, "tool_calls", None
                ):
                    final_response = str(message.content)

        if "tools" in chunk:
            for message in chunk["tools"].get("messages", []):
                tool_call_id = getattr(message, "tool_call_id", "")
                resource = {}
                tool_name = "unknown"
                # The tool name is recoverable from the preceding intent while
                # keeping raw tool arguments out of the result.
                for event in reversed(events):
                    if event["event"] == "tool_intent":
                        tool_name = event["tool_name"]
                        resource = TOOL_RESOURCE_MAP.get(tool_name, {})
                        break
                is_error = getattr(message, "status", None) == "error"
                events.append(
                    {
                        "event": "fga_deny" if is_error else "fga_allow",
                        "tool_name": tool_name,
                        "resource": resource.get("object", ""),
                        "resource_type": resource.get("resource_type", "tool"),
                        "tool_call_id": tool_call_id,
                    }
                )
                if resource.get("resource_type") == "agent":
                    events.append(
                        {
                            "event": "agent_call_denied" if is_error else "agent_call_started",
                            "parent_agent_id": agent_id,
                            "child_agent_id": resource.get("object", ""),
                            "invocation_mode": "tool",
                        }
                    )
                    if not is_error:
                        events.append(
                            {
                                "event": "agent_call_completed",
                                "parent_agent_id": agent_id,
                                "child_agent_id": resource.get("object", ""),
                                "invocation_mode": "tool",
                            }
                        )

        if "handoff" in chunk:
            for message in chunk["handoff"].get("messages", []):
                agent_call = getattr(message, "additional_kwargs", {}).get("agent_call")
                if agent_call:
                    event_name = agent_call.get("event", "agent_call_completed")
                    if event_name == "agent_call_completed":
                        events.append(
                            {
                                "event": "agent_call_started",
                                "parent_agent_id": agent_call.get("parent_agent_id", agent_id),
                                "child_agent_id": agent_call.get("child_agent_id", ""),
                                "invocation_mode": "handoff",
                            }
                        )
                    events.append(
                        {
                            "event": event_name,
                            "parent_agent_id": agent_call.get("parent_agent_id", agent_id),
                            "child_agent_id": agent_call.get("child_agent_id", ""),
                            "invocation_mode": "handoff",
                        }
                    )
                if getattr(message, "content", None):
                    final_response = str(message.content)

    return {
        "thread_id": thread_id,
        "response": final_response,
        "interrupted": interrupted,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "Allowed (Standard Read)",
        "description": "alice searches documents via research-bot — should succeed",
        "user_msg": "Search for documents about Q3 revenue.",
        "user_id": "user:alice",
        "agent_id": "agent:research-bot",
        "expected_tags": ["fga-allow"],
        "expect_hitl": False,
    },
    {
        "name": "Denied (No Permission)",
        "description": "bob tries to fetch a user profile — has no tool executor permission",
        "user_msg": "Fetch the profile for user u-5678.",
        "user_id": "user:bob",
        "agent_id": "agent:research-bot",
        "expected_tags": ["fga-deny"],
        "expect_hitl": False,
    },
    {
        "name": "Elevated (HITL Approval)",
        "description": "alice writes to report-2024 — requires HITL approval",
        "user_msg": "Write the following content to document report-2024: 'Q3 quarterly summary: Revenue increased 15% year-over-year.'",
        "user_id": "user:alice",
        "agent_id": "agent:research-bot",
        "expected_tags": ["fga-allow", "hitl-requested", "hitl-approved"],
        "expect_hitl": True,
        "hitl_response": {
            "decision": "conditional",
            "reviewer": "jane",
            "conditions": "Backup first",
        },
    },
    {
        "name": "Admin (Full Access + HITL)",
        "description": "admin-bot deletes report-2024 — admin with HITL",
        "user_msg": "Delete document report-2024 immediately.",
        "user_id": "user:alice",
        "agent_id": "agent:admin-bot",
        "expected_tags": ["fga-allow", "hitl-requested", "hitl-approved"],
        "expect_hitl": True,
        "hitl_response": {
            "decision": "approve",
            "reviewer": "security-team",
            "conditions": "Backup confirmed",
        },
    },
]


async def run_scenario(
    scenario: dict,
    graph: Any,
    scenario_index: int,
) -> dict:
    """Execute a single scenario, handling HITL interrupts."""
    name = scenario["name"]
    thread_id = f"scenario-{scenario_index}-{uuid.uuid4().hex[:8]}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": scenario["user_id"],
            "agent_id": scenario["agent_id"],
        },
        "run_name": f"fga_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
        "tags": ["fga-governance-demo"],
    }

    print(f"\n{'=' * 60}")
    print(f"Scenario: {name}")
    print(f"  User: {scenario['user_id']}, Agent: {scenario['agent_id']}")
    print(f"  Message: {scenario['user_msg']}")
    print("=" * 60)

    # First invocation
    result = None
    interrupted = False

    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=scenario["user_msg"])]},
        config=config,
        stream_mode="updates",
    ):
        if isinstance(chunk, dict) and "__interrupt" in chunk:
            interrupted = True
            break
        result = chunk

    # Check state for interrupts (handles both streaming and non-streaming detection)
    if not interrupted:
        state = await graph.aget_state(config)
        if state.tasks and any(
            getattr(t, "interrupts", None) for t in state.tasks
        ):
            interrupted = True

    if interrupted:
        state = await graph.aget_state(config)
        for task in state.tasks or []:
            for intr in getattr(task, "interrupts", []):
                print(f"\n  [INTERRUPT] {intr.value}")

    if interrupted and scenario.get("expect_hitl"):
        hitl_response = scenario["hitl_response"]
        print(f"  [HUMAN] Responding with: {hitl_response}")

        # Resume with human decision
        async for chunk in graph.astream(
            Command(resume=hitl_response),
            config=config,
            stream_mode="updates",
        ):
            if isinstance(chunk, dict) and "__interrupt" in chunk:
                print(f"  [INTERRUPT] Unexpected second interrupt")
                break
            result = chunk

    # Extract final response
    if result:
        for node_name, node_output in result.items():
            if node_name.startswith("__"):
                continue
            if not isinstance(node_output, dict):
                continue
            messages = node_output.get("messages", [])
            if messages:
                final_msg = messages[-1]
                content = getattr(final_msg, "content", str(final_msg))
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"\n  Final response ({node_name}): {content}")
    else:
        print("\n  No final response captured.")

    return {"scenario": name, "thread_id": thread_id, "interrupted": interrupted}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@ls.traceable(name="fga_governance_demo")
async def main():
    base_url = os.environ.get("BASE_URL")
    if base_url and base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/")[:-3]
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        **({"base_url": base_url} if base_url else {}),
    )

    fga_store = create_demo_store()
    graph = build_fga_graph(ALL_TOOLS, llm, fga_store)

    print("OpenFGA Authorization Governance Demo")
    print("=" * 60)
    print(f"FGA Store loaded with {len(fga_store.list_tuples())} tuples")
    print(f"Tools registered: {[t.name for t in ALL_TOOLS]}")
    print(f"Resource mappings: {list(TOOL_RESOURCE_MAP.keys())}")

    results = []
    for i, scenario in enumerate(SCENARIOS):
        result = await run_scenario(scenario, graph, i)
        results.append(result)

    # Summary
    print(f"\n\n{'=' * 60}")
    print("SCENARIO SUMMARY")
    print("=" * 60)
    for r in results:
        hitl_marker = " (HITL)" if r["interrupted"] else ""
        print(f"  {r['scenario']}: thread={r['thread_id']}{hitl_marker}")


if __name__ == "__main__":
    asyncio.run(main())
