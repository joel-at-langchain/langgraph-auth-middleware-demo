"""Tenant-scoped review queue backed by actual LangGraph interrupts.

Only proposal snapshots cross session boundaries. Chat history, checkpoints and
raw run streams remain owned by the initiating session.
"""

import asyncio
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import langsmith as ls
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.types import Command
from starlette.exceptions import HTTPException

from demo.store import AccessDenied
from demo.tracing import TRACE_SCHEMA_VERSION, tag_review
from demo.policy import policy_execution, policy_metadata


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Approval:
    id: str
    proposal: dict
    conversation_id: str
    request_run_id: str
    created_at: str = field(default_factory=now)
    status: str = "pending"
    reviewer_id: str | None = None
    decision: str | None = None
    comment: str = ""
    resolution_run_id: str | None = None
    resolved_at: str | None = None
    outcome: str | None = None
    demo_key: str | None = None


class ApprovalInbox:
    def __init__(self, runtime):
        self.runtime = runtime
        self.items = {}
        self.demo_conversations = {}
        self.demo_creations = {}
        self.demo_locks = {}

    @property
    def store(self):
        return self.runtime.store

    def publish(self, conversation, run, proposal):
        approval_id = proposal["approval_id"]
        self.items.setdefault(approval_id, Approval(
            approval_id, deepcopy(proposal), conversation.id, run.id, demo_key=conversation.demo_key,
        ))
        conversation.pending_run_id = run.id

    def active_owner(self, item):
        conversation = self.runtime.conversations.get(item.conversation_id)
        return conversation is not None and time.monotonic() - self.runtime.sessions.get(conversation.session_id, 0) < 7200

    def cleanup(self):
        self.items = {key: item for key, item in self.items.items() if item.conversation_id in self.runtime.conversations}
        self.demo_conversations = {key: cid for key, cid in self.demo_conversations.items() if cid in self.runtime.conversations}

    def reviewable(self, actor, proposal):
        try:
            self.store.authorization.authorize_review(actor, proposal, phase="inbox_read")
            return True
        except AccessDenied:
            return False

    def can_review(self, actor):
        # Polling still enforces the real policy, but does not flood LangSmith.
        with ls.tracing_context(enabled=False):
            return any(self.reviewable(actor, {
                "tenant_id": actor.tenant_id, "account_id": account["id"],
                "resource": self.store.briefs[account["id"]]["id"], "operation": "save_account_brief",
            }) for account in self.store.accounts.values() if account["tenant_id"] == actor.tenant_id)

    def serialize(self, item):
        proposal = item.proposal
        return {
            "id": item.id, "status": item.status, "created_at": item.created_at,
            "request_run_id": item.request_run_id, "conversation_id": item.conversation_id,
            "resolution_run_id": item.resolution_run_id, "resolved_at": item.resolved_at,
            "requester": self.store.users[proposal["user_id"]]["name"],
            "requester_id": proposal["user_id"], "agent_id": proposal["agent_id"],
            "account_id": proposal["account_id"], "account_name": proposal["account_name"],
            "operation": proposal["operation"], "content": proposal["content"],
            "expected_version": proposal["expected_version"], "reviewer_id": item.reviewer_id,
            "decision": item.decision, "comment": item.comment, "outcome": item.outcome,
            "demo": item.demo_key is not None,
        }

    def list_for(self, actor):
        visible = []
        with ls.tracing_context(enabled=False):
            if not self.can_review(actor):
                raise HTTPException(403, "Select an eligible lead to view approvals")
            for item in self.items.values():
                if item.proposal["tenant_id"] == actor.tenant_id and self.active_owner(item) and self.reviewable(actor, item.proposal):
                    visible.append(item)
        visible.sort(key=lambda item: item.created_at, reverse=True)
        pending = [item for item in visible if item.status in {"pending", "resuming"}]
        recent = [item for item in visible if item.status not in {"pending", "resuming"}]
        return {"items": [self.serialize(item) for item in (pending + recent)[:100]],
                "pending_count": len(pending), "reviewer_id": actor.user_id,
                "truncated": len(visible) > 100}

    def authorized_item(self, actor, approval_id, *, config=None, emit=None):
        item = self.items.get(approval_id)
        if item is None or not self.active_owner(item) or item.proposal["tenant_id"] != actor.tenant_id:
            raise HTTPException(404, "Approval is unavailable")
        try:
            self.store.authorization.authorize_review(actor, item.proposal, phase="inbox_decision", config=config, emit=emit)
        except AccessDenied:
            # Do not reveal whether a guessed/foreign/now-revoked item exists.
            raise HTTPException(404, "Approval is unavailable") from None
        return item

    def review(self, actor, approval_id, decision, comment):
        """One correlated review root; start resumption only after this span ends."""
        from demo.agent import emitter

        item = self.items.get(approval_id)
        if item is None or not self.active_owner(item) or item.proposal["tenant_id"] != actor.tenant_id:
            raise HTTPException(404, "Approval is unavailable")
        original = self.runtime.runs.get(item.request_run_id)
        labels = dict(original.sample_labels) if original else {}
        tags = ["customer-operations", "tenant-governance", "phase:review"]
        if labels:
            tags.append("tool-outcome-batch")
            if labels.get("sample_batch_id"):
                tags.append("batch:" + labels["sample_batch_id"])
        trace_config = {"run_name": "approvals.review", "callbacks": self.runtime.trace_callbacks,
                        "tags": tags, "metadata": {**labels, **asdict(actor),
                            "trace_schema_version": TRACE_SCHEMA_VERSION, "interaction_phase": "review",
                            "thread_id": item.conversation_id, "conversation_id": item.conversation_id,
                            "approval_id": item.id, "request_run_id": item.request_run_id,
                            "reviewer_id": actor.user_id},
                        "configurable": {"thread_id": item.conversation_id,
                                         **asdict(actor), "governance_action_id": uuid.uuid4().hex,
                                         "call_id": item.proposal["call_id"], "invocation_mode": "review"}}

        def decide(_request, config: RunnableConfig):
            try:
                self.authorized_item(actor, approval_id, config=config,
                                     emit=emitter(actor, config, "review_account_brief"))
                if decision not in {"approve", "deny", "conditional"}:
                    raise HTTPException(400, "Invalid review decision")
                self._claim(item, actor, decision, comment)
            except HTTPException as exc:
                outcome = "denied" if exc.status_code == 404 else "stale" if item.status == "stale" else "rejected"
                tag_review(config, outcome)
                return {"status_code": exc.status_code, "detail": exc.detail, "review_outcome": outcome}
            tag_review(config, "accepted", decision)
            return {"decision": "allow", "review_outcome": "accepted", "approval_decision": decision}

        with policy_execution(trace_config) as (prepared, _state):
            result = RunnableLambda(decide, name="approvals.review").invoke(
                {"actor": asdict(actor), "approval_id": item.id}, config=prepared,
            )
        if "status_code" in result:
            raise HTTPException(result["status_code"], result["detail"])
        # No awaits separate authorization, version binding and claim. This runs
        # outside the review Runnable context: resumed turns remain separate roots.
        conversation = self.runtime.conversations[item.conversation_id]
        run = self.runtime.start(conversation, Command(resume={
            "approval_id": item.id, "reviewer": actor.user_id, "decision": decision, "comment": comment,
        }), reviewing=item)
        item.resolution_run_id = run.id
        return item, run

    def _claim(self, item, actor, decision, comment):
        """No awaits between validation and claim: one winner per pending item."""
        conversation = self.runtime.conversations[item.conversation_id]
        pending = conversation.pending
        if (item.status != "pending" or conversation.busy or not pending
                or pending["approval_id"] != item.id or conversation.pending_run_id != item.request_run_id):
            raise HTTPException(409, "This approval is no longer pending")
        if any(item.proposal.get(key) != value for key, value in policy_metadata().items()):
            item.status, item.outcome, item.resolved_at = "stale", "policy_changed", now()
            conversation.pending = None
            conversation.failed = True
            raise HTTPException(409, "The policy changed. Ask the agent for a new proposal.")
        if self.store.briefs[item.proposal["account_id"]]["version"] != item.proposal["expected_version"]:
            item.status, item.outcome, item.resolved_at = "stale", "brief_version_changed", now()
            conversation.pending = None
            conversation.failed = True
            raise HTTPException(409, "The brief changed. Ask the agent for a new proposal.")
        item.status, item.reviewer_id, item.decision, item.comment = "resuming", actor.user_id, decision, comment
        conversation.pending = None

    def observe_resolution(self, item, event):
        if item is None or item.status != "resuming" or event["data"].get("call_id") != item.proposal["call_id"]:
            return
        data = event["data"]
        if event["event"] == "tool_completed":
            status = {"saved": "approved", "archived": "approved", "rejected": "rejected",
                      "pending_secondary_approval": "held"}.get(data.get("status"))
            if status:
                item.status, item.outcome, item.resolved_at = status, data["status"], now()
        elif event["event"] in {"tool_failed", "tool_denied"}:
            item.status = "stale" if data.get("reason_code") == "stale_approval" else "failed"
            item.outcome, item.resolved_at = data.get("reason_code", "execution_failed"), now()

    def finish_resolution(self, item):
        if item is not None and item.status == "resuming":
            item.status, item.outcome, item.resolved_at = "failed", "execution_did_not_confirm_change", now()

    def demo_scenarios(self, actor):
        tenant = actor.tenant_id
        for index in range(3):
            account_id = f"account:{tenant}/AC-{100 + index}"
            account = self.store.accounts[account_id]
            operation = "archive_account_brief" if index == 2 else "save_account_brief"
            args = {"account_id": account_id}
            if index == 0:
                args["content"] = (f"Renewal risk brief for {account['name']}. Resolve intermittent SSO failures and "
                                   f"complete regional failover validation before renewal on {account['renewal_date']}. "
                                   f"Agree fix/test dates with support; confirm customer acceptance. "
                                   f"Sources: case:{tenant}/100-1, case:{tenant}/100-2.")
            elif index == 1:
                args["content"] = (f"Meeting-ready brief for {account['name']}. The usage export is verified and "
                                   f"the quarterly service review is closed. Confirm the renewal meeting for "
                                   f"{account['renewal_date']} and revalidate success criteria. "
                                   f"Sources: case:{tenant}/101-1, case:{tenant}/101-2, document:{tenant}/101-technical.")
            yield {
                "key": f"{tenant}/{index}", "name": operation, "args": args,
                "requester": f"user:{tenant}/{'lead' if index == 2 else 'csm'}",
                "prompt": f"{'Archive the outdated' if index == 2 else 'Save a reviewed'} account brief for {account['name']}.",
                "proposal": {"tenant_id": tenant, "account_id": account_id, "operation": operation,
                             "resource": self.store.briefs[account_id]["id"]},
            }

    async def populate(self, actor, session_id):
        if not self.can_review(actor):
            raise HTTPException(403, "Select an eligible lead to populate approvals")
        lock = self.demo_locks.setdefault(actor.tenant_id, asyncio.Lock())
        async with lock:
            created, reused, waiting = [], [], []
            for scenario in self.demo_scenarios(actor):
                with ls.tracing_context(enabled=False):
                    if not self.reviewable(actor, scenario["proposal"]):
                        continue
                cid = self.demo_conversations.get(scenario["key"])
                existing = self.runtime.conversations.get(cid)
                if existing and (existing.pending or existing.busy) and self.runtime.session_active(existing.session_id):
                    reused.append(existing.id)
                    continue
                if self.demo_creations.get(actor.tenant_id, 0) >= 15:
                    # At most five complete rounds per tenant per server lifetime.
                    break
                if sum(c.session_id == session_id for c in self.runtime.conversations.values()) >= 20:
                    break
                requester = self.store.actor(actor.tenant_id, scenario["requester"])
                conversation = self.runtime.new_conversation(session_id, requester)
                conversation.mode = "approval_demo"
                conversation.demo_key = scenario["key"]
                conversation.demo_action = {"name": scenario["name"], "args": scenario["args"], "call_id": uuid.uuid4().hex}
                self.demo_conversations[scenario["key"]] = conversation.id
                self.demo_creations[actor.tenant_id] = self.demo_creations.get(actor.tenant_id, 0) + 1
                run = self.runtime.start(conversation, {"messages": [HumanMessage(content=scenario["prompt"])]})
                waiting.append(run)
                created.append(conversation.id)

            async def wait_for_pause(run):
                async with run.changed:
                    await run.changed.wait_for(lambda: run.done)

            if waiting:
                try:
                    await asyncio.wait_for(asyncio.gather(*(wait_for_pause(run) for run in waiting)), timeout=10)
                except asyncio.TimeoutError:
                    pass  # Runs continue; the visible inbox refreshes their actual state.
            return {"created": len(created), "reused": len(reused),
                    "limit_reached": self.demo_creations.get(actor.tenant_id, 0) >= 15 or
                                     sum(c.session_id == session_id for c in self.runtime.conversations.values()) >= 20,
                    "inbox": self.list_for(actor)}
