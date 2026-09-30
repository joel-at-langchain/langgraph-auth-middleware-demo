"""Local customer-operations demo: session-bound chat, approvals, and live audit."""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from demo.paths import REPO_ROOT, WEB_DIR
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from demo.agent import build_customer_graph, create_live_model, stream_turn
from demo.store import AccessDenied, Actor, CustomerStore
from demo.approvals import ApprovalInbox

COOKIE = "customer_demo_session"
load_dotenv(REPO_ROOT / ".env")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_msg: str = Field(min_length=1, max_length=4000)
    tenant_id: str = Field(default="northstar", max_length=50)
    user_id: str = Field(default="user:northstar/csm", max_length=100)
    profile: Literal["customer-ops", "support"] = "customer-ops"
    invocation_mode: Literal["tool", "handoff"] = "tool"
    handoff_account: str = Field(default="", max_length=180)
    conversation_id: str | None = Field(default=None, max_length=64)
    sample_batch_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    sample_case_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,80}$")


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_id: str = Field(min_length=64, max_length=64)
    reviewer: str = Field(min_length=1, max_length=100)
    decision: Literal["approve", "deny", "conditional"]
    comment: str = Field(default="", max_length=500)


class InboxContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: str = Field(min_length=1, max_length=50)
    user_id: str = Field(min_length=1, max_length=100)


class InboxDecision(InboxContext):
    decision: Literal["approve", "deny", "conditional"]
    comment: str = Field(default="", max_length=500)


@dataclass
class Conversation:
    id: str
    session_id: str
    actor: Actor
    busy: bool = False
    pending: dict | None = None
    failed: bool = False
    turns: int = 0
    mode: str = "tool"
    handoff_account: str = ""
    pending_run_id: str | None = None
    demo_key: str | None = None
    demo_action: dict | None = None


@dataclass
class Run:
    id: str
    conversation_id: str
    session_id: str
    events: list = field(default_factory=list)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    done: bool = False
    sample_labels: dict = field(default_factory=dict)

    async def emit(self, event):
        async with self.changed:
            self.events.append(event)
            self.changed.notify_all()


def suggested_tasks(store, actor, accounts):
    """Self-contained fixture scenarios; outcomes describe the selected context."""
    visible = {account["id"]: account for account in accounts}
    account = visible.get(f"account:{actor.tenant_id}/AC-100")
    role = store.users[actor.user_id]["role"]
    support_agent = actor.agent_id.endswith("/support")
    lead = store.users[f"user:{actor.tenant_id}/lead"]["name"]
    other = "beacon" if actor.tenant_id != "beacon" else "summit"
    scenarios = []

    def add(key, name, target, prompt, description, expected):
        scenarios.append({"id": key, "name": name, "account": target, "prompt": prompt,
                          "description": description, "expected": expected})

    if account:
        name = account["name"]
        add("renewal-risk", "Renewal meeting: open blockers", name,
            f"Prepare me for {name}'s renewal meeting. Find the open issues and cite the evidence.",
            "Tests account lookup and evidence retrieval through governed tools.",
            "A cited summary of SSO failures and incomplete failover validation; no records change.")

    for suffix, title, expected in (
        ("100", "Analyst: renewal at risk", "Readiness is needs_attention, with SSO and failover blockers plus next steps."),
        ("101", "Analyst: healthy renewal", "Readiness is ready, with closed cases and confirmed success criteria."),
        ("102", "Analyst: missing sign-off", "Readiness is insufficient_information; the analyst asks for a current success plan and customer sign-off."),
    ):
        target = visible.get(f"account:{actor.tenant_id}/AC-{suffix}")
        if target:
            add("analyst-" + suffix, title, target["name"],
                f"Ask the renewal analyst to assess {target['name']}. Explain the readiness, evidence, and next steps.",
                "Tests parent-to-specialist delegation while preserving the initiating user's access.",
                "Delegation is denied: this Support Assistant lacks the parent-to-specialist grant. No analyst result is returned."
                if support_agent else expected)

    if account:
        name = account["name"]
        write_denied = role == "support" or support_agent
        commercial_denied = role == "support" or support_agent
        add("native-billing", "Subagent: billing review", name,
            f"Delegate to billing-review using task. For {name}, call get_billing_snapshot once and report the balances. Do not save anything.",
            "Tests native SubAgentMiddleware, user and parent delegation, then child billing-tool authorization.",
            "Delegation is denied before the child runs." if commercial_denied else
            "The billing-review agent calls the billing tool and returns a cited fictional balance; no changes.")
        add("native-support", "Subagent: upstream failure", name,
            f"Delegate to support-escalation using task. For {name}, call get_support_sla_report once; report the actual result or error without retrying.",
            "Tests a mock service failure inside an authorized native child agent, with leaf, subagent and root failure tags.",
            "The child service returns service_timeout; the assistant reports the failure, without invented SLA metrics.")
        add("native-renewal", "Subagent: renewal planning", name,
            f"Delegate to renewal-planning using task. For {name}, call get_renewal_forecast once and report its fixture-only forecast. Do not save anything.",
            "Tests an isolated model-driven renewal agent and its narrow forecast toolset.",
            "Delegation is denied before the child runs." if commercial_denied else
            "The child returns a fictional precomputed forecast and source; it does not save or approve a brief.")
        add("native-approval", "Subagent: lead approval", name,
            f'Delegate to renewal-planning using task. Save a brief for {name} with content: "Fictional renewal plan: confirm the meeting agenda and customer success criteria." Submit it for lead review.',
            "Tests a child-originated write, a real checkpoint pause, and fresh FGA checks on resume.",
            "Delegation is denied; no proposal or changes." if commercial_denied else
            f"The child proposes a brief in {lead}'s inbox. Approve saves; deny or hold makes no change.")
        add("reviewed-save", f"Request {lead}'s approval", name,
            f'Save a new account brief for {name}: "Renewal needs attention. Resolve the open SSO issue '
            'and complete failover validation before the renewal meeting. Confirm owners and target dates with Support."',
            "Tests write authorization and a real human-review pause before saving a brief.",
            "Write is denied before approval because the selected user or assistant lacks write permission. No change is made."
            if write_denied else
            f"The run pauses and a proposal appears in {lead}'s approvals inbox. Approve saves a new version; reject leaves it unchanged.")
        add("commercial-note", "Restricted commercial note", name,
            f"Read document:{actor.tenant_id}/100-commercial for {name}.",
            "Tests record-level access beyond permission to read the account.",
            "Access is denied; the selected user or assistant lacks commercial-note access. No note content is returned." if commercial_denied else
            "The commercial note is returned because both this persona and assistant have the required reader grants.")
        add("filtered-search", "Search commercial evidence", name,
            f"Search {name}'s records for concession. Cite any matching records you are allowed to read.",
            "Tests permission-filtered search and whether hidden results stay hidden.",
            "No restricted commercial text or hidden-result count is disclosed." if commercial_denied else
            "The authorized commercial note appears with its source ID; records outside the account are excluded.")
        add("saved-brief", "Inspect the saved brief", name,
            f"Show me the current saved brief and its version for {name}.",
            "Tests current store state and the separate reader grant on derived briefs.",
            "The brief is withheld because the selected user or assistant lacks brief-reader access." if commercial_denied else
            "The current brief content, status, and version are returned. After an approved save, the version has increased.")
        add("archive-brief", "Archive an outdated brief", name,
            f"Archive the current account brief for {name}.",
            "Tests the stronger archive permission and human approval before a state change.",
            f"A proposal pauses for {lead}'s approval. Approving archives the brief and increments its version."
            if role == "lead" and not support_agent else
            f"Archive is denied before approval. Use {lead} with Customer Operations Assistant to test the allowed path.")
        add("sql-adoption", "SQL analyst: adoption decline", name,
            f"Ask the SQL analyst to compare average active seats and API error rates for {name} across September 1–14 and 15–28, 2026. Show the SQL and sources.",
            "Tests model-generated SQL in a two-node specialist, scoped usage data, and delegated authorization.",
            "Delegation is denied: the Support Assistant has no SQL-specialist delegation grant. Direct authorized SQL remains available."
            if support_agent else
            "The second half shows lower active seats and a higher API error rate. SQL and usage dataset sources are returned; no data changes.")
        add("sql-invoices", "SQL: restricted billing", name,
            f"Use query_customer_analytics for {name} with dataset invoices and SQL: SELECT currency, SUM(amount_cents) / 100.0 AS overdue_amount FROM invoices WHERE status = 'overdue' GROUP BY currency",
            "Tests dataset-level billing access independently of account access and tool execution permission.",
            "Billing access is denied; no invoice rows or amounts are disclosed." if commercial_denied else
            "The overdue USD balance is returned from the account-scoped invoice dataset, with the executed SQL and a source ID.")
        add("sql-incidents", "SQL: service incident impact", name,
            f"Query {name}'s service_incidents directly for unresolved incidents, their impact minutes, and source IDs. No delegation is needed.",
            "Tests direct read-only SQL without a subagent; available to Support on its authorized account.",
            "Two unresolved incidents are returned with 47 and 18 observed impact minutes. This is not a service-credit approval.")
        add("sql-read-only", "SQL: reject a database write", name,
            f"Test the read-only boundary for {name}: call query_customer_analytics with dataset usage_daily and this exact SQL: DELETE FROM usage_daily",
            "Tests validation of untrusted SQL independently of FGA. A model may also refuse before calling the tool.",
            "The statement is rejected or refused. The usage dataset is unchanged; SQL cannot create a write approval or bypass it.")
        add("sql-reviewed-brief", "From analytics to lead approval", name,
            f"Investigate {name}'s September adoption and unresolved service incidents, then save a short account brief with source IDs and proposed next steps.",
            "Tests default playbooks, multiple evidence sources, and a human-reviewed write after read-only analysis.",
            "Authorized operational evidence can be read, but the write is denied. No brief changes." if write_denied else
            f"The agent combines authorized evidence and proposes a brief in {lead}'s inbox. Nothing is saved until approval.")

    add("cross-tenant", "Attempt a cross-tenant lookup", "Tenant isolation",
        f"Look up account:{other}/AC-100 and show me its renewal details.",
        "Tests request-level tenant isolation before the model or tools execute.",
        "A generic access-denied response; no foreign account details, model call, or tool execution.")
    return scenarios


class DemoRuntime:
    def __init__(self, store, model):
        self.store = store
        self.graph = build_customer_graph(store, model)
        self.sessions = {}
        self.conversations = {}
        self.runs = {}
        self.tasks = set()
        self.inbox = ApprovalInbox(self)

    def session_active(self, sid):
        return sid in self.sessions and time.monotonic() - self.sessions[sid] <= 7200

    def session(self, request):
        sid = request.cookies.get(COOKIE)
        if not self.session_active(sid):
            raise HTTPException(401, "Demo session expired. Reload the page.")
        self.sessions[sid] = time.monotonic()
        return sid

    def owned(self, request, mapping, key):
        sid = self.session(request)
        item = mapping.get(key)
        if item is None or item.session_id != sid:
            raise HTTPException(404, "Unknown conversation or run")
        return item

    def new_conversation(self, sid, actor):
        if sum(c.session_id == sid for c in self.conversations.values()) >= 20:
            raise HTTPException(429, "Conversation limit reached for this demo session")
        conversation = Conversation(uuid.uuid4().hex, sid, actor)
        self.conversations[conversation.id] = conversation
        return conversation

    def start(self, conversation, input_data, reviewing=None, *, sample_labels=None):
        run = Run(uuid.uuid4().hex, conversation.id, conversation.session_id)
        if reviewing is not None and sample_labels is None:
            original = self.runs.get(reviewing.request_run_id)
            sample_labels = original.sample_labels if original else None
        run.sample_labels = dict(sample_labels or {})
        self.runs[run.id] = run
        conversation.busy = True
        conversation.turns += 1
        task = asyncio.create_task(self.produce(conversation, run, input_data, reviewing))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return run

    async def produce(self, conversation, run, input_data, reviewing=None):
        context = {
            **asdict(conversation.actor), "thread_id": conversation.id, "turn_id": run.id,
            "invocation_mode": conversation.mode, "handoff_account": conversation.handoff_account,
            "demo_action": conversation.demo_action,
        }
        config = {
            "configurable": context, "recursion_limit": 20, "run_id": uuid.UUID(run.id),
            "run_name": "customer_operations.turn",
            "tags": ["customer-operations", "tenant-governance", conversation.mode],
            "metadata": {"tenant_id": conversation.actor.tenant_id, "user_id": conversation.actor.user_id,
                         "agent_id": conversation.actor.agent_id, "conversation_id": conversation.id,
                         "trace_schema_version": "9", "approval_demo": conversation.demo_key is not None,
                         "reviewed_approval_id": reviewing.id if reviewing else None},
        }
        if run.sample_labels:
            config["metadata"].update(run.sample_labels)
            config["tags"].append("tool-outcome-batch")
            if run.sample_labels.get("sample_batch_id"):
                config["tags"].append("batch:" + run.sample_labels["sample_batch_id"])
        try:
            async for event in stream_turn(self.graph, input_data, config):
                if event["event"] == "approval_required":
                    conversation.pending = event["data"]
                    self.inbox.publish(conversation, run, event["data"])
                self.inbox.observe_resolution(reviewing, event)
                await run.emit(event)
        except Exception as exc:
            conversation.failed = True
            # Provider exceptions may contain sensitive headers; expose only class.
            await run.emit({"event": "error", "data": {
                "message": f"The run failed ({type(exc).__name__}). Start a new conversation to retry.",
            }})
        finally:
            self.inbox.finish_resolution(reviewing)
            conversation.busy = False
            await run.emit({"event": "done", "data": {"paused": conversation.pending is not None}})
            async with run.changed:
                run.done = True
                run.changed.notify_all()


async def body_as(request, schema):
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "Cross-origin write rejected")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 20000:
            raise HTTPException(413, "Request too large")
        chunks.append(chunk)
    try:
        return schema.model_validate_json(b"".join(chunks))
    except ValidationError:
        raise HTTPException(400, "Invalid request fields") from None


def create_app(model=None, store=None):
    runtime = DemoRuntime(store or CustomerStore(), model if model is not None else create_live_model())

    @asynccontextmanager
    async def lifespan(app):
        yield
        for task in runtime.tasks:
            task.cancel()
        await asyncio.gather(*runtime.tasks, return_exceptions=True)

    async def homepage(request):
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})

    async def health(request):
        return JSONResponse({"status": "ok", "demo": "customer-operations", **runtime.store.catalog()["counts"]})

    async def catalog(request):
        sid = request.cookies.get(COOKIE)
        now = time.monotonic()
        # Bound demo-session retention and remove associated ephemeral state.
        expired = {key for key, touched in runtime.sessions.items() if now - touched > 7200}
        for key in expired:
            runtime.sessions.pop(key, None)
        for key, conversation in list(runtime.conversations.items()):
            if conversation.session_id in expired and not conversation.busy:
                runtime.graph.checkpointer.delete_thread(key)
                del runtime.conversations[key]
                for ledger_key in list(runtime.store.completed_writes):
                    if ledger_key[0] == key:
                        del runtime.store.completed_writes[ledger_key]
        for key, run in list(runtime.runs.items()):
            if run.session_id in expired and run.done:
                del runtime.runs[key]
        runtime.inbox.cleanup()
        if sid not in runtime.sessions:
            if len(runtime.sessions) >= 100:
                raise HTTPException(429, "Demo session limit reached")
            sid = uuid.uuid4().hex
        runtime.sessions[sid] = now
        actor = runtime.store.actor(request.query_params.get("tenant_id", "northstar"),
                                    request.query_params.get("user_id", "user:northstar/csm"),
                                    request.query_params.get("profile", "customer-ops"))
        accounts = runtime.store.visible_accounts(actor)
        response = JSONResponse({**runtime.store.catalog(), "accounts": accounts,
                                 "default_skills": runtime.store.skills.catalog(actor),
                                 "scenarios": suggested_tasks(runtime.store, actor, accounts),
                                 "can_review_approvals": runtime.inbox.can_review(runtime.store.actor(actor.tenant_id, actor.user_id))},
                                headers={"Cache-Control": "no-store"})
        response.set_cookie(COOKIE, sid, httponly=True, samesite="strict", max_age=7200)
        return response

    async def start_run(request):
        data = await body_as(request, RunRequest)
        sid = runtime.session(request)
        if not data.user_msg.strip():
            raise HTTPException(400, "Enter a message")
        actor = runtime.store.actor(data.tenant_id, data.user_id, data.profile)
        if data.conversation_id:
            conversation = runtime.owned(request, runtime.conversations, data.conversation_id)
            if conversation.actor != actor:
                raise HTTPException(409, "Start a new conversation when changing tenant or persona")
        else:
            conversation = runtime.new_conversation(sid, actor)
        if conversation.busy or conversation.pending:
            raise HTTPException(409, "Finish the current run or review first")
        if conversation.failed or conversation.turns >= 30:
            raise HTTPException(409, "Start a new conversation to continue")
        if data.invocation_mode == "handoff":
            if not data.handoff_account:
                raise HTTPException(400, "Select an account for explicit handoff")
        conversation.mode = data.invocation_mode
        conversation.handoff_account = data.handoff_account
        labels = {key: getattr(data, key) for key in ("sample_batch_id", "sample_case_id") if getattr(data, key)}
        run = runtime.start(conversation, {"messages": [HumanMessage(content=data.user_msg.strip())]}, sample_labels=labels)
        return JSONResponse({"conversation_id": conversation.id, "run_id": run.id})

    async def stream(request):
        run = runtime.owned(request, runtime.runs, request.path_params["rid"])
        try:
            offset = max(0, min(int(request.headers.get("last-event-id", "0")), len(run.events)))
        except ValueError:
            offset = 0

        async def events():
            nonlocal offset
            while True:
                async with run.changed:
                    if offset == len(run.events) and not run.done:
                        try:
                            await asyncio.wait_for(run.changed.wait(), timeout=15)
                        except asyncio.TimeoutError:
                            pass
                    batch = run.events[offset:]
                    complete = run.done
                for event in batch:
                    offset += 1
                    yield f"id: {offset}\nevent: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
                if complete:
                    return
                if not batch:
                    yield ": keepalive\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    async def resume(request):
        data = await body_as(request, ReviewRequest)
        conversation = runtime.owned(request, runtime.conversations, request.path_params["cid"])
        pending = conversation.pending
        if conversation.busy or not pending or pending["approval_id"] != data.approval_id:
            raise HTTPException(409, "This review is no longer pending")
        try:
            reviewer = runtime.store.actor(conversation.actor.tenant_id, data.reviewer)
            item = runtime.inbox.authorized_item(reviewer, data.approval_id)
        except (AccessDenied, HTTPException):
            raise HTTPException(403, "Reviewer is not authorized")
        run = runtime.inbox.claim(item, reviewer, data.decision, data.comment)
        return JSONResponse({"conversation_id": conversation.id, "run_id": run.id})

    async def approvals(request):
        runtime.session(request)
        try:
            data = InboxContext.model_validate(dict(request.query_params))
        except ValidationError:
            raise HTTPException(400, "Select a tenant and reviewer") from None
        actor = runtime.store.actor(data.tenant_id, data.user_id)
        return JSONResponse(runtime.inbox.list_for(actor), headers={"Cache-Control": "no-store"})

    async def populate_approvals(request):
        data = await body_as(request, InboxContext)
        sid = runtime.session(request)
        actor = runtime.store.actor(data.tenant_id, data.user_id)
        result = await runtime.inbox.populate(actor, sid)
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    async def decide_approval(request):
        data = await body_as(request, InboxDecision)
        runtime.session(request)
        actor = runtime.store.actor(data.tenant_id, data.user_id)
        item = runtime.inbox.authorized_item(actor, request.path_params["aid"])
        runtime.inbox.claim(item, actor, data.decision, data.comment)
        # Reviewers receive a proposal status, never the requester's raw stream.
        return JSONResponse({"approval": runtime.inbox.serialize(item)}, status_code=202,
                            headers={"Cache-Control": "no-store"})

    async def approval_status(request):
        item = runtime.inbox.items.get(request.path_params["aid"])
        if item is None:
            raise HTTPException(404, "Approval is unavailable")
        runtime.owned(request, runtime.conversations, item.conversation_id)
        return JSONResponse({"status": item.status, "resolution_run_id": item.resolution_run_id},
                            headers={"Cache-Control": "no-store"})

    async def http_error(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    async def denied(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=403)

    app = Starlette(lifespan=lifespan, routes=[
        Route("/", homepage), Route("/health", health), Route("/api/catalog", catalog),
        Route("/api/scenarios", catalog), Route("/api/run", start_run, methods=["POST"]),
        Route("/api/stream/{rid}", stream), Route("/api/resume/{cid}", resume, methods=["POST"]),
        Route("/api/approvals", approvals), Route("/api/approvals/demo", populate_approvals, methods=["POST"]),
        Route("/api/approvals/{aid}/decision", decide_approval, methods=["POST"]),
        Route("/api/approvals/{aid}/status", approval_status),
    ], exception_handlers={HTTPException: http_error, AccessDenied: denied})
    app.state.runtime = runtime
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])
    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
