"""Local customer-operations demo: session-bound chat, approvals, and live audit."""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
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

from customer_agent import build_customer_graph, create_live_model, stream_turn
from customer_store import AccessDenied, Actor, CustomerStore
from approval_inbox import ApprovalInbox

COOKIE = "customer_demo_session"
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_msg: str = Field(min_length=1, max_length=4000)
    tenant_id: str = Field(default="northstar", max_length=50)
    user_id: str = Field(default="user:northstar/csm", max_length=100)
    profile: Literal["customer-ops", "support"] = "customer-ops"
    invocation_mode: Literal["tool", "handoff"] = "tool"
    handoff_account: str = Field(default="", max_length=180)
    conversation_id: str | None = Field(default=None, max_length=64)


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

    async def emit(self, event):
        async with self.changed:
            self.events.append(event)
            self.changed.notify_all()


def suggested_tasks(actor, accounts):
    account = accounts[0]["name"] if accounts else "your assigned account"
    other = "beacon" if actor.tenant_id != "beacon" else "summit"
    return [
        {"name": "Prepare for a renewal", "prompt": f"Prepare me for {account}'s renewal meeting. Find the open issues and cite the evidence."},
        {"name": "Delegate to the analyst", "prompt": f"Ask the renewal analyst to assess {account}. Explain the blockers and next steps."},
        {"name": "Save a reviewed brief", "prompt": "Save the findings we just discussed as the account brief."},
        {"name": "Inspect a commercial note", "prompt": f"Read document:{actor.tenant_id}/100-commercial for {account}."},
        {"name": "Try a different tenant", "prompt": f"Look up account:{other}/AC-100 and show me its renewal details."},
        {"name": "Verify the saved brief", "prompt": f"Show me the current saved brief and its version for {account}."},
    ]


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

    def start(self, conversation, input_data, reviewing=None):
        run = Run(uuid.uuid4().hex, conversation.id, conversation.session_id)
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
                         "trace_schema_version": "4", "approval_demo": conversation.demo_key is not None,
                         "reviewed_approval_id": reviewing.id if reviewing else None},
        }
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
        return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store"})

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
                                 "scenarios": suggested_tasks(actor, accounts),
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
        run = runtime.start(conversation, {"messages": [HumanMessage(content=data.user_msg.strip())]})
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
