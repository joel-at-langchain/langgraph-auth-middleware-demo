"""Real SubAgentMiddleware dispatch, FGA, native child loops and checkpoints."""

import asyncio
import json
import unittest
import uuid
from collections import Counter
from unittest.mock import patch

import httpx
import langsmith as ls
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langgraph.types import Command

from customer_agent import build_customer_graph, stream_turn
from customer_store import CustomerStore
from customer_subagents import SUBAGENTS
from server import create_app
from test_customer_observability import StreamingTestModel, all_runs
from test_customer_operations import ScriptedCustomerModel
from test_customer_rejection_tags import RecordingTracer

ACCOUNT = "account:northstar/AC-101"


def task(kind="billing-review", account=ACCOUNT, **extra):
    return [("task", {"description": f"For {account}, perform the requested check once.", "subagent_type": kind, **extra})]


class NativeSubagentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tracing = ls.tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.store = CustomerStore()
        self.collector = RunCollectorCallbackHandler()
        self.tracer = RecordingTracer()
        self.config = {"run_name": "customer_operations.turn", "callbacks": [self.collector, self.tracer], "configurable": {
            "thread_id": uuid.uuid4().hex, "turn_id": uuid.uuid4().hex,
            "tenant_id": "northstar", "user_id": "user:northstar/csm", "agent_id": "agent:northstar/customer-ops",
        }}

    async def execute(self, model=None, *, graph=None, resume=None):
        graph = graph or build_customer_graph(self.store, model)
        return [event async for event in stream_turn(graph, resume if resume is not None else {
            "messages": [HumanMessage(content="Perform the requested native specialist check.")],
        }, self.config)]

    async def test_native_middleware_runs_three_real_child_model_loops(self):
        for kind, operation in zip(SUBAGENTS, ("get_billing_snapshot", "get_support_sla_report", "get_renewal_forecast")):
            with self.subTest(kind=kind):
                self.collector.traced_runs.clear()
                model = ScriptedCustomerModel(task(kind), [(operation, {"account_id": ACCOUNT})])
                events = await self.execute(model)
                self.assertEqual(len(model.seen), 4)  # Parent, child, child final, parent final.
                self.assertIn("agent_call_completed", [e["event"] for e in events])
                started = next(e["data"] for e in events if e["event"] == "agent_call_started")
                self.assertEqual(started["middleware"], "deepagents.SubAgentMiddleware")
                leaf = next(e["data"] for e in events if e["event"] == "tool_completed" and e["data"]["tool_name"] == operation)
                self.assertEqual(leaf["agent_id"], f"agent:northstar/{kind}")
                self.assertEqual(leaf["parent_agent_id"], "agent:northstar/customer-ops")
                spans = list(all_runs(self.collector.traced_runs))
                self.assertEqual(len(self.collector.traced_runs), 1)
                self.assertTrue(any(r.name == kind for r in spans))
                self.assertTrue(any(r.name == "tools.task." + kind for r in spans))
                self.assertFalse(any("tool-rejected" in r.tags for r in spans))

    async def test_parent_and_user_delegation_denials_stop_before_child(self):
        for grant in (("agent:northstar/customer-ops", "delegate", "agent:northstar/billing-review"),
                      ("team:northstar/success#member", "delegate", "agent:northstar/billing-review")):
            with self.subTest(grant=grant):
                self.store.fga.delete_tuple(*grant)
                model = ScriptedCustomerModel(task())
                events = await self.execute(model)
                self.assertEqual(len(model.seen), 2)
                self.assertIn("agent_call_denied", [e["event"] for e in events])
                span = next(r for r in reversed(list(self.tracer.completed.values())) if r["name"] == "tools.task.billing-review")
                self.assertIn("tool-rejected", span["tags"])
                self.assertIn("rejection-authorization", span["tags"])
                self.store.fga.write_tuple(*grant)

    async def test_child_and_parent_resource_grants_are_both_required(self):
        for subject in ("agent:northstar/billing-review", "agent:northstar/customer-ops"):
            with self.subTest(subject=subject):
                self.store.fga.delete_tuple(subject, "reader", "service:northstar/billing")
                events = await self.execute(ScriptedCustomerModel(task(), [("get_billing_snapshot", {"account_id": ACCOUNT})]))
                self.assertIn("tool_denied", [e["event"] for e in events])
                self.assertNotIn("service_call_started", [e["event"] for e in events])
                span = next(r for r in reversed(list(self.tracer.completed.values())) if r["name"] == "tools.task.billing-review")
                self.assertIn("subagent-rejected", span["tags"])
                root = next(r for r in reversed(list(self.tracer.completed.values())) if r["name"] == "customer_operations.turn")
                self.assertEqual(root["metadata"]["rejected_tool_call_count"], 1)
                self.store.fga.write_tuple(subject, "reader", "service:northstar/billing")

    async def test_child_cannot_switch_account_or_tenant(self):
        for reference in ("account:northstar/AC-100", "account:beacon/AC-101", "northstar/AC-100", "beacon/AC-101"):
            events = await self.execute(ScriptedCustomerModel(task(), [("get_billing_snapshot", {"account_id": reference})]))
            self.assertIn("tool_denied", [e["event"] for e in events])
            self.assertNotIn("service_call_started", [e["event"] for e in events])

    async def test_omitted_prefix_only_normalizes_exact_delegated_account(self):
        events = await self.execute(ScriptedCustomerModel(task(), [("get_billing_snapshot", {"account_id": "northstar/AC-101"})]))
        self.assertIn("service_call_completed", [e["event"] for e in events])
        self.assertNotIn("tool_denied", [e["event"] for e in events])
        leaf = next(r for r in all_runs(self.collector.traced_runs) if r.name == "get_billing_snapshot")
        self.assertEqual(json.loads(leaf.outputs["output"].content)["account_id"], ACCOUNT)

    async def test_nested_execution_failure_tags_leaf_task_and_root(self):
        account = "account:northstar/AC-100"
        events = await self.execute(ScriptedCustomerModel(task("support-escalation", account),
                                   [("get_support_sla_report", {"account_id": account})]))
        self.assertIn("tool_failed", [e["event"] for e in events])
        spans = list(all_runs(self.collector.traced_runs))
        leaf = next(r for r in spans if r.name == "get_support_sla_report")
        owner = next(r for r in spans if r.name == "tools.task.support-escalation")
        self.assertIn("tool-failed", leaf.tags)
        self.assertIn("subagent-failed", owner.tags)
        self.assertIn("contains-tool-failure", self.collector.traced_runs[-1].tags)
        self.assertEqual(len(self.collector.traced_runs), 1)

    async def test_parent_filtered_brief_cannot_reappear_through_child(self):
        self.store.fga.delete_tuple("agent:northstar/customer-ops", "reader", "brief:northstar/101")
        model = ScriptedCustomerModel(task("renewal-planning"), [("get_customer_account", {"account_id": ACCOUNT})])
        await self.execute(model)
        child_result = next(m for m in model.seen[2] if m.type == "tool")
        self.assertNotIn("saved_brief", json.loads(child_result.content))

    async def test_delegation_revoked_after_child_model_blocks_tool(self):
        model = ScriptedCustomerModel(task(), [("get_billing_snapshot", {"account_id": ACCOUNT})])
        original = model.ainvoke

        async def revoke(messages, config=None):
            result = await original(messages, config)
            if len(model.seen) == 2:
                self.store.fga.delete_tuple("agent:northstar/customer-ops", "delegate", "agent:northstar/billing-review")
            return result

        with patch.object(model, "ainvoke", side_effect=revoke):
            events = await self.execute(model)
        self.assertIn("tool_denied", [e["event"] for e in events])
        self.assertNotIn("service_call_started", [e["event"] for e in events])

    async def test_revoked_delegation_on_resume_never_writes(self):
        model = ScriptedCustomerModel(task("renewal-planning"),
            [("save_account_brief", {"account_id": ACCOUNT, "content": "Fictional plan"})])
        graph = build_customer_graph(self.store, model)
        events = await self.execute(graph=graph)
        proposal = next(e["data"] for e in events if e["event"] == "approval_required")
        self.store.fga.delete_tuple("agent:northstar/customer-ops", "delegate", "agent:northstar/renewal-planning")
        events = await self.execute(graph=graph, resume=Command(resume={
            "approval_id": proposal["approval_id"], "reviewer": "user:northstar/lead", "decision": "approve",
        }))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)
        self.assertNotIn("hitl_response", [e["event"] for e in events])
        self.assertIn("contains-tool-rejection", self.collector.traced_runs[-1].tags)

    async def test_validation_and_unknown_subagent_fail_inside_task_span(self):
        for calls in (task("unknown"), task(user_id="user:beacon/lead"),
                      [("task", {"description": "No explicit account", "subagent_type": "billing-review"})]):
            events = await self.execute(ScriptedCustomerModel(calls))
            self.assertNotIn("agent_call_started", [e["event"] for e in events])
            self.assertTrue(any(e["event"] in {"tool_rejected", "agent_call_denied"} for e in events))

    async def test_native_child_interrupt_approve_deny_and_hold(self):
        for decision in ("approve", "deny", "conditional"):
            with self.subTest(decision=decision):
                self.config["configurable"]["thread_id"] = uuid.uuid4().hex
                before = self.store.briefs[ACCOUNT]["version"]
                model = ScriptedCustomerModel(task("renewal-planning"),
                    [("save_account_brief", {"account_id": ACCOUNT, "content": "Fictional reviewed renewal plan."})])
                graph = build_customer_graph(self.store, model)
                events = await self.execute(graph=graph)
                proposal = next(e["data"] for e in events if e["event"] == "approval_required")
                self.assertEqual(proposal["agent_id"], "agent:northstar/renewal-planning")
                self.assertEqual(self.store.briefs[ACCOUNT]["version"], before)
                self.assertNotIn("contains-tool-rejection", self.collector.traced_runs[-1].tags)
                events = await self.execute(graph=graph, resume=Command(resume={
                    "approval_id": proposal["approval_id"], "reviewer": "user:northstar/lead", "decision": decision,
                }))
                self.assertTrue(any(e["event"] == "hitl_response" for e in events))
                self.assertEqual(self.store.briefs[ACCOUNT]["version"], before + (decision == "approve"))
                self.assertEqual("contains-tool-rejection" in self.collector.traced_runs[-1].tags, decision == "deny")

    async def test_http_entrypoint_native_approval_and_resume_labels(self):
        model = ScriptedCustomerModel(task("renewal-planning"),
            [("save_account_brief", {"account_id": ACCOUNT, "content": "Fictional HTTP proposal"})])
        app = create_app(model=model)
        runtime = app.state.runtime
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            await client.get("/api/catalog")
            response = await client.post("/api/run", json={"user_msg": "Ask renewal-planning to save a Westhaven Energy brief.",
                "sample_batch_id": "offline-native", "sample_case_id": "hitl"})
            self.assertEqual(response.status_code, 200)
            ids = response.json()
            await asyncio.gather(*tuple(runtime.tasks))
            listing = await client.get("/api/approvals", params={"tenant_id": "northstar", "user_id": "user:northstar/lead"})
            item, = listing.json()["items"]
            self.assertEqual(item["agent_id"], "agent:northstar/renewal-planning")
            resumed = await client.post("/api/resume/" + ids["conversation_id"], json={
                "approval_id": item["id"], "reviewer": "user:northstar/lead", "decision": "approve"})
            self.assertEqual(resumed.status_code, 200)
            await asyncio.gather(*tuple(runtime.tasks))
            self.assertEqual(runtime.inbox.items[item["id"]].status, "approved")
            self.assertEqual(runtime.store.briefs[ACCOUNT]["version"], 2)
            self.assertEqual(runtime.runs[resumed.json()["run_id"]].sample_labels["sample_batch_id"], "offline-native")

    async def test_child_tokens_are_not_streamed_as_parent_answer(self):
        kind = "billing-review"
        model = StreamingTestModel(batches=[
            [AIMessageChunk(content="", tool_call_chunks=[{"name": "task", "args": json.dumps(task()[0][1]), "id": "parent-task", "index": 0}])],
            [AIMessageChunk(content="", tool_call_chunks=[{"name": "get_billing_snapshot", "args": json.dumps({"account_id": ACCOUNT}), "id": "child-read", "index": 0}])],
            [AIMessageChunk(content="CHILD_ONLY_REPORT")],
            [AIMessageChunk(content="Public "), AIMessageChunk(content="summary.")],
        ])
        events = await self.execute(model)
        deltas = [e["data"]["content"] for e in events if e["event"] == "response_delta"]
        self.assertEqual("".join(deltas), "Public summary.")
        self.assertGreater(len(deltas), 1)

    def test_live_example_matrix(self):
        from run_subagent_samples import cases

        matrix = cases()
        self.assertEqual(len(matrix), 20)
        self.assertEqual(len({c["case_id"] for c in matrix}), 20)
        self.assertEqual(Counter(c["outcome"] for c in matrix), {"success": 8, "authorization": 6, "execution": 3, "approval": 3})
        self.assertEqual({c["subagent_type"] for c in matrix}, set(SUBAGENTS))
        self.assertEqual({c["decision"] for c in matrix if c["outcome"] == "approval"}, {"approve", "deny", "conditional"})


if __name__ == "__main__":
    unittest.main()
