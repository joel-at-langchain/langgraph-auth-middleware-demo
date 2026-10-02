"""Canonical outcomes and identity, using real callbacks without remote tracing."""

import asyncio
import json
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import langsmith as ls
import httpx
from langchain_core.messages import HumanMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langgraph.types import Command

from demo.agent import build_customer_graph, stream_turn
from demo.store import CustomerStore
from demo.server import create_app
from demo.tracing import tag_tool_rejection
from scripts.trace_verification import review_filter, verify_review
from tests.test_customer_operations import ScriptedCustomerModel, call
from tests.test_customer_rejection_tags import RecordingTracer
from tests.test_customer_subagents import task


class TraceSemanticsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        context = ls.tracing_context(enabled=False)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.store = CustomerStore()
        self.tracer = RecordingTracer()
        self.collector = RunCollectorCallbackHandler()
        self.config = {"run_name": "customer_operations.turn", "callbacks": [self.tracer, self.collector],
                       "configurable": {"thread_id": uuid.uuid4().hex, "turn_id": uuid.uuid4().hex,
                                        "tenant_id": "northstar", "user_id": "user:northstar/csm",
                                        "agent_id": "agent:northstar/customer-ops"}}

    async def execute(self, model, text="Perform the requested check"):
        graph = build_customer_graph(self.store, model)
        return [event async for event in stream_turn(graph, {"messages": [HumanMessage(content=text)]}, self.config)]

    def named(self, name):
        return [run for run in self.tracer.completed.values() if run["name"] == name]

    async def test_auth_actor_matches_input_metadata_and_completion_event(self):
        await self.execute(ScriptedCustomerModel(task(), call("get_billing_snapshot", account_id="account:northstar/AC-101")))
        auth = [r for r in self.tracer.completed.values() if r["name"].startswith("authorization.")]
        self.assertTrue(auth)
        for run in auth:
            actor = run["inputs"].get("actor")
            if actor is None:
                continue
            with self.subTest(name=run["name"], actor=actor):
                for key in ("agent_id", "user_id", "tenant_id", "parent_agent_id"):
                    self.assertEqual(run["metadata"].get(key), actor.get(key))
                terminal = [e for e in run["events"] if e["name"] in {
                    "authorization_completed", "tenant_verification_completed"}]
                self.assertEqual(len(terminal), 1)
                self.assertEqual(terminal[0]["kwargs"]["agent_id"], actor["agent_id"])

    async def test_final_authorization_decision_is_exclusive(self):
        self.config["configurable"]["user_id"] = "user:northstar/support"
        await self.execute(ScriptedCustomerModel(call("query_customer_analytics", datasets=["invoices"],
                                                     sql="SELECT COUNT(*) FROM invoices")))
        denied = [r for r in self.named("authorization.authorize_transaction") if r["outputs"]["decision"] == "deny"]
        self.assertTrue(denied)
        for run in denied:
            self.assertEqual({t for t in run["tags"] if t.startswith("auth-decision:")}, {"auth-decision:deny"})
            self.assertEqual({t for t in run["tags"] if t.startswith("fga-")}, {"fga-deny"})
            self.assertEqual(run["tags"].count("fga-deny"), 1)
            self.assertEqual(run["metadata"]["auth_decision"], "deny")

    async def test_filtered_skill_check_does_not_deny_discovery_transaction(self):
        self.store.fga.delete_tuple("agent:northstar/customer-ops", "reader", "skill:northstar/sql-analysis")
        await self.execute(ScriptedCustomerModel())
        discovery = [r for r in self.named("authorization.authorize_transaction") if r["metadata"].get("auth_phase") == "discover"]
        self.assertTrue(discovery)
        for run in discovery:
            self.assertIn("auth-decision:allow", run["tags"])
            self.assertNotIn("fga-deny", run["tags"])
        self.assertTrue(any(e["name"] == "fga_decision" and e["kwargs"]["decision"] == "deny"
                            for r in discovery for e in r["events"]))

    async def test_request_denial_is_filterable_but_not_a_tool_rejection(self):
        model = ScriptedCustomerModel()
        await self.execute(model, "Look up account:beacon/AC-100")
        root, = self.named("customer_operations.turn")
        self.assertIn("request-outcome:denied", root["tags"])
        self.assertEqual(root["metadata"]["request_outcome"], "denied")
        self.assertNotIn("contains-tool-rejection", root["tags"])
        self.assertEqual(model.seen, [])
        await self.execute(model, "Hello")
        roots = self.named("customer_operations.turn")
        self.assertEqual(sum("request-outcome:denied" in r["tags"] for r in roots), 1)
        self.assertNotIn("metadata", self.config)

    async def test_subagent_execution_failure_is_error_not_rejection(self):
        await self.execute(ScriptedCustomerModel(task("support-escalation", "account:northstar/AC-100"),
                                                call("get_support_sla_report")))
        owner, = self.named("tools.task.support-escalation")
        child, = self.named("support-escalation")
        for run in (owner, child):
            self.assertEqual(run["metadata"]["subagent_outcome"], "error")
            self.assertIn("subagent-outcome:error", run["tags"])
            self.assertIn("contains-tool-failure", run["tags"])
        root, = self.named("customer_operations.turn")
        self.assertEqual(root["metadata"]["rejected_tool_call_count"], 1)

    async def test_child_model_failure_records_actual_subagent_error(self):
        class FailingChild(ScriptedCustomerModel):
            async def ainvoke(self, messages, config=None):
                if len(self.seen) == 1:
                    raise RuntimeError("Synthetic child model failure")
                return await super().ainvoke(messages, config)

        with self.assertRaisesRegex(RuntimeError, "Synthetic child model failure"):
            await self.execute(FailingChild(task()))
        for name in ("tools.task.billing-review", "billing-review"):
            run, = self.named(name)
            self.assertEqual(run["metadata"]["subagent_outcome"], "error")

    async def test_successful_leaf_does_not_inherit_subagent_terminal_outcome(self):
        await self.execute(ScriptedCustomerModel(task(), call("get_billing_snapshot", account_id="account:northstar/AC-101")))
        leaf, = self.named("get_billing_snapshot")
        self.assertEqual(leaf["metadata"]["tool_outcome"], "completed")
        self.assertNotIn("subagent_outcome", leaf["metadata"])
        self.assertNotIn("subagent_boundary", leaf["metadata"])

    async def test_native_approval_waiting_then_hold_is_not_an_execution_failure(self):
        model = ScriptedCustomerModel(task("renewal-planning"),
                                      call("save_account_brief", account_id="account:northstar/AC-101", content="Fictional brief"))
        graph = build_customer_graph(self.store, model)
        events = [e async for e in stream_turn(graph, {"messages": [HumanMessage(content="Save a fictional brief")]}, self.config)]
        proposal = next(e["data"] for e in events if e["event"] == "approval_required")
        for name in ("tools.task.renewal-planning", "renewal-planning"):
            run, = self.named(name)
            self.assertEqual(run["metadata"]["subagent_outcome"], "waiting")
            self.assertEqual(run["metadata"]["approval_outcome"], "pending")
            self.assertNotIn("contains-tool-rejection", run["tags"])
        self.tracer.completed.clear()
        resume = Command(resume={"approval_id": proposal["approval_id"], "reviewer": "user:northstar/lead", "decision": "conditional"})
        [e async for e in stream_turn(graph, resume, self.config)]
        for name in ("tools.task.renewal-planning", "renewal-planning"):
            run, = self.named(name)
            self.assertEqual(run["metadata"]["subagent_outcome"], "completed")
            self.assertEqual(run["metadata"]["approval_outcome"], "held")
            self.assertNotIn("contains-tool-rejection", run["tags"])

    async def test_subagent_aggregation_does_not_depend_on_span_names(self):
        def renamed(config, current, event, data):
            for tracer in (self.tracer, self.collector):
                for run in list(tracer.run_map.values()):
                    if run.name in {"tools.task.billing-review", "billing-review"}:
                        run.name = "renamed-specialist-boundary"
            tag_tool_rejection(config, current, event, data)

        self.store.fga.delete_tuple("agent:northstar/customer-ops", "delegate", "agent:northstar/billing-review")
        with patch("demo.agent.tag_tool_rejection", side_effect=renamed):
            await self.execute(ScriptedCustomerModel(task()))
        owner, = self.named("renamed-specialist-boundary")
        self.assertIn("subagent-rejected", owner["tags"])
        self.assertEqual(owner["metadata"]["tool_name"], "task")
        self.assertEqual(owner["metadata"]["subagent_outcome"], "denied")
        root, = self.named("customer_operations.turn")
        self.assertEqual(root["metadata"]["rejected_tools"], ["task"])

    async def test_handled_child_validation_is_completed_with_rejection_aggregate(self):
        await self.execute(ScriptedCustomerModel(task(), [("get_billing_snapshot", {})]))
        owner, = self.named("tools.task.billing-review")
        child, = self.named("billing-review")
        for run in (owner, child):
            self.assertEqual(run["metadata"]["subagent_outcome"], "completed")
            self.assertIn("contains-tool-rejection", run["tags"])
            self.assertNotIn("contains-tool-failure", run["tags"])
        leaf, = self.named("get_billing_snapshot")
        self.assertEqual(leaf["metadata"]["tool_outcome"], "rejected")

    async def reviewed_run(self, decision, *, stale=False, reviewer="user:northstar/lead"):
        self.tracer.completed.clear()
        self.collector.traced_runs.clear()
        app = create_app(model=ScriptedCustomerModel(call("save_account_brief", content="PRIVATE_BRIEF")),
                         callbacks=[self.tracer, self.collector])
        runtime = app.state.runtime

        async def drain():
            while runtime.tasks:
                await asyncio.gather(*tuple(runtime.tasks))

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            await client.get("/api/catalog")
            response = await client.post("/api/run", json={"user_msg": "Save a brief for Meridian Retail",
                                        "sample_batch_id": "semantics-audit", "sample_case_id": "approval-test"})
            self.assertEqual(response.status_code, 200)
            ids = response.json()
            await drain()
            item, = runtime.inbox.items.values()
            if stale:
                runtime.store.briefs[item.proposal["account_id"]]["version"] += 1
            response = await client.post(f"/api/approvals/{item.id}/decision", json={
                "tenant_id": "northstar", "user_id": reviewer, "decision": decision, "comment": "PRIVATE_COMMENT"})
            await drain()
        return response, ids, item

    async def test_review_stages_share_correlation_but_remain_separate_roots(self):
        for decision, outcome in (("approve", "approved"), ("deny", "denied"), ("conditional", "held")):
            with self.subTest(decision=decision):
                response, ids, item = await self.reviewed_run(decision)
                self.assertEqual(response.status_code, 202)
                roots = [r for r in self.tracer.completed.values() if r["parent_run_id"] is None]
                self.assertEqual(len(roots), 3)
                self.assertEqual({r["metadata"]["interaction_phase"] for r in roots}, {"request", "review", "resume"})
                for root in roots:
                    metadata = root["metadata"]
                    self.assertEqual(metadata["conversation_id"], ids["conversation_id"])
                    self.assertEqual(metadata["thread_id"], ids["conversation_id"])
                    self.assertEqual(metadata["request_run_id"], ids["run_id"])
                    self.assertEqual(metadata["sample_batch_id"], "semantics-audit")
                    self.assertEqual(metadata["sample_case_id"], "approval-test")
                    self.assertEqual(metadata["approval_id"], item.id)
                    self.assertEqual(metadata["trace_schema_version"], "10")
                    self.assertNotIn("PRIVATE_BRIEF", json.dumps(metadata))
                    self.assertNotIn("PRIVATE_COMMENT", json.dumps(metadata))
                review, = self.named("approvals.review")
                self.assertEqual(review["metadata"]["review_outcome"], "accepted")
                self.assertEqual(review["metadata"]["approval_outcome"], outcome)
                self.assertEqual(review["metadata"]["approval_decision"], decision)
                self.assertNotIn("PRIVATE_COMMENT", json.dumps(review, default=str))
                self.assertNotIn("PRIVATE_BRIEF", json.dumps(review, default=str))
                auth, = [r for r in self.named("authorization.authorize_transaction")
                         if r["metadata"].get("auth_phase") == "inbox_decision"]
                self.assertIsNotNone(auth["parent_run_id"])
                self.assertEqual(auth["metadata"]["auth_decision"], "allow")
                resume, = [r for r in roots if r["metadata"]["interaction_phase"] == "resume"]
                self.assertEqual(resume["metadata"]["approval_outcome"], outcome)
                self.assertEqual("contains-tool-rejection" in resume["tags"], decision == "deny")
                remote = next(r for r in self.collector.traced_runs if r.name == "approvals.review")
                resumed = next(r for r in self.collector.traced_runs if r.extra["metadata"]["interaction_phase"] == "resume")
                self.assertLessEqual(remote.end_time, resumed.start_time)
                verify_review(remote, {"tenant_id": "northstar", "conversation_id": ids["conversation_id"],
                              "approval_id": item.id, "case_id": "approval-test", "decision": decision}, "semantics-audit")

    async def test_unauthorized_or_stale_review_never_claims_human_approval(self):
        for stale, reviewer, code, outcome in ((False, "user:northstar/csm", 404, "denied"),
                                              (True, "user:northstar/lead", 409, "stale")):
            response, _, item = await self.reviewed_run("approve", stale=stale, reviewer=reviewer)
            self.assertEqual(response.status_code, code)
            review, = self.named("approvals.review")
            self.assertEqual(review["metadata"]["review_outcome"], outcome)
            self.assertNotIn("approval_decision", review["metadata"])
            self.assertNotIn("approval_outcome", review["metadata"])
            self.assertIsNone(item.resolution_run_id)
            self.assertEqual(len(self.collector.traced_runs), 2)

    def test_review_verifier_retains_historical_support(self):
        legacy = SimpleNamespace(name="authorization.authorize_transaction", end_time=True, parent_run_id=None,
                                 inputs={"actor": {"user_id": "user:northstar/lead"}}, outputs={"decision": "allow"})
        verify_review(legacy, {"tenant_id": "northstar"}, "old-batch")
        query = review_filter("2026-10-01T12:00:00+00:00")
        self.assertIn('eq(name,"approvals.review")', query)
        self.assertIn('eq(name,"authorization.authorize_transaction")', query)


if __name__ == "__main__":
    unittest.main()
