"""Use real tracer callbacks, replacing only LangSmith network persistence."""

import asyncio
from copy import deepcopy
import json
import unittest
import uuid
from unittest.mock import Mock, patch

import langsmith as ls
from langchain_core.messages import HumanMessage
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langgraph.types import Command

from demo.agent import build_customer_graph, stream_turn
from demo.store import CustomerStore
from demo.services import SERVICE_TOOLS
from demo.tracing import tag_tool_rejection
from tests.test_customer_analytics import SQLModel
from tests.test_customer_observability import all_runs
from tests.test_customer_operations import ACCOUNT, ScriptedCustomerModel, call


class RecordingTracer(LangChainTracer):
    def __init__(self, **kwargs):
        kwargs.setdefault("client", Mock())
        kwargs.setdefault("project_name", "offline-rejection-tests")
        super().__init__(**kwargs)
        self.completed = {}

    def copy_with_metadata_defaults(self, **kwargs):
        cloned = super().copy_with_metadata_defaults(**kwargs)
        cloned.completed = self.completed
        return cloned

    def _persist_run_single(self, run):
        pass

    def _update_run_single(self, run):
        self.completed[run.id] = {"name": run.name, "parent_run_id": run.parent_run_id,
                                  "tags": list(run.tags or []), "metadata": deepcopy(run.extra.get("metadata", {})),
                                  "error": run.error, "inputs": deepcopy(run.inputs),
                                  "outputs": deepcopy(run.outputs), "events": deepcopy(run.events)}


class RejectionTagTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tracing = ls.tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.store = CustomerStore()
        self.tracer = RecordingTracer()
        self.collector = RunCollectorCallbackHandler()
        self.config = {"run_name": "customer_operations.turn", "tags": ["customer-operations"],
                       "metadata": {"demo_label": "keep"},
                       "callbacks": [self.tracer, self.collector], "configurable": {
                           "thread_id": uuid.uuid4().hex, "turn_id": uuid.uuid4().hex,
                           "tenant_id": "northstar", "user_id": "user:northstar/csm",
                           "agent_id": "agent:northstar/customer-ops", "invocation_mode": "tool",
                       }}

    async def execute(self, model=None, *, graph=None, config=None, text="Review Meridian Retail"):
        graph = graph or build_customer_graph(self.store, model)
        return [event async for event in stream_turn(graph, {"messages": [HumanMessage(content=text)]}, config or self.config)]

    def completed(self, name):
        return [run for run in self.tracer.completed.values() if run["name"] == name]

    def assert_untagged(self, run):
        self.assertNotIn("tool-rejected", run["tags"])
        self.assertNotIn("contains-tool-rejection", run["tags"])
        self.assertNotIn("tool_rejected", run["metadata"])
        self.assertNotIn("contains_tool_rejection", run["metadata"])

    async def test_successful_sql_never_gets_rejection_tags(self):
        await self.execute(ScriptedCustomerModel(call("query_customer_analytics", datasets=["usage_daily"],
                                                     sql="SELECT COUNT(*) FROM usage_daily")))
        for run in self.tracer.completed.values():
            self.assert_untagged(run)

    async def test_authorization_denial_tags_tool_and_root_and_persists_metadata(self):
        self.config["configurable"]["user_id"] = "user:northstar/support"
        events = await self.execute(ScriptedCustomerModel(call("query_customer_analytics", datasets=["invoices"],
                                                              sql="SELECT SUM(amount_cents) FROM invoices")))
        tool, = self.completed("tools.query_customer_analytics")
        root, = self.completed("customer_operations.turn")
        self.assertTrue({"tool-rejected", "rejection-authorization"} <= set(tool["tags"]))
        self.assertTrue({"customer-operations", "contains-tool-rejection", "rejection-authorization"} <= set(root["tags"]))
        self.assertNotIn("tool-rejected", root["tags"])
        denial, = [e["data"] for e in events if e["event"] == "tool_denied"]
        self.assertEqual(tool["metadata"]["call_id"], denial["call_id"])
        self.assertEqual(tool["metadata"]["reason_code"], denial["reason_code"])
        self.assertEqual(root["metadata"]["rejected_tool_call_count"], 1)
        self.assertEqual(root["metadata"]["rejected_tools"], ["query_customer_analytics"])
        self.assertEqual(root["metadata"]["tool_rejections"][0]["call_id"], denial["call_id"])
        # A denied tool can retain its original error status; the handled turn is
        # successful. Outcome tags distinguish policy from runtime failures.
        self.assertIsNotNone(tool["error"])
        self.assertIsNone(root["error"])
        auth = [r for r in self.completed("authorization.authorize_transaction") if "fga-deny" in r["tags"]]
        self.assertTrue(auth)
        self.assertTrue(all(r["error"] is None for r in auth))
        collector_root, = self.collector.traced_runs
        self.assertIn("contains-tool-rejection", collector_root.tags)
        self.assertEqual(collector_root.extra["metadata"]["rejected_tool_call_count"], 1)

    async def test_sql_validation_is_distinct_and_does_not_copy_arguments_into_summary(self):
        await self.execute(ScriptedCustomerModel(call("query_customer_analytics", datasets=["usage_daily"],
                                                     sql="DELETE FROM usage_daily /* PRIVATE_TEST_MARKER */")))
        tool, = self.completed("tools.query_customer_analytics")
        root, = self.completed("customer_operations.turn")
        self.assertTrue({"tool-rejected", "rejection-sql-validation"} <= set(tool["tags"]))
        self.assertNotIn("rejection-authorization", root["tags"])
        self.assertEqual(tool["metadata"]["reason_code"], "unsafe_or_invalid_sql")
        self.assertNotIn("PRIVATE_TEST_MARKER", json.dumps(root["metadata"]))
        self.assertNotIn("sql", root["metadata"]["tool_rejections"][0])

    async def test_delegation_denial_tags_the_requested_parent_tool(self):
        self.config["configurable"]["agent_id"] = "agent:northstar/support"
        await self.execute(SQLModel(call("analyze_customer_data", datasets=["usage_daily"], question="Count days")))
        tool, = self.completed("tools.analyze_customer_data")
        self.assertIn("tool-rejected", tool["tags"])
        self.assertIn("agent-call-denied", tool["tags"])
        self.assertEqual(tool["metadata"]["reason_code"], "delegation_not_granted")

    async def test_delegated_sql_validation_and_unsupported_generation_are_tagged(self):
        for sql in ("DELETE FROM usage_daily", "UNSUPPORTED"):
            with self.subTest(sql=sql):
                self.tracer.completed.clear()
                self.config["configurable"]["thread_id"] = uuid.uuid4().hex
                events = await self.execute(SQLModel(call("analyze_customer_data", datasets=["usage_daily"], question="Count days"), sql=sql))
                tool, = self.completed("tools.analyze_customer_data")
                root, = self.completed("customer_operations.turn")
                self.assertIn("rejection-sql-validation", tool["tags"])
                self.assertEqual(root["metadata"]["rejected_tools"], ["analyze_customer_data"])
                self.assertEqual(root["metadata"]["rejected_tool_call_count"], 1)
                self.assertEqual([e["event"] for e in events].count("tool_rejected"), 1)

    async def test_duplicate_signals_and_mixed_calls_preserve_counts_without_sibling_inheritance(self):
        self.config["configurable"]["user_id"] = "user:northstar/support"
        batches = (call("query_customer_analytics", datasets=["invoices"], sql="SELECT COUNT(*) FROM invoices")
                   + call("get_customer_account")
                   + call("query_customer_analytics", datasets=["invoices"], sql="SELECT SUM(amount_cents) FROM invoices"))

        def twice(*args):
            tag_tool_rejection(*args)
            tag_tool_rejection(*args)

        with patch("demo.agent.tag_tool_rejection", side_effect=twice):
            await self.execute(ScriptedCustomerModel(batches))
        root, = self.completed("customer_operations.turn")
        self.assertEqual(root["metadata"]["rejected_tool_call_count"], 2)
        self.assertEqual(len(root["metadata"]["tool_rejections"]), 2)
        self.assertEqual(root["tags"].count("contains-tool-rejection"), 1)
        successful, = self.completed("tools.get_customer_account")
        self.assert_untagged(successful)
        self.assertEqual(self.config["tags"], ["customer-operations"])
        self.assertEqual(self.config["metadata"], {"demo_label": "keep"})
        for run in self.completed("skills.apply_model_middleware"):
            self.assert_untagged(run)

    async def test_concurrent_runs_and_next_turn_do_not_share_outcomes(self):
        denied_config = {**self.config, "configurable": {**self.config["configurable"], "user_id": "user:northstar/support"}}
        allowed_config = {**self.config, "configurable": {**self.config["configurable"], "thread_id": uuid.uuid4().hex}}
        batch = call("query_customer_analytics", datasets=["invoices"], sql="SELECT COUNT(*) FROM invoices")
        await asyncio.gather(self.execute(ScriptedCustomerModel(batch), config=denied_config),
                             self.execute(ScriptedCustomerModel(batch), config=allowed_config))
        roots = self.completed("customer_operations.turn")
        self.assertEqual(sum("contains-tool-rejection" in r["tags"] for r in roots), 1)
        await self.execute(ScriptedCustomerModel(), config=denied_config)
        roots = self.completed("customer_operations.turn")
        self.assertEqual(sum("contains-tool-rejection" in r["tags"] for r in roots), 1)

    async def test_request_denial_and_filtered_skill_discovery_are_not_tool_rejections(self):
        await self.execute(ScriptedCustomerModel(), text="Look up account:beacon/AC-100")
        self.store.fga.delete_tuple("agent:northstar/customer-ops", "reader", "skill:northstar/sql-analysis")
        await self.execute(ScriptedCustomerModel())
        self.assertTrue(any("fga-deny" in r["tags"] for r in self.tracer.completed.values()))
        for root in self.completed("customer_operations.turn"):
            self.assert_untagged(root)

    async def test_unexpected_execution_error_gets_shared_tags_and_distinct_category(self):
        with patch.object(self.store.analytics, "query", side_effect=RuntimeError("Synthetic execution failure")):
            with self.assertRaises(RuntimeError):
                await self.execute(ScriptedCustomerModel(call("query_customer_analytics", datasets=["usage_daily"],
                                                             sql="SELECT COUNT(*) FROM usage_daily")))
        root, = self.completed("customer_operations.turn")
        self.assertIsNotNone(root["error"])
        self.assertIn("contains-tool-rejection", root["tags"])
        self.assertIn("contains-tool-failure", root["tags"])
        self.assertIn("rejection-execution", root["tags"])
        self.assertNotIn("rejection-authorization", root["tags"])
        tool, = self.completed("tools.query_customer_analytics")
        self.assertIn("tool-failed", tool["tags"])
        self.assertEqual(tool["metadata"]["reason_code"], "execution_error")

    async def test_collector_only_and_no_tracer_execution_work(self):
        self.config["configurable"]["user_id"] = "user:northstar/support"
        batch = call("query_customer_analytics", datasets=["invoices"], sql="SELECT COUNT(*) FROM invoices")
        self.config["callbacks"] = [self.collector]
        await self.execute(ScriptedCustomerModel(batch))
        root, = self.collector.traced_runs
        self.assertIn("contains-tool-rejection", root.tags)
        tool, = [r for r in all_runs([root]) if r.name == "tools.query_customer_analytics"]
        self.assertIn("tool-rejected", tool.tags)
        self.config["callbacks"] = []
        events = await self.execute(ScriptedCustomerModel(batch))
        self.assertIn("tool_denied", [e["event"] for e in events])

    async def test_all_five_services_share_outcome_tags_but_successes_remain_clean(self):
        for tool, service in SERVICE_TOOLS.items():
            for failed in (False, True):
                with self.subTest(tool=tool, failed=failed):
                    self.tracer.completed.clear()
                    self.config["configurable"].update(thread_id=uuid.uuid4().hex,
                        user_id="user:northstar/support" if failed and service["restricted"] else "user:northstar/csm")
                    account = ACCOUNT if failed else "account:northstar/AC-101"
                    events = await self.execute(ScriptedCustomerModel(call(tool, account_id=account)))
                    root, = self.completed("customer_operations.turn")
                    span, = self.completed("tools." + tool)
                    self.assertIsNone(root["error"])  # Expected mock failures are handled tool results.
                    if failed:
                        category = "authorization" if service["restricted"] else "execution"
                        self.assertIn("tool-rejected", span["tags"])
                        self.assertIn("rejection-" + category, span["tags"])
                        self.assertIn("contains-tool-rejection", root["tags"])
                        self.assertEqual(root["metadata"]["rejected_tool_call_count"], 1)
                        self.assertNotIn("service_call_completed", [e["event"] for e in events])
                    else:
                        self.assert_untagged(root)
                        self.assert_untagged(span)
                        self.assertIn("service_call_completed", [e["event"] for e in events])

    async def test_invalid_arguments_are_tagged_inside_tool_before_any_store_access(self):
        model = ScriptedCustomerModel([("get_customer_account", {"account_id": ACCOUNT, "tenant_override": "beacon"})])
        with patch.object(self.store, "get_account") as read:
            await self.execute(model)
            read.assert_not_called()
        tool, = self.completed("tools.get_customer_account")
        self.assertIn("rejection-input-validation", tool["tags"])
        self.assertEqual(tool["metadata"]["reason_code"], "invalid_request")

    async def test_approval_pause_is_not_failure_but_human_rejection_is_tagged(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("save_account_brief", content="Fictional test brief")))
        events = await self.execute(graph=graph)
        root, = self.completed("customer_operations.turn")
        self.assert_untagged(root)
        proposal, = [e["data"] for e in events if e["event"] == "approval_required"]
        self.tracer.completed.clear()
        resume = Command(resume={"approval_id": proposal["approval_id"], "reviewer": "user:northstar/lead", "decision": "deny"})
        [event async for event in stream_turn(graph, resume, self.config)]
        tool, = self.completed("tools.save_account_brief")
        self.assertIn("rejection-approval", tool["tags"])
        self.assertNotIn("tool-failed", tool["tags"])
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)


if __name__ == "__main__":
    unittest.main()
