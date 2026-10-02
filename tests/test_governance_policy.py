"""Policy provenance, deterministic signals, and execution-time enforcement."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

import httpx
import langsmith as ls
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from demo.agent import build_customer_graph, stream_turn
from demo.paths import REPO_ROOT
from demo.policy import (MAX_ACTIONS, PolicyExecution, active_policy, load_bundle,
                         resource_evidence, _execution)
from demo.server import create_app
from demo.store import CustomerStore
from tests.test_customer_operations import ACCOUNT, ScriptedCustomerModel, call
from tests.test_customer_rejection_tags import RecordingTracer
from tests.test_customer_subagents import task
from scripts.policy_samples import cases as policy_cases, verify_policy_evidence
from scripts.trace_batch import plan_batch
from types import SimpleNamespace


def decision(index, *, tool="query_customer_analytics", target=ACCOUNT, foreign=False, denied=True):
    return {"action_id": str(index), "call_id": "model-id", "decision": "deny" if denied else "allow",
            "reason_code": "missing_grant" if denied else "granted", "resource_keys": [target],
            "cross_tenant": foreign, "path": ("agent:northstar/customer-ops", "", tool, ""),
            "operation": tool}


class PolicyUnitTests(unittest.TestCase):
    def test_bundle_identity_hash_and_strict_manifest(self):
        manifest = (REPO_ROOT / "policies/demo-governance.json").read_text()
        document = (REPO_ROOT / "policies/demo-governance.md").read_text()
        policy = active_policy()
        self.assertEqual(policy.version, "1.0.0")
        self.assertEqual(len(policy.bundle_hash), 64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "demo-governance.json").write_text(manifest)
            (path / "demo-governance.md").write_text(document)
            self.assertEqual(load_bundle(path), policy)
            (path / "demo-governance.md").write_text(document + "\nChanged rationale.\n")
            self.assertNotEqual(load_bundle(path).bundle_hash, policy.bundle_hash)
            for mutate in (
                lambda data: data.update(policy_version="2.0.0"),
                lambda data: data.update(evaluator_version="unknown"),
                lambda data: data["rules"][0].update(mode="block"),
                lambda data: data["rules"][0].update(threshold=True),
                lambda data: data["rules"][0].update(threshold=0),
                lambda data: data["rules"][0].update(expression="untrusted code"),
                lambda data: data["rules"].append(data["rules"][0]),
            ):
                data = json.loads(manifest)
                mutate(data)
                (path / "demo-governance.json").write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_bundle(path)

    def test_exact_threshold_duplicate_decisions_and_configured_threshold(self):
        state = PolicyExecution(active_policy())
        self.assertEqual(state.observe(decision(1)), [])
        self.assertEqual(state.observe(decision(1)), [])
        self.assertEqual(state.observe(decision(2)), [])
        findings = state.observe(decision(3))
        self.assertEqual([f["control_id"] for f in findings], ["MON-001"])
        self.assertEqual(findings[0]["count"], 3)
        self.assertEqual(state.summary("completed")["policy_denied_action_count"], 3)
        rules = tuple(replace(r, threshold=4) if r.control_id == "MON-001" else r for r in active_policy().rules)
        state = PolicyExecution(replace(active_policy(), rules=rules))
        for i in range(3):
            self.assertEqual(state.observe(decision(i)), [])
        self.assertEqual(state.observe(decision(3))[0]["count"], 4)

    def test_same_target_different_paths_and_tenant_isolation(self):
        state = PolicyExecution(active_policy())
        state.observe(decision(1))
        state.observe(decision(2, tool="task", target="account:northstar/AC-101"))
        self.assertFalse(state.summary("completed")["security_signals"])
        findings = state.observe(decision(3, tool="task"))
        multi, = [f for f in findings if f["control_id"] == "MON-002"]
        self.assertEqual(multi["resource_key"], ACCOUNT)
        self.assertEqual(multi["count"], 2)
        self.assertEqual(len(multi["evidence"]), 2)

    def test_concurrent_duplicate_events_bounded_evidence_and_state(self):
        state = PolicyExecution(active_policy())
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(state.observe, [decision(i % 80) for i in range(160)]))
        summary = state.summary("completed")
        self.assertEqual(summary["policy_denied_action_count"], 80)
        finding, = summary["security_signals"]
        self.assertEqual(len(finding["evidence"]), 50)
        self.assertTrue(finding["evidence_truncated"])
        for i in range(80, MAX_ACTIONS + 1):
            state.observe(decision(i))
        summary = state.summary("completed")
        self.assertEqual(summary["policy_evaluation_status"], "incomplete")
        self.assertTrue(summary["policy_actions_truncated"])
        self.assertEqual(summary["policy_denied_action_count"], MAX_ACTIONS)

    def test_canonical_resource_resolution_does_not_infer_unknown_foreign_targets(self):
        store = CustomerStore()
        actor = store.actor("northstar", "user:northstar/csm")
        for reference, expected in (("Meridian Retail", ([ACCOUNT], False)),
                                    ("AC-100", ([ACCOUNT], False)),
                                    ("account:beacon/AC-100", (["account:beacon/AC-100"], True)),
                                    ("account:beacon/AC-999", ([], False))):
            self.assertEqual(resource_evidence(store, actor, {"resource": reference}), expected)

    def test_focused_batch_and_verifier_accept_only_complete_consistent_evidence(self):
        cases = plan_batch(6, "governance")["ordinary"]
        self.assertEqual(len(cases), 6)
        self.assertEqual(sum(bool(c["expected_security_signals"]) for c in cases), 3)
        metadata = {**active_policy().metadata(), **PolicyExecution(active_policy()).summary("completed")}
        root = SimpleNamespace(extra={"metadata": metadata}, tags=[], name="customer_operations.turn")
        child = SimpleNamespace(extra={"metadata": active_policy().metadata()}, name="tool")
        verify_policy_evidence(root, [child])
        metadata["policy_evaluation_status"] = "incomplete"
        with self.assertRaises(AssertionError):
            verify_policy_evidence(root, [child])
        metadata["policy_evaluation_status"] = "completed"
        child.extra["metadata"]["policy_version"] = "wrong-version"
        with self.assertRaises(AssertionError):
            verify_policy_evidence(root, [child])


class GovernanceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        context = ls.tracing_context(enabled=False)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.store = CustomerStore()
        self.tracer = RecordingTracer()
        self.config = {"run_name": "customer_operations.turn", "callbacks": [self.tracer],
                       "metadata": {"policy_version": "caller-cannot-override", "demo_label": "preserve"},
                       "configurable": {"thread_id": uuid.uuid4().hex, "turn_id": uuid.uuid4().hex,
                           "tenant_id": "northstar", "user_id": "user:northstar/support",
                           "agent_id": "agent:northstar/customer-ops"}}

    async def execute(self, model, *, text="Perform the requested check", config=None, graph=None, resume=None):
        graph = graph or build_customer_graph(self.store, model)
        data = Command(resume=resume) if resume else {"messages": [HumanMessage(content=text)]}
        return [event async for event in stream_turn(graph, data, config or self.config)]

    def named(self, name):
        return [run for run in self.tracer.completed.values() if run["name"] == name]

    def root(self):
        root, = self.named("customer_operations.turn")
        return root

    def assert_no_signals(self):
        self.assertEqual(self.root()["metadata"]["security_signals"], [])
        for run in self.tracer.completed.values():
            self.assertFalse(any(tag.startswith("security-signal:") for tag in run["tags"]))

    async def test_three_denials_signal_once_and_prevent_sql_execution(self):
        calls = sum([call("query_customer_analytics", datasets=["invoices"], sql="SELECT COUNT(*) FROM invoices")
                     for _ in range(3)], []) + call("get_customer_account")
        with patch.object(self.store.analytics, "_execute", wraps=self.store.analytics._execute) as execute:
            await self.execute(ScriptedCustomerModel(calls))
        execute.assert_not_called()
        metadata = self.root()["metadata"]
        self.assertEqual(metadata["policy_denied_action_count"], 3)
        self.assertEqual(metadata["policy_evaluation_status"], "completed")
        self.assertEqual([s["control_id"] for s in metadata["security_signals"]], ["MON-001"])
        self.assertEqual(len(self.named("governance.evaluate_signals")), 1)
        clean, = self.named("tools.get_customer_account")
        self.assertNotIn("security_signals", clean["metadata"])
        self.assertFalse(any(t.startswith("security-signal:") for t in clean["tags"]))
        for run in self.tracer.completed.values():
            for key, value in active_policy().metadata().items():
                self.assertEqual(run["metadata"].get(key), value, (run["name"], key))
        finding, = metadata["security_signals"]
        self.assertEqual(len({a["action_id"] for a in finding["evidence"]}), 3)
        self.assertTrue(all(a.get("run_id") and a.get("trace_id") for a in finding["evidence"]))
        self.assertNotIn("SELECT", json.dumps(metadata))
        self.assertEqual(self.config["metadata"]["policy_version"], "caller-cannot-override")

    async def test_direct_and_native_denied_paths_are_canonical_and_stop_service(self):
        model = ScriptedCustomerModel(call("get_billing_snapshot") + task("billing-review", ACCOUNT))
        events = await self.execute(model)
        self.assertNotIn("service_call_started", [e["event"] for e in events])
        finding, = self.root()["metadata"]["security_signals"]
        self.assertEqual(finding["control_id"], "MON-002")
        self.assertEqual({a["operation"] for a in finding["evidence"]}, {"get_billing_snapshot", "task"})
        owner, = self.named("tools.task.billing-review")
        self.assertIn(finding["tag"], owner["tags"])

    async def test_cross_tenant_gate_and_native_early_denial_without_disclosure(self):
        for reference, expected in (("account:beacon/AC-100", True), ("account:beacon/AC-999", False)):
            self.tracer.completed.clear()
            model = ScriptedCustomerModel()
            events = await self.execute(model, text="Look up " + reference)
            self.assertEqual(model.seen, [])
            signals = self.root()["metadata"]["security_signals"]
            self.assertEqual(bool(signals), expected)
            if expected:
                self.assertEqual(signals[0]["control_id"], "MON-003")
            self.assertNotIn("security_signals", json.dumps(events))
            self.assertIn("Access denied", next(e["data"]["content"] for e in events if e["event"] == "agent_response"))
        self.tracer.completed.clear()
        await self.execute(ScriptedCustomerModel(task("billing-review", "account:beacon/AC-100")))
        self.assertEqual(self.root()["metadata"]["security_signals"][0]["control_id"], "MON-003")

    async def test_discovery_sql_validation_service_failures_and_success_are_not_denials(self):
        self.store.fga.delete_tuple("agent:northstar/customer-ops", "reader", "skill:northstar/sql-analysis")
        calls = call("get_customer_account") + call("get_support_sla_report") + call("get_crm_sync_status")
        calls += call("query_customer_analytics", datasets=["usage_daily"], sql="DELETE FROM usage_daily")
        await self.execute(ScriptedCustomerModel(calls))
        self.assert_no_signals()
        self.assertEqual(self.root()["metadata"]["policy_denied_action_count"], 0)
        self.assertEqual(self.root()["metadata"]["policy_evaluation_status"], "completed")

    async def test_detector_is_independent_of_tracing_and_projection_failures(self):
        calls = call("get_billing_snapshot") * 3
        config = {**self.config, "callbacks": []}
        with patch("demo.tracing.tag_policy_summary") as summary:
            await self.execute(ScriptedCustomerModel(calls), config=config)
        self.assertEqual(summary.call_args.args[2]["policy_denied_action_count"], 3)
        self.assertEqual(summary.call_args.args[2]["security_signals"][0]["control_id"], "MON-001")
        with patch("demo.tracing.tag_policy_findings", side_effect=RuntimeError("offline projection")):
            events = await self.execute(ScriptedCustomerModel(calls))
        self.assertEqual(self.root()["metadata"]["policy_denied_action_count"], 3)
        self.assertTrue(self.root()["metadata"]["policy_trace_projection_failed"])
        self.assertNotIn("service_call_started", [e["event"] for e in events])

    async def test_concurrent_runs_and_following_turn_reset_observation_window(self):
        allowed = {**self.config, "configurable": {**self.config["configurable"],
            "thread_id": uuid.uuid4().hex, "user_id": "user:northstar/csm"}}
        await asyncio.gather(self.execute(ScriptedCustomerModel(call("get_billing_snapshot") * 3)),
                             self.execute(ScriptedCustomerModel(call("get_billing_snapshot") * 3), config=allowed))
        roots = self.named("customer_operations.turn")
        self.assertEqual(sorted(r["metadata"]["policy_denied_action_count"] for r in roots), [0, 3])
        self.tracer.completed.clear()
        await self.execute(ScriptedCustomerModel(call("get_billing_snapshot")))
        self.assert_no_signals()
        self.assertEqual(self.root()["metadata"]["policy_denied_action_count"], 1)

    async def test_child_signal_owned_by_child_and_parent_without_sibling_leakage(self):
        self.config["configurable"]["user_id"] = "user:northstar/csm"
        self.store.fga.delete_tuple("agent:northstar/billing-review", "reader", "service:northstar/billing")
        model = ScriptedCustomerModel(task(), call("get_billing_snapshot", account_id="account:northstar/AC-101") * 3
                                      + call("get_customer_account", account_id="account:northstar/AC-101"))
        await self.execute(model)
        for name in ("billing-review", "tools.task.billing-review", "customer_operations.turn"):
            run, = self.named(name)
            self.assertIn("security-signal:repeated-authorization-denials", run["tags"])
        leaf, = self.named("get_customer_account")
        self.assertNotIn("security_signals", leaf["metadata"])
        self.assertEqual(self.root()["metadata"]["policy_denied_action_count"], 3)
        for run in self.tracer.completed.values():
            for key, value in active_policy().metadata().items():
                self.assertEqual(run["metadata"].get(key), value, (run["name"], key))

    async def test_reused_model_call_ids_do_not_hide_distinct_attempts(self):
        class ReusedIDs(ScriptedCustomerModel):
            async def ainvoke(self, messages, config=None):
                message = await super().ainvoke(messages, config)
                for tool in message.tool_calls:
                    tool["id"] = "reused-model-id"
                return message
        await self.execute(ReusedIDs(call("get_billing_snapshot") * 3))
        self.assertEqual(self.root()["metadata"]["policy_denied_action_count"], 3)
        finding, = self.root()["metadata"]["security_signals"]
        self.assertEqual(len({a["action_id"] for a in finding["evidence"]}), 3)

    async def test_failed_execution_has_error_status_not_a_clean_evaluation(self):
        class FailedModel(ScriptedCustomerModel):
            async def ainvoke(self, messages, config=None):
                raise RuntimeError("synthetic model failure")
        with self.assertRaises(RuntimeError):
            await self.execute(FailedModel())
        self.assertEqual(self.root()["metadata"]["policy_evaluation_status"], "error")

    async def test_later_raw_sdk_children_do_not_inherit_security_findings(self):
        tracer = self.tracer
        captured = []
        class InspectRawChild(ScriptedCustomerModel):
            async def ainvoke(self, messages, config=None):
                if self.seen:
                    root, = [r for r in tracer.run_map.values() if r.name == "customer_operations.turn"]
                    raw = root.create_child("raw-sdk-middleware")
                    captured.append(dict(raw.extra["metadata"]))
                return await super().ainvoke(messages, config)
        await self.execute(InspectRawChild(call("get_billing_snapshot") * 3))
        self.assertTrue(captured)
        self.assertNotIn("security_signals", captured[0])
        self.assertEqual(captured[0]["policy_version"], "1.0.0")
        self.assertTrue(self.root()["metadata"]["security_signals"])

    async def test_unrelated_denial_does_not_inherit_an_earlier_cross_tenant_finding(self):
        await self.execute(ScriptedCustomerModel(call("get_customer_account", account_id="account:beacon/AC-100")
                                                 + call("get_billing_snapshot")))
        leaf, = self.named("tools.get_billing_snapshot")
        self.assertNotIn("security_signals", leaf["metadata"])
        self.assertEqual(len(self.named("governance.evaluate_signals")), 1)

    async def test_stream_yields_without_leaking_policy_context_and_can_close_in_another_task(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("get_customer_account")))
        stream = stream_turn(graph, {"messages": [HumanMessage(content="Review Meridian Retail")]}, self.config)
        await anext(stream)
        self.assertIsNone(_execution.get())
        await asyncio.create_task(stream.aclose())
        self.assertIsNone(_execution.get())

    async def test_approval_policy_binding_interruption_hold_and_revocation(self):
        self.config["configurable"]["user_id"] = "user:northstar/csm"
        for decision, revoke in (("conditional", False), ("deny", False), ("approve", True)):
            self.tracer.completed.clear()
            graph = build_customer_graph(self.store, ScriptedCustomerModel(call("save_account_brief", content="Fictional brief")))
            events = await self.execute(None, graph=graph)
            proposal = next(e["data"] for e in events if e["event"] == "approval_required")
            self.assertEqual(self.root()["metadata"]["policy_evaluation_status"], "interrupted")
            self.assert_no_signals()
            for key, value in active_policy().metadata().items():
                self.assertEqual(proposal[key], value)
            self.tracer.completed.clear()
            if revoke:
                self.store.fga.delete_tuple("agent:northstar/customer-ops", "executor", "tool:northstar/save_account_brief")
            await self.execute(None, graph=graph, resume={"approval_id": proposal["approval_id"],
                "reviewer": "user:northstar/lead", "decision": decision})
            self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)
            self.assert_no_signals()

    async def test_request_review_resume_share_identity_and_changed_policy_is_stale(self):
        for changed in (False, True):
            self.tracer.completed.clear()
            app = create_app(model=ScriptedCustomerModel(call("save_account_brief", content="Fictional brief")),
                             callbacks=[self.tracer])
            runtime = app.state.runtime
            async def drain():
                while runtime.tasks:
                    await asyncio.gather(*tuple(runtime.tasks))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                await client.get("/api/catalog")
                await client.post("/api/run", json={"user_msg": "Save a brief for Meridian Retail"})
                await drain()
                item, = runtime.inbox.items.values()
                policy = replace(active_policy(), bundle_hash="a" * 64) if changed else active_policy()
                with patch("demo.policy.active_policy", return_value=policy):
                    response = await client.post(f"/api/approvals/{item.id}/decision", json={
                        "tenant_id": "northstar", "user_id": "user:northstar/lead", "decision": "approve"})
                    await drain()
                if changed:
                    self.assertEqual(response.status_code, 409)
                    self.assertEqual(item.outcome, "policy_changed")
                    self.assertEqual(runtime.store.briefs[ACCOUNT]["version"], 1)
                else:
                    self.assertEqual(response.status_code, 202)
                    roots = [r for r in self.tracer.completed.values() if r["parent_run_id"] is None]
                    self.assertEqual(len(roots), 3)
                    for root in roots:
                        for key, value in active_policy().metadata().items():
                            self.assertEqual(root["metadata"][key], value)
                        self.assertIn(root["metadata"]["policy_evaluation_status"], {"completed", "interrupted"})

    async def test_all_governance_examples_via_normal_http_entry_point(self):
        scripts = [call("get_billing_snapshot") * 3,
                   call("get_billing_snapshot") + task("billing-review", ACCOUNT), [],
                   call("get_customer_account"), call("get_support_sla_report"),
                   call("query_customer_analytics", datasets=["usage_daily"],
                        sql="SELECT missing_demo_column FROM usage_daily")]
        for case, calls in zip(policy_cases(), scripts):
            with self.subTest(case=case["case_id"]):
                self.tracer.completed.clear()
                app = create_app(model=ScriptedCustomerModel(calls), callbacks=[self.tracer])
                runtime = app.state.runtime
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                    await client.get("/api/catalog")
                    payload = {"user_msg": case["prompt"], "tenant_id": case["tenant_id"],
                               "user_id": case["user_id"], "profile": case["profile"]}
                    invalid = await client.post("/api/run", json={**payload, "policy_version": "forged"})
                    self.assertEqual(invalid.status_code, 400)
                    response = await client.post("/api/run", json=payload)
                    self.assertEqual(response.status_code, 200)
                    while runtime.tasks:
                        await asyncio.gather(*tuple(runtime.tasks))
                    actual = sorted(t for t in self.root()["tags"] if t.startswith("security-signal:"))
                    self.assertEqual(actual, sorted(case["expected_security_signals"]))
                    events = runtime.runs[response.json()["run_id"]].events
                    self.assertIn(case["expected_event"], [e["event"] for e in events])
                    spans = [SimpleNamespace(extra={"metadata": r["metadata"]}, tags=r["tags"], name=r["name"])
                             for r in self.tracer.completed.values() if r["parent_run_id"] is not None]
                    root = SimpleNamespace(extra={"metadata": self.root()["metadata"]}, tags=self.root()["tags"])
                    verify_policy_evidence(root, spans)

    async def test_detector_errors_are_incomplete_not_compliant_and_do_not_allow_execution(self):
        with patch.object(PolicyExecution, "observe", side_effect=RuntimeError("detector fault")):
            events = await self.execute(ScriptedCustomerModel(call("get_billing_snapshot")))
        self.assertEqual(self.root()["metadata"]["policy_evaluation_status"], "incomplete")
        self.assertNotIn("service_call_started", [e["event"] for e in events])


if __name__ == "__main__":
    unittest.main()
