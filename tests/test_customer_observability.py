"""Offline trace-tree, authorization-boundary, and incremental-stream contracts."""

import asyncio
import json
import unittest
import uuid
from typing import Any
from unittest.mock import Mock

import langsmith as ls
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.types import Command

from demo.agent import build_customer_graph, response_text, stream_turn
from demo.store import CustomerStore
from demo.server import Conversation, DemoRuntime
from tests.test_customer_operations import ACCOUNT, NOTE, ScriptedCustomerModel, call


class StreamingTestModel(BaseChatModel):
    """Uses real LangChain model callbacks, but no network or external model."""

    batches: list[list[AIMessageChunk]]
    gate: Any = None
    fail_after_first: bool = False
    completed: int = 0
    started: int = 0

    @property
    def _llm_type(self):
        return "offline-streaming-test-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("The test must use actual model streaming")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        batch = self.batches[self.started]
        self.started += 1
        for index, chunk in enumerate(batch):
            if index == 1 and self.gate is not None:
                await self.gate.wait()
            if index == 1 and self.fail_after_first:
                raise RuntimeError("Synthetic provider failure")
            yield ChatGenerationChunk(message=chunk)
        self.completed += 1


def all_runs(roots):
    for run in roots:
        yield run
        yield from all_runs(run.child_runs)


class ObservabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tracing = ls.tracing_context(enabled=False)
        self.tracing.__enter__()
        self.store = CustomerStore()
        self.collector = RunCollectorCallbackHandler()
        self.config = {
            "run_name": "customer_operations.turn", "callbacks": [self.collector],
            "configurable": {
                "thread_id": uuid.uuid4().hex, "turn_id": uuid.uuid4().hex,
                "tenant_id": "northstar", "user_id": "user:northstar/csm",
                "agent_id": "agent:northstar/customer-ops", "invocation_mode": "tool",
            },
        }

    def tearDown(self):
        self.tracing.__exit__(None, None, None)

    async def execute(self, model, *, graph=None, input_data=None):
        graph = graph or build_customer_graph(self.store, model)
        data = input_data if input_data is not None else {"messages": [HumanMessage(content="Review Meridian Retail")]}
        return [event async for event in stream_turn(graph, data, self.config)]

    def auth_runs(self, scope="tool"):
        return [run for run in all_runs(self.collector.traced_runs)
                if run.name == "authorization.authorize_transaction"
                and run.extra.get("metadata", {}).get("auth_scope") == scope]

    async def test_authorization_is_a_named_child_span_of_each_tool(self):
        events = await self.execute(ScriptedCustomerModel(call("get_customer_account") + call("search_account_records")))
        runs = {run.id: run for run in all_runs(self.collector.traced_runs)}
        auth = self.auth_runs()
        self.assertEqual(len(auth), 2)
        for run in auth:
            operation = run.inputs["operation"]
            self.assertEqual(runs[run.parent_run_id].name, "tools." + operation)
            self.assertEqual(run.outputs["decision"], "allow")
            self.assertEqual(run.extra["metadata"]["auth_phase"], "invoke")
            self.assertNotIn("content", json.dumps(run.inputs))
            self.assertLessEqual(run.end_time, runs[run.parent_run_id].end_time)
        names = {run.name for run in runs.values()}
        self.assertFalse(names & {"Unnamed", "RunnableLambda", "entry", "agent", "tools"}, names)
        self.assertIn("customer_operations.route_next_step", names)
        self.assertIn("customer_operations.route_invocation", names)
        for operation in ("get_customer_account", "search_account_records"):
            sequence = [e["event"] for e in events if e["data"].get("tool_name") == operation]
            self.assertLess(sequence.index("authorization_started"), sequence.index("fga_check"))
            self.assertLess(sequence.index("fga_decision"), sequence.index("authorization_completed"))
            self.assertLess(sequence.index("authorization_completed"), sequence.index("tool_completed"))

    async def test_denied_transaction_has_explicit_output_and_no_authorized_scope(self):
        await self.execute(ScriptedCustomerModel(call("get_customer_account", account_id="account:beacon/AC-100")))
        run, = self.auth_runs()
        self.assertEqual(run.outputs, {"decision": "deny", "reason_code": "resource_unavailable"})
        self.assertIsNone(run.error)  # A policy denial is a decision, not middleware failure.

    async def test_fga_events_belong_to_auth_span_in_langsmith_context(self):
        class LocalLangSmithTracer(LangChainTracer):
            def _persist_run_single(self, run):
                pass  # Keep the real tracing context without sending anything.

            def _update_run_single(self, run):
                captured.append(run)

        captured = []
        tracer = LocalLangSmithTracer(client=Mock(), project_name="offline-test")
        self.config["callbacks"].append(tracer)
        await self.execute(ScriptedCustomerModel(call("get_customer_account")))
        auth, = [r for r in captured if r.name == "authorization.authorize_transaction"
                 and r.extra["metadata"]["auth_scope"] == "tool"]
        names = [event["name"] for event in auth.events]
        self.assertIn("authorization_started", names)
        self.assertIn("fga_check", names)
        self.assertIn("fga_decision", names)
        self.assertIn("authorization_completed", names)
        tool, = [r for r in captured if r.name == "tools.get_customer_account"]
        self.assertNotIn("fga_decision", [event["name"] for event in tool.events])
        self.assertEqual(auth.extra["metadata"]["call_id"], tool.extra["metadata"]["call_id"])

    async def test_every_no_tool_turn_has_request_and_tenant_middleware_before_model(self):
        model = StreamingTestModel(batches=[[AIMessageChunk(content="Hello.")]])
        await self.execute(model, input_data={"messages": [HumanMessage(content="Hello")]})
        request, = self.auth_runs(scope="request")
        self.assertEqual(request.inputs["references"], [])
        self.assertEqual(request.outputs["decision"], "allow")
        tenant, = [r for r in request.child_runs if r.name == "authorization.verify_tenant"]
        self.assertEqual(tenant.outputs["decision"], "allow")
        self.assertTrue(all(item["allowed"] for item in tenant.outputs["checks"]))
        model_run, = [r for r in all_runs(self.collector.traced_runs) if r.run_type == "llm"]
        self.assertLessEqual(request.end_time, model_run.start_time)
        self.assertEqual(self.auth_runs(), [])
        self.assertNotIn("Hello", json.dumps(request.inputs))

    async def test_wrong_tenant_account_name_denied_before_model_can_skip_tools(self):
        self.config["configurable"].update(tenant_id="beacon", user_id="user:beacon/csm", agent_id="agent:beacon/customer-ops")
        model = ScriptedCustomerModel()  # It would return a no-tool answer if invoked.
        events = await self.execute(model)
        self.assertEqual(model.seen, [])
        request, = self.auth_runs(scope="request")
        self.assertEqual(request.inputs["references"], ["Meridian Retail"])
        self.assertEqual(request.outputs, {"decision": "deny", "reason_code": "resource_unavailable"})
        tenant, = [r for r in request.child_runs if r.name == "authorization.verify_tenant"]
        self.assertEqual(tenant.outputs["decision"], "allow")  # Valid Beacon actor, unavailable account.
        self.assertIn("request_denied", [e["event"] for e in events])
        self.assertNotIn("tool_intent", [e["event"] for e in events])
        self.assertNotIn("account:northstar", json.dumps(events) + json.dumps(request.outputs))
        self.assertIn("Access denied", next(e["data"]["content"] for e in events if e["event"] == "agent_response"))

    async def test_correct_tenant_account_name_passes_request_preflight(self):
        model = ScriptedCustomerModel()
        await self.execute(model)
        self.assertEqual(len(model.seen), 1)
        request, = self.auth_runs(scope="request")
        self.assertEqual(request.outputs["checked_references"], ["Meridian Retail"])
        self.assertEqual(request.outputs["decision"], "allow")

    async def test_foreign_and_unknown_qualified_accounts_have_same_denial(self):
        results = []
        for reference in ("account:beacon/AC-100", "account:unknown/AC-999"):
            model = ScriptedCustomerModel()
            events = await self.execute(model, input_data={"messages": [HumanMessage(content="Look up " + reference)]})
            self.assertEqual(model.seen, [])
            results.append(next(e["data"]["content"] for e in events if e["event"] == "agent_response"))
        self.assertEqual(results[0], results[1])
        self.assertTrue(all(r.outputs == {"decision": "deny", "reason_code": "resource_unavailable"}
                            for r in self.auth_runs(scope="request")))

    async def test_invalid_context_or_revoked_membership_blocks_greeting(self):
        for subject in ("user:northstar/csm", "agent:northstar/customer-ops"):
            self.store = CustomerStore()
            self.store.fga.delete_tuple(subject, "member", "tenant:northstar")
            model = ScriptedCustomerModel()
            events = await self.execute(model, input_data={"messages": [HumanMessage(content="Hello")]})
            self.assertEqual(model.seen, [])
            self.assertIn("request_denied", [e["event"] for e in events])
            tenant = next(e["data"] for e in events if e["event"] == "tenant_verification_completed")
            self.assertEqual(tenant["decision"], "deny")
        self.store = CustomerStore()
        self.config["configurable"]["user_id"] = "user:beacon/csm"
        # Even an accidental membership grant cannot change the stored identity's tenant.
        self.store.fga.write_tuple("user:beacon/csm", "member", "tenant:northstar")
        model = ScriptedCustomerModel()
        await self.execute(model, input_data={"messages": [HumanMessage(content="Hello")]})
        self.assertEqual(model.seen, [])

    async def test_request_decision_is_not_reused_across_turns(self):
        model = ScriptedCustomerModel()
        graph = build_customer_graph(self.store, model)
        await self.execute(None, graph=graph, input_data={"messages": [HumanMessage(content="Hello")]})
        self.store.fga.delete_tuple("user:northstar/csm", "member", "tenant:northstar")
        events = await self.execute(None, graph=graph, input_data={"messages": [HumanMessage(content="And now?")]})
        self.assertEqual(len(model.seen), 1)
        self.assertIn("request_denied", [e["event"] for e in events])
        self.assertEqual([r.outputs["decision"] for r in self.auth_runs(scope="request")], ["allow", "deny"])

    async def test_membership_revocation_during_review_blocks_write_and_next_model(self):
        model = ScriptedCustomerModel(call("save_account_brief", content="Reviewed text"))
        graph = build_customer_graph(self.store, model)
        events = await self.execute(None, graph=graph)
        proposal = next(e["data"] for e in events if e["event"] == "approval_required")
        calls_before = len(model.seen)
        self.store.fga.delete_tuple("user:northstar/csm", "member", "tenant:northstar")
        events = await self.execute(None, graph=graph, input_data=Command(resume={
            "approval_id": proposal["approval_id"], "reviewer": "user:northstar/lead", "decision": "approve",
        }))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)
        self.assertEqual(len(model.seen), calls_before)
        self.assertTrue(any(e["event"] == "tenant_verification_completed" and e["data"]["decision"] == "deny" for e in events))

    def test_explicit_reference_matching_is_scoped_and_deduplicated(self):
        extract = self.store.authorization.explicit_account_references
        self.assertEqual(extract("Compare MERIDIAN   RETAIL with account:beacon/AC-100, AC-101 and Meridian Retail."),
                         ["MERIDIAN RETAIL", "account:beacon/AC-100", "AC-101"])
        self.assertEqual(extract("NotMeridian Retailish; Meridian."), [])
        self.assertEqual(extract("Read document:northstar/100-commercial"), [])

    async def test_filtering_does_not_expose_hidden_identifiers_in_auth_trace(self):
        self.config["configurable"]["user_id"] = "user:northstar/support"
        await self.execute(ScriptedCustomerModel(call("search_account_records", query="concession")))
        run, = self.auth_runs()
        self.assertNotIn(NOTE, json.dumps(run.inputs) + json.dumps(run.outputs))
        self.assertEqual(len(run.outputs["record_ids"]), 3)

    async def test_specialist_evidence_has_its_own_actor_and_auth_transactions(self):
        await self.execute(ScriptedCustomerModel(call("assess_renewal_readiness")))
        auth = self.auth_runs()
        self.assertEqual(len(auth), 3)
        parent, *child = auth
        self.assertEqual(parent.inputs["operation"], "assess_renewal_readiness")
        self.assertEqual({r.inputs["operation"] for r in child}, {"get_customer_account", "search_account_records"})
        self.assertTrue(all(r.inputs["actor"]["agent_id"] == "agent:northstar/renewal-analyst" for r in child))
        self.assertTrue(all(r.inputs["actor"]["user_id"] == "user:northstar/csm" for r in auth))

    async def test_resume_runs_a_fresh_middleware_transaction_without_brief_content(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("save_account_brief", content="Do not copy this into auth inputs.")))
        events = await self.execute(None, graph=graph)
        proposal = next(e["data"] for e in events if e["event"] == "approval_required")
        await self.execute(None, graph=graph, input_data=Command(resume={
            "approval_id": proposal["approval_id"], "reviewer": "user:northstar/lead", "decision": "approve",
        }))
        resumed = [r for r in self.auth_runs() if r.inputs["phase"] == "resume"]
        self.assertEqual(len(resumed), 1)
        self.assertEqual(resumed[0].inputs["reviewer"], "user:northstar/lead")
        self.assertNotIn("Do not copy", json.dumps(resumed[0].inputs))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 2)

    async def test_response_delta_arrives_before_model_finishes(self):
        gate = asyncio.Event()
        model = StreamingTestModel(batches=[[AIMessageChunk(content="Meridian "), AIMessageChunk(content="Retail.")]], gate=gate)
        graph = build_customer_graph(self.store, model)
        stream = stream_turn(graph, {"messages": [HumanMessage(content="Hello")]}, self.config)
        events = []
        try:
            while not any(e["event"] == "response_delta" for e in events):
                events.append(await asyncio.wait_for(anext(stream), 3))
            self.assertEqual(model.completed, 0)
            self.assertEqual(events[-1]["data"]["content"], "Meridian ")
            self.assertFalse(any(e["event"] == "agent_response" for e in events))
        finally:
            gate.set()
        events.extend([event async for event in stream])
        deltas = [e["data"] for e in events if e["event"] == "response_delta"]
        final, = [e["data"] for e in events if e["event"] == "agent_response"]
        self.assertEqual("".join(d["content"] for d in deltas), final["content"])
        self.assertEqual(final["content"], "Meridian Retail.")
        self.assertEqual({d["message_id"] for d in deltas}, {final["message_id"]})
        model_runs = [r for r in all_runs(self.collector.traced_runs) if r.run_type == "llm"]
        self.assertEqual([r.name for r in model_runs], ["customer_operations.generate_response"])

    async def test_tool_call_chunks_are_not_rendered_and_preamble_has_separate_bubble(self):
        model = StreamingTestModel(batches=[
            [AIMessageChunk(content="I'll check the account."), AIMessageChunk(content="", tool_call_chunks=[{
                "name": "get_customer_account", "id": "lookup-1", "args": json.dumps({"account_id": ACCOUNT}), "index": 0,
            }])],
            [AIMessageChunk(content="Meridian "), AIMessageChunk(content="Retail.")],
        ])
        events = await self.execute(model)
        text = "".join(e["data"]["content"] for e in events if e["event"] == "response_delta")
        self.assertEqual(text, "I'll check the account.Meridian Retail.")
        starts = [e["data"]["message_id"] for e in events if e["event"] == "response_start"]
        self.assertEqual(len(set(starts)), 2)
        ends = [e["data"]["intermediate"] for e in events if e["event"] == "response_end"]
        self.assertEqual(ends, [True, False])
        self.assertEqual(len([e for e in events if e["event"] == "agent_response"]), 1)

    def test_only_text_content_blocks_are_visible(self):
        self.assertEqual(response_text([
            {"type": "thinking", "thinking": "private reasoning", "text": "also private"},
            {"type": "tool_use", "text": "private args"},
            {"type": "text", "text": "Visible"},
        ]), "Visible")

    async def test_server_records_deltas_before_completion_and_orders_terminal_events(self):
        gate = asyncio.Event()
        model = StreamingTestModel(batches=[[AIMessageChunk(content="First "), AIMessageChunk(content="second.")]], gate=gate)
        runtime = DemoRuntime(self.store, model)
        conversation = Conversation(uuid.uuid4().hex, "test-session", self.store.actor("northstar", "user:northstar/csm"))
        run = runtime.start(conversation, {"messages": [HumanMessage(content="Hello")]})
        async def wait_for_delta():
            async with run.changed:
                await run.changed.wait_for(lambda: any(e["event"] == "response_delta" for e in run.events))
        try:
            await asyncio.wait_for(wait_for_delta(), 3)
            self.assertFalse(run.done)
            self.assertEqual(model.completed, 0)
            self.assertNotIn("agent_response", [e["event"] for e in run.events])
        finally:
            gate.set()
            await asyncio.gather(*runtime.tasks)
        self.assertEqual([e["event"] for e in run.events][-3:], ["response_end", "agent_response", "done"])

    async def test_failure_after_partial_output_has_no_fake_final_response(self):
        model = StreamingTestModel(batches=[[AIMessageChunk(content="Partial "), AIMessageChunk(content="not delivered")]], fail_after_first=True)
        runtime = DemoRuntime(self.store, model)
        conversation = Conversation(uuid.uuid4().hex, "test-session", self.store.actor("northstar", "user:northstar/csm"))
        run = runtime.start(conversation, {"messages": [HumanMessage(content="Hello")]})
        await asyncio.gather(*runtime.tasks)
        names = [e["event"] for e in run.events]
        self.assertIn("response_delta", names)
        self.assertNotIn("agent_response", names)
        self.assertEqual(names[-2:], ["error", "done"])
        self.assertNotIn("Synthetic provider failure", json.dumps(run.events))


if __name__ == "__main__":
    unittest.main()
