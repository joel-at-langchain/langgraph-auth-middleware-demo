"""Behavioral checks through the served parent graph and HTTP session boundary."""

import asyncio
import json
import unittest
import uuid
from collections import deque

import httpx
import langsmith as ls
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from customer_agent import build_customer_graph, stream_turn
from customer_store import CustomerStore
from fga_store import FGAStore
from server import create_app

ACCOUNT = "account:northstar/AC-100"
NOTE = "document:northstar/100-commercial"


class ScriptedCustomerModel:
    """Script only model choices. All policy, tools, graph, and stores are real."""

    def __init__(self, *batches):
        self.batches = deque(batches)
        self.seen = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages, config=None):
        self.seen.append(messages)
        if isinstance(messages[-1], HumanMessage):
            batch = self.batches.popleft() if self.batches else []
            return AIMessage(content="" if batch else "Ready to help.", tool_calls=[
                {"name": name, "args": args, "id": uuid.uuid4().hex} for name, args in batch
            ])
        results = [message.content for message in messages if isinstance(message, ToolMessage)]
        return AIMessage(content="\n".join(results[-4:]))


def call(name, **args):
    return [(name, {"account_id": ACCOUNT, **args})]


class CustomerGraphTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.trace_context = ls.tracing_context(enabled=False)
        self.trace_context.__enter__()
        self.store = CustomerStore()
        self.config = {"configurable": {
            "thread_id": uuid.uuid4().hex, "turn_id": uuid.uuid4().hex,
            "tenant_id": "northstar", "user_id": "user:northstar/csm",
            "agent_id": "agent:northstar/customer-ops", "invocation_mode": "tool",
        }}

    def tearDown(self):
        self.trace_context.__exit__(None, None, None)

    async def run_turn(self, graph, text="Review the account", resume=None):
        input_data = Command(resume=resume) if resume else {"messages": [HumanMessage(content=text)]}
        return [event async for event in stream_turn(graph, input_data, self.config)]

    def response(self, events):
        return "\n".join(event["data"]["content"] for event in events if event["event"] == "agent_response")

    def proposal(self, events):
        return next(event["data"] for event in events if event["event"] == "approval_required")

    def review(self, proposal, decision="approve", reviewer="user:northstar/lead"):
        return {"approval_id": proposal["approval_id"], "decision": decision, "reviewer": reviewer}

    async def test_delegation_gathers_evidence_and_emits_real_lifecycle(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("assess_renewal_readiness")))
        events = await self.run_turn(graph)
        result = json.loads(self.response(events))
        self.assertEqual(result["readiness"], "needs_attention")
        self.assertEqual(result["days_to_renewal"], 30)
        self.assertEqual(len(result["risk_factors"]), 2)
        names = [event["event"] for event in events]
        started = names.index("agent_call_started")
        finished = names.index("agent_call_completed")
        child_checks = [i for i, event in enumerate(events) if event["event"] == "fga_check"
                        and event["data"]["agent_id"].endswith("renewal-analyst")]
        self.assertTrue(child_checks)
        self.assertTrue(all(started < i < finished for i in child_checks))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)

    async def test_user_denial_is_not_rescued_by_parent_or_child(self):
        self.config["configurable"]["user_id"] = "user:northstar/support"
        model = ScriptedCustomerModel(call("search_account_records", record_id=NOTE), call("save_account_brief", content="Forbidden write"))
        graph = build_customer_graph(self.store, model)
        events = await self.run_turn(graph)
        self.assertIn("Access denied", self.response(events))
        self.assertNotIn("6%", self.response(events))
        events = await self.run_turn(graph)
        self.assertFalse(any(e["event"] == "approval_required" for e in events))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)

    async def test_search_filters_before_matching_and_counting(self):
        self.config["configurable"]["user_id"] = "user:northstar/support"
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("search_account_records", query="concession")))
        events = await self.run_turn(graph)
        result = json.loads(self.response(events))
        self.assertEqual(result["records"], [])
        self.assertEqual(result["count"], 0)
        self.assertNotIn(NOTE, json.dumps(events))

    async def test_saved_brief_does_not_launder_restricted_details(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("save_account_brief", content="Procurement requested a 6% concession.")))
        proposal = self.proposal(await self.run_turn(graph))
        await self.run_turn(graph, resume=self.review(proposal))
        self.config["configurable"].update(thread_id=uuid.uuid4().hex, user_id="user:northstar/support")
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("get_customer_account"), call("assess_renewal_readiness")))
        result = json.loads(self.response(await self.run_turn(graph)))
        self.assertNotIn("saved_brief", result)
        self.assertNotIn("6%", json.dumps(result))
        assessment = json.loads(self.response(await self.run_turn(graph)).split("\n")[-1])
        self.assertNotIn(NOTE, assessment["source_ids"])

    async def test_cross_tenant_id_is_denied_without_record_data(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("get_customer_account", account_id="account:beacon/AC-100")))
        events = await self.run_turn(graph)
        self.assertIn("Access denied", self.response(events))
        self.assertNotIn("Juniper", self.response(events))

    async def test_short_reference_is_tenant_scoped(self):
        self.config["configurable"].update(tenant_id="beacon", user_id="user:beacon/csm", agent_id="agent:beacon/customer-ops")
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("get_customer_account", account_id="AC-100")))
        result = json.loads(self.response(await self.run_turn(graph)))
        self.assertEqual(result["name"], "Juniper Manufacturing")

    async def test_tool_and_handoff_enforce_same_delegation_policy(self):
        for profile in ("customer-ops", "support"):
            for mode in ("tool", "handoff"):
                self.config["configurable"].update(thread_id=uuid.uuid4().hex, invocation_mode=mode,
                                                  handoff_account=ACCOUNT, agent_id=f"agent:northstar/{profile}")
                graph = build_customer_graph(self.store, ScriptedCustomerModel(call("assess_renewal_readiness")))
                events = await self.run_turn(graph)
                names = [event["event"] for event in events]
                if profile == "support":
                    self.assertIn("agent_call_denied", names)
                    self.assertNotIn("agent_call_started", names)
                else:
                    self.assertIn("agent_call_completed", names)

    async def test_approved_write_persists_and_followup_retains_history(self):
        model = ScriptedCustomerModel(call("save_account_brief", content="Resolve SSO failures before renewal."), call("get_customer_account"))
        graph = build_customer_graph(self.store, model)
        proposal = self.proposal(await self.run_turn(graph, "Save a brief"))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)
        events = await self.run_turn(graph, resume=self.review(proposal))
        self.assertIn('"status": "saved"', self.response(events))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 2)
        events = await self.run_turn(graph, "Show what we saved")
        self.assertIn("Resolve SSO failures", self.response(events))
        self.assertGreater(sum(isinstance(m, HumanMessage) for m in model.seen[-1]), 1)

    async def test_rejection_and_conditional_leave_brief_unchanged(self):
        for decision in ("deny", "conditional"):
            self.config["configurable"]["thread_id"] = uuid.uuid4().hex
            graph = build_customer_graph(self.store, ScriptedCustomerModel(call("save_account_brief", content="Not approved")))
            proposal = self.proposal(await self.run_turn(graph))
            await self.run_turn(graph, resume=self.review(proposal, decision))
            self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)

    async def test_reviewer_scope_and_changed_version_reject_approval(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("save_account_brief", content="New brief")))
        proposal = self.proposal(await self.run_turn(graph))
        events = await self.run_turn(graph, resume=self.review(proposal, reviewer="user:beacon/lead"))
        self.assertIn("Access denied", self.response(events))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)
        self.config["configurable"]["thread_id"] = uuid.uuid4().hex
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("save_account_brief", content="Stale proposal")))
        proposal = self.proposal(await self.run_turn(graph))
        self.store.briefs[ACCOUNT]["version"] = 2
        events = await self.run_turn(graph, resume=self.review(proposal))
        self.assertIn("stale_approval", self.response(events))
        self.assertNotEqual(self.store.briefs[ACCOUNT]["content"], "Stale proposal")

    async def test_revoke_user_before_resume_blocks_write(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("save_account_brief", content="Revoked")))
        proposal = self.proposal(await self.run_turn(graph))
        self.store.fga.delete_tuple("user:northstar/csm", "member", "team:northstar/success")
        events = await self.run_turn(graph, resume=self.review(proposal))
        self.assertIn("Access denied", self.response(events))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 1)

    async def test_two_write_calls_do_not_duplicate_first_on_replay(self):
        batch = call("save_account_brief", content="First approved brief") + call("archive_account_brief")
        self.config["configurable"]["user_id"] = "user:northstar/lead"
        graph = build_customer_graph(self.store, ScriptedCustomerModel(batch))
        first = self.proposal(await self.run_turn(graph))
        second = self.proposal(await self.run_turn(graph, resume=self.review(first)))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 2)
        await self.run_turn(graph, resume=self.review(second))
        self.assertEqual(self.store.briefs[ACCOUNT]["version"], 3)
        self.assertEqual(self.store.briefs[ACCOUNT]["status"], "archived")
        self.assertEqual(len(self.store.briefs[ACCOUNT]["history"]), 2)

    async def test_multiple_tools_have_distinct_audit_call_ids(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("get_customer_account") + call("search_account_records")))
        events = await self.run_turn(graph)
        ids = {event["data"]["tool_name"]: event["data"]["call_id"] for event in events if event["event"] == "tool_completed"}
        self.assertEqual(len(set(ids.values())), 2)

    async def test_bad_arguments_are_failure_not_policy_denial(self):
        graph = build_customer_graph(self.store, ScriptedCustomerModel(call("get_customer_account", user_id="user:beacon/lead")))
        events = await self.run_turn(graph)
        self.assertIn("invalid_request", self.response(events))
        # Request/model tenant checks still run; malformed tool arguments never
        # reach the tool's own policy transaction or generate a policy denial.
        self.assertFalse(any(e["event"] == "fga_decision" and
                             e["data"]["tool_name"] == "get_customer_account" for e in events))

    async def test_healthy_and_missing_evidence_assessments(self):
        for ref, expected in (("AC-101", "ready"), ("AC-102", "insufficient_information")):
            self.config["configurable"]["thread_id"] = uuid.uuid4().hex
            graph = build_customer_graph(self.store, ScriptedCustomerModel(call("assess_renewal_readiness", account_id=ref)))
            result = json.loads(self.response(await self.run_turn(graph)))
            self.assertEqual(result["readiness"], expected)


class ServerBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.trace_context = ls.tracing_context(enabled=False)
        self.trace_context.__enter__()
        self.app = create_app(model=ScriptedCustomerModel(call("get_customer_account")))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")
        self.other = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")

    async def asyncTearDown(self):
        await asyncio.gather(*self.app.state.runtime.tasks, return_exceptions=True)
        await self.client.aclose()
        await self.other.aclose()
        self.trace_context.__exit__(None, None, None)

    async def test_session_bound_turn_stream_and_followup(self):
        catalog = await self.client.get("/api/catalog")
        self.assertIn("HttpOnly", catalog.headers["set-cookie"])
        started = (await self.client.post("/api/run", json={"user_msg": "Review Meridian Retail"})).json()
        stream = await self.client.get("/api/stream/" + started["run_id"])
        self.assertIn("Meridian Retail", stream.text)
        self.assertIn("event: done", stream.text)
        await self.other.get("/api/catalog")
        forbidden = await self.other.get("/api/stream/" + started["run_id"])
        self.assertEqual(forbidden.status_code, 404)
        forbidden = await self.other.post("/api/run", json={"user_msg": "Continue", "conversation_id": started["conversation_id"]})
        self.assertEqual(forbidden.status_code, 404)
        switched = await self.client.post("/api/run", json={"user_msg": "Continue", "conversation_id": started["conversation_id"], "user_id": "user:northstar/lead"})
        self.assertEqual(switched.status_code, 409)
        continued = await self.client.post("/api/run", json={"user_msg": "Continue", "conversation_id": started["conversation_id"]})
        self.assertEqual(continued.status_code, 200)
        await self.client.get("/api/stream/" + continued.json()["run_id"])

    async def test_request_validation_and_origin(self):
        await self.client.get("/api/catalog")
        for payload in ({"user_msg": "x", "invocation_mode": "arbitrary"}, {"user_msg": "x", "invocation_mode": []}, {"user_msg": "x" * 4001}):
            self.assertEqual((await self.client.post("/api/run", json=payload)).status_code, 400)
        response = await self.client.post("/api/run", json={"user_msg": "x"}, headers={"Origin": "https://other.example"})
        self.assertEqual(response.status_code, 403)

    async def test_approval_ownership_reviewer_and_replay(self):
        runtime = self.app.state.runtime
        runtime.graph = build_customer_graph(runtime.store, ScriptedCustomerModel(call("save_account_brief", content="Approved content")))
        await self.client.get("/api/catalog")
        ids = (await self.client.post("/api/run", json={"user_msg": "Save a brief"})).json()
        await self.client.get("/api/stream/" + ids["run_id"])
        pending = runtime.conversations[ids["conversation_id"]].pending
        body = {"approval_id": pending["approval_id"], "reviewer": "user:northstar/lead", "decision": "approve"}
        route = "/api/resume/" + ids["conversation_id"]
        await self.other.get("/api/catalog")
        self.assertEqual((await self.other.post(route, json=body)).status_code, 404)
        self.assertEqual((await self.client.post(route, json={**body, "reviewer": "user:beacon/lead"})).status_code, 403)
        approved = await self.client.post(route, json=body)
        self.assertEqual(approved.status_code, 200)
        await self.client.get("/api/stream/" + approved.json()["run_id"])
        self.assertEqual((await self.client.post(route, json=body)).status_code, 409)
        self.assertEqual(runtime.store.briefs[ACCOUNT]["version"], 2)


class FGAUsersetTests(unittest.TestCase):
    def test_cycles_fail_closed_and_valid_grant_has_correct_path(self):
        store = FGAStore()
        store.write_tuple("team:a#member", "member", "team:b")
        store.write_tuple("team:b#member", "member", "team:a")
        store.write_tuple("team:a#member", "reader", "account:x")
        self.assertFalse(store.check("user:test", "reader", "account:x").allowed)
        store.write_tuple("user:test", "member", "team:b")
        result = store.check("user:test", "reader", "account:x")
        self.assertTrue(result.allowed)
        self.assertEqual(result.grant_path[-1].object, "account:x")


if __name__ == "__main__":
    unittest.main()
