"""Fixed mock services, fresh authorization, and the balanced HTTP trace matrix."""

import asyncio
from collections import Counter
import json
import unittest

import httpx
import langsmith as ls

from customer_agent import SPECS
from customer_services import FAULTS, SERVICE_TOOLS, MockServiceFailure
from customer_store import AccessDenied, CustomerStore, TENANTS
from run_tool_trace_batch import cases
from server import RunRequest, create_app
from test_customer_operations import ScriptedCustomerModel


class MockServiceTests(unittest.TestCase):
    def setUp(self):
        self.tracing = ls.tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.store = CustomerStore()

    def test_all_services_are_real_named_tools_and_tenant_scoped(self):
        self.assertTrue(set(SERVICE_TOOLS) <= set(SPECS))
        for tenant, *_ in TENANTS:
            actor = self.store.actor(tenant, f"user:{tenant}/csm")
            for tool in SERVICE_TOOLS:
                result = self.store.services.read(actor, tool, "AC-101")
                self.assertEqual(result["account_id"], f"account:{tenant}/AC-101")
                self.assertTrue(result["mock_service"])
                self.assertTrue(all(source.startswith(f"service:{tenant}/") for source in result["source_ids"]))
            with self.assertRaises(AccessDenied):
                self.store.services.read(actor, "get_billing_snapshot", "account:unknown/AC-101")

    def test_faults_are_fixed_and_authorization_precedes_mock_upstream(self):
        actor = self.store.actor("northstar", "user:northstar/csm")
        for tool, reason in FAULTS.items():
            with self.subTest(tool=tool), self.assertRaises(MockServiceFailure) as error:
                self.store.services.read(actor, tool, "AC-100")
            self.assertEqual(error.exception.code, reason)
            self.store.fga.delete_tuple(actor.agent_id, "executor", "tool:northstar/" + tool)
            events = []
            with self.assertRaises(AccessDenied):
                self.store.services.read(actor, tool, "AC-100", emit=lambda name, **data: events.append(name))
            self.assertNotIn("service_call_started", events)

    def test_user_and_agent_service_readers_are_both_required(self):
        actor = self.store.actor("northstar", "user:northstar/csm")
        for subject in ("team:northstar/success#member", actor.agent_id):
            self.store.fga.delete_tuple(subject, "reader", "service:northstar/billing")
            with self.assertRaises(AccessDenied):
                self.store.services.read(actor, "get_billing_snapshot", "AC-101")
            self.store.fga.write_tuple(subject, "reader", "service:northstar/billing")
        support = self.store.actor("northstar", "user:northstar/support")
        for tool in ("get_billing_snapshot", "get_renewal_forecast"):
            with self.assertRaises(AccessDenied):
                self.store.services.read(support, tool, "AC-100")

    def test_batch_has_exactly_25_success_and_25_unsuccessful_across_five_names(self):
        matrix = cases()
        self.assertEqual(len(matrix), 50)
        self.assertEqual(len({case["case_id"] for case in matrix}), 50)
        self.assertEqual(sum(case["unsuccessful"] for case in matrix), 25)
        self.assertEqual(Counter(case["tool_name"] for case in matrix), dict.fromkeys(SERVICE_TOOLS, 10))
        self.assertEqual(Counter(case["category"] for case in matrix), {None: 25, "execution": 15, "authorization": 10})
        for case in matrix:
            RunRequest(user_msg=case["prompt"], tenant_id=case["tenant_id"], user_id=case["user_id"],
                       sample_batch_id="test-batch", sample_case_id=case["case_id"])


class ServiceHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_entire_batch_matrix_through_standard_http_main_agent_boundary(self):
        with ls.tracing_context(enabled=False):
            outcomes = []
            for case in cases():
                index = 100 if case["unsuccessful"] else 101
                model = ScriptedCustomerModel([(case["tool_name"], {"account_id": f"account:{case['tenant_id']}/AC-{index}"})])
                app = create_app(model=model)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                    await client.get("/api/catalog")
                    response = await client.post("/api/run", json={"user_msg": case["prompt"], "tenant_id": case["tenant_id"],
                        "user_id": case["user_id"], "sample_batch_id": "offline-test", "sample_case_id": case["case_id"]})
                    self.assertEqual(response.status_code, 200)
                    ids = response.json()
                    await asyncio.gather(*tuple(app.state.runtime.tasks))
                    run = app.state.runtime.runs[ids["run_id"]]
                    self.assertEqual(run.sample_labels["sample_case_id"], case["case_id"])
                    names = [e["event"] for e in run.events]
                    failed = bool({"tool_denied", "tool_failed"} & set(names))
                    self.assertEqual(failed, case["unsuccessful"], case["case_id"])
                    self.assertIn("agent_response", names)
                    self.assertNotIn("error", names)
                    self.assertNotIn("approval_required", names)
                    self.assertNotIn("request_denied", names)
                    if not failed:
                        result = next(e["data"]["content"] for e in run.events if e["event"] == "agent_response")
                        self.assertTrue(json.loads(result)["mock_service"])
                    outcomes.append(failed)
            self.assertEqual(sum(outcomes), 25)


if __name__ == "__main__":
    unittest.main()
