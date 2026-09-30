"""Validate the 100-root plan and HITL cases through the normal HTTP entry point."""

import asyncio
from collections import Counter
import unittest

import httpx
import langsmith as ls

from scripts.mixed_samples import approval_cases, approval_prompt, bound_inbox_item, bound_proposal, check_resolution, service_cases
from demo.server import create_app
from tests.test_customer_operations import ScriptedCustomerModel


class MatrixTests(unittest.TestCase):
    def test_exact_count_and_outcomes(self):
        services, approvals = service_cases(), approval_cases()
        self.assertEqual(len(services) + 3 * len(approvals), 100)
        self.assertEqual(len({c["case_id"] for c in services + approvals}), 80)
        self.assertEqual(Counter(c["category"] for c in services), {None: 35, "execution": 21, "authorization": 14})
        self.assertEqual(Counter(c["decision"] for c in approvals), {"approve": 5, "deny": 3, "conditional": 2})
        self.assertEqual(len({c["tenant_id"] for c in approvals}), 3)


class HITLHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_workflows_use_real_interrupts_and_bound_reviews(self):
        with ls.tracing_context(enabled=False):
            for case in approval_cases():
                with self.subTest(case=case["case_id"]):
                    batch_id = "offline-mixed"
                    content = f"Fictional demo review [{batch_id}/{case['case_id']}]. Confirm meeting agenda."
                    model = ScriptedCustomerModel([("save_account_brief", {
                        "account_id": case["account_id"], "content": content,
                    })])
                    app = create_app(model=model)
                    runtime = app.state.runtime
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                        await client.get("/api/catalog")
                        response = await client.post("/api/run", json={
                            "tenant_id": case["tenant_id"], "user_id": case["user_id"],
                            "user_msg": approval_prompt(case, batch_id),
                            "sample_batch_id": batch_id, "sample_case_id": case["case_id"],
                        })
                        self.assertEqual(response.status_code, 200)
                        ids = response.json()
                        await asyncio.gather(*tuple(runtime.tasks))
                        events = runtime.runs[ids["run_id"]].events
                        proposal = bound_proposal(case, ids, events, batch_id)
                        self.assertEqual(runtime.store.briefs[case["account_id"]]["version"], 1)
                        for field in ("tenant_id", "user_id", "account_id"):
                            with self.assertRaises(ValueError):
                                bound_proposal({**case, field: "foreign"}, ids, events, batch_id)
                        # A model-copied timestamp is not the security boundary; server IDs are.
                        self.assertEqual(bound_proposal(case, ids, events, "another-timestamp"), proposal)
                        with self.assertRaises(ValueError):
                            bound_proposal(case, {**ids, "conversation_id": "foreign"}, events, batch_id)
                        inbox = (await client.get("/api/approvals", params={
                            "tenant_id": case["tenant_id"], "user_id": case["reviewer"],
                        })).json()["items"]
                        self.assertEqual(bound_inbox_item(case, ids, inbox)["id"], proposal["approval_id"])
                        with self.assertRaises(ValueError):
                            bound_inbox_item(case, {**ids, "run_id": "foreign"}, inbox)
                        with self.assertRaises(ValueError):
                            bound_inbox_item(case, {**ids, "conversation_id": "foreign"}, inbox)
                        resumed = await client.post("/api/resume/" + ids["conversation_id"], json={
                            "approval_id": proposal["approval_id"], "reviewer": case["reviewer"],
                            "decision": case["decision"],
                        })
                        self.assertEqual(resumed.status_code, 200)
                        await asyncio.gather(*tuple(runtime.tasks))
                        resume_ids = resumed.json()
                        self.assertEqual(resume_ids["conversation_id"], ids["conversation_id"])
                        resolution_events = runtime.runs[resume_ids["run_id"]].events
                        status = (await client.get(f"/api/approvals/{proposal['approval_id']}/status")).json()
                        result = check_resolution(case, resolution_events, status["status"])
                        # Scripted offline model has no token stream; live verifier requires one.
                        self.assertTrue(all(v for k, v in result["checks"].items() if k != "streamed_response"), result)
                        brief = runtime.store.briefs[case["account_id"]]
                        self.assertEqual(brief["version"], 2 if case["decision"] == "approve" else 1)
                        if case["decision"] == "approve":
                            self.assertEqual(brief["content"], content)
                        self.assertEqual(len(runtime.runs), 2)


if __name__ == "__main__":
    unittest.main()
