"""Inbox behavior through the HTTP/session boundary and real graph interrupts."""

import asyncio
import time
import unittest

import httpx
import langsmith as ls
from langchain_core.messages import HumanMessage

from demo.agent import build_customer_graph, DEMO_APPROVAL_NODE
from demo.server import create_app
from tests.test_customer_operations import ACCOUNT, ScriptedCustomerModel, call

LEAD = {"tenant_id": "northstar", "user_id": "user:northstar/lead"}


class ApprovalInboxTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tracing = ls.tracing_context(enabled=False)
        self.tracing.__enter__()
        self.model = ScriptedCustomerModel()
        self.app = create_app(model=self.model)
        self.runtime = self.app.state.runtime
        self.owner = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")
        self.lead = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")
        await self.owner.get("/api/catalog")
        await self.lead.get("/api/catalog")

    async def drain(self):
        while self.runtime.tasks:
            await asyncio.gather(*tuple(self.runtime.tasks))

    async def asyncTearDown(self):
        await self.drain()
        await self.owner.aclose()
        await self.lead.aclose()
        self.tracing.__exit__(None, None, None)

    async def populate(self, client=None, context=None):
        response = await (client or self.owner).post("/api/approvals/demo", json=context or LEAD)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def listing(self, client=None, context=None):
        response = await (client or self.lead).get("/api/approvals", params=context or LEAD)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def decide(self, item, decision="approve", client=None, context=None):
        return await (client or self.lead).post(f"/api/approvals/{item['id']}/decision", json={
            **(context or LEAD), "decision": decision, "comment": "Reviewed in the fictional demo.",
        })

    async def test_demo_populates_real_interrupts_without_writing_or_model_calls(self):
        result = await self.populate()
        self.assertEqual(result["created"], 3)
        self.assertEqual(result["inbox"]["pending_count"], 3)
        self.assertEqual({i["operation"] for i in result["inbox"]["items"]}, {"save_account_brief", "archive_account_brief"})
        self.assertEqual(self.model.seen, [])
        self.assertTrue(all(brief["version"] == 1 for brief in self.runtime.store.briefs.values()))
        for item in result["inbox"]["items"]:
            snapshot = await self.runtime.graph.aget_state({"configurable": {"thread_id": item["conversation_id"]}})
            self.assertEqual(snapshot.next, (DEMO_APPROVAL_NODE,))
            self.assertEqual(snapshot.tasks[0].interrupts[0].value["approval_id"], item["id"])
            self.assertEqual(item["status"], "pending")
            self.assertTrue(self.runtime.runs[item["request_run_id"]].done)

    async def test_repeated_and_concurrent_population_reuses_pending_scenarios(self):
        first, second = await asyncio.gather(self.populate(), self.populate(self.lead))
        self.assertEqual(first["created"] + second["created"], 3)
        self.assertEqual(first["reused"] + second["reused"], 3)
        self.assertEqual(len(self.runtime.conversations), 3)
        self.assertEqual(len(self.runtime.inbox.items), 3)

    async def test_cross_session_lead_review_does_not_share_chat_or_stream(self):
        await self.populate()
        listing = await self.listing()
        item = next(i for i in listing["items"] if i["account_id"] == ACCOUNT)
        self.assertEqual(item["requester_id"], "user:northstar/csm")
        self.assertNotIn("messages", item)
        self.assertEqual((await self.lead.get("/api/stream/" + item["request_run_id"])).status_code, 404)
        self.assertEqual((await self.lead.get(f"/api/approvals/{item['id']}/status")).status_code, 404)
        self.assertEqual((await self.decide(item)).status_code, 202)
        await self.drain()
        self.assertEqual(self.runtime.store.briefs[ACCOUNT]["version"], 2)
        saved = next(i for i in (await self.listing())["items"] if i["id"] == item["id"])
        self.assertEqual((saved["status"], saved["outcome"]), ("approved", "saved"))
        self.assertEqual(saved["reviewer_id"], LEAD["user_id"])
        self.assertEqual((await self.lead.get("/api/stream/" + saved["resolution_run_id"])).status_code, 404)
        status = await self.owner.get(f"/api/approvals/{item['id']}/status")
        self.assertEqual(status.json()["resolution_run_id"], saved["resolution_run_id"])

    async def test_nonreviewers_foreign_tenants_and_guessed_ids_are_denied(self):
        item = (await self.populate())["inbox"]["items"][0]
        csm = {**LEAD, "user_id": "user:northstar/csm"}
        self.assertEqual((await self.lead.get("/api/approvals", params=csm)).status_code, 403)
        self.assertEqual((await self.lead.post("/api/approvals/demo", json=csm)).status_code, 403)
        denied = await self.decide(item, context=csm)
        foreign = await self.decide(item, context={"tenant_id": "beacon", "user_id": "user:beacon/lead"})
        unknown = await self.decide({"id": "0" * 64})
        self.assertEqual([denied.status_code, foreign.status_code, unknown.status_code], [404, 404, 404])
        self.assertEqual(denied.json(), foreign.json())
        self.assertEqual(denied.json(), unknown.json())
        self.assertEqual((await self.listing(context={"tenant_id": "beacon", "user_id": "user:beacon/lead"}))["items"], [])

    async def test_reviewer_and_reader_grants_are_rechecked_on_list_and_action(self):
        item = next(i for i in (await self.populate())["inbox"]["items"] if i["account_id"] == ACCOUNT)
        self.runtime.store.fga.delete_tuple(LEAD["user_id"], "reviewer", ACCOUNT)
        self.assertNotIn(item["id"], {i["id"] for i in (await self.listing())["items"]})
        self.assertEqual((await self.decide(item)).status_code, 404)
        self.runtime.store.fga.write_tuple(LEAD["user_id"], "reviewer", ACCOUNT)
        self.runtime.store.fga.delete_tuple("agent:northstar/customer-ops", "reader", "brief:northstar/100")
        self.assertNotIn(item["id"], {i["id"] for i in (await self.listing())["items"]})
        self.assertEqual((await self.decide(item)).status_code, 404)
        self.runtime.store.fga.delete_tuple(LEAD["user_id"], "member", "tenant:northstar")
        self.assertEqual((await self.lead.get("/api/approvals", params=LEAD)).status_code, 403)

    async def test_two_reviewers_only_commit_once(self):
        item = next(i for i in (await self.populate())["inbox"]["items"] if i["account_id"] == ACCOUNT)
        results = await asyncio.gather(self.decide(item), self.decide(item, client=self.owner))
        self.assertEqual(sorted(r.status_code for r in results), [202, 409])
        await self.drain()
        self.assertEqual(self.runtime.store.briefs[ACCOUNT]["version"], 2)
        self.assertEqual(len(self.runtime.store.briefs[ACCOUNT]["history"]), 1)
        self.assertEqual((await self.decide(item)).status_code, 409)

    async def test_normal_chat_approval_is_indexed_and_inline_inbox_race_is_safe(self):
        self.runtime.graph = build_customer_graph(self.runtime.store, ScriptedCustomerModel(call("save_account_brief", content="Real chat proposal")))
        started = (await self.owner.post("/api/run", json={"user_msg": "Save a brief for Meridian Retail"})).json()
        await self.drain()
        item, = (await self.listing())["items"]
        self.assertEqual(item["content"], "Real chat proposal")
        self.assertFalse(item["demo"])
        results = await asyncio.gather(
            self.owner.post("/api/resume/" + started["conversation_id"], json={
                "approval_id": item["id"], "reviewer": LEAD["user_id"], "decision": "approve",
            }), self.decide(item),
        )
        self.assertEqual(sum(r.status_code in {200, 202} for r in results), 1)
        self.assertEqual(sum(r.status_code == 409 for r in results), 1)
        await self.drain()
        self.assertEqual(self.runtime.store.briefs[ACCOUNT]["version"], 2)

    async def test_reject_hold_and_archive_report_actual_outcomes(self):
        items = sorted((await self.populate())["inbox"]["items"], key=lambda item: item["account_id"])
        for item, decision in zip(items, ("deny", "conditional", "approve")):
            self.assertEqual((await self.decide(item, decision)).status_code, 202)
        await self.drain()
        status = {item["account_id"]: item["status"] for item in (await self.listing())["items"]}
        self.assertEqual(list(status[aid] for aid in sorted(status)), ["rejected", "held", "approved"])
        self.assertEqual([self.runtime.store.briefs[i["account_id"]]["version"] for i in items], [1, 1, 2])
        self.assertEqual(self.runtime.store.briefs[items[2]["account_id"]]["status"], "archived")

    async def test_stale_proposal_cannot_be_approved(self):
        item = next(i for i in (await self.populate())["inbox"]["items"] if i["account_id"] == ACCOUNT)
        self.runtime.store.briefs[ACCOUNT]["version"] = 2
        self.assertEqual((await self.decide(item)).status_code, 409)
        self.assertEqual(self.runtime.inbox.items[item["id"]].status, "stale")
        self.assertNotEqual(self.runtime.store.briefs[ACCOUNT]["content"], item["content"])

    async def test_revoked_requester_fails_resume_without_mutation(self):
        item = next(i for i in (await self.populate())["inbox"]["items"] if i["account_id"] == ACCOUNT)
        self.runtime.store.fga.delete_tuple("user:northstar/csm", "member", "team:northstar/success")
        self.assertEqual((await self.decide(item)).status_code, 202)
        await self.drain()
        self.assertEqual(self.runtime.inbox.items[item["id"]].status, "failed")
        self.assertEqual(self.runtime.store.briefs[ACCOUNT]["version"], 1)

    async def test_successful_write_is_not_mislabeled_if_followup_model_fails(self):
        class BrokenAfterWrite(ScriptedCustomerModel):
            async def ainvoke(self, messages, config=None):
                if not isinstance(messages[-1], HumanMessage):
                    raise RuntimeError("Synthetic model failure")
                return await super().ainvoke(messages, config)

        self.runtime.graph = build_customer_graph(self.runtime.store, BrokenAfterWrite(call("save_account_brief", content="Saved before model failure")))
        await self.owner.post("/api/run", json={"user_msg": "Save the account brief"})
        await self.drain()
        item, = (await self.listing())["items"]
        await self.decide(item)
        await self.drain()
        self.assertEqual(self.runtime.inbox.items[item["id"]].status, "approved")
        self.assertEqual(self.runtime.store.briefs[ACCOUNT]["version"], 2)

    async def test_expired_owner_and_cleanup_remove_proposals(self):
        item = (await self.populate())["inbox"]["items"][0]
        sid = self.runtime.conversations[item["conversation_id"]].session_id
        self.runtime.sessions[sid] = time.monotonic() - 7201
        self.assertEqual((await self.listing())["items"], [])
        self.assertEqual((await self.decide(item)).status_code, 404)
        await self.lead.get("/api/catalog")
        self.assertEqual(self.runtime.inbox.items, {})

    async def test_demo_limit_private_route_and_origin_validation(self):
        self.runtime.inbox.demo_creations["northstar"] = 15
        limited = await self.populate()
        self.assertTrue(limited["limit_reached"])
        self.assertEqual(limited["created"], 0)
        self.assertEqual((await self.lead.post("/api/approvals/demo", json={**LEAD, "operation": "arbitrary"})).status_code, 400)
        self.assertEqual((await self.lead.post("/api/approvals/demo", json=LEAD, headers={"Origin": "https://other.example"})).status_code, 403)
        self.assertEqual((await self.owner.post("/api/run", json={"user_msg": "x", "invocation_mode": "approval_demo"})).status_code, 400)


if __name__ == "__main__":
    unittest.main()
