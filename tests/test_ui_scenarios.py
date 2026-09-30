"""Tenant scenario catalog and policy-aligned expectations; no external services."""

import asyncio
import unittest

import httpx
import langsmith as ls

from demo.store import AccessDenied, CustomerStore, TENANTS
from demo.server import create_app, suggested_tasks
from tests.test_customer_operations import ScriptedCustomerModel, call


class ScenarioCatalogTests(unittest.TestCase):
    def setUp(self):
        self.store = CustomerStore()
        self.tracing = ls.tracing_context(enabled=False)
        self.tracing.__enter__()

    def tearDown(self):
        self.tracing.__exit__(None, None, None)

    def scenarios(self, tenant, role="csm", profile="customer-ops"):
        actor = self.store.actor(tenant, f"user:{tenant}/{role}", profile)
        return actor, {s["id"]: s for s in suggested_tasks(self.store, actor, self.store.visible_accounts(actor))}

    def test_each_tenant_has_self_contained_named_scenarios_and_its_own_lead(self):
        for tenant, _, _, people, accounts in TENANTS:
            with self.subTest(tenant=tenant):
                _, scenarios = self.scenarios(tenant)
                self.assertEqual(len(scenarios), 19)
                self.assertEqual(scenarios["analyst-101"]["account"], accounts[1])
                self.assertEqual(scenarios["analyst-102"]["account"], accounts[2])
                save = scenarios["reviewed-save"]
                self.assertIn(people[2], save["name"])
                self.assertIn(people[2], save["expected"])
                self.assertIn(accounts[0], save["prompt"])
                self.assertIn("Confirm owners and target dates", save["prompt"])
                self.assertNotIn("just discussed", save["prompt"])
                for item in scenarios.values():
                    self.assertTrue(all(item[k] for k in ("id", "name", "account", "prompt", "description", "expected")))
                    self.assertLessEqual(len(item["prompt"]), 4000)
                    for foreign in TENANTS:
                        if foreign[0] != tenant:
                            for name in foreign[4]:
                                self.assertNotIn(name, str(item))

    def test_support_sees_no_hidden_accounts_or_commercial_content_in_hints(self):
        for tenant, _, _, _, accounts in TENANTS:
            _, scenarios = self.scenarios(tenant, "support")
            self.assertNotIn("analyst-101", scenarios)
            self.assertNotIn("analyst-102", scenarios)
            for name in accounts[1:]:
                self.assertNotIn(name, str(scenarios))
            self.assertIn("denied", scenarios["reviewed-save"]["expected"])
            self.assertIn("denied", scenarios["commercial-note"]["expected"])
            self.assertNotIn("6%", str(scenarios))
            self.assertIn("withheld", scenarios["saved-brief"]["expected"])

    def test_expectations_match_seeded_policy_for_every_persona_and_profile(self):
        for tenant, *_ in TENANTS:
            for role in ("csm", "support", "lead"):
                for profile in ("customer-ops", "support"):
                    with self.subTest(tenant=tenant, role=role, profile=profile):
                        actor, scenarios = self.scenarios(tenant, role, profile)
                        for key, operation in (("reviewed-save", "save_account_brief"), ("archive-brief", "archive_account_brief"),
                                               ("analyst-100", "assess_renewal_readiness"), ("commercial-note", "search_account_records")):
                            kwargs = {"record_id": f"document:{tenant}/100-commercial"} if key == "commercial-note" else {}
                            try:
                                self.store.authorization.authorize(actor, operation, f"account:{tenant}/AC-100", **kwargs)
                                allowed = True
                            except AccessDenied:
                                allowed = False
                            self.assertEqual("denied" not in scenarios[key]["expected"], allowed)

    def test_no_visible_accounts_has_only_non_disclosing_isolation_scenario(self):
        actor = self.store.actor("northstar", "user:northstar/csm")
        scenarios = suggested_tasks(self.store, actor, [])
        self.assertEqual([s["id"] for s in scenarios], ["cross-tenant"])
        self.assertNotIn("Meridian", str(scenarios))


class ScenarioHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_prompt_can_start_a_real_approval_in_every_tenant(self):
        with ls.tracing_context(enabled=False):
            for tenant, _, _, people, accounts in TENANTS:
                account_id = f"account:{tenant}/AC-100"
                app = create_app(model=ScriptedCustomerModel(call("save_account_brief", account_id=account_id,
                                                                 content="Confirm owners and target dates with Support.")))
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                    context = {"tenant_id": tenant, "user_id": f"user:{tenant}/csm"}
                    catalog = (await client.get("/api/catalog", params=context)).json()
                    scenario = next(s for s in catalog["scenarios"] if s["id"] == "reviewed-save")
                    started = await client.post("/api/run", json={**context, "user_msg": scenario["prompt"]})
                    self.assertEqual(started.status_code, 200)
                    await asyncio.gather(*tuple(app.state.runtime.tasks))
                    inbox = await client.get("/api/approvals", params={"tenant_id": tenant, "user_id": f"user:{tenant}/lead"})
                    item, = inbox.json()["items"]
                    self.assertEqual(item["status"], "pending")
                    self.assertEqual(item["requester"], people[0])
                    self.assertEqual(item["account_name"], accounts[0])
                    self.assertEqual(app.state.runtime.store.briefs[account_id]["version"], 1)


if __name__ == "__main__":
    unittest.main()
