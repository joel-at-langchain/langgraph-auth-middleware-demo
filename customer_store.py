"""Fictional tenant data and policy-enforced customer operations (in memory)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Callable

from fga_store import FGAStore
from customer_auth import CustomerAuthorizationMiddleware

REFERENCE_DATE = date(2026, 9, 28)
TOOLS = (
    "get_customer_account", "search_account_records", "save_account_brief",
    "archive_account_brief", "assess_renewal_readiness",
)
TENANTS = (
    ("northstar", "Northstar Cloud", "Managed infrastructure",
     ("Maya Chen", "Jordan Ellis", "Priya Shah"),
     ("Meridian Retail", "Westhaven Energy", "Orchard Travel")),
    ("beacon", "Beacon Data", "Analytics platform",
     ("Elena Ruiz", "Marcus Reed", "Dev Patel"),
     ("Juniper Manufacturing", "Cobalt Foods", "Pinecrest Media")),
    ("summit", "Summit Identity", "Workforce identity",
     ("Avery Brooks", "Sam Okafor", "Noor Hassan"),
     ("Alder Services", "Granite Insurance", "Willow Studios")),
)


@dataclass(frozen=True)
class Actor:
    tenant_id: str
    user_id: str
    agent_id: str
    parent_agent_id: str | None = None


class AccessDenied(Exception):
    def __init__(self, code="resource_unavailable"):
        self.code = code
        super().__init__("Access denied: the requested resource or operation is unavailable in this context.")


class StaleApproval(Exception):
    pass


class CustomerStore:
    """Reset by constructing a fresh store. All IDs are tenant-qualified."""

    def __init__(self):
        self.fga = FGAStore()
        self.tenants = {}
        self.users = {}
        self.accounts = {}
        self.records = {}
        self.briefs = {}
        # Write replay protection; keys are conversation ID + model tool-call ID.
        self.completed_writes = {}
        self._seed()
        self.authorization = CustomerAuthorizationMiddleware(self)

    def _seed(self):
        fga = self.fga
        for tenant, name, product, people, customers in TENANTS:
            self.tenants[tenant] = {"id": tenant, "name": name, "product": product}
            tenant_object = f"tenant:{tenant}"
            cs_team, support_team = f"team:{tenant}/success", f"team:{tenant}/support"
            persona_ids = []
            for role, person in zip(("csm", "support", "lead"), people):
                uid = f"user:{tenant}/{role}"
                persona_ids.append(uid)
                self.users[uid] = {"id": uid, "name": person, "role": role, "tenant_id": tenant}
                fga.write_tuple(uid, "member", tenant_object)
                fga.write_tuple(uid, "member", support_team if role == "support" else cs_team)
            csm, support, lead = persona_ids
            ops, support_agent, specialist = (
                f"agent:{tenant}/{profile}" for profile in ("customer-ops", "support", "renewal-analyst")
            )
            for agent in (ops, support_agent, specialist):
                fga.write_tuple(agent, "member", tenant_object)
                capabilities = TOOLS if agent == ops else TOOLS[:2]
                if agent == support_agent:
                    capabilities = (*TOOLS[:2], "assess_renewal_readiness")
                for capability in capabilities:
                    fga.write_tuple(agent, "executor", f"tool:{tenant}/{capability}")
            fga.write_tuple(ops, "delegate", specialist)
            for index, customer in enumerate(customers):
                aid = f"account:{tenant}/AC-{index + 100}"
                status = ("attention", "healthy", "incomplete")[index]
                self.accounts[aid] = {
                    "id": aid, "reference": f"AC-{index + 100}", "tenant_id": tenant,
                    "name": customer, "owner": people[0], "service": product,
                    "renewal_date": ("2026-10-28", "2026-11-27", "2026-10-20")[index],
                    "success_plan_confirmed": index != 2,
                    "service_tier": ("Enterprise", "Business", "Enterprise")[index],
                }
                fga.write_tuple(f"{cs_team}#member", "writer", aid)
                fga.write_tuple(lead, "admin", aid)
                fga.write_tuple(lead, "reviewer", aid)
                if index == 0:
                    fga.write_tuple(f"{support_team}#member", "reader", aid)
                fga.write_tuple(ops, "admin", aid)
                for agent in (support_agent, specialist):
                    fga.write_tuple(agent, "reader", aid)
                cases = (
                    ("Intermittent SSO login failures", "critical", "open",
                     "SSO login failures affect the rollout. The workaround is temporary; engineering has not confirmed a fix date."),
                    ("Regional failover validation", "medium", "open",
                     "The customer needs a completed failover test before the renewal meeting."),
                ) if index == 0 else (
                    ("Usage export verification", "low", "closed", "The customer verified the corrected usage export."),
                    ("Quarterly service review", "low", "closed", "Service targets were met and the review is complete."),
                )
                for number, (title, severity, case_status, content) in enumerate(cases, 1):
                    rid = f"case:{tenant}/{index + 100}-{number}"
                    self.records[rid] = {
                        "id": rid, "tenant_id": tenant, "account_id": aid, "kind": "case",
                        "title": title, "severity": severity, "status": case_status,
                        "content": content, "owner": people[1], "restricted": False,
                    }
                notes = (
                    ("Technical success plan", False,
                     "Success criteria and service owners are confirmed. Review current cases before the renewal meeting."
                     if status != "incomplete" else
                     "A current success plan and customer sign-off have not been recorded. Confirm the decision maker before assessing readiness."),
                    ("Restricted commercial note", True,
                     "Procurement requested a 6% concession. Any revised terms require commercial approval; no commitment has been made."),
                )
                for suffix, (title, restricted, content) in zip(("technical", "commercial"), notes):
                    rid = f"document:{tenant}/{index + 100}-{suffix}"
                    self.records[rid] = {
                        "id": rid, "tenant_id": tenant, "account_id": aid, "kind": "document",
                        "title": title, "content": content, "restricted": restricted,
                    }
                    if restricted:
                        fga.write_tuple(f"{cs_team}#member", "reader", rid)
                        fga.write_tuple(ops, "reader", rid)
                        fga.write_tuple(specialist, "reader", rid)
                self.briefs[aid] = {
                    "id": f"brief:{tenant}/{index + 100}", "account_id": aid,
                    "version": 1, "status": "active", "content": f"Prior account brief for {customer}: refresh before the next meeting.",
                    "history": [],
                }
                # Derived briefs can contain commercial evidence. Account-reader
                # access alone must not expose that information through a summary.
                for subject in (f"{cs_team}#member", ops, specialist):
                    fga.write_tuple(subject, "reader", self.briefs[aid]["id"])

    def actor(self, tenant_id, user_id, profile="customer-ops") -> Actor:
        if tenant_id not in self.tenants or profile not in {"customer-ops", "support"}:
            raise AccessDenied("invalid_context")
        if self.users.get(user_id, {}).get("tenant_id") != tenant_id:
            raise AccessDenied("invalid_context")
        return Actor(tenant_id, user_id, f"agent:{tenant_id}/{profile}")

    def resolve_account(self, actor: Actor, reference: str) -> dict:
        text = reference.strip().casefold()
        for account in self.accounts.values():
            if account["tenant_id"] == actor.tenant_id and text in {
                account["id"].casefold(), account["name"].casefold(), account["reference"].casefold(),
            }:
                return account
        raise AccessDenied()

    def check(self, actor: Actor, resource: str, relation: str, tool: str, emit: Callable | None = None):
        checks = [
            (actor.user_id, "member", f"tenant:{actor.tenant_id}"),
            (actor.agent_id, "member", f"tenant:{actor.tenant_id}"),
            (actor.user_id, relation, resource),
            (actor.agent_id, relation, resource),
            (actor.agent_id, "executor", f"tool:{actor.tenant_id}/{tool}"),
        ]
        decisions = []
        for subject, requested, obj in checks:
            result = self.fga.check(subject, requested, obj)
            decisions.append({
                "subject": subject, "relation": requested, "object": obj,
                "allowed": result.allowed,
                "grant_path": [asdict(grant) for grant in result.grant_path],
            })
        allowed = all(item["allowed"] for item in decisions)
        if emit:
            emit("fga_check", resource=resource, relation=relation, checks=decisions)
            emit("fga_decision", resource=resource, relation=relation,
                 decision="allow" if allowed else "deny", reason_code="granted" if allowed else "missing_grant")
        if not allowed:
            raise AccessDenied("missing_grant")

    def account(self, actor, reference, relation, tool, emit=None):
        try:
            account = self.resolve_account(actor, reference)
        except AccessDenied:
            if emit:
                emit("fga_decision", resource=reference, decision="deny", reason_code="resource_unavailable")
            raise
        self.check(actor, account["id"], relation, tool, emit)
        return account

    def get_account(self, actor, reference, emit=None, *, config=None):
        scope = self.authorization.authorize(actor, "get_customer_account", reference, emit=emit, config=config)
        account = self.accounts[scope["account_id"]]
        result = {**deepcopy(account), "demo_reference_date": str(REFERENCE_DATE)}
        if "brief_id" in scope:
            brief = self.briefs[account["id"]]
            result["saved_brief"] = {key: deepcopy(value) for key, value in brief.items() if key != "history"}
        return result

    def search(self, actor, reference, query="", record_id=None, emit=None, *, config=None):
        scope = self.authorization.authorize(actor, "search_account_records", reference,
                                             record_id=record_id, emit=emit, config=config)
        matches = []
        for record_id in scope["record_ids"]:
            record = self.records[record_id]
            if query and not any(word in (record["title"] + " " + record["content"]).casefold()
                                 for word in query.casefold().split()):
                continue
            matches.append(deepcopy(record))
        return {"account_id": scope["account_id"], "records": matches, "count": len(matches)}

    def delegate(self, actor, reference, emit, *, config=None):
        scope = self.authorization.authorize(actor, "assess_renewal_readiness", reference, emit=emit, config=config)
        return self.accounts[scope["account_id"]], replace(
            actor, agent_id=scope["child_agent_id"], parent_agent_id=actor.agent_id,
        )

    def reviewers(self, actor, account_id):
        return [deepcopy(user) for user in self.users.values()
                if user["tenant_id"] == actor.tenant_id
                and self.fga.check(user["id"], "reviewer", account_id).allowed]

    def catalog(self):
        return {
            "reference_date": str(REFERENCE_DATE),
            "tenants": [dict(tenant, users=[deepcopy(user) for user in self.users.values() if user["tenant_id"] == tenant["id"]])
                        for tenant in self.tenants.values()],
            "profiles": [{"id": "customer-ops", "name": "Customer Operations Assistant"},
                         {"id": "support", "name": "Support Assistant · read only"}],
            "counts": {"tenants": len(self.tenants), "users": len(self.users), "teams": 6,
                       "accounts": len(self.accounts), "records": len(self.records),
                       "relationship_tuples": len(self.fga.list_tuples())},
        }

    def visible_accounts(self, actor):
        return [{"id": account["id"], "name": account["name"]} for account in self.accounts.values()
                if account["tenant_id"] == actor.tenant_id
                and self.fga.check(actor.user_id, "reader", account["id"]).allowed
                and self.fga.check(actor.agent_id, "reader", account["id"]).allowed]
