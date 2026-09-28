"""Named authorization middleware shared by tools, specialists, and reviews.

Scopes are evaluated immediately before use, never cached in graph state. Only
authorized identifiers leave this boundary; customer content is not trace input.
"""

from dataclasses import asdict
import re

from langchain_core.runnables import RunnableConfig, RunnableLambda


class CustomerAuthorizationMiddleware:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def explicit_account_references(text):
        """Conservative fixture-name matching, not an NLP authorization boundary.

        The static demo vocabulary recognizes supplied names without looking up
        foreign records or returning their IDs. Unknown/implicit references still
        require tool authorization. A match never changes the trusted actor.
        """
        from customer_store import TENANTS

        matches = [(match.start(), match.group()) for match in re.finditer(
            r"(?<![\w:])account:[a-z0-9_-]+/[a-z0-9_-]+|(?<![\w:/-])AC-\d+(?![\w/-])",
            text, re.IGNORECASE,
        )]
        for tenant in TENANTS:
            for name in tenant[4]:
                pattern = r"(?<!\w)" + re.escape(name).replace(r"\ ", r"\s+") + r"(?!\w)"
                matches.extend((match.start(), " ".join(match.group().split()))
                               for match in re.finditer(pattern, text, re.IGNORECASE))
        references, seen = [], set()
        for _, reference in sorted(matches):
            if reference.casefold() not in seen:
                references.append(reference)
                seen.add(reference.casefold())
        return references

    @staticmethod
    def _trace_config(config, actor, name, **metadata):
        child = {
            **(config or {}), "run_name": name,
            "tags": [*(config or {}).get("tags", []), "authorization-middleware"],
            "metadata": {**(config or {}).get("metadata", {}), **asdict(actor), **metadata},
        }
        child.pop("run_id", None)
        return child

    def verify_tenant(self, actor, *, emit=None, config=None):
        from customer_store import AccessDenied

        def verify(_request):
            tenant = f"tenant:{actor.tenant_id}"
            if emit:
                emit("tenant_verification_started", resource=tenant)
            valid_context = (
                actor.tenant_id in self.store.tenants
                and self.store.users.get(actor.user_id, {}).get("tenant_id") == actor.tenant_id
                and actor.agent_id.startswith(f"agent:{actor.tenant_id}/")
            )
            checks = []
            for subject in (actor.user_id, actor.agent_id):
                result = self.store.fga.check(subject, "member", tenant)
                checks.append({"subject": subject, "relation": "member", "object": tenant,
                               "allowed": result.allowed,
                               "grant_path": [asdict(grant) for grant in result.grant_path]})
            allowed = valid_context and all(check["allowed"] for check in checks)
            result = {"decision": "allow" if allowed else "deny",
                      "reason_code": "granted" if allowed else "invalid_tenant_context",
                      "context_matches_tenant": valid_context, "checks": checks}
            if emit:
                emit("fga_check", resource=tenant, relation="member", checks=checks)
                emit("fga_decision", resource=tenant, relation="member",
                     decision=result["decision"], reason_code=result["reason_code"])
                emit("tenant_verification_completed", resource=tenant,
                     decision=result["decision"], reason_code=result["reason_code"])
            return result

        result = RunnableLambda(verify, name="authorization.verify_tenant").invoke(
            {"actor": asdict(actor)},
            config=self._trace_config(config, actor, "authorization.verify_tenant"),
        )
        if result["decision"] != "allow":
            raise AccessDenied(result["reason_code"])
        return result

    def authorize_request(self, actor, references, *, emit=None, config=None):
        def evaluate(_config):
            # Visibility preflight only; actual tool permissions remain mandatory.
            for reference in references:
                self.store.account(actor, reference, "reader", "get_customer_account", emit)
            return {"checked_references": references}

        request = {"actor": asdict(actor), "operation": "authorize_request",
                   "references": references, "phase": "request"}
        return self._transaction(actor, request, evaluate, emit, config, scope="request")

    def authorize_review(self, actor, proposal, *, emit=None, config=None, phase="review"):
        """Reading a proposal and deciding it require the same reviewer scope."""
        from customer_store import AccessDenied

        def evaluate(_config):
            if proposal.get("tenant_id") != actor.tenant_id or proposal.get("operation") not in {
                "save_account_brief", "archive_account_brief",
            }:
                raise AccessDenied()
            account = self.store.account(actor, proposal["account_id"], "reader", "get_customer_account", emit)
            brief_id = self.store.briefs[account["id"]]["id"]
            if proposal.get("resource") != brief_id:
                raise AccessDenied()
            self.store.check(actor, brief_id, "reader", "get_customer_account", emit)
            grant = self.store.fga.check(actor.user_id, "reviewer", account["id"])
            if emit:
                emit("fga_check", resource=account["id"], relation="reviewer", checks=[{
                    "subject": actor.user_id, "relation": "reviewer", "object": account["id"],
                    "allowed": grant.allowed, "grant_path": [asdict(item) for item in grant.grant_path],
                }])
                emit("fga_decision", resource=account["id"], relation="reviewer",
                     decision="allow" if grant.allowed else "deny",
                     reason_code="granted" if grant.allowed else "reviewer_not_authorized")
            if not grant.allowed:
                raise AccessDenied("reviewer_not_authorized")
            return {"account_id": account["id"], "brief_id": brief_id}

        request = {"actor": asdict(actor), "operation": "review_account_brief",
                   "resource": proposal.get("account_id", ""), "phase": phase,
                   "approval_id": proposal.get("approval_id", "")}
        return self._transaction(actor, request, evaluate, emit, config, scope="approval")

    def authorize(self, actor, operation, reference, *, record_id=None,
                  reviewer=None, phase="invoke", emit=None, config=None):
        request = {
            "actor": asdict(actor), "operation": operation, "resource": reference,
            "phase": phase,
        }
        if record_id is not None:
            request["record_id"] = record_id
        if reviewer is not None:
            request["reviewer"] = reviewer

        return self._transaction(actor, request,
                                 lambda _config: self._evaluate(actor, operation, reference, record_id, reviewer, phase, emit),
                                 emit, config, scope="tool")

    def _transaction(self, actor, request, evaluate, emit, config, *, scope):
        # Import here to keep the store's public Actor/AccessDenied API stable.
        from customer_store import AccessDenied

        resource = request.get("resource") or ", ".join(request.get("references", [])) or f"tenant:{actor.tenant_id}"
        phase = request["phase"]

        def authorize_transaction(_request, config: RunnableConfig):
            if emit:
                emit("authorization_started", resource=resource, phase=phase, auth_scope=scope)
            try:
                self.verify_tenant(actor, emit=emit, config=config)
                result = {"decision": "allow", "reason_code": "granted", **evaluate(config)}
            except AccessDenied as exc:
                result = {"decision": "deny", "reason_code": exc.code}
            if emit:
                emit("authorization_completed", resource=resource, phase=phase, auth_scope=scope,
                     decision=result["decision"], reason_code=result["reason_code"])
            return result

        result = RunnableLambda(authorize_transaction, name="authorization.authorize_transaction").invoke(
            request, config=self._trace_config(config, actor, "authorization.authorize_transaction",
                                              auth_operation=request["operation"], auth_phase=phase, auth_scope=scope),
        )
        if result["decision"] != "allow":
            raise AccessDenied(result["reason_code"])
        return result

    def _evaluate(self, actor, operation, reference, record_id, reviewer, phase, emit):
        from customer_store import AccessDenied, TOOLS

        if operation not in TOOLS or phase not in {"invoke", "resume"}:
            raise AccessDenied("invalid_operation")
        relation = ("admin" if operation == "archive_account_brief" else
                    "writer" if operation == "save_account_brief" else "reader")
        account = self.store.account(actor, reference, relation, operation, emit)
        scope = {"account_id": account["id"]}

        if phase == "resume":
            if operation not in {"save_account_brief", "archive_account_brief"}:
                raise AccessDenied("invalid_operation")
            if reviewer not in {user["id"] for user in self.store.reviewers(actor, account["id"])}:
                raise AccessDenied("reviewer_not_authorized")

        if operation == "get_customer_account":
            brief_id = self.store.briefs[account["id"]]["id"]
            try:
                # Omit inaccessible derived data without revealing its identifier.
                self.store.check(actor, brief_id, "reader", operation)
            except AccessDenied:
                return scope
            self.store.check(actor, brief_id, "reader", operation, emit)
            scope["brief_id"] = brief_id

        elif operation == "search_account_records":
            candidates = [record for record in self.store.records.values()
                          if record["account_id"] == account["id"]]
            if record_id:
                candidates = [record for record in candidates if record["id"] == record_id]
                if not candidates:
                    if emit:
                        emit("fga_decision", resource=record_id, decision="deny", reason_code="resource_unavailable")
                    raise AccessDenied()
            visible_ids = []
            for record in candidates:
                resource = record["id"] if record["restricted"] else account["id"]
                try:
                    self.store.check(actor, resource, "reader", operation, emit if record_id else None)
                except AccessDenied:
                    if record_id:
                        raise
                    continue
                visible_ids.append(record["id"])
            scope["record_ids"] = visible_ids

        elif operation == "assess_renewal_readiness":
            specialist = f"agent:{actor.tenant_id}/renewal-analyst"
            result = self.store.fga.check(actor.agent_id, "delegate", specialist)
            if emit:
                emit("fga_check", resource=specialist, relation="delegate", checks=[{
                    "subject": actor.agent_id, "relation": "delegate", "object": specialist,
                    "allowed": result.allowed, "grant_path": [asdict(grant) for grant in result.grant_path],
                }])
                emit("fga_decision", resource=specialist, relation="delegate",
                     decision="allow" if result.allowed else "deny",
                     reason_code="granted" if result.allowed else "delegation_not_granted")
            if not result.allowed:
                raise AccessDenied("delegation_not_granted")
            scope["child_agent_id"] = specialist

        return scope
