"""Named authorization middleware shared by tools, specialists, and reviews.

Scopes are evaluated immediately before use, never cached in graph state. Only
authorized identifiers leave this boundary; customer content is not trace input.
"""

from dataclasses import asdict, replace
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
                  reviewer=None, phase="invoke", datasets=None, emit=None, config=None):
        request = {
            "actor": asdict(actor), "operation": operation, "resource": reference,
            "phase": phase,
        }
        if record_id is not None:
            request["record_id"] = record_id
        if reviewer is not None:
            request["reviewer"] = reviewer
        if datasets is not None:
            request["datasets"] = datasets

        return self._transaction(actor, request,
                                 lambda _config: self._evaluate(actor, operation, reference, record_id, reviewer, phase, emit, datasets),
                                 emit, config, scope="tool")

    def authorize_subagent(self, parent, subagent_type, reference, *, emit=None, config=None):
        from customer_store import AccessDenied
        from customer_subagents import SUBAGENTS

        def evaluate(child_config):
            if subagent_type not in SUBAGENTS or parent.parent_agent_id:
                raise AccessDenied("subagent_not_allowed")
            child_id = f"agent:{parent.tenant_id}/{subagent_type}"
            self._check_delegation(parent, child_id, emit)
            grant = self.store.fga.check(parent.user_id, "delegate", child_id)
            if emit:
                emit("fga_check", resource=child_id, relation="delegate", checks=[{
                    "subject": parent.user_id, "relation": "delegate", "object": child_id,
                    "allowed": grant.allowed, "grant_path": [asdict(t) for t in grant.grant_path],
                }])
                emit("fga_decision", resource=child_id, relation="delegate",
                     decision="allow" if grant.allowed else "deny",
                     reason_code="granted" if grant.allowed else "user_delegation_not_granted")
            if not grant.allowed:
                raise AccessDenied("user_delegation_not_granted")
            account = self.store.account(parent, reference, "reader", "task", emit)
            child = replace(parent, agent_id=child_id, parent_agent_id=parent.agent_id)
            self.verify_tenant(child, emit=emit, config=child_config)
            self.store.check(child, account["id"], "reader", "get_customer_account", emit)
            return {"account_id": account["id"], "child_agent_id": child_id}

        return self._transaction(parent, {"actor": asdict(parent), "operation": "task",
            "resource": reference, "phase": "delegate", "subagent_type": subagent_type},
            evaluate, emit, config, scope="subagent")

    def authorize_child_tool(self, child, operation, args, *, emit=None, config=None):
        from customer_store import AccessDenied
        from customer_subagents import SUBAGENTS

        context = config["configurable"]
        kind = context["subagent_type"]
        parent = replace(child, agent_id=child.parent_agent_id, parent_agent_id=None)
        scope = self.authorize_subagent(parent, kind, context["delegated_account_id"], emit=emit, config=config)
        if child.agent_id != scope["child_agent_id"] or operation not in SUBAGENTS[kind]["tools"]:
            raise AccessDenied("subagent_tool_not_allowed")
        reference = args["account_id"].strip()
        # Normalize only a shorthand for the exact server-authorized account.
        # This does not create a general alias or allow a different tenant/ID.
        if reference.casefold() == scope["account_id"].removeprefix("account:").casefold():
            reference = scope["account_id"]
        requested = self.store.resolve_account(child, reference)
        if requested["id"] != scope["account_id"]:
            raise AccessDenied("subagent_scope_mismatch")
        # The child cannot lend its capabilities to a less privileged parent.
        return self.authorize(parent, operation, requested["id"], record_id=args.get("record_id"),
                              emit=emit, config=config)

    def authorize_skills(self, actor, *, emit=None, config=None):
        from customer_skills import SKILLS
        from customer_store import AccessDenied

        def evaluate(_config):
            visible = []
            for key in SKILLS:
                resource = f"skill:{actor.tenant_id}/{key}"
                try:
                    self.store.check(actor, resource, "reader", "load_agent_skills", emit)
                except AccessDenied:
                    continue
                visible.append(key)
            return {"skills": visible}

        return self._transaction(actor, {"operation": "load_agent_skills", "phase": "discover",
                                        "resource": f"tenant:{actor.tenant_id}"},
                                 evaluate, emit, config, scope="skill")

    def authorize_skill_file(self, actor, file_path, *, operation, emit=None, config=None):
        from customer_skills import SKILL_FILES
        from customer_store import AccessDenied

        key = SKILL_FILES.get(file_path)
        resource = f"skill:{actor.tenant_id}/{key}" if key else file_path

        def evaluate(_config):
            if key is None or operation not in {"load_agent_skills", "read_file"}:
                raise AccessDenied()
            # Revoking the loader grant also blocks reads of previously known
            # paths; read_file requires an additional executor grant of its own.
            self.store.check(actor, resource, "reader", "load_agent_skills", emit)
            if operation == "read_file":
                self.store.check(actor, resource, "reader", operation, emit)
            return {"skill_id": key, "file_path": file_path}

        return self._transaction(actor, {"actor": asdict(actor), "operation": operation,
                                        "resource": resource, "phase": "read" if operation == "read_file" else "discover"},
                                 evaluate, emit, config, scope="skill")

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

    def _evaluate(self, actor, operation, reference, record_id, reviewer, phase, emit, datasets=None):
        from customer_store import AccessDenied, TOOLS
        from customer_analytics import ANALYTICS_TOOLS, SCHEMAS, dataset_id
        from customer_services import SERVICE_TOOLS

        if operation not in TOOLS or phase not in {"invoke", "resume"}:
            raise AccessDenied("invalid_operation")
        relation = ("admin" if operation == "archive_account_brief" else
                    "writer" if operation == "save_account_brief" else "reader")
        account = self.store.account(actor, reference, relation, operation, emit)
        scope = {"account_id": account["id"]}

        if operation in SERVICE_TOOLS:
            if phase != "invoke":
                raise AccessDenied("invalid_operation")
            resource = f"service:{actor.tenant_id}/{SERVICE_TOOLS[operation]['service']}"
            self.store.check(actor, resource, "reader", operation, emit)
            return {**scope, "service_id": resource}

        if operation in ANALYTICS_TOOLS:
            if phase != "invoke":
                raise AccessDenied("invalid_operation")
            # Schema discovery can filter. Explicit dataset requests must fail
            # closed rather than silently answering only part of a question.
            if datasets is None and operation != "get_analytics_schema":
                raise AccessDenied("invalid_operation")
            if datasets is not None and (not isinstance(datasets, list) or not 1 <= len(datasets) <= 3
                                         or any(not isinstance(name, str) or name not in SCHEMAS for name in datasets)):
                raise AccessDenied("invalid_operation")
            subjects = [actor]
            if actor.parent_agent_id:
                parent = replace(actor, agent_id=actor.parent_agent_id, parent_agent_id=None)
                self.store.account(parent, account["id"], "reader", operation, emit)
                self._check_delegation(parent, actor.agent_id, emit)
                subjects.append(parent)
            allowed = []
            for name in dict.fromkeys(datasets if datasets is not None else SCHEMAS):
                try:
                    for subject in subjects:
                        self.store.check(subject, dataset_id(account["id"], name), "reader", operation,
                                         emit if datasets is not None else None)
                except AccessDenied:
                    if datasets is not None:
                        raise
                    continue
                allowed.append(name)
            scope["datasets"] = allowed
            if operation == "analyze_customer_data":
                specialist = f"agent:{actor.tenant_id}/sql-analyst"
                self._check_delegation(actor, specialist, emit)
                scope["child_agent_id"] = specialist
            return scope

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
            self._check_delegation(actor, specialist, emit)
            scope["child_agent_id"] = specialist

        return scope

    def _check_delegation(self, actor, specialist, emit):
        from customer_store import AccessDenied

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
