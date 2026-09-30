# Native subagents

Add three isolated, model-driven specialists through the installed Deep Agents
`SubAgentMiddleware`. Keep the existing renewal/SQL tool-wrapped graphs.

| Specialist | Allowed work | Write boundary |
|---|---|---|
| billing-review | Billing snapshots and usage-export readiness | Read-only |
| support-escalation | Cases, account facts, SLA evidence | Recommendations only |
| renewal-planning | Forecasts, CRM status, account briefs | Existing lead approval |

## Enforcement and observability

1. Main `/api/run` selects `task(description, subagent_type)`.
2. Governed task checks tenant, user and parent delegation, executor and account grants.
3. Native middleware runs a separate model/tool loop with only its explicit toolset.
4. Each model/tool boundary rechecks delegation. Each tool intersects user, parent
   and child access and rejects changes to the delegated account.
5. Writes use the existing approval inbox, checkpoint resume and version-bound
   proposal. Denial/hold makes no change; waiting is not a tool failure.

Spans identify the selected specialist (`tools.task.<specialist>`) and native
agent loop. Leaf rejection tags remain backward compatible. Owning task/agent
spans receive aggregate rejection tags without inflating leaf failure counts.

## Acceptance checks

- Authorized calls return evidence and preserve streamed parent responses.
- Missing user/parent delegation blocks the child before model execution.
- Missing child/parent resource grants block the leaf tool; no scope escalation.
- Revocations apply before each model/tool call and after an approval pause.
- Unknown agents, extra identity arguments and ambiguous accounts are rejected.
- Mock service failures retain execution tags; human denials retain approval tags.
- Approval and conditional hold never produce false rejection tags.
- Twenty live examples cover all three tenants and all four outcome classes.
  HITL examples produce extra review/resume roots; report examples and traces separately.

No new dependencies, real external service mutations, shell tools, or unrestricted
filesystem tools. Existing dirty changes and earlier trace batches are preserved.
