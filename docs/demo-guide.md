# Customer Operations — Demo Guide

Open http://localhost:8000 and refresh an older open tab.

The demo has 3 fictional tenants, 9 users, 6 teams, 9 accounts, 18 support
cases, 18 account documents, 297 analytics rows, and 576 relationship tuples.
The analytics service executes real SQL in isolated in-memory SQLite databases;
business fixtures, permissions, and approvals remain in memory. Restarting the
server restores the seed data.
Dates are evaluated against **September 28, 2026**, so results remain repeatable.

## Scenario picker

The left sidebar lists scenarios for the selected tenant, using its account and
lead names. Hover over a scenario to see what it tests and its expected result;
the same description is associated with the button for screen readers. Clicking
only fills **Message the assistant** and opens the chat view. It does not submit
the request, change persona/profile, or approve anything.

The analytics scenarios cover SQL-generated adoption analysis, restricted
billing, direct incident queries, rejected SQL writes, and analytics-backed briefs
that require lead approval. **Default agent skills** lists the playbooks available
to the selected persona/profile; full instructions are read on demand.

Keep **Invocation controls → Chat · model chooses tools** selected. Expectations
describe the current persona/profile with seeded permissions and update when you
switch context. For example, **Request Priya Shah's approval** queues a real brief
proposal when acting as Maya with Customer Operations Assistant, but is denied
for Jordan or the read-only Support Assistant. The save prompt is self-contained;
no earlier conversation is required.

Other scenarios cover at-risk and healthy renewals, missing sign-off, restricted
commercial notes, filtered search, saved-brief verification, archive permissions,
and cross-tenant denial. Only visible accounts appear in the scenario picker.
Changing context clears the previous tenant's draft and scenarios before loading
the new ones.

## Analytics service and SQL analyst

The LangGraph agent exposes three analytics tools:

- `get_analytics_schema`: discovers visible tables and columns for one account.
- `query_customer_analytics`: runs supplied SQL through the read-only service.
- `analyze_customer_data`: delegates a natural-language question to a two-node
  specialist, which generates SQL and then executes it under fresh authorization.

There are three fictional tables, with the same schema across all tenants:

| Table | Seed data | Access |
| --- | --- | --- |
| `usage_daily` | 28 September snapshots per account; active/licensed seats, API requests, errors | Account readers with dataset grants |
| `invoices` | July–September invoices; integer USD cents, due date, payment status | Success team and Customer Operations/SQL Analyst only |
| `service_incidents` | Two account-specific incidents with status and observed impact minutes | Account readers with dataset grants |

Try these in **Northstar Cloud → Maya Chen → Customer Operations Assistant**:

1. “Ask the SQL analyst to compare Meridian Retail's average active seats and API
   error rates for September 1–14 versus 15–28, 2026. Show the SQL and sources.”
   Expected: average active seats fall from 820 to 590 and API error rates rise.
2. “Query Meridian Retail's overdue invoices and report the outstanding amount
   in dollars.” Expected: USD 24,000 overdue as of the fixed demo date.
3. “Which unresolved service incidents affected Meridian Retail, and how many
   impact minutes were recorded?” Expected: two incidents, 47 and 18 minutes.
4. “Use adoption and incident evidence to draft and save a brief for Meridian
   Retail.” Expected: authorized evidence, followed by a real approval pause for
   Priya. Nothing is saved before approval.

Switch to **Jordan Ellis**: incident/usage data for Meridian remains available,
but invoice access is denied, including through the more privileged SQL Analyst.
Switch to **Support Assistant**: direct operational SQL works, but SQL specialist
delegation is denied even when the user is Maya. Other tenants' cards use their
own account and lead names; healthy and incomplete accounts have different data.

For a direct query, ask the assistant to call `query_customer_analytics` for
Meridian Retail, dataset `usage_daily`, with:

```sql
SELECT usage_date, active_seats, licensed_seats,
       ROUND(error_requests * 100.0 / NULLIF(api_requests, 0), 2) AS error_pct
FROM usage_daily
ORDER BY usage_date
```

The account scope is enforced **before SQL execution**, not by adding a model-
generated `WHERE tenant_id` clause. Each connection contains only the selected
account's authorized, explicitly requested datasets. Hidden tables are not copied.
Both user and agent need account/dataset access, the agent needs tool execution
permission, and delegation requires a separate parent-to-child grant. Child
reads also recheck the parent permissions; generation does not freeze access.

The SQLite authorizer blocks writes, schema introspection, PRAGMA, ATTACH,
extensions, file functions, and recursive queries. One statement, an allowlist
of SQL functions, compile/VM limits, a 100-row cap, and a 32 KB result budget keep
execution bounded. Results include executed SQL, sources, date, and truncation.
Invalid/unsafe queries return a generic error, not raw SQLite details. This is a
local demo sandbox, not a production database security architecture. No SQL write
approval exists; writes continue to use the existing reviewed brief tools.

## Read-only mock services and a balanced trace batch

Five named tools are available through the ordinary main agent. These
are local functions with fictional fixtures, not network endpoints or MCP servers.
All calls check tenant membership, account access, tool execution permission, and
both user/agent reader grants on `service:<tenant>/<service>` before any result
or simulated upstream failure. They never modify accounts or contact real systems.

| Tool | Successful fixture | Unsuccessful fixture |
| --- | --- | --- |
| `get_billing_snapshot` | CSM, AC-101: billing balances | Support persona, AC-100: service reader denied |
| `get_support_sla_report` | CSM, AC-101: SLA metrics | CSM, AC-100: simulated timeout |
| `get_crm_sync_status` | CSM, AC-101: current sync status | CSM, AC-100: simulated unavailable service |
| `get_usage_export_status` | CSM, AC-101: ready export manifest | CSM, AC-100: simulated invalid upstream response |
| `get_renewal_forecast` | CSM, AC-101: precomputed fictional forecast | Support persona, AC-100: service reader denied |

These fixtures apply independently in all three tenants. Service lifecycle events
carry `mock_service=true`. Faults are fixed server-owned fixtures, not a prompt or
tool argument that enables arbitrary failure injection. Expected service failures
return a bounded error code to the assistant; unknown execution exceptions still
propagate to the existing run-error handler after their tool span is tagged.

For example, as Maya: “Call get_support_sla_report once for Meridian Retail and
report its actual result or error. Do not retry.” The same request for Westhaven
Energy succeeds. Billing/forecast failure cases use the Support **persona**, with
the Customer Operations **assistant profile**, to demonstrate independent user
and agent permissions.

Generate exactly 50 new traces through `/api/run` and `/api/stream`, using the real
model and standard main-agent routing:

`python -m scripts.tool_samples --output trace_batches/my-new-batch.jsonl`

The matrix repeats each tool's success and failure case five times, rotating
tenants: 25 successes, 15 simulated service failures, and 10 authorization denials.
Concurrency defaults to three. No HTTP POST/turn retries or replacement traces
are generated; deviations fail the batch instead of silently manufacturing a
50/50 result. Run IDs are recorded immediately after acceptance in an exclusively
created JSONL manifest. A model refusal without invoking the tool fails verification.

Verify existing traces, without generating any more:

`python -m scripts.tool_samples --verify-only trace_batches/my-new-batch.jsonl`

The verifier reads those 50 IDs from LangSmith and checks persisted root/tool tags,
categories, and counts, writing a separate `.verified.json` report. Filter by the
manifest's `sample_batch_id` metadata or `batch:<batch-id>` tag. `sample_case_id`
identifies each case. These bounded request labels are observability only and
never influence tool selection, authorization, or failure behavior.

## Default runtime skills

The main agent uses **Deep Agents `SkillsMiddleware`** (`deepagents==0.7.20`) for
three trusted, version-controlled playbooks: **Evidence-backed account briefing**,
**Scoped SQL analysis**, and **Human-reviewed account changes**. Their files live
under `demo/playbooks/`, with standard `name` and `description` YAML frontmatter.

The existing LangGraph explicitly invokes the middleware's public lifecycle hooks:

1. `abefore_agent` discovers authorized skills and parses their frontmatter.
2. `awrap_model_call` appends skill summaries and virtual paths to the system
   message. Full instructions are **not** eagerly injected.
3. When applicable, the model calls `read_file` for the listed `SKILL.md`. A fresh
   authorization transaction precedes returning instructions as a tool result.

This integrates the native middleware, not the full `create_deep_agent` stack.
Existing routing, approval checkpoints, tool governance, and streaming remain in
place. No shell, filesystem writes, arbitrary file reads, or installer is enabled.
The SQL specialist retains its own task-specific prompt and does not automatically
inherit the parent's skills.

Each middleware invocation gets a read-only backend scoped to its trusted actor.
It accepts only `/skills/account-briefing/SKILL.md`,
`/skills/sql-analysis/SKILL.md`, and `/skills/approval-workflow/SKILL.md`.
Model-provided paths are exact registry lookups, never OS paths or URLs. Discovery
requires `skill:<tenant>/<name>` reader grants for both user and assistant, plus
the assistant's `load_agent_skills` executor grant. Full reads recheck those grants
and additionally require the assistant's `read_file` executor grant.

Metadata is refreshed **before each main model call**, rather than using the SDK's
usual once-per-thread discovery cache. Revoked skills disappear from the next
summary and new reads are denied; instructions already in conversation history
cannot be retroactively erased. Skills guide behavior but never grant tool/data
access. SQL and write checks remain active even if a playbook is absent. Bundled
files are cached when the store starts; restart the server after editing them.

Try: “Read the sql-analysis skill instructions, then explain how you handle invoice
amounts and daily active seats. Do not run a query or save anything.” Expect
`skills_discovered`, a governed `read_file`, then `skill_instructions_loaded` and
a streamed explanation about integer cents and averaging seat snapshots.

## Start with a complete conversation

Select **Northstar Cloud → Maya Chen → Customer Operations Assistant**.
Keep the route on **Chat · model chooses tools**.

1. “Prepare me for Meridian Retail's renewal meeting. Find the open issues and
   cite the evidence.”
2. “Ask the renewal analyst to assess this same account. Explain the blockers
   and next steps.”
3. “Save that assessment as the account brief, including sources and next steps.”
4. Review the proposed content, select **Priya Shah**, and click **Approve**.
5. “Show me the saved brief and its current version for that account.”

The first request reads actual fixture records. The second delegates to the
two-node specialist, preserving the initiating user's access. The third pauses
before any change. Approval saves version 2; the follow-up reads the saved content.

The expected risk factors are unresolved SSO failures and failover validation,
with renewal 30 days away. Source references appear in the answer and audit feed.

## Explore individual capabilities

Unless specified otherwise, use Northstar Cloud, Maya Chen, and the Customer
Operations Assistant. Changing tenant, persona, or profile starts a fresh chat.

| Capability | Setup and query | What to look for |
| --- | --- | --- |
| Account lookup | “Look up Meridian Retail's renewal date, owner, and current brief.” | Account-specific retrieval; saved brief is versioned. |
| Evidence search | “Find Meridian Retail's open support issues and technical success plan.” | Search over real fixtures with source IDs. |
| Specialist delegation | “Ask the renewal analyst to assess Meridian Retail.” | Parent → specialist → evidence reads → assessment. Readiness: needs attention. |
| Healthy account | “Ask the renewal analyst to assess Westhaven Energy.” | Ready; 60 days to renewal, no open cases. |
| Missing evidence | “Ask the renewal analyst to assess Orchard Travel.” | Insufficient information; missing success-plan sign-off, 22 days to renewal. |
| Restricted record denied | Switch to **Jordan Ellis**, keeping the Customer Operations Assistant. “Read document:northstar/100-commercial for Meridian Retail.” | Record access denied despite the agent having broader permissions. |
| Filtered search | As **Jordan Ellis**: “Search Meridian Retail's records for concession.” | No commercial-note content or hidden-result count. Shared support records remain accessible. |
| User write denied | As **Jordan Ellis**: “Save an account brief for Meridian Retail saying the open SSO issue needs a resolution date.” | Denied before the review panel; no mutation. |
| Agent delegation denied | Switch back to **Maya Chen**, then select **Support Assistant · read only**. “Ask the renewal analyst to assess Meridian Retail.” | Agent-call denial: the user can read the account, but this parent has no delegation grant. |
| Cross-tenant isolation | As Maya in Northstar: “Look up account:beacon/AC-100 and show me its renewal details.” | Denied; Beacon account data is not returned. |
| Wrong-tenant name before routing | Select **Beacon Data → Elena Ruiz**: “Tell me about Meridian Retail.” | Request middleware denies before any model/tool call; tenant membership verification is still visible. |
| No-tool tenant verification | Any valid persona: “Hello, what can you help me with?” | Request and tenant middleware execute even when the model does not use tools. |
| Scoped short references | In Northstar: “Look up AC-100.” Then switch to **Beacon Data → Elena Ruiz** and repeat. | Meridian Retail versus Juniper Manufacturing; the reference resolves inside the selected tenant. |
| Human rejection | Ask Maya to save a new brief, then select Priya and **Reject**. | No write; reading the account shows the previous version. |
| Review hold | Ask to save, then select **Hold for further review**. | Pending-secondary-approval response; no write. This release has no second-review workflow—submit a new save request when ready. |
| Archive | Switch to **Priya Shah**. “Archive the current account brief for Meridian Retail.” Approve the proposed action as Priya. | Archived status, incremented version, retained history. Maya lacks the archive permission. |

Saved briefs use separate reader grants because they can contain commercial
information. Support users cannot obtain restricted details by reading a saved
summary. These are simulated personas; selecting a reviewer demonstrates the
policy, not a real authenticated sign-in.

## Lead approvals inbox

1. Open **Approvals inbox**, then **Switch to Priya Shah** (Northstar's lead).
   Other tenants have their own leads: **Dev Patel** in Beacon and **Noor Hassan**
   in Summit.
2. Click **Populate demo approvals**. Three real agent runs pause for review:
   a Meridian Retail renewal-risk brief, a Westhaven Energy meeting-ready brief,
   and an Orchard Travel archive request. The first two are requested by Maya;
   the archive is requested by Priya. Other tenants use their own accounts/users.
3. Select a request to inspect its exact proposed content, expected brief version,
   requester, source run ID, and conversation ID. Add an optional reviewer note.
4. **Approve request** resumes the original graph and applies the change only
   after fresh authorization and version checks. **Reject request** makes no
   change. **Hold request** also makes no change; it does not create a second-stage
   review queue. Request a new proposal when ready.
5. The item first shows **Applying decision…**, then its confirmed result. The
   detail includes the decision run ID for finding the matching LangSmith trace.

The populate button creates genuine tool authorization/interrupt traces through
the main graph, using a private deterministic route rather than a model call.
It never approves or changes a brief. Clicking again reuses pending samples;
resolved samples can be generated again using the current brief versions. Generation
is capped at 15 demo runs per tenant per server lifetime and 20 conversations per
browser session. Restarting the server resets these limits and all fictional data.

Ordinary chat requests to save/archive briefs appear in the same inbox. An
eligible lead can review a proposal from another browser session, but cannot read
the requesting session's chat history or raw event stream. Reviewer access requires
both reviewer and reader grants for the account and brief, and is rechecked on
every list and decision. Switching to a non-lead or another tenant hides the
proposals; this remains a simulated-persona demo, not production sign-in.

The visible inbox refreshes every four seconds. A waiting original chat reconnects
to its decision stream when a lead acts from another session. Switching persona
or starting a new chat leaves paused proposals queued, but does not restore the
old chat UI. Competing decisions are claimed once; version conflicts become
**stale**, and revoked requester permissions become **failed** without a write.
Approvals expire with their originating in-memory session (two hours idle).

## Explicit graph handoff

Expand **Invocation controls**, choose **Explicit analyst handoff**, and select
an account. Send “Assess the selected account.” This route invokes the specialist
through the parent graph without an LLM routing decision. The selected account
determines the task; the prompt does not change its target.

Repeat using the Support Assistant profile to see the same delegation denial.
Switch the route back to **Chat · model chooses tools** for normal conversation
and follow-ups.

## Traces and test runners

Start with [README.md](../README.md) for portable setup and the supported
`generate_traces.py` command. For example:

```sh
python generate_traces.py --count 100 --include-hitl --dry-run
python generate_traces.py --count 100 --include-hitl --verify
```

This uses the normal model-routed `/api/run` entry point. HITL decisions require
the explicit flag; counts include request, review, and resume roots. The older
fixed-matrix commands below remain available for specialized experiments.

The right-hand activity feed displays real execution events. Expand an FGA check
to see the tenant, initiating user, acting agent, resource, call ID, and matching
membership/grant path. Failures, access denials, and review decisions have
distinct events.

With tracing enabled, runs appear in the configured `LANGSMITH_PROJECT`
as **customer_operations.turn**, tagged
**customer-operations**, **tenant-governance**, and **tool** or **handoff**.
The specialist graph is named **renewal_analyst.assess_account**.
New runs carry `trace_schema_version=9`; older traces are
unchanged. Update saved run-name filters if they target the old root name.

### Trace naming and authorization middleware

Application spans use stable `component.action` names. Tenant, user, resource,
call ID, operation, and phase belong in metadata/inputs, not dynamically generated
span names. Model-facing tool names remain unchanged.

For a lookup, the relevant hierarchy is:

```text
customer_operations.turn
├─ customer_operations.authorize_request
│  ├─ authorization.authorize_transaction  [auth_scope=request]
│  │  └─ authorization.verify_tenant
│  └─ customer_operations.route_invocation
├─ customer_operations.respond
│  ├─ authorization.verify_tenant
│  ├─ customer_operations.generate_response  [model]
│  └─ customer_operations.route_next_step
├─ customer_operations.execute_tools
│  └─ tools.get_customer_account
│     └─ authorization.authorize_transaction  [auth_scope=tool]
│        └─ authorization.verify_tenant
└─ customer_operations.respond
   └─ customer_operations.generate_response  [model]
```

`customer_operations.route_invocation` selects chat or explicit handoff.
LangGraph may also show internal plumbing such as `__start__`; these are framework
spans, not unnamed application callbacks.

Every new turn runs request authorization **before the model**, including
greetings and model replies that never select a tool. `authorization.verify_tenant`
checks the trusted user/agent context and both tenant-membership grants. The
request transaction also checks explicitly mentioned account references against
the selected tenant. A denial ends the turn with a generic response and no model
or tool execution. Context is checked again before subsequent model invocations
and within tool transactions, including approval resumption.

For **Beacon → Elena → “Tell me about Meridian Retail”**, expect tenant verification
to **allow** (Elena is a valid Beacon member), followed by request authorization
**deny / resource_unavailable** (the supplied account is unavailable in Beacon).
The trace does not disclose the account's owning tenant or its canonical ID.

The in-memory request gate recognizes exact fixture account names, qualified
account IDs, and short `AC-…` references. This is a conservative demo convenience,
not general natural-language intent parsing: mentions in quotes/negations are
also checked, while unknown names, aliases, misspellings, and implicit references
must be resolved through governed tools. Request preflight is visibility-only;
it never substitutes for the tool's operation-specific authorization. Explicit
handoff checks the selected account rather than interpreting the message text.

Open **authorization.authorize_transaction** to see the actual authorization
scope (`request`, `tool`, `skill`, or `approval`), phase, decision, and FGA events with matching grant
paths. Tenant-membership events are nested under **authorization.verify_tenant**.
This middleware finishes
before the operation consumes its authorized scope. It covers account reads,
record filtering, delegation, specialist evidence reads, and writes. Approval
resumption runs a fresh transaction (`auth_phase=resume`) including reviewer scope.
Policy denial is an explicit `decision=deny` result, not a middleware crash.
Inputs contain identity and requested references—not brief text—and outputs
contain only an allowed scope or a denial. Hidden record IDs/counts remain absent.

Seeded approval traces use `customer_operations.prepare_demo_approval` and
`approval_demo=true` metadata. The decision run includes `reviewed_approval_id`;
reviewer checks use `auth_scope=approval`. Inbox polling enforces the same FGA
policy without creating repetitive traces for each refresh.

SQL traces add `tools.analyze_customer_data` → `sql_analyst.analyze_account` →
`sql_analyst.generate_query` / `sql_analyst.execute_query`. The model call is
`sql_analyst.generate_sql_statement`; actual database execution is
`analytics.execute_read_only_query`, with nested authorization middleware.
Direct SQL uses `tools.query_customer_analytics` and the same execution boundary.
Skills appear under `skills.discover_metadata` and `skills.apply_model_middleware`;
the latter contains `customer_operations.generate_response`. On-demand reads use
`tools.read_file` → `skills.read_instructions` →
`authorization.authorize_transaction` → `authorization.verify_tenant`.
Skill authorization has `auth_scope=skill` and `auth_phase=discover` or `read`.
The activity feed distinguishes `skills_discovered` from
`skill_instructions_loaded`, alongside SQL completion/rejection events.
Internal SQL-generation tokens are never streamed into the assistant chat bubble;
the main assistant's explanation still streams normally.

### Finding failed or rejected tool calls in LangSmith

For whole traces, filter the root **customer_operations.turn** runs by the tag
`contains-tool-rejection`. To inspect individual rejected calls, include child
runs and filter by `tool-rejected`. These tags serve as shared
**unsuccessful-tool** filters, covering failures as well as policy rejections.
Use the category tags to distinguish the causes:

- `rejection-authorization`: FGA or delegation denial.
- `rejection-sql-validation`: unsafe/invalid SQL or rejected analyst output.
- `rejection-input-validation`: malformed arguments or invalid read windows.
- `rejection-approval`: human review rejection or stale approval.
- `rejection-execution`: mock-service faults or unexpected tool exceptions.

Execution failures additionally have `tool-failed` on the tool span and
`contains-tool-failure` on the root, with `failure-execution` on both. Multiple
category tags can occur on one root when different calls fail for different reasons.

The rejected tool span carries `tool_name`, `call_id`, `reason_code`, and
`rejection_category` metadata. The root carries `rejected_tool_call_count`,
`rejected_tools`, `rejection_categories`, and a `tool_rejections` summary (first
100 calls; `rejection_details_truncated` indicates additional calls). Summaries
exclude SQL, arguments, record contents, and exception text. Repeated events for
one tool span count once. Existing `fga-deny` tags remain on authorization spans.

Tags are recorded from actual execution events, not model prose. A root tag means
the turn **contained** an unsuccessful tool call, not that the final response failed.
Existing error statuses are unchanged. Successful sibling calls and later turns
do not inherit rejection tags. Pending human review and cancellation are not
failures. Request-gate denials before a tool invocation, filtered-out skills/records,
and invented tool names with no actual registered tool span are excluded.
JSON argument schemas remain unchanged for models; Pydantic validation runs
inside the traced tool coroutine so invalid arguments receive outcome tags too.
No historical runs are backfilled. These tags persist with normal trace completion;
they do not require a separate LangSmith update request.

Try `sql-adoption` (successful, untagged) and `sql-billing-denied` (tagged as
authorization rejection) using the live sample runner below.

### Streaming responses

Normal chat renders real model text incrementally, with a responding cursor.
Tool-call arguments and reasoning blocks are never rendered as chat text. If the
model speaks before calling a tool, that text becomes a labeled **tool update**;
the final answer gets its own bubble. Errors mark partial responses as interrupted.

The SSE contract adds `response_start`, `response_delta`, and `response_end`, keyed
by `message_id`. The existing `agent_response` event still carries the full final
text for CLI clients; the UI reconciles it into the same bubble rather than
duplicating it. Reconnection uses event IDs to resume without appending text twice.
The explicit handoff is deterministic and does not invoke a model, so its already
computed result is delivered as one completed response (no simulated typing).
Request-gate denials are also deterministic completed responses; allowed model
responses continue to stream normally.

Offline verification:

`LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false python -m unittest discover -s tests -t . -p 'test_*.py'`

Frontend streaming, replay, and inbox tests: `node --test tests/ui/test_chat_stream.cjs`.

Live samples through the same HTTP/main-agent path as the UI:

`python -m scripts.samples`

To reproduce the no-tool authorization cases:

`python -m scripts.samples --case wrong-tenant-name --case greeting`

The runner reports event counts and first/last text timing, and requires text
deltas for model-driven samples.

Native skill discovery and on-demand reading through the main agent:

`python -m scripts.samples --case skill-loading`

SQL samples through the same HTTP/main-agent path (no database changes):

`python -m scripts.samples --case sql-adoption --case sql-incidents --case sql-billing-denied --case sql-delegation-denied`

The live workflow sample also approves a fictional brief in Beacon Data:

`python -m scripts.samples --case workflow --include-write`

Server startup in your activated virtual environment:

`python server.py`

The served implementation is in `demo/store.py`, `demo/auth.py`,
`demo/agent.py`, `demo/approvals.py`, and `demo/server.py` (launched by `server.py`).
Earlier generic-summary experiments live in `examples/legacy/` and are not part
of the served demo.
`demo/analytics.py` and `demo/skills.py` provide the local services.
Persistent SQL storage and protocol-level mock MCP servers remain future phases.

## Native SubAgentMiddleware specialists

Alongside the renewal and SQL analysts, three native agents
use Deep Agents' actual `SubAgentMiddleware` and isolated model/tool loops:

| Agent | Tools and limits |
|---|---|
| `billing-review` | Account facts, billing snapshot, usage-export status; read-only |
| `support-escalation` | Account facts, visible cases, SLA report; recommendations only |
| `renewal-planning` | Account facts, visible cases, forecast, CRM status; saving a brief requires lead approval |

The parent invokes `task(description, subagent_type)`. Include exactly one
account reference in the description. Neither identity fields nor recursive
delegation are exposed. Each child tool is constrained to that account and the
intersection of user, parent and child permissions. Both user and parent need
FGA `delegate` grants; child model/tool calls recheck them, including on resume.
An omitted `account:` prefix is normalized only if the remaining tenant/account
identifier exactly matches the server-authorized delegated account. Other
accounts and foreign tenant references remain denied.
Support personas/assistants can delegate to support-escalation, but not billing
review or renewal planning. Customer-success personas using Customer Operations
can delegate to all three.

Try the left-side scenarios, or ask:

- "Delegate to billing-review using task. For Westhaven Energy, call get_billing_snapshot once and report the balances."
- "Delegate to support-escalation using task. For Meridian Retail, call get_support_sla_report once and report its actual result or error without retrying."
- "Delegate to renewal-planning using task. Save a brief for Westhaven Energy: confirm the renewal meeting agenda and success criteria. Submit it for lead review."

Select Jordan + Customer Operations for a user-delegation denial; select Maya +
Support Assistant for a parent-delegation denial when requesting billing or renewal.

Spans include `tools.task.<specialist>`, the named native child agent, and the
actual child tool name. Leaf tags remain `tool-rejected` / `rejection-<category>`;
execution failures also carry `tool-failed` / `failure-execution`. Owning task
and child-agent spans carry `subagent-rejected` and aggregate rejection tags;
execution failures additionally carry `subagent-failed`. Parent roots retain
`contains-tool-rejection` / `contains-tool-failure` and add
`contains-subagent-rejection`. An aggregate does not inflate rejected-tool counts.
Waiting for approval, approved writes, and conditional holds are not failures;
human denial uses `rejection-approval`.

Generate 20 real main-entrypoint examples (8 successes, 6 delegation denials,
3 mock child service failures, 3 HITL outcomes):

```sh
python -m scripts.subagent_samples --output trace_batches/native-example.jsonl
python -m scripts.subagent_samples --verify-only trace_batches/native-example.jsonl
```

For an explicit follow-up subset, supply repeated case numbers, for example
`--case 06 --case 07 --case 08`. This creates a separate, labeled batch; it does
not overwrite or silently replace earlier results. Verification reports both
actual persisted tags and whether the intended scenario outcome passed.

The runner approves one fictional brief, denies one and holds one. It reuses a
single session, never retries POSTs, and records accepted run IDs immediately.
Twenty examples produce 26 root traces:20 requests,3 review-authorizations and
3 resumes. Initial/resumed runs share conversation IDs and batch labels. The
read-only verification report also records the three review roots by exact
proposal ID. Use a new output path for each batch; existing artifacts are never
overwritten. Offline tests: `python -m unittest -q tests.test_customer_subagents`.
