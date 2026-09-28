# Customer Operations — Demo Guide

Open http://localhost:8000 and refresh an older open tab.

The demo has 3 fictional tenants, 9 users, 6 teams, 9 accounts, 18 support
cases, 18 account documents, and 171 relationship tuples. All records and
permissions are in memory. Restarting the server restores the seed data.
Dates are evaluated against **September 28, 2026**, so results remain repeatable.

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

The right-hand activity feed displays real execution events. Expand an FGA check
to see the tenant, initiating user, acting agent, resource, call ID, and matching
membership/grant path. Failures, access denials, and review decisions have
distinct events.

With the existing tracing configuration, runs appear in LangSmith's
**auth-guardrail** project as **customer_operations.turn**, tagged
**customer-operations**, **tenant-governance**, and **tool** or **handoff**.
The specialist graph is named **renewal_analyst.assess_account**.
New runs carry `trace_schema_version=4`; older traces are
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
scope (`request`, `tool`, or `approval`), phase, decision, and FGA events with matching grant
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

`LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false /private/tmp/mcp-dev-venv/bin/python -m unittest -v test_approval_inbox test_customer_operations test_customer_observability test_agent_to_agent`

Frontend streaming, replay, and inbox tests: `node --test test_chat_stream.cjs`.

Live samples through the same HTTP/main-agent path as the UI:

`/private/tmp/mcp-dev-venv/bin/python run_customer_samples.py`

To reproduce the no-tool authorization cases:

`/private/tmp/mcp-dev-venv/bin/python run_customer_samples.py --case wrong-tenant-name --case greeting`

The runner reports event counts and first/last text timing, and requires text
deltas for model-driven samples.

The live workflow sample also approves a fictional brief in Beacon Data:

`/private/tmp/mcp-dev-venv/bin/python run_customer_samples.py --case workflow --include-write`

Server startup in the prepared local environment:

`/private/tmp/mcp-dev-venv/bin/python server.py`

The served implementation is in `customer_store.py`, `customer_auth.py`,
`customer_agent.py`, `approval_inbox.py`, and `server.py`. Earlier generic-summary scripts remain as legacy regression demos.
SQL persistence and mock MCP services are still future phases.
