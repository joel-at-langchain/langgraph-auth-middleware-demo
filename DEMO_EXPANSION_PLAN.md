# Customer Operations Governance Demo — Implementation Plan

Date: 2026-09-28
Status: first in-memory release implemented on 2026-09-28. See DEMO_GUIDE.md for
the running workflow and exploration queries. The served application now uses
customer_store.py, customer_agent.py, and server.py; the original generic-summary
scripts remain as legacy regression demos.

September 29 expansion: implemented a local SQLite analytics service (usage,
invoices, service incidents), a two-node SQL-generation/execution specialist,
dataset-level FGA checks, and three default runtime playbooks. The database is
an ephemeral authorized projection of fixtures, not persistent shared storage.
Added direct-query/delegation/denial/approval UI scenarios and security regression
tests. Kept the existing LangGraph framework and streaming/approval paths; that
SQL phase required no new dependencies. See DEMO_GUIDE.md for the current behavior. Persistent SQL and
protocol-level mock MCP services remain future phases. The original phase-one
plan below is retained as historical context.

September 29 middleware update: replaced eager skill-body injection with native
Deep Agents `SkillsMiddleware` discovery and model hooks, using the approved
`deepagents==0.7.20` dependency and compatible LangChain/LangGraph updates. The
main graph, interrupts, approvals, and streaming remain intact. An invocation-local,
FGA-filtered read-only backend exposes only three fixed virtual skill paths.
Metadata refreshes before each model call; full instructions require a governed
`read_file` call with fresh reader and executor checks. Added progressive-loading,
revocation, concurrency, trace, and streaming tests plus a live skill-loading sample.
No shell tools, filesystem writes, or full `create_deep_agent` stack are enabled.

September 29 outcome-coverage update: added five read-only mock service tools
for billing, support SLA, CRM sync, usage-export status, and renewal forecasts.
They use fresh tenant/account/service/executor authorization and fixed fictional
fault fixtures. Shared outcome tagging now covers execution errors, input
validation, and rejected reviews as well as FGA/SQL denials, without tagging
pending reviews. A real-model HTTP runner records exactly 50 accepted turns in
a manifest and separately verifies a 25/25 outcome split in LangSmith. No fake
trace records, protocol-level MCP servers, or new dependencies were added.

## Direction

Turn the demo into a multi-tenant customer-operations assistant. A customer
success manager can investigate an account, ask a specialist to assess renewal
readiness, and save a brief after review. Each step illustrates access to a
specific business resource or permission to delegate to another agent.

Use fictional data throughout. Keep the current LangGraph, Starlette, LangSmith,
and in-memory stores for this phase. SQL and mock MCP services come later.

First implementation priority: replace the current generic names, canned tools,
and summary task with one complete account-review workflow. Expand the seed
data and direct conversation experience around that workflow.

## Current behavior to build on

- The UI already accepts custom prompts, but clears the feed and creates a new
  thread for every run. Direct use needs an ongoing conversation, not a new
  sample for each message.
- Profile and search tools return canned records. Write and delete tools return
  success-shaped responses without modifying a business-data store.
- Document permission checks target `document:report-2024` regardless of the
  requested document ID. New tools must authorize the actual selected record.
- The FGA store supports direct tuples and a reader/writer/admin hierarchy. It
  has no tenant model or team-membership traversal yet.
- Delegation can run through ToolNode or an explicit graph route. The current
  handoff is selected by configuration and bypasses the main LLM; it is not an
  autonomous agent routing decision.
- User denial can currently fall back to agent permission. That must change for
  operations performed on a user's behalf in the tenant-aware workflow.
- The child takes the first three sentences of supplied text. Give it a small,
  structured business task and real fixture evidence instead.
- Some UI/helper lifecycle events are reconstructed after completion, and tool
  errors are classified as FGA denials. Use actual policy and execution events
  before relying on them to demonstrate more complex workflows.

## Business setting and seed data

The application serves three independent fictional SaaS companies. Each company
is a tenant; the customers it manages are accounts inside that tenant.

| Tenant | Product context | Example customer account |
| --- | --- | --- |
| Northstar Cloud | Managed infrastructure | Meridian Retail |
| Beacon Data | Analytics platform | Juniper Manufacturing |
| Summit Identity | Workforce identity | Alder Services |

Start with 3 tenants, 9 users, 6 teams, 9 accounts, 18 support cases, and 18
account documents. Seed a handful of saved briefs and roughly 80–120 relationship
tuples, driven by the intended access cases rather than a tuple-count target.

Each tenant has customer-success and support teams, named people, account
assignments, and different capabilities. In Northstar, for example:

- Maya Chen: customer success manager assigned to Meridian Retail; can read
  account records, request a readiness assessment, and propose a saved brief.
- Jordan Ellis: support engineer; can read assigned cases and shared technical
  notes, but cannot read restricted commercial notes or save renewal briefs.
- Priya Shah: customer-success lead; can review a proposed brief within her
  tenant. Review authority does not grant access to another tenant.

Fixtures should include a healthy renewal, one blocked by an unresolved support
issue, one with missing evidence, and a restricted commercial note. Reuse a
short account reference across two tenants to exercise tenant-scoped lookup.
Use stable tenant-qualified IDs internally and a fixed demo reference date for
renewal calculations. The same seed must give repeatable results.

Separate business records from relationship tuples. Business records carry
tenant, account, owner, status, dates, and source references. Tuples describe
membership, resource access, agent capabilities, delegation, and review rights.
Keep both stores resettable and in memory for now.

## Names, tools, and tasks

The main graph becomes the **Customer Operations Assistant**. Its specialist
becomes the **Renewal Readiness Analyst**. Both have tenant-scoped agent identities.
A restricted support profile supplies a realistic delegation-denial case
without introducing another graph.

Replace the existing four business tools and one delegation capability:

| Current capability | Replacement | Observable behavior |
| --- | --- | --- |
| `fetch_user_profile` | `get_customer_account` | Resolve an account and return permitted ownership, renewal, and service details. |
| `search_documents` | `search_account_records` | Search seeded notes and support cases; return authorized evidence and source IDs. |
| `write_document` | `save_account_brief` | Save or update an actual in-memory brief after resource permission checks and human approval. |
| `delete_document` | `archive_account_brief` | Mark an existing brief archived after the appropriate permission and review checks. |
| `delegate_to_summary_agent` | `assess_renewal_readiness` | Delegate a scoped account assessment to the specialist and return structured findings. |

Look up records by validated IDs. Resolve tenant and actor context from the demo
session, never from model-supplied identity claims. Search results must be
filtered before they reach the model, including snippets and result counts.
Unknown or inaccessible references should not disclose another tenant's data.

The specialist remains a two-node graph:

1. **Collect account evidence:** fetch permitted account details, open cases,
   and notes through the same governed record-access functions as the main agent.
2. **Assess renewal readiness:** produce readiness status, evidence-backed risk
   factors, missing information, recommended next steps, and source IDs.

Use explicit deterministic assessment rules initially. For example, an unresolved
critical case close to renewal requires attention; incomplete evidence produces
an insufficient-information result. These are documented demo rules, not a
predictive score. The main LLM explains the result in natural language and can
ask follow-up questions. The specialist does not save a brief as a side effect.

Support both delegation through a tool and explicit graph handoff using the
same account request, actor context, authorization functions, and result shape.
Retain the handoff selector as an advanced demo control and describe its routing
accurately. Direct user interaction means chatting with the main agent.

## Relationship model and decisions

Implement the small relationship subset the demo needs: tenant membership, team
membership, account reader/editor grants, document readers, brief reviewers,
tool executors, and parent-to-specialist delegation. Team-based access requires
adding bounded membership resolution to the current direct-tuple store; merely
seeding team tuples will not make the existing checker understand them.

Example relationships, expressed as subject / relation / object:

| Subject | Relation | Object |
| --- | --- | --- |
| Maya Chen | member | Northstar Cloud tenant |
| Maya Chen | member | Northstar customer-success team |
| Northstar customer-success team members | reader | Meridian Retail account |
| Maya Chen | editor | Meridian Retail account briefs |
| Priya Shah | reviewer | Meridian Retail account briefs |
| Northstar Customer Operations Assistant | delegate | Northstar Renewal Readiness Analyst |
| Northstar Renewal Readiness Analyst | reader | Meridian Retail account |
| Maya Chen | reader | Meridian restricted commercial note |

Access requires tenant membership plus a matching resource grant; tenant
membership alone must not expose every account. Shared notes may explicitly
inherit account-reader access. Restricted notes require separate grants.

For user-initiated operations, require both the user's resource permission and
the acting agent's capability. For delegation, also require the parent-to-child
relationship, and constrain the child's reads to the user's authorized scope
and the child's own permissions. Preserve the initiating user and tenant at
every hop. No user-denial fallback to a more privileged agent.

Human review is an additional condition after FGA permission succeeds. Bind a
decision to the specific operation, resource, and content version; recheck
permission on resume. Conditional review remains pending and performs no write.
The reviewer must be an eligible persona from the active tenant.

Use a shared authorization path for tool calls and graph handoffs, with stable
decision reason codes and explanations of the matched relationships. Keep this
an explicitly limited FGA simulator, not a claim of full OpenFGA compatibility.

## Direct agent experience

Make **Chat with Customer Operations** the default view. Keep suggested tasks as
optional prompt starters that use the same execution path as typed messages.

- Select a fictional tenant and persona from seeded options and start a chat.
- Preserve conversation history so “What is blocking the renewal?” can be
  followed by “Ask the analyst to review it” and “Save that brief.”
- Create conversation IDs on the server and bind each to its demo session,
  tenant, persona, and agent profile. Validate that binding on message, stream,
  and approval requests. Changing tenant or persona starts a fresh conversation.
- Give each turn its own run/stream lifecycle while retaining conversation state;
  closing an SSE stream must not erase the conversation's identity binding.
- Show ordinary assistant responses alongside an expandable activity feed for
  record access, delegation, and approval decisions.
- Label persona selection as simulated identity. A real sign-in integration is
  outside this phase; the local selector is not production authentication.

The live UI uses the configured model provider. Automated tests use a scripted
model through the same main graph. Share the invocation service and event
projection between the web server and test/sample runners so these paths agree.

## Suggested tasks and expected outcomes

| Request | Expected result |
| --- | --- |
| “Prepare me for Meridian Retail's renewal meeting. What is unresolved?” | Retrieve permitted account records and cite the open case and account notes. |
| “Ask the renewal analyst whether we're ready, then explain the blockers.” | Governed delegation, specialist evidence reads, assessment, and parent response. |
| “Save that as the account brief.” | Pause for an eligible review; an approval produces a stored brief visible on the next read. |
| “Archive the superseded brief after the new one is approved.” | Review and archive the selected brief; retain its history. |
| Jordan asks for Meridian's restricted commercial note. | Denial without exposing the note; shared technical evidence remains usable. |
| A Northstar user requests an account belonging to Beacon Data. | Tenant isolation enforced before any record content reaches an agent. |
| The support agent profile requests a readiness assessment. | Delegation denied even if the user can read the underlying account. |
| A reviewer rejects a proposed brief. | Explicit rejection in the conversation; no business-record mutation. |

## Delivery order and acceptance

1. **Realistic workflow:** add seed fixtures and a small business-record store;
   rename tools and agent roles; implement actual reads, searches, brief writes,
   and archives; replace sentence extraction with the two-node assessment.
2. **Tenant and relationship coverage:** add the three workspaces and personas,
   team-derived grants, actual-resource checks, shared delegation policy, and
   review rules. Use these checks from the first tenant-aware tool invocation.
3. **Direct conversation:** extend the server's session lifecycle, make chat the
   default UI, and load selectors and suggested tasks from the same seed data.
4. **Trace and scenario alignment:** update the existing runners, tests, and
   evaluators for renamed resources and the new authorization semantics. Emit
   lifecycle events at execution time; correlate them by call ID. Distinguish
   policy denial, approval pending/rejected, and execution failure.

Primary files: `fga_store.py` (relationship policy), `langgraph_fga_governance.py`
(tools, governance, parent invocation), `agent_to_agent.py` (specialist),
`server.py` and `index.html` (direct conversation), and the existing scenario,
test, and evaluator files. Add only small seed-data and business-store modules
as needed. Correct stale names and resource assumptions in evaluators without
automatically modifying registered evaluators in LangSmith.

Acceptance criteria:

- Typed prompts and sample starters both work through the main agent; no sample
  bypasses governance by invoking middleware or the child graph directly.
- A three-turn investigate → assess → save conversation retains account context.
- Tenant isolation, team access, restricted documents, delegation denial, and
  review approval/rejection are covered by behavioral tests.
- A broader agent grant never rescues a denied user request. Tool delegation and
  handoff apply the same policy for equivalent actor/resource context.
- Denied requests neither execute the child nor expose forbidden record content.
- Approved writes persist in the in-memory store; rejected or conditional
  requests leave business records unchanged. Reset restores the fixtures.
- LangSmith traces identify tenant, initiating user, parent/child agent, resource,
  call ID, invocation mode, and decision reason. Start/completion/failure events
  reflect actual execution rather than inferred ToolMessage status.
- Parallel calls and resumed approvals retain correct call-to-resource
  correlation. Multi-tenant conversations do not share history or checkpoints.
- Scripted tests run offline; a small live-model smoke batch verifies free-form
  tool selection, follow-up behavior, and uploaded trace structure separately.

## Later phases

**Local SQL:** replace the in-memory business store with a real local SQLite
database populated with fictional records, retaining the tool contracts and
policy tests. Add explicit schema/seed/reset commands. Consider PostgreSQL only
if the demo needs database-specific isolation behavior.

**Mock MCP services:** expose account, support, and knowledge capabilities as
separate local MCP servers backed by the same fictional data. Preserve tenant
and user context across the transport and enforce authorization at the service
boundary. Add transport-specific failures and service identities at that stage.
The current phase needs ordinary local functions, not simulated MCP networking.
