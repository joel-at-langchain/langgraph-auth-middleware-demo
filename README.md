# FGA-governed customer operations

A local, multi-tenant agent demo showing **fine-grained authorization (FGA)**
across requests, tools, delegated agents, skills, SQL, and human approvals—with
streaming responses and inspectable LangSmith traces.

## Quick start

Prerequisites: Python 3.12 and [Socket Firewall](https://github.com/SocketDev/sfw-free)
(`sfw`) for dependency installation. Node.js 18+ is only needed for UI tests.
From the repo root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
sfw pip install -r requirements.txt
```

Configure the following in your shell or a local, git-ignored `.env` file. Replace
the placeholders with your own credentials; never commit them:

```dotenv
ANTHROPIC_API_KEY=<your-model-provider-key>
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your-langsmith-key>
LANGSMITH_PROJECT=fga-governance-demo
```

Optional: `DEMO_MODEL` overrides `claude-sonnet-4-5-20250929`; `BASE_URL` selects an
Anthropic-compatible model endpoint. Set `LANGSMITH_ENDPOINT` for your LangSmith
region/deployment. The server and trace verifier must use the same LangSmith
configuration. Existing shell variables take precedence over `.env`.

```sh
python server.py
```

Open [127.0.0.1:8000](http://127.0.0.1:8000). Start with **Northstar → Maya Chen →
Customer Operations Assistant**. The sidebar scenarios populate the message;
hover for the policy being tested and expected result. Try:

- “Ask the SQL analyst to compare Meridian Retail's active seats in the first and second halves of September 2026.”
- “Delegate to billing-review using task. For Westhaven Energy, call get_billing_snapshot once.”
- “Save a brief for Westhaven Energy: confirm the renewal meeting agenda.” Then review it as **Priya Shah** in the approvals inbox.

## What it demonstrates

| Area | Implementation |
| --- | --- |
| Tenant and tool authorization | Explicit request/transaction middleware; user **and** agent grants; 576 in-memory relationship tuples across three fictional tenants |
| Delegation | Native `SubAgentMiddleware` specialists: billing review, support escalation, renewal planning; separate tool-wrapped renewal and SQL analysts |
| Data and services | Account documents/cases, scoped read-only SQLite analytics, five mock billing/support/CRM/usage/forecast tools with deterministic failures |
| Skills | Deep Agents `SkillsMiddleware` discovers permitted playbooks; governed `read_file` loads full instructions on demand |
| Human review | Checkpointed brief saves/archives; lead inbox; approve, reject, or hold; authorization/version checks on resume |
| Observability | Streaming SSE, visible FGA checks, named spans, failure/rejection tags, shared thread IDs across request/resume |

The parent is a custom LangGraph integrating Deep Agents middleware—not the full
`create_deep_agent` stack. Permissions intersect across user, parent, and child;
delegation never grants extra access. See the [demo guide](docs/demo-guide.md) for the
scenario catalog, policy details, and trace hierarchy.

## Populate a tracing project

Keep the server running; use another terminal with the same environment:

```sh
# Preview cases and root-trace budget: no requests, files, or model cost.
python generate_traces.py --count 100 --include-hitl --dry-run

# Generate a varied batch, exercise fictional reviews, then verify saved tags.
python generate_traces.py --count 100 --include-hitl --verify

# Smaller, read-only batches; no automatic human decisions by default.
python generate_traces.py --suite subagents --count 20 --verify
python generate_traces.py --suite tools --count 50 --output trace_batches/tool-demo.jsonl
python generate_traces.py --suite sql --count 12 --verify

# Verify an existing completed manifest without creating more traces.
python generate_traces.py --verify-only trace_batches/tool-demo.jsonl
```

Defaults: **20 roots**, `mixed` suite, three workers, no HITL. `mixed` covers
services, native subagents, SQL, legacy analysts, skills, and tenant isolation.
Counts range from 1–500; workers from 1–6. `--include-hitl` requires `mixed` and
at least nine roots, and authorizes automated fictional lead decisions/brief writes.
Live generation uses the real model and incurs provider/tracing usage.

**Count means root traces, not conversations or spans.** One completed HITL
workflow contributes request + reviewer authorization + resume = three roots.
The 100-root HITL plan has 70 ordinary requests and ten review workflows. If a
model never reaches a planned review, that unexpected trace is retained; distinct
read-only filler requests use the unused root slots. Exact outcome ratios are
not guaranteed—this is real model routing, not manufactured trace data.

Each run gets a unique batch ID and an exclusive JSONL manifest in `trace_batches/`.
Use `--output trace_batches/my-batch.jsonl` to choose a new path. Verification
writes a separate `.verified.json` report with persisted tags, root IDs, approval
outcomes, and scenario mismatches. Generated artifacts are ignored by Git.
Exit codes: `0` completed, `1` execution/verification error, `2` invalid CLI usage
or unexpected scenario outcomes with `--strict`. Expected denials/failures pass.

No automatic POST retries, artifact overwrites, or server restarts. Sessions rotate
below the per-session cap. On interruption, inspect the manifest before starting
a **new** batch; accepted runs may already exist. Partial batches are not resumed.
An already-existing verification report is preserved; use it instead of rerunning.

### Find the traces in LangSmith

Filter agent roots by `sample_batch_id` metadata or `batch:<batch-id>` tag; use
`sample_case_id` for individual examples. Reviewer-authorization roots do not
carry batch labels: the verification report correlates them by exact approval ID.
Request/resume agent roots share `thread_id` / `conversation_id`.

- `contains-tool-rejection` on roots; `tool-rejected` on affected tool spans.
- `rejection-authorization`, `rejection-sql-validation`, `rejection-input-validation`, `rejection-approval`, or `rejection-execution` identifies the cause.
- `contains-tool-failure` / `tool-failed` additionally marks execution errors.
- `contains-subagent-rejection` / `subagent-rejected` identifies delegated failures.

Pending review and holds are not failures. Request-level tenant denials are not
tool rejections. Verification checks the stored trace, not just the final answer.

## Development

```sh
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false python -m unittest discover -s tests -t . -p 'test_*.py'
node --test tests/ui/test_chat_stream.cjs
```

```text
demo/                Runtime: agent, authorization, stores, services, approvals
  playbooks/         Three bundled runtime skills (SKILL.md)
scripts/             Trace-batch engine and focused live sample runners
tests/               Offline Python regressions; UI tests in tests/ui/
web/                 Chat and approvals UI
docs/demo-guide.md   Scenarios, policy details, and trace interpretation
examples/legacy/     Earlier generic-agent and evaluator experiments
server.py            Stable server launcher
generate_traces.py   Stable trace-generation CLI
```

`demo/paths.py` resolves configuration, UI, and playbooks from the repository,
not the working directory. The `.env` stays at the repository root. Default trace
artifacts go to `trace_batches/`; explicit `--output` paths are relative to your
working directory. `local/scratch/` preserves personal experiments. Both folders
are git-ignored and excluded from the demo and test discovery.

For focused samples, run `python -m scripts.samples --help` from the repo root.
The legacy examples are retained for reference, not used by the running demo;
some legacy runners register evaluators or automatically review fictional writes.
Use `generate_traces.py` for the current demo. Superseded implementation plans
have been removed; their history remains in Git.

## Demo boundaries

Personas and FGA storage are simulated—not production authentication or a hosted
FGA service. Services are local mocks, not actual MCP servers. SQL executes against
isolated in-memory SQLite; no external customer systems are contacted. Model and
LangSmith calls **do** use configured external services. Dates are fixed to
September 28, 2026. Restarting resets chats, approvals, and fictional data but
does not remove LangSmith traces. A hold makes no write and has no second-review
workflow. Keep this demo bound to loopback.
