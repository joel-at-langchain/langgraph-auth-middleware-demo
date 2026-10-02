"""Reusable trace-batch engine. See generate_traces.py --help for the public CLI.

Counts are root traces, including review/resume roots when HITL is enabled.
No POST retries. Existing artifacts are never overwritten. Verification is read-only.
"""

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
from demo.paths import REPO_ROOT, TRACE_DIR
from threading import Lock, local
import time
from urllib.parse import urlparse
import uuid
import warnings

import httpx

from demo.store import TENANTS
from scripts.samples import events_for
from scripts.subagent_samples import cases as native_cases
from scripts.tool_samples import FAILURE_EVENTS, cases as service_cases
from scripts.trace_verification import review_filter, verify_review
from scripts.policy_samples import cases as policy_cases, verify_policy_evidence

SUITES = {"mixed": None, "tools": "service", "subagents": "native-subagent", "sql": "sql", "governance": "governance"}


def validate_base_url(value):
    url = urlparse(value)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"} or url.username or url.password
            or url.path not in {"", "/"} or url.query or url.fragment):
        raise ValueError("Use only a local HTTP demo server")
    # Accessing port also validates malformed/out-of-range ports.
    if url.port == 0:
        raise ValueError("Use a nonzero port")
    return value.rstrip("/")


def plan_batch(count=20, suite="mixed", include_hitl=False):
    """Offline, deterministic case selection; no configuration or network access."""
    if not 1 <= count <= 500 or suite not in SUITES:
        raise ValueError("Count must be 1–500 and suite must be mixed/tools/subagents/sql/governance")
    if include_hitl and (suite != "mixed" or count < 9):
        raise ValueError("HITL requires --suite mixed and --count at least 9")
    approvals = approval_cases()[:min(10, max(3, count // 10))] if include_hitl else []
    groups = {}
    for case in policy_cases() if suite == "governance" else ordinary_cases():
        if SUITES[suite] is None or case["group"] == SUITES[suite]:
            groups.setdefault(case["group"], deque()).append(case)
    # Round-robin groups so a small mixed batch still reaches every capability.
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.popleft())
    ordinary = [{**ordered[i % len(ordered)], "case_id": f"{i + 1:03}-{ordered[i % len(ordered)]['case_id']}"}
                for i in range(count - 3 * len(approvals))]
    return {"expected_roots": count, "suite": suite, "include_hitl": include_hitl,
            "ordinary": ordinary, "approvals": approvals}


def plan_summary(plan):
    return {"expected_roots": plan["expected_roots"], "suite": plan["suite"],
            "ordinary_requests": len(plan["ordinary"]), "approval_workflows": len(plan["approvals"]),
            "planned_review_roots": len(plan["approvals"]), "planned_resume_roots": len(plan["approvals"]),
            "request_groups": dict(Counter(c["group"] for c in plan["ordinary"] + plan["approvals"])),
            "ordinary_expected_categories": dict(Counter(c["expected_category"] or "no-tool-rejection"
                                                          for c in plan["ordinary"])),
            "review_decisions": dict(Counter(c["decision"] for c in plan["approvals"]))}


def check_server(base_url):
    with httpx.Client(base_url=base_url, timeout=10, follow_redirects=False, trust_env=False) as client:
        response = client.get("/health")
        response.raise_for_status()
        health = response.json()
        if health.get("status") != "ok" or health.get("demo") != "customer-operations":
            raise ValueError("Expected a healthy customer-operations demo")


def now():
    return datetime.now(timezone.utc).isoformat()


def ordinary_cases():
    cases = []

    def add(group, tool, prompt, tenant="northstar", *, role="csm", profile="customer-ops", category=None,
            expected_event="tool_completed", **extra):
        cases.append({"case_id": f"{len(cases) + 1:02}-{group}", "group": group, "tool_name": tool,
                      "prompt": prompt, "tenant_id": tenant, "user_id": f"user:{tenant}/{role}", "profile": profile,
                      "expected_category": category, "expected_event": expected_event, **extra})

    for case in service_cases()[:30]:
        add("service", case["tool_name"], case["prompt"], case["tenant_id"], role=case["user_id"].rsplit("/", 1)[1],
            category=case["category"], expected_event="tool_failed" if case["category"] == "execution" else
            "tool_denied" if case["category"] == "authorization" else "tool_completed")
    for case in native_cases()[:17]:
        outcome = case["outcome"]
        add("native-subagent", "task" if outcome == "authorization" else case["tool_name"], case["prompt"],
            case["tenant_id"], role=case["user_id"].rsplit("/", 1)[1], profile=case["profile"],
            category=outcome if outcome in {"authorization", "execution"} else None,
            expected_event="agent_call_denied" if outcome == "authorization" else
            "tool_failed" if outcome == "execution" else "tool_completed", subagent_type=case["subagent_type"])
    for tenant, _, _, _, names in TENANTS:
        prefix = f"For {names[0]}, "
        suffix = " Call the specified tool once, report its actual result or error, do not retry or save anything."
        add("sql", "query_customer_analytics", prefix + "use query_customer_analytics with datasets [usage_daily] and SQL: SELECT COUNT(*) AS days FROM usage_daily." + suffix, tenant)
        add("sql", "query_customer_analytics", prefix + "use query_customer_analytics with datasets [invoices] and SQL: SELECT SUM(amount_cents) FROM invoices WHERE status = 'overdue'." + suffix,
            tenant, role="support", category="authorization", expected_event="tool_denied")
        add("sql", "query_customer_analytics", prefix + "test query validation by sending query_customer_analytics with datasets [usage_daily] and this exact SQL: SELECT missing_demo_column FROM usage_daily. Do not correct the query." + suffix,
            tenant, category="sql-validation", expected_event="tool_rejected")
        add("sql", "analyze_customer_data", prefix + "use analyze_customer_data with datasets [usage_daily] to compare average active seats for September 1–14 and 15–28, 2026. Include SQL and sources." + suffix, tenant)
    for tenant, _, _, _, names in TENANTS:
        prompt = f"Call assess_renewal_readiness once for {names[0]}. Report its result or delegation denial. Do not call task, retry, or save anything."
        add("renewal-analyst", "assess_renewal_readiness", prompt, tenant)
        add("renewal-analyst", "assess_renewal_readiness", prompt, tenant, profile="support", category="authorization", expected_event="agent_call_denied")
    for index, skill in enumerate(("sql-analysis", "account-briefing", "not-granted")):
        add("skill", "read_file", f"Call read_file once with file_path /skills/{skill}/SKILL.md and limit 1000. Briefly report its instructions or actual denial; do not try alternate paths or tools.",
            TENANTS[index][0], category="authorization" if index == 2 else None,
            expected_event="tool_denied" if index == 2 else "tool_completed")
    add("tenant-isolation", None, "Tell me about Meridian Retail.", "beacon", expected_event="request_denied")
    add("tenant-isolation", None, "Look up account:summit/AC-100 and show its renewal details.", "northstar", expected_event="request_denied")
    return cases


def approval_cases():
    cases = []
    for index, decision in enumerate(("approve", "deny", "conditional", "approve", "deny", "conditional", "approve", "deny", "conditional", "approve")):
        tenant, _, _, _, names = TENANTS[index % 3]
        native = index % 2 == 1
        kind = "native-approval" if native else "direct-approval"
        case_id = f"hitl-{index + 1:02}-{kind}"
        action = (f'For {names[1]}, call save_account_brief once with content "Fictional demo {case_id}: confirm renewal meeting agenda and customer success criteria." '
                  "Submit it for normal lead review. After review, report the outcome without saving or proposing a second time.")
        prompt = ("Use task to delegate to renewal-planning. The child must perform this task: " if native else "") + action
        cases.append({"case_id": case_id, "group": kind, "prompt": prompt, "tenant_id": tenant,
                      "user_id": f"user:{tenant}/csm", "profile": "customer-ops", "tool_name": "save_account_brief",
                      "account_id": f"account:{tenant}/AC-101", "decision": decision,
                      "expected_event": "approval_required", "expected_category": None,
                      **({"subagent_type": "renewal-planning"} if native else {})})
    return cases


def evidence_for(events):
    details = {}
    for event in events:
        if event["event"] not in FAILURE_EVENTS:
            continue
        data = event["data"]
        category = {"tool_denied": "authorization", "agent_call_denied": "authorization",
                    "tool_failed": "execution", "tool_rejected": data.get("rejection_category", "sql-validation")}[event["event"]]
        # The parent's error ToolMessage also emits a terminal signal for a
        # validation error; it is not a second execution failure or tool span.
        category = {"unsafe_or_invalid_sql": "sql-validation", "invalid_request": "input-validation",
                    "stale_approval": "approval"}.get(data.get("reason_code"), category)
        key = (data.get("tool_name"), data.get("call_id"))
        details[key] = {"tool_name": key[0], "call_id": key[1], "category": category,
                        "reason_code": data.get("reason_code")}
    return list(details.values())


def check_events(case, events):
    evidence = evidence_for(events)
    categories = sorted({e["category"] for e in evidence})
    names = [e["event"] for e in events]
    checks = {
        "expected_event": any(e["event"] == case["expected_event"] and
                              (case["tool_name"] is None or case["expected_event"] == "approval_required" or
                               e["data"].get("tool_name") == case["tool_name"]) for e in events),
        "expected_categories": categories == ([case["expected_category"]] if case["expected_category"] else []),
        "no_server_error": "error" not in names,
        "expected_pause": ("approval_required" in names) == (case["expected_event"] == "approval_required"),
        "streamed_response": case["expected_event"] in {"approval_required", "request_denied"} or "response_delta" in names,
    }
    if case.get("subagent_type"):
        checks["native_dispatch"] = (any(e["event"] == "agent_call_denied" and e["data"].get("tool_name") == "task" for e in events)
            if case["expected_event"] == "agent_call_denied" else any(e["event"] == "agent_call_started" and
                e["data"].get("subagent_type") == case["subagent_type"] for e in events))
    return {"passed": all(checks.values()), "checks": checks, "categories": categories, "rejections": evidence,
            "event_counts": dict(Counter(names)), "request_denied": "request_denied" in names,
            "native_started": any(e["event"] == "agent_call_started" and e["data"].get("subagent_type") for e in events)}


def bound_proposal(case, ids, events):
    proposals = [event["data"] for event in events if event["event"] == "approval_required"]
    if len(proposals) != 1:
        raise ValueError("Expected exactly one proposal")
    proposal = proposals[0]
    expected = {"tenant_id": case["tenant_id"], "user_id": case["user_id"], "account_id": case["account_id"],
                "operation": "save_account_brief", "conversation_id": ids["conversation_id"],
                "agent_id": f"agent:{case['tenant_id']}/{case.get('subagent_type', 'customer-ops')}"}
    if any(proposal.get(k) != v for k, v in expected.items()):
        raise ValueError("Proposal does not match this batch's authorized scope")
    return proposal


def run_batch(base_url, path, *, count=20, suite="mixed", include_hitl=False, workers=3):
    base_url = validate_base_url(base_url)
    plan = plan_batch(count, suite, include_hitl)
    if not 1 <= workers <= 6:
        raise ValueError("Workers must be 1–6")
    if path.exists():
        raise FileExistsError("Use a new manifest path")
    check_server(base_url)
    batch_id = f"governance-{count}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    path.parent.mkdir(parents=True, exist_ok=True)
    lock, worker, clients = Lock(), local(), []
    with path.open("x", encoding="utf-8") as output:
        def record(row):
            with lock:
                output.write(json.dumps({"batch_id": batch_id, **row}) + "\n")
                output.flush()

        record({"kind": "batch", "manifest_version": 2, "started_at": now(), **plan_summary(plan),
                "workers": workers})

        def client_for_worker():
            if not getattr(worker, "client", None) or worker.conversations >= 18:
                if getattr(worker, "client", None):
                    worker.client.close()
                client = httpx.Client(base_url=base_url, timeout=240, follow_redirects=False, trust_env=False)
                with lock:
                    clients.append(client)
                client.get("/api/catalog").raise_for_status()
                worker.client, worker.conversations = client, 0
            worker.conversations += 1
            return worker.client

        def start(case, client):
            record({"kind": "request_submitted", "case_id": case["case_id"]})
            response = client.post("/api/run", json={
                "user_msg": case["prompt"], "tenant_id": case["tenant_id"], "user_id": case["user_id"],
                "profile": case["profile"], "sample_batch_id": batch_id, "sample_case_id": case["case_id"],
            })
            response.raise_for_status()
            ids = response.json()
            record({"kind": "started", "stage": "request", **case, **ids})
            return ids

        def finish(case, ids, events, stage):
            result = {"kind": "result", "stage": stage, **case, **ids, **check_events(case, events)}
            record(result)
            print(f"{case['case_id']} {stage}: {'PASS' if result['passed'] else 'UNEXPECTED'}", flush=True)
            return result

        print("Batch: " + batch_id, flush=True)
        print("Manifest: " + str(path), flush=True)
        results, reviews = [], 0
        review_outcomes = []
        try:
            # Serial writes avoid version races; only proposals created here are reviewed.
            for case in plan["approvals"]:
                client = client_for_worker()
                ids = start(case, client)
                events = events_for(client, ids["run_id"])
                results.append(finish(case, ids, events, "request"))
                if not any(e["event"] == "approval_required" for e in events):
                    continue  # Preserve the actual outcome; unused roots are filled below.
                proposal = bound_proposal(case, ids, events)
                record({"kind": "paused", **case, **ids, "approval_id": proposal["approval_id"],
                        "expected_version": proposal["expected_version"]})
                record({"kind": "review_submitted", "case_id": case["case_id"], "approval_id": proposal["approval_id"]})
                response = client.post("/api/resume/" + ids["conversation_id"], json={
                    "approval_id": proposal["approval_id"], "reviewer": f"user:{case['tenant_id']}/lead",
                    "decision": case["decision"], "comment": "Fictional mixed-governance batch: " + case["case_id"],
                })
                response.raise_for_status()
                resumed = response.json()
                resumed_case = {**case, "expected_event": "tool_rejected" if case["decision"] == "deny" else "tool_completed",
                                "expected_category": "approval" if case["decision"] == "deny" else None}
                record({"kind": "started", "stage": "resume", **resumed_case, **resumed,
                        "approval_id": proposal["approval_id"], "request_run_id": ids["run_id"]})
                reviews += 1
                results.append(finish(resumed_case, resumed, events_for(client, resumed["run_id"]), "resume"))
                status = client.get(f"/api/approvals/{proposal['approval_id']}/status")
                status.raise_for_status()
                expected = {"approve": "approved", "deny": "rejected", "conditional": "held"}[case["decision"]]
                outcome = {"kind": "review_outcome", "case_id": case["case_id"], "decision": case["decision"],
                           "status": status.json()["status"], "passed": status.json()["status"] == expected}
                record(outcome)
                review_outcomes.append(outcome)

            ordinary = list(plan["ordinary"])
            # Every accepted request is a root, including an unexpected refusal.
            # Fill only unused review/resume slots, never replay a failed example.
            remaining = count - (len(results) + reviews)
            fillers = [c for c in ordinary_cases() if c["group"] == "service" and c["expected_category"] is None]
            for index in range(remaining - len(ordinary)):
                ordinary.append({**fillers[index % len(fillers)], "case_id": f"fill-{index + 1:02}", "group": "unused-slot-fill"})
            if len(ordinary) != remaining:
                raise ValueError("Trace budget mismatch")

            def run_case(case):
                client = client_for_worker()
                ids = start(case, client)
                return finish(case, ids, events_for(client, ids["run_id"]), "request")

            # Only keep a worker-sized window in flight. On ambiguous errors,
            # finish recording accepted in-flight runs but submit no more POSTs.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                queue = iter(ordinary)
                pending = {pool.submit(run_case, case) for case in islice(queue, workers)}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    completed = [future.result() for future in done]
                    results.extend(completed)
                    for _ in completed:
                        case = next(queue, None)
                        if case:
                            pending.add(pool.submit(run_case, case))
            if len(results) + reviews != count:
                raise ValueError("Trace budget mismatch")
            summary = {"kind": "summary", "finished_at": now(), "agent_roots": len(results),
                       "review_roots": reviews, "total_roots": count,
                       "expected_outcome_passes": sum(r["passed"] for r in results),
                       "unexpected_cases": [r["case_id"] + "/" + r["stage"] for r in results if not r["passed"]] +
                                           [r["case_id"] + "/review" for r in review_outcomes if not r["passed"]]}
            record(summary)
            print(json.dumps(summary), flush=True)
            return summary
        except Exception as error:
            record({"kind": "interrupted", "finished_at": now(), "error_type": type(error).__name__})
            raise
        finally:
            for client in clients:
                client.close()


def descendants(root):
    for child in root.child_runs or []:
        yield child
        yield from descendants(child)


def verify_batch(path):
    from dotenv import load_dotenv
    from langsmith import Client, utils

    if not __debug__:
        raise ValueError("Verification requires Python without -O")
    target = path.with_suffix(".verified.json")
    if target.exists():
        raise FileExistsError("Verification report already exists")
    load_dotenv(REPO_ROOT / ".env")  # Existing configuration by reference only.
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    batch, summary = rows[0], rows[-1]
    expected_roots = batch["expected_roots"]
    assert batch["kind"] == "batch" and 1 <= expected_roots <= 500
    assert summary["kind"] == "summary" and summary["total_roots"] == expected_roots
    started = [r for r in rows if r["kind"] == "started"]
    observed = {r["run_id"]: r for r in rows if r["kind"] == "result"}
    pauses = {r["approval_id"]: r for r in rows if r["kind"] == "paused"}
    assert len(started) == len(observed) == summary["agent_roots"]
    assert len({r["run_id"] for r in started}) == len(started)
    assert len(pauses) == summary["review_roots"]
    assert summary["agent_roots"] + summary["review_roots"] == expected_roots
    client = Client()

    def verify(case):
        root = None
        for attempt in range(4):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    root = client.read_run(case["run_id"], load_child_runs=True)
                if root.end_time:
                    break
            except Exception:
                if attempt == 3:
                    raise
            time.sleep(2)
        assert root is not None and root.end_time and root.parent_run_id is None
        metadata = root.extra.get("metadata", {})
        actual = observed[case["run_id"]]
        assert root.name == "customer_operations.turn"
        assert metadata["sample_batch_id"] == batch["batch_id"]
        assert metadata["conversation_id"] == metadata["thread_id"] == case["conversation_id"]
        spans = list(descendants(root))
        verify_policy_evidence(root, spans)
        if "expected_security_signals" in case:
            assert metadata.get("policy_id"), "Restart the demo to load the governance policy before this suite"
            assert sorted(t for t in root.tags if t.startswith("security-signal:")) == sorted(case["expected_security_signals"]), case["case_id"]
        rejected = [r for r in spans if r.run_type == "tool" and "tool-rejected" in r.tags]
        categories = {r.extra["metadata"]["rejection_category"] for r in rejected}
        assert categories == set(actual["categories"]), case["case_id"]
        assert ("contains-tool-rejection" in root.tags) == bool(rejected)
        assert ("contains-tool-failure" in root.tags) == ("execution" in categories)
        assert metadata.get("rejected_tool_call_count", 0) == len(rejected)
        for category in categories:
            assert "rejection-" + category in root.tags
        for tool in rejected:
            category = tool.extra["metadata"]["rejection_category"]
            assert "rejection-" + category in tool.tags
            assert ("tool-failed" in tool.tags) == (category == "execution")
        if actual["request_denied"]:
            assert not any(r.run_type in {"tool", "llm"} for r in spans)
            if metadata.get("trace_schema_version") == "10":
                assert metadata["request_outcome"] == "denied"
                assert "request-outcome:denied" in root.tags
        if case.get("subagent_type") and any(r.name.startswith("tools.task.") for r in spans):
            owner = next(r for r in spans if r.name == "tools.task." + case["subagent_type"])
            assert ("subagent-rejected" in owner.tags) == bool(rejected)
            if actual["native_started"]:
                assert any(r.name == case["subagent_type"] for r in spans)
        if case["stage"] == "resume":
            assert metadata["reviewed_approval_id"] == case["approval_id"]
            leaves = [r for r in spans if r.run_type == "tool" and r.name.removeprefix("tools.") == "save_account_brief"]
            result = leaves[-1].outputs.get("output", leaves[-1].outputs)
            if isinstance(result, dict) and "content" in result:
                result = result["content"]
            if isinstance(result, str):
                result = json.loads(result)
            assert result["changed"] == (case["decision"] == "approve")
            if result["changed"]:
                assert result["version"] == pauses[case["approval_id"]]["expected_version"] + 1
        return {"case_id": case["case_id"], "stage": case["stage"], "group": case["group"], "run_id": str(root.id),
                "conversation_id": case["conversation_id"], "tags": root.tags, "categories": sorted(categories),
                "policy_version": metadata.get("policy_version"), "security_signals": metadata.get("security_signals", []),
                "request_denied": actual["request_denied"], "expected_outcome_passed": actual["passed"],
                "rejected_tools": [{"name": r.name, "tags": r.tags} for r in rejected]}

    verified = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(verify, case) for case in started]):
            verified.append(future.result())
            print(f"Verified agent roots: {len(verified)}/{len(started)}", flush=True)
    query = review_filter(summary["finished_at"])
    reviews = []
    for attempt in range(4) if pauses else ():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            reviews = [r for r in client.list_runs(project_name=utils.get_tracer_project(), is_root=True,
                start_time=datetime.fromisoformat(batch["started_at"]), filter=query, limit=500)
                if (r.inputs or {}).get("approval_id") in pauses]
        if len(reviews) == summary["review_roots"] and all(r.end_time for r in reviews):
            break
        time.sleep(2)
    assert len(reviews) == summary["review_roots"] and len({r.inputs["approval_id"] for r in reviews}) == len(reviews)
    for root in reviews:
        case = pauses[root.inputs["approval_id"]]
        verify_review(root, case, batch["batch_id"])
        verified.append({"case_id": case["case_id"], "stage": "review", "group": case["group"], "run_id": str(root.id),
                         "conversation_id": case["conversation_id"], "tags": root.tags, "categories": [], "expected_outcome_passed": True})
    assert len(verified) == len({r["run_id"] for r in verified}) == expected_roots
    report = {"batch_id": batch["batch_id"], "verified_roots": expected_roots, "agent_roots": len(started), "review_roots": len(reviews),
              "root_groups": dict(Counter(r["group"] for r in verified)),
              "rejection_categories": dict(Counter(c for r in verified for c in r["categories"])),
              "roots_with_tool_rejections": sum(bool(r["categories"]) for r in verified),
              "request_denials": sum(r.get("request_denied", False) for r in verified),
              "review_decisions": dict(Counter(r["decision"] for r in pauses.values())),
              "unexpected_cases": summary["unexpected_cases"], "runs": verified}
    with target.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(f"Verified all {expected_roots} traces. Report: " + str(target), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Populate LangSmith through the local demo's normal agent entry point.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="loopback demo URL (default: %(default)s)")
    parser.add_argument("--count", type=int, default=20, help="total root-trace budget, 1–500 (default: %(default)s)")
    parser.add_argument("--suite", choices=SUITES, default="mixed", help="scenario selection (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=3, help="concurrent ordinary requests, 1–6 (default: %(default)s)")
    parser.add_argument("--include-hitl", action="store_true", help="allow fictional lead decisions and brief writes; mixed suite, count >=9")
    parser.add_argument("--output", type=Path, help="new .jsonl manifest; default: unique file in trace_batches/")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the plan without network access, files, or model cost")
    mode.add_argument("--verify", action="store_true", help="verify saved LangSmith tags/counts after generation")
    mode.add_argument("--verify-only", type=Path, metavar="MANIFEST", help="verify a completed batch without generating traces")
    parser.add_argument("--strict", action="store_true", help="exit 2 on unexpected case outcomes (expected denials/failures pass)")
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            if args.output or args.include_hitl:
                parser.error("--verify-only cannot be combined with --output or --include-hitl")
            report = verify_batch(args.verify_only)
            return 2 if args.strict and report["unexpected_cases"] else 0
        try:
            base_url = validate_base_url(args.base_url)
            plan = plan_batch(args.count, args.suite, args.include_hitl)
            if not 1 <= args.workers <= 6:
                raise ValueError("Workers must be 1–6")
            if args.output and args.output.suffix != ".jsonl":
                raise ValueError("Output must be a new .jsonl file")
        except ValueError:
            parser.error("Use a loopback HTTP URL, count 1–500, workers 1–6, and .jsonl output; HITL requires mixed/count >=9")
        if args.dry_run:
            print(json.dumps({**plan_summary(plan), "cases": plan["approvals"] + plan["ordinary"]}, indent=2))
            return 0
        path = args.output or TRACE_DIR / (f"{args.suite}-{args.count}-" +
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6] + ".jsonl")
        if path.exists() or (args.verify and path.with_suffix(".verified.json").exists()):
            raise FileExistsError("Choose an unused output path")
        print(json.dumps(plan_summary(plan)), flush=True)
        summary = run_batch(base_url, path, count=args.count, suite=args.suite,
                            include_hitl=args.include_hitl, workers=args.workers)
        if args.verify:
            verify_batch(path)
        if summary["unexpected_cases"]:
            print("Unexpected outcomes retained: " + ", ".join(summary["unexpected_cases"]), flush=True)
        return 2 if args.strict and summary["unexpected_cases"] else 0
    except Exception as error:
        print("Batch command stopped: " + type(error).__name__ +
              ". Check server/configuration and unused output paths; inspect any manifest before rerunning. No POST retries were made.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
