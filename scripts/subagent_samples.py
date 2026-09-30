"""Twenty live native-subagent examples through /api/run, including real HITL.

20 request roots + 3 review authorization roots + 3 resumed roots = 26 traces.
No automatic POST retries. Verification reads existing traces only.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from demo.paths import REPO_ROOT
import time
from urllib.parse import urlparse
import uuid
import warnings

import httpx

from demo.store import TENANTS
from scripts.samples import events_for
from scripts.tool_samples import FAILURE_EVENTS


def cases():
    result = []

    def add(kind, tool, outcome, tenant_index, *, role="csm", profile="customer-ops", account_index=1, decision=None):
        tenant, _, _, _, accounts = TENANTS[tenant_index % 3]
        case_id = f"{len(result) + 1:02}-{kind}-{outcome}"
        account_id = f"account:{tenant}/AC-{100 + account_index}"
        action = (f'Call save_account_brief once with content "Fictional {case_id}: confirm renewal meeting agenda and customer success criteria." '
                  "Submit it for lead review; after the decision, report the outcome without submitting another proposal."
                  if outcome == "approval" else
                  f"Call {tool} once and report its actual result or error. Do not save anything.")
        prompt = (f"Use task to delegate to the native {kind} subagent. Its task is: For {accounts[account_index]} "
                  f"({account_id}), {action} Do not retry, substitute a direct tool, or use a different analyst. "
                  "If delegation is denied, report that denial. Keep the final answer brief.")
        result.append({"case_id": case_id, "subagent_type": kind, "tool_name": tool, "outcome": outcome,
                       "tenant_id": tenant, "user_id": f"user:{tenant}/{role}", "profile": profile,
                       "account_id": account_id, "decision": decision, "prompt": prompt})

    success = [("billing-review", "get_billing_snapshot"), ("support-escalation", "get_support_sla_report"),
               ("renewal-planning", "get_renewal_forecast"), ("billing-review", "get_usage_export_status"),
               ("support-escalation", "search_account_records"), ("renewal-planning", "get_crm_sync_status"),
               ("billing-review", "get_billing_snapshot"), ("renewal-planning", "get_customer_account")]
    for index, (kind, tool) in enumerate(success):
        add(kind, tool, "success", index)
    for tenant in range(3):
        add("billing-review", "get_billing_snapshot", "authorization", tenant, role="support", account_index=0)
        add("renewal-planning", "get_renewal_forecast", "authorization", tenant, profile="support")
    for tenant, (kind, tool) in enumerate((("billing-review", "get_usage_export_status"),
                                         ("support-escalation", "get_support_sla_report"),
                                         ("renewal-planning", "get_crm_sync_status"))):
        add(kind, tool, "execution", tenant, account_index=0)
    for tenant, decision in enumerate(("approve", "deny", "conditional")):
        add("renewal-planning", "save_account_brief", "approval", tenant, decision=decision)
    return result


def check_events(case, events, *, resumed=False):
    names = [e["event"] for e in events]
    outcome = case["outcome"]
    category = ("approval" if case["decision"] == "deny" else None) if resumed else (
        outcome if outcome in {"authorization", "execution"} else None)
    failures = [e for e in events if e["event"] in FAILURE_EVENTS]
    expected_event = ("agent_call_denied" if category == "authorization" else
                      "tool_failed" if category == "execution" else
                      "tool_rejected" if category == "approval" else "tool_completed")
    paused = outcome == "approval" and not resumed
    native_started = [e for e in events if e["event"] == "agent_call_started"
                      and e["data"].get("subagent_type") == case["subagent_type"]]
    checks = {
        "no_request_error": not set(names) & {"request_denied", "error"},
        "native_boundary": ("agent_call_denied" in names and not native_started) if category == "authorization" else bool(native_started),
        "correct_failure_split": bool(failures) == bool(category),
        "expected_tool_event": ("approval_required" in names if paused else any(
            e["event"] == expected_event and e["data"].get("tool_name") == (
                "task" if category == "authorization" else case["tool_name"]) for e in events)),
        "expected_pause": ("approval_required" in names) == paused,
        "streamed_parent_response": paused or "response_delta" in names,
    }
    if resumed:
        status = {"approve": "saved", "deny": "rejected", "conditional": "pending_secondary_approval"}[case["decision"]]
        checks["review_outcome"] = any(e["event"] == "tool_completed" and e["data"].get("status") == status for e in events)
    observed_categories = {("execution" if e["event"] == "tool_failed" else
                            e["data"].get("rejection_category", "approval") if e["event"] == "tool_rejected" else
                            "authorization") for e in failures}
    return {"passed": all(checks.values()), "checks": checks, "category": category,
            "observed_categories": sorted(observed_categories), "child_started": bool(native_started),
            "event_counts": dict(Counter(names)), "evidence": [
                {"event": e["event"], **{k: e["data"][k] for k in (
                    "tool_name", "agent_id", "parent_agent_id", "subagent_type", "call_id", "reason_code", "status",
                ) if k in e["data"]}} for e in events
                if e["event"] in FAILURE_EVENTS | {"agent_call_started", "agent_call_completed", "tool_completed", "hitl_response"}
            ]}


def run_batch(base_url, path, selected=None):
    matrix = [case for case in cases() if not selected or case["case_id"][:2] in selected]
    batch_id = "native-subagents-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        def record(row):
            output.write(json.dumps({"batch_id": batch_id, **row}) + "\n")
            output.flush()

        record({"kind": "batch", "started_at": datetime.now(timezone.utc).isoformat(), "examples": len(matrix),
                "expected_trace_roots": len(matrix) + 2 * sum(c["outcome"] == "approval" for c in matrix)})
        print("Batch: " + batch_id, flush=True)
        results = []
        # One in-memory session, exactly20 conversations. No cookies in artifacts.
        with httpx.Client(base_url=base_url, timeout=240, follow_redirects=False) as client:
            client.get("/api/catalog").raise_for_status()
            for case in matrix:
                ids = {}
                try:
                    response = client.post("/api/run", json={
                        "user_msg": case["prompt"], "tenant_id": case["tenant_id"], "user_id": case["user_id"],
                        "profile": case["profile"], "sample_batch_id": batch_id, "sample_case_id": case["case_id"],
                    })
                    response.raise_for_status()
                    ids = response.json()
                    record({"kind": "started", "stage": "request", **case, **ids})
                    events = events_for(client, ids["run_id"])
                    result = check_events(case, events)
                    record({"kind": "result", "stage": "request", **case, **ids, **result})
                    if result["passed"] and case["outcome"] == "approval":
                        proposal, = [e["data"] for e in events if e["event"] == "approval_required"]
                        expected = {"tenant_id": case["tenant_id"], "user_id": case["user_id"],
                                    "account_id": case["account_id"], "conversation_id": ids["conversation_id"],
                                    "operation": "save_account_brief", "agent_id": f"agent:{case['tenant_id']}/renewal-planning"}
                        if any(proposal.get(k) != v for k, v in expected.items()):
                            raise ValueError("Review proposal does not match this batch request")
                        record({"kind": "paused", **case, **ids, "approval_id": proposal["approval_id"],
                                "expected_version": proposal["expected_version"]})
                        response = client.post("/api/resume/" + ids["conversation_id"], json={
                            "approval_id": proposal["approval_id"], "reviewer": f"user:{case['tenant_id']}/lead",
                            "decision": case["decision"], "comment": "Fictional native-subagent batch review: " + case["case_id"],
                        })
                        response.raise_for_status()
                        resumed = response.json()
                        record({"kind": "started", "stage": "resume", **case, **resumed,
                                "approval_id": proposal["approval_id"], "request_run_id": ids["run_id"]})
                        result = check_events(case, events_for(client, resumed["run_id"]), resumed=True)
                        status = client.get(f"/api/approvals/{proposal['approval_id']}/status")
                        status.raise_for_status()
                        result["checks"]["inbox_outcome"] = status.json()["status"] == {
                            "approve": "approved", "deny": "rejected", "conditional": "held"}[case["decision"]]
                        result["passed"] = all(result["checks"].values())
                        record({"kind": "result", "stage": "resume", **case, **resumed, **result})
                except Exception as error:
                    result = {"passed": False, "error_type": type(error).__name__}
                    record({"kind": "error", **case, **ids, **result})
                results.append(result)
                print(f"{case['case_id']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
        summary = {"kind": "summary", "examples": len(results), "passed": sum(r["passed"] for r in results),
                   "finished_at": datetime.now(timezone.utc).isoformat()}
        record(summary)
        print(json.dumps(summary), flush=True)
        return summary["passed"] == len(matrix)


def descendants(root):
    for child in root.child_runs or []:
        yield child
        yield from descendants(child)


def verify_batch(path):
    from dotenv import load_dotenv
    from langsmith import Client, utils

    load_dotenv(REPO_ROOT / ".env")  # Use existing configuration by reference only.
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    batch = rows[0]
    started = [row for row in rows if row["kind"] == "started"]
    pauses = {row["approval_id"]: row for row in rows if row["kind"] == "paused"}
    results = {(r["case_id"], r["stage"]): r for r in rows if r["kind"] == "result"}
    requests = [row for row in started if row["stage"] == "request"]
    assert len(requests) == batch["examples"]
    assert len(started) == len(requests) + len(pauses)
    assert len({r["run_id"] for r in started}) == len(started)
    client = Client()

    def verify(case):
        for attempt in range(4):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                root = client.read_run(case["run_id"], load_child_runs=True)
            if root.end_time:
                break
            time.sleep(2)
        assert root.end_time and root.parent_run_id is None
        assert root.name == "customer_operations.turn"
        metadata = root.extra.get("metadata", {})
        assert metadata["sample_batch_id"] == batch["batch_id"]
        assert metadata["conversation_id"] == case["conversation_id"]
        assert metadata["thread_id"] == case["conversation_id"]
        observed = results[(case["case_id"], case["stage"])]
        # Older manifests lack explicit observed categories. Derive these from
        # recorded events; never relabel an unexpected result as a passed case.
        categories = observed.get("observed_categories")
        if categories is None:
            categories = sorted({("execution" if e["event"] == "tool_failed" else
                                  "approval" if e["event"] == "tool_rejected" else "authorization")
                                 for e in observed["evidence"] if e["event"] in FAILURE_EVENTS})
        assert len(categories) <= 1
        category = next(iter(categories), None)
        child_started = any(e["event"] == "agent_call_started" for e in observed["evidence"])
        spans = list(descendants(root))
        owner = next(r for r in spans if r.name == "tools.task." + case["subagent_type"])
        children = [r for r in spans if r.name == case["subagent_type"] and r.run_type == "chain"]
        assert bool(children) == child_started
        assert ("contains-tool-rejection" in root.tags) == bool(category)
        assert ("subagent-rejected" in owner.tags) == bool(category)
        tagged = [r for r in spans if r.run_type == "tool" and "tool-rejected" in r.tags]
        assert metadata.get("rejected_tool_call_count", 0) == len(tagged)
        if category:
            assert "rejection-" + category in root.tags
            assert "rejection-" + category in owner.tags
            assert tagged
        if category == "execution":
            assert "contains-tool-failure" in root.tags and "subagent-failed" in owner.tags
            assert any("tool-failed" in r.tags for r in tagged)
        if child_started:
            leaf = next(r for r in spans if r.run_type == "tool" and r.name == case["tool_name"])
            assert leaf.extra["metadata"]["agent_id"] == f"agent:{case['tenant_id']}/{case['subagent_type']}"
            if case["stage"] == "resume":
                content = leaf.outputs.get("output", leaf.outputs)
                if isinstance(content, dict) and "content" in content:
                    content = content["content"]
                if isinstance(content, str):
                    content = json.loads(content)
                assert content["changed"] == (case["decision"] == "approve")
        return {"case_id": case["case_id"], "stage": case["stage"], "run_id": str(root.id),
                "conversation_id": case["conversation_id"], "category": category, "expected_outcome_passed": observed["passed"],
                "expected_category": observed["category"], "root_tags": root.tags,
                "task_tags": owner.tags, "child_agent_spans": [str(r.id) for r in children]}

    verified = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(verify, case) for case in started]):
            verified.append(future.result())
            print(f"Verified agent roots: {len(verified)}/{len(started)}", flush=True)
    finished = rows[-1]["finished_at"]
    query = ('and(eq(name,"authorization.authorize_transaction"),eq(metadata_key,"auth_phase"),'
             f'eq(metadata_value,"inbox_decision"),lt(start_time,{json.dumps(finished)}))')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reviews = [r for r in client.list_runs(project_name=utils.get_tracer_project(), is_root=True,
            start_time=datetime.fromisoformat(batch["started_at"]), filter=query, limit=100)
            if (r.inputs or {}).get("approval_id") in pauses]
    assert len(reviews) == len(pauses) and len({r.inputs["approval_id"] for r in reviews}) == len(pauses)
    for root in reviews:
        case = pauses[root.inputs["approval_id"]]
        assert root.end_time and root.outputs["decision"] == "allow"
        verified.append({"case_id": case["case_id"], "stage": "review_authorization", "run_id": str(root.id),
                         "conversation_id": case["conversation_id"], "root_tags": root.tags, "category": None})
    assert len({r["run_id"] for r in verified}) == len(started) + len(pauses)
    latest = {r["case_id"]: r for r in rows if r["kind"] in {"result", "error"}}
    report = {"batch_id": batch["batch_id"], "verified_examples": len(requests), "verified_trace_roots": len(verified),
              "expected_outcome_passes": sum(r["passed"] for r in latest.values()),
              "unexpected_cases": [key for key, row in latest.items() if not row["passed"]],
              "observed_rejection_categories": dict(Counter(r["category"] for r in verified if r["category"])),
              "human_review": dict(Counter(r["decision"] for r in pauses.values())), "runs": verified}
    report_path = path.with_suffix(".verified.json")
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(f"Verified {len(requests)} examples / {len(verified)} roots; {len(report['unexpected_cases'])} unexpected cases retained. Report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--case", action="append", choices=[c["case_id"][:2] for c in cases()], help="Run only these explicitly selected case numbers")
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_batch(args.verify_only)
            return
        url = urlparse(args.base_url)
        if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"} or url.username
                or url.password or url.path not in {"", "/"} or url.query or url.fragment):
            parser.error("Use only a local HTTP demo server")
        if not args.output:
            parser.error("Specify a new --output path")
        if not run_batch(args.base_url, args.output, args.case):
            raise SystemExit("Unexpected outcomes recorded; no replacement runs generated.")
    except Exception as error:
        print("Batch command failed: " + type(error).__name__, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
