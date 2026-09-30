"""Generate 100 roots through real agent HTTP requests and real HITL resumes.

70 ordinary turns + 10 workflows (request, review authorization, resume).
No turn/decision retries. --verify-only never generates agent runs.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock, local
import time
from urllib.parse import urlparse
import uuid
import warnings

import httpx

from customer_store import TENANTS
from run_customer_samples import events_for
from run_tool_trace_batch import FAILURE_EVENTS, cases, check_events


def now():
    return datetime.now(timezone.utc).isoformat()


def service_cases():
    return [{**case, "case_id": f"service-{i:02}-{case['tool_name']}"}
            for i, case in enumerate(cases() + cases()[:20], 1)]


def approval_cases():
    decisions = ["approve", "deny", "conditional", "approve", "deny",
                 "approve", "conditional", "approve", "deny", "approve"]
    result = []
    for index, decision in enumerate(decisions):
        tenant, _, _, _, accounts = TENANTS[index % len(TENANTS)]
        result.append({"case_id": f"hitl-{index + 1:02}-{decision}", "tenant_id": tenant,
                       "user_id": f"user:{tenant}/csm", "reviewer": f"user:{tenant}/lead",
                       "account_id": f"account:{tenant}/AC-101", "account_name": accounts[1],
                       "tool_name": "save_account_brief", "decision": decision})
    return result


def approval_prompt(case, batch_id):
    content = (f"Fictional demo review [{batch_id}/{case['case_id']}]. "
               f"Prepare the next renewal meeting for {case['account_name']}; "
               "confirm the agenda and revalidate success criteria with the customer.")
    return (f"Use save_account_brief once for {case['account_name']} ({case['account_id']}) "
            f"with this exact content: {content}\nSubmit it for the normal lead approval. "
            "Do not archive, delegate, retry, or save a second time. After review, briefly "
            "report its actual outcome; a denial or conditional hold must not trigger another proposal.")


def bound_proposal(case, ids, events, batch_id):
    proposals = [e["data"] for e in events if e["event"] == "approval_required"]
    if len(proposals) != 1 or any(e["event"] in FAILURE_EVENTS | {"error", "request_denied"} for e in events):
        raise ValueError("Expected exactly one clean approval pause")
    proposal = proposals[0]
    expected = {"tenant_id": case["tenant_id"], "user_id": case["user_id"],
                "account_id": case["account_id"], "operation": "save_account_brief",
                "conversation_id": ids["conversation_id"]}
    if any(proposal.get(key) != value for key, value in expected.items()):
        raise ValueError("Proposal context does not match this batch request")
    if f"/{case['case_id']}]" not in proposal.get("content", ""):
        raise ValueError("Proposal missing case marker")
    if not any(e["event"] == "done" and e["data"].get("paused") for e in events):
        raise ValueError("Request did not pause")
    return proposal


def check_resolution(case, events, status):
    decision = case["decision"]
    expected = {"approve": "saved", "deny": "rejected", "conditional": "pending_secondary_approval"}[decision]
    failures = [e for e in events if e["event"] in FAILURE_EVENTS]
    checks = {
        "expected_tool_outcome": any(e["event"] == "tool_completed" and
                                     e["data"].get("tool_name") == "save_account_brief" and
                                     e["data"].get("status") == expected for e in events),
        "expected_failure_split": bool(failures) == (decision == "deny"),
        "review_applied": any(e["event"] == "hitl_response" and e["data"].get("decision") == decision for e in events),
        "inbox_resolved": status == {"approve": "approved", "deny": "rejected", "conditional": "held"}[decision],
        "no_second_proposal": not any(e["event"] in {"approval_required", "request_denied", "error"} for e in events),
        "streamed_response": any(e["event"] == "response_delta" for e in events),
    }
    return {"checks": checks, "passed": all(checks.values()), "observed_unsuccessful": bool(failures),
            "event_counts": dict(Counter(e["event"] for e in events))}


def bound_inbox_item(case, ids, items):
    matches = [item for item in items if item["request_run_id"] == ids["run_id"]]
    if len(matches) != 1:
        raise ValueError("Expected exactly one inbox proposal for the accepted request")
    item = matches[0]
    expected = {"conversation_id": ids["conversation_id"], "requester_id": case["user_id"],
                "account_id": case["account_id"], "operation": "save_account_brief", "status": "pending"}
    if any(item.get(key) != value for key, value in expected.items()):
        raise ValueError("Inbox proposal does not match the pending batch request")
    if f"/{case['case_id']}]" not in item.get("content", ""):
        raise ValueError("Inbox proposal missing case marker")
    return item


def run_batch(base_url, output, workers, continue_batch=False):
    previous = [json.loads(line) for line in output.read_text().splitlines()] if continue_batch else []
    batch_id = (previous[0]["batch_id"] if previous else
                "mixed-hitl-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    completed = {r["case_id"]: r for r in previous if r["kind"] == "result" and r["passed"]}
    accepted = {r["case_id"]: r for r in previous if r["kind"] == "started" and r["stage"] != "resume"}
    submitted = {r["case_id"] for r in previous if r["kind"] == "decision_submitted"}
    if any(r["kind"] == "summary" and r["service_passed"] == 70 and r["workflow_passed"] == 10 for r in previous):
        raise ValueError("Batch already finished; use --verify-only")
    for case_id, row in accepted.items():
        if case_id not in completed and (row["stage"] == "service" or case_id in submitted):
            raise ValueError("Ambiguous interrupted case; inspect existing IDs before continuing")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    worker_state = local()
    clients = []

    def service_client():
        if not getattr(worker_state, "client", None) or worker_state.turns >= 20:
            if getattr(worker_state, "client", None):
                worker_state.client.close()
            client = httpx.Client(base_url=base_url, timeout=180, follow_redirects=False)
            with lock:
                clients.append(client)
            client.get("/api/catalog").raise_for_status()
            worker_state.client, worker_state.turns = client, 0
        worker_state.turns += 1
        return worker_state.client
    with output.open("a" if continue_batch else "x", encoding="utf-8") as manifest:
        def record(row):
            with lock:
                manifest.write(json.dumps({"batch_id": batch_id, **row}) + "\n")
                manifest.flush()

        if not previous:
            record({"kind": "batch", "started_at": now(), "expected_roots": 100,
                    "service_turns": 70, "approval_workflows": 10,
                    "expected_categories": {"execution": 21, "authorization": 14, "approval": 3}})
        else:
            record({"kind": "continued", "continued_at": now(), "completed_cases": len(completed)})

        def recover_approval(client, case, ids):
            context = {"tenant_id": case["tenant_id"], "user_id": case["reviewer"]}
            response = client.get("/api/approvals", params=context)
            response.raise_for_status()
            item = bound_inbox_item(case, ids, response.json()["items"])
            record({"kind": "paused", **case, "run_id": ids["run_id"],
                    "conversation_id": ids["conversation_id"], "approval_id": item["id"],
                    "expected_version": item["expected_version"], "passed": True, "recovered_from_inbox": True})
            record({"kind": "decision_submitted", **case, "request_run_id": ids["run_id"],
                    "approval_id": item["id"], "submitted_at": now()})
            response = client.post(f"/api/approvals/{item['id']}/decision", json={
                **context, "decision": case["decision"],
                "comment": f"Fictional batch review: {batch_id}/{case['case_id']}.",
            })
            response.raise_for_status()
            resolution = response.json()["approval"]
            resumed = {"conversation_id": ids["conversation_id"], "run_id": resolution["resolution_run_id"]}
            record({"kind": "started", **case, **resumed, "stage": "resume",
                    "request_run_id": ids["run_id"], "approval_id": item["id"],
                    "unsuccessful": case["decision"] == "deny",
                    "category": "approval" if case["decision"] == "deny" else None})
            deadline = time.monotonic() + 180
            while resolution["status"] == "resuming" and time.monotonic() < deadline:
                time.sleep(2)
                response = client.get("/api/approvals", params=context)
                response.raise_for_status()
                resolution = next(i for i in response.json()["items"] if i["id"] == item["id"])
            expected = {"approve": "approved", "deny": "rejected", "conditional": "held"}[case["decision"]]
            return {**resumed, "passed": resolution["status"] == expected,
                    "observed_unsuccessful": case["decision"] == "deny",
                    "recovered_via_reviewer_inbox": True, "stream_verification": "unavailable_after_session_closed",
                    "inbox_status": resolution["status"]}

        def start(client, case, prompt, stage):
            response = client.post("/api/run", json={
                "user_msg": prompt, "tenant_id": case["tenant_id"], "user_id": case["user_id"],
                "profile": "customer-ops", "invocation_mode": "tool",
                "sample_batch_id": batch_id, "sample_case_id": case["case_id"],
            })
            response.raise_for_status()
            ids = response.json()
            record({"kind": "started", **case, **ids, "stage": stage})
            return ids

        def service(case):
            if case["case_id"] in completed:
                return completed[case["case_id"]]
            ids = {}
            phase = "catalog"
            try:
                client = service_client()
                phase = "request_submission"
                ids = start(client, case, case["prompt"], "service")
                phase = "stream"
                result = {**check_events(case, events_for(client, ids["run_id"])), **ids}
            except Exception as error:
                result = {**ids, "passed": False, "error_type": type(error).__name__, "failure_phase": phase}
                if isinstance(error, httpx.HTTPStatusError):
                    result["http_status"] = error.response.status_code
            record({"kind": "result", "stage": "service", **case, **result})
            print(f"{case['case_id']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
            return result

        def approval(case):
            if case["case_id"] in completed:
                return completed[case["case_id"]]
            if case["case_id"] in accepted:
                with httpx.Client(base_url=base_url, timeout=180, follow_redirects=False) as client:
                    client.get("/api/catalog").raise_for_status()
                    result = recover_approval(client, case, accepted[case["case_id"]])
                record({"kind": "result", "stage": "workflow", **case, **result})
                print(f"{case['case_id']}: {'PASS' if result['passed'] else 'FAIL'} (recovered existing proposal)", flush=True)
                return result
            ids = {}
            try:
                # Serialize these workflows, so approved writes cannot stale one another.
                with httpx.Client(base_url=base_url, timeout=180, follow_redirects=False) as client:
                    client.get("/api/catalog").raise_for_status()
                    ids = start(client, case, approval_prompt(case, batch_id), "request")
                    events = events_for(client, ids["run_id"])
                    proposal = bound_proposal(case, ids, events, batch_id)
                    record({"kind": "paused", **case, **ids, "approval_id": proposal["approval_id"],
                            "expected_version": proposal["expected_version"], "passed": True})
                    record({"kind": "decision_submitted", **case, **ids,
                            "approval_id": proposal["approval_id"], "submitted_at": now()})
                    response = client.post("/api/resume/" + ids["conversation_id"], json={
                        "approval_id": proposal["approval_id"], "reviewer": case["reviewer"],
                        "decision": case["decision"], "comment": f"Fictional batch review: {batch_id}/{case['case_id']}.",
                    })
                    response.raise_for_status()
                    resumed = response.json()
                    record({"kind": "started", **case, **resumed, "stage": "resume",
                            "request_run_id": ids["run_id"], "approval_id": proposal["approval_id"],
                            "unsuccessful": case["decision"] == "deny",
                            "category": "approval" if case["decision"] == "deny" else None})
                    events = events_for(client, resumed["run_id"])
                    status = client.get(f"/api/approvals/{proposal['approval_id']}/status")
                    status.raise_for_status()
                    result = {**resumed, **check_resolution(case, events, status.json()["status"])}
                    result["checks"]["same_conversation"] = ids["conversation_id"] == resumed["conversation_id"]
                    result["passed"] = all(result["checks"].values())
            except Exception as error:
                result = {**ids, "passed": False, "error_type": type(error).__name__}
            record({"kind": "result", "stage": "workflow", **case, **result})
            print(f"{case['case_id']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
            return result

        print(f"Batch: {batch_id}; 70 service turns + 10 three-trace HITL workflows", flush=True)
        # HITL first: detect any model/approval mismatch before spending the regular batch.
        approval_results = []
        for case in approval_cases():
            result = approval(case)
            approval_results.append(result)
            if not result["passed"]:
                print("Stopped after unexpected HITL outcome; no retries/replacements.", flush=True)
                return False
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = [future.result() for future in as_completed([pool.submit(service, case) for case in service_cases()])]
        finally:
            for client in clients:
                client.close()
        summary = {"kind": "summary", "finished_at": now(), "service_passed": sum(r["passed"] for r in results),
                   "workflow_passed": sum(r["passed"] for r in approval_results),
                   "service_unsuccessful": sum(r.get("observed_unsuccessful", False) for r in results)}
        record(summary)
        print(json.dumps(summary), flush=True)
        return summary["service_passed"] == 70 and summary["workflow_passed"] == 10 and summary["service_unsuccessful"] == 35


def descendants(root):
    for run in root.child_runs or []:
        yield run
        yield from descendants(run)


def verify_batch(path):
    from dotenv import load_dotenv
    from langsmith import Client, utils

    load_dotenv(Path(__file__).parent / ".env")  # Configuration by reference; no secret output.
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    batch = rows[0]
    started = [r for r in rows if r["kind"] == "started"]
    pauses = {r["approval_id"]: r for r in rows if r["kind"] == "paused"}
    if len(started) != 90 or len({r["run_id"] for r in started}) != 90 or len(pauses) != 10:
        raise ValueError("Expected 90 accepted agent turns and 10 bound approval pauses")
    summaries = [r for r in rows if r["kind"] == "summary"]
    if not summaries or summaries[-1]["service_passed"] != 70 or summaries[-1]["workflow_passed"] != 10:
        raise ValueError("Batch did not complete")
    client = Client()
    # Review roots have no batch labels. Exact proposal IDs bind them to these requests.
    review_filter = ('and(eq(name, "authorization.authorize_transaction"),'
                     'eq(metadata_key, "auth_phase"),eq(metadata_value, "inbox_decision"),'
                     f'lt(start_time, {json.dumps(summaries[-1]["finished_at"])}))')
    decisions = []
    for attempt in range(4):
        decisions = [r for r in client.list_runs(
            project_name=utils.get_tracer_project(), is_root=True,
            start_time=datetime.fromisoformat(batch["started_at"]), filter=review_filter, limit=100,
        ) if (r.inputs or {}).get("approval_id") in pauses]
        if len(decisions) == 10:
            break
        time.sleep(2)
    if len(decisions) != 10 or len({r.inputs["approval_id"] for r in decisions}) != 10:
        raise ValueError("Expected one persisted reviewer-authorization root per proposal")
    print("Located all 10 reviewer authorization traces.", flush=True)

    def verify(case):
        for attempt in range(4):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    root = client.read_run(case["run_id"], load_child_runs=True)
                if root.end_time is not None:
                    break
            except Exception:
                if attempt == 3:
                    raise
            time.sleep(2)
        else:
            raise ValueError("Incomplete persisted root")
        metadata = root.extra.get("metadata", {})
        tool_runs = [run for run in descendants(root) if run.run_type == "tool"]
        matching = [run for run in tool_runs if run.name == "tools." + case["tool_name"]]
        tagged = [run for run in tool_runs if "tool-rejected" in (run.tags or [])]
        unsuccessful = case.get("unsuccessful", False)
        assert root.parent_run_id is None and root.name == "customer_operations.turn"
        assert metadata.get("conversation_id") == case["conversation_id"]
        assert metadata.get("thread_id") == case["conversation_id"]
        if case["stage"] != "resume":
            assert metadata.get("sample_batch_id") == batch["batch_id"]
            assert metadata.get("sample_case_id") == case["case_id"]
        else:
            assert metadata.get("reviewed_approval_id") == case["approval_id"]
        assert matching and bool(tagged) == unsuccessful
        assert ("contains-tool-rejection" in root.tags) == unsuccessful
        assert metadata.get("rejected_tool_call_count", 0) == len(tagged)
        if unsuccessful:
            assert "rejection-" + case["category"] in root.tags
            assert any("rejection-" + case["category"] in run.tags for run in matching)
        execution_failed = case.get("category") == "execution"
        assert ("contains-tool-failure" in root.tags) == execution_failed
        assert any("tool-failed" in run.tags for run in matching) == execution_failed
        if case["stage"] == "resume":
            output = matching[-1].outputs
            # StructuredTool output is a ToolMessage serialized inside an output envelope.
            if "output" in output:
                output = output["output"]
            if isinstance(output, dict) and "content" in output:
                output = output["content"]
            if isinstance(output, str):
                output = json.loads(output)
            assert output["changed"] == (case["decision"] == "approve")
            if case["decision"] == "approve":
                assert output["version"] == pauses[case["approval_id"]]["expected_version"] + 1
        return {"case_id": case["case_id"], "stage": case["stage"], "run_id": str(root.id),
                "conversation_id": case["conversation_id"], "category": case.get("category"),
                "unsuccessful": unsuccessful, "root_tags": root.tags,
                "tool_spans": [{"name": run.name, "tags": run.tags} for run in tool_runs]}

    verified = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(verify, case) for case in started]):
            verified.append(future.result())
            print(f"Verified agent traces: {len(verified)}/90", flush=True)
    for root in decisions:
        case = pauses[root.inputs["approval_id"]]
        assert root.end_time and root.outputs["decision"] == "allow"
        assert root.inputs["actor"]["user_id"] == case["reviewer"]
        assert "contains-tool-rejection" not in root.tags
        verified.append({"case_id": case["case_id"], "stage": "review_authorization", "run_id": str(root.id),
                         "approval_id": root.inputs["approval_id"], "conversation_id": case["conversation_id"],
                         "root_tags": root.tags, "unsuccessful": False, "category": None})
    assert len({r["run_id"] for r in verified}) == 100
    assert Counter(r["category"] for r in verified if r["unsuccessful"]) == batch["expected_categories"]
    report = {"batch_id": batch["batch_id"], "verified_roots": 100, "agent_turns": 90,
              "review_authorizations": 10, "ordinary_successes": 35, "execution_failures": 21,
              "authorization_denials": 14, "approval_pauses": 10,
              "human_decisions": {"approved": 5, "denied": 3, "held": 2},
              "tagged_unsuccessful_roots": 38, "runs": sorted(verified, key=lambda r: (r["case_id"], r["stage"]))}
    report_path = path.with_suffix(".verified.json")
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(f"Verified all 100 roots. Report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--continue-batch", action="store_true", help="Continue only unsubmitted cases in an existing manifest")
    args = parser.parse_args()
    if args.verify_only:
        try:
            verify_batch(args.verify_only)
        except Exception as error:
            print("Verification failed: " + type(error).__name__, flush=True)
            raise SystemExit(1)
        return
    url = urlparse(args.base_url)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}
            or url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment):
        parser.error("Target must be a loopback HTTP demo server")
    if not args.output:
        parser.error("Specify a new --output manifest path")
    if not run_batch(args.base_url, args.output, args.workers, args.continue_batch):
        raise SystemExit("Batch incomplete; inspect the manifest. No replacement runs generated.")


if __name__ == "__main__":
    main()
