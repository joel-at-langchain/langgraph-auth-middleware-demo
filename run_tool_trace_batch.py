"""Create exactly 50 real HTTP/main-agent traces; verify their persisted tags.

Run with --output trace_batches/<unique-name>.jsonl. No POST/turn retries are
performed. --verify-only <manifest> checks existing runs without generating more.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
import time
from urllib.parse import urlparse
import uuid
import warnings

import httpx

from customer_services import SERVICE_TOOLS
from customer_store import TENANTS
from run_customer_samples import events_for

QUESTIONS = {
    "get_billing_snapshot": "report the open and overdue balance",
    "get_support_sla_report": "report the support SLA result",
    "get_crm_sync_status": "check whether CRM synchronization is current",
    "get_usage_export_status": "check the existing usage export's status",
    "get_renewal_forecast": "report the precomputed renewal forecast",
}
FAILURE_EVENTS = {"tool_denied", "agent_call_denied", "tool_rejected", "tool_failed"}


def cases():
    result = []
    for repetition in range(5):
        for tool_index, (tool, service) in enumerate(SERVICE_TOOLS.items()):
            tenant, _, _, _, accounts = TENANTS[(repetition + tool_index) % len(TENANTS)]
            for unsuccessful in (False, True):
                role = "support" if unsuccessful and service["restricted"] else "csm"
                index = 0 if unsuccessful else 1
                account = f"account:{tenant}/AC-{100 + index}"
                case_id = f"{len(result) + 1:02}-{tool}-{'unsuccessful' if unsuccessful else 'success'}"
                prompt = (f"Call {tool} once for {accounts[index]} ({account}) to {QUESTIONS[tool]}. "
                          "Report only this service's result, or its actual error if unavailable; do not retry, "
                          "delegate, substitute another service, or save anything. Two sentences maximum.")
                result.append({"case_id": case_id, "tool_name": tool, "unsuccessful": unsuccessful,
                               "category": ("authorization" if service["restricted"] else "execution") if unsuccessful else None,
                               "tenant_id": tenant, "user_id": f"user:{tenant}/{role}", "prompt": prompt})
    return result


def check_events(case, events):
    failures = [e for e in events if e["event"] in FAILURE_EVENTS]
    expected = "tool_denied" if case["category"] == "authorization" else "tool_failed" if case["unsuccessful"] else "tool_completed"
    checks = {
        "expected_tool_outcome": any(e["event"] == expected and e["data"].get("tool_name") == case["tool_name"] for e in events),
        "expected_failure_split": bool(failures) == case["unsuccessful"],
        "streamed_response": any(e["event"] == "response_delta" for e in events),
        "no_approval_or_request_denial": not any(e["event"] in {"approval_required", "request_denied", "error"} for e in events),
    }
    return {"checks": checks, "passed": all(checks.values()), "observed_unsuccessful": bool(failures),
            "event_counts": dict(Counter(e["event"] for e in events))}


def run_batch(base_url, output, workers):
    batch_id = "tool-outcomes-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    lock = Lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as manifest:
        def record(row):
            with lock:
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()

        record({"kind": "batch", "batch_id": batch_id, "total": 50, "expected_unsuccessful": 25})

        def run_case(case):
            ids = {}
            try:
                with httpx.Client(base_url=base_url, timeout=180) as client:
                    client.get("/api/catalog").raise_for_status()
                    response = client.post("/api/run", json={
                        "user_msg": case["prompt"], "tenant_id": case["tenant_id"], "user_id": case["user_id"],
                        "profile": "customer-ops", "invocation_mode": "tool",
                        "sample_batch_id": batch_id, "sample_case_id": case["case_id"],
                    })
                    response.raise_for_status()
                    ids = response.json()
                    record({"kind": "started", "batch_id": batch_id, **case, **ids})
                    result = {"kind": "result", "batch_id": batch_id, **case, **ids,
                              **check_events(case, events_for(client, ids["run_id"]))}
            except Exception as error:
                # Never print provider headers, cookies, credentials, or raw errors.
                result = {"kind": "result", "batch_id": batch_id, **case, **ids,
                          "passed": False, "error_type": type(error).__name__}
            record(result)
            print(f"{case['case_id']}: {'PASS' if result['passed'] else 'FAIL'} run={ids.get('run_id', 'unavailable')}", flush=True)
            return result

        print(f"Batch: {batch_id} (50 requests, 25 expected unsuccessful)", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = [future.result() for future in as_completed([pool.submit(run_case, case) for case in cases()])]
        summary = {"kind": "summary", "batch_id": batch_id, "total": len(results),
                   "runs_created": len({r["run_id"] for r in results if r.get("run_id")}),
                   "passed": sum(r["passed"] for r in results),
                   "observed_unsuccessful": sum(r.get("observed_unsuccessful", False) for r in results)}
        record(summary)
        print(json.dumps(summary), flush=True)
    return summary["runs_created"] == 50 and summary["passed"] == 50 and summary["observed_unsuccessful"] == 25


def verify_batch(manifest_path):
    from dotenv import load_dotenv
    from langsmith import Client

    load_dotenv(Path(__file__).parent / ".env")  # Existing configuration by reference only.
    with manifest_path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    started = [row for row in rows if row["kind"] == "started"]
    if len(started) != 50 or len({row["run_id"] for row in started}) != 50:
        raise ValueError("Manifest must contain exactly 50 distinct accepted runs")
    client = Client()

    def children(run):
        for child in run.child_runs or []:
            yield child
            yield from children(child)

    def verify(case):
        root = None
        # Bounded ingestion wait; retry reads only, never generate replacement runs.
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
        if root is None or root.end_time is None:
            raise ValueError("Trace is not complete")
        metadata = root.extra.get("metadata", {})
        tool_runs = [run for run in children(root) if run.run_type == "tool"]
        matching = [run for run in tool_runs if run.name == "tools." + case["tool_name"]]
        tagged = [run for run in tool_runs if "tool-rejected" in (run.tags or [])]
        assert metadata.get("sample_batch_id") == case["batch_id"]
        assert metadata.get("sample_case_id") == case["case_id"]
        assert bool(tagged) == case["unsuccessful"] and matching
        assert ("contains-tool-rejection" in (root.tags or [])) == case["unsuccessful"]
        assert metadata.get("rejected_tool_call_count", 0) == len(tagged)
        if case["unsuccessful"]:
            assert any("rejection-" + case["category"] in (run.tags or []) for run in matching)
            assert "rejection-" + case["category"] in root.tags
        return {"case_id": case["case_id"], "run_id": str(root.id), "root_tags": root.tags,
                "unsuccessful": bool(tagged), "tool_spans": [{"name": run.name, "tags": run.tags} for run in tool_runs]}

    with ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(verify, started))
    assert sum(row["unsuccessful"] for row in verified) == 25
    report = {"batch_id": started[0]["batch_id"], "verified_traces": 50, "successful": 25, "unsuccessful": 25,
              "runs": verified}
    report_path = manifest_path.with_suffix(".verified.json")
    with report_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(f"Verified 50 persisted LangSmith traces: 25 successful, 25 unsuccessful. Report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        try:
            verify_batch(args.verify_only)
        except Exception as error:
            print("Verification failed: " + type(error).__name__)
            raise SystemExit(1)
        return
    url = urlparse(args.base_url)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"} or url.username or url.password:
        parser.error("Batch requests target only a local HTTP demo server")
    if not args.output:
        parser.error("Specify a new --output path for the run-ID manifest")
    if not run_batch(args.base_url, args.output, args.workers):
        raise SystemExit("Batch did not meet the expected split. Inspect the manifest; no extra runs were generated.")


if __name__ == "__main__":
    main()
