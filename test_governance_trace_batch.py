"""Offline matrix, category and review-binding contracts for the mixed batch."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

import run_governance_trace_batch as batch
from run_governance_trace_batch import approval_cases, bound_proposal, check_events, evidence_for, ordinary_cases


class GovernanceBatchTests(unittest.TestCase):
    def test_exact_hundred_roots_and_varied_matrix(self):
        ordinary, approvals = ordinary_cases(), approval_cases()
        self.assertEqual(len(ordinary) + 3 * len(approvals), 100)
        self.assertEqual(len({c["case_id"] for c in ordinary + approvals}), 80)
        self.assertEqual(Counter(c["group"] for c in ordinary), {
            "service": 30, "native-subagent": 17, "sql": 12, "renewal-analyst": 6, "skill": 3, "tenant-isolation": 2,
        })
        self.assertEqual(Counter(c["decision"] for c in approvals), {"approve": 4, "deny": 3, "conditional": 3})
        self.assertEqual(sum(bool(c.get("subagent_type")) for c in approvals), 5)
        self.assertEqual(Counter(c["expected_category"] for c in ordinary), {
            None: 36, "authorization": 19, "execution": 12, "sql-validation": 3,
        })

    def test_validation_duplicate_signal_is_not_an_execution_failure(self):
        events = [{"event": "tool_rejected", "data": {"tool_name": "query_customer_analytics", "call_id": "a",
                   "reason_code": "unsafe_or_invalid_sql", "rejection_category": "sql-validation"}},
                  {"event": "tool_failed", "data": {"tool_name": "query_customer_analytics", "call_id": "a",
                   "reason_code": "unsafe_or_invalid_sql"}}]
        self.assertEqual(len(evidence_for(events)), 1)
        self.assertEqual(evidence_for(events)[0]["category"], "sql-validation")
        self.assertEqual(evidence_for([{"event": "approval_required", "data": {}}]), [])

    def test_request_denial_is_not_a_tool_rejection(self):
        result = check_events(ordinary_cases()[-1], [{"event": "request_denied", "data": {}}])
        self.assertTrue(result["passed"])
        self.assertEqual(result["categories"], [])

    def test_review_binding_rejects_foreign_context(self):
        for case in approval_cases():
            ids = {"conversation_id": "owned-conversation"}
            proposal = {"tenant_id": case["tenant_id"], "user_id": case["user_id"], "account_id": case["account_id"],
                        "operation": "save_account_brief", "conversation_id": ids["conversation_id"],
                        "agent_id": f"agent:{case['tenant_id']}/{case.get('subagent_type', 'customer-ops')}"}
            events = [{"event": "approval_required", "data": proposal}]
            self.assertEqual(bound_proposal(case, ids, events), proposal)
            for field in proposal:
                with self.subTest(case=case["case_id"], field=field), self.assertRaises(ValueError):
                    bound_proposal(case, ids, [{"event": "approval_required", "data": {**proposal, field: "foreign"}}])


class TraceCLITests(unittest.TestCase):
    def call(self, args):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            code = batch.main(args)
        return code, output.getvalue()

    def test_dry_run_has_no_network_or_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(batch.httpx, "Client") as client:
            path = Path(tmp) / "missing" / "batch.jsonl"
            code, output = self.call(["--dry-run", "--count", "100", "--include-hitl", "--output", str(path)])
            plan = json.loads(output)
            self.assertEqual((code, plan["ordinary_requests"], plan["approval_workflows"]), (0, 70, 10))
            self.assertEqual(plan["expected_roots"], 100)
            self.assertFalse(path.parent.exists())
            client.assert_not_called()

    def test_matrix_budget_and_unique_case_labels(self):
        for count in (1, 9, 20, 50, 100, 500):
            for suite in batch.SUITES:
                for hitl in (False, True) if suite == "mixed" and count >= 9 else (False,):
                    plan = batch.plan_batch(count, suite, hitl)
                    cases = plan["ordinary"] + plan["approvals"]
                    self.assertEqual(len(plan["ordinary"]) + 3 * len(plan["approvals"]), count)
                    self.assertEqual(len({c["case_id"] for c in cases}), len(cases))
                    if suite != "mixed":
                        self.assertEqual({c["group"] for c in cases}, {batch.SUITES[suite]})
        self.assertEqual(len({c["group"] for c in batch.plan_batch(20)["ordinary"]}), 6)
        self.assertEqual(batch.plan_batch()["approvals"], [])

    def test_invalid_options_fail_before_network(self):
        options = (["--count", "0"], ["--count", "501"], ["--workers", "7"],
                   ["--include-hitl", "--count", "8"], ["--include-hitl", "--suite", "tools"],
                   ["--base-url", "https://127.0.0.1"], ["--base-url", "http://example.com"],
                   ["--base-url", "http://user:password@localhost"], ["--base-url", "http://localhost/api"],
                   ["--base-url", "http://localhost/?x=1"], ["--base-url", "http://localhost:99999"],
                   ["--output", "batch.txt"], ["--dry-run", "--verify"])
        with patch.object(batch.httpx, "Client") as client:
            for args in options:
                with self.subTest(args=args), self.assertRaises(SystemExit) as result:
                    self.call(list(args))
                self.assertEqual(result.exception.code, 2)
            client.assert_not_called()

    def test_strict_unexpected_outcomes_and_verification_only(self):
        with patch.object(batch, "run_batch", return_value={"unexpected_cases": ["case/request"]}) as run:
            self.assertEqual(self.call(["--strict"])[0], 2)
            self.assertEqual(self.call([])[0], 0)
            self.assertFalse(run.call_args.kwargs["include_hitl"])
        with patch.object(batch, "verify_batch", return_value={"unexpected_cases": []}) as verify, \
                patch.object(batch, "run_batch") as run:
            self.assertEqual(self.call(["--verify-only", "prior.jsonl", "--strict"])[0], 0)
            verify.assert_called_once_with(Path("prior.jsonl"))
            run.assert_not_called()

    def test_artifact_collisions_do_not_submit_requests(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(batch, "run_batch") as run:
            path = Path(tmp) / "batch.jsonl"
            path.touch()
            self.assertEqual(self.call(["--output", str(path)])[0], 1)
            path = Path(tmp) / "other.jsonl"
            path.with_suffix(".verified.json").touch()
            self.assertEqual(self.call(["--output", str(path), "--verify"])[0], 1)
            run.assert_not_called()


class BatchExecutionTests(unittest.TestCase):
    def simulate(self, path, *, count=20, hitl=False, missing_pause=False, fail_post=False, bad_scope=False):
        plan = batch.plan_batch(count, "mixed", hitl)
        cases = {c["case_id"]: c for c in plan["ordinary"] + plan["approvals"]}
        runs, statuses, submissions, client_options = {}, {}, [], []
        real_client = httpx.Client

        def handle(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok", "demo": "customer-operations"})
            if request.url.path == "/api/catalog":
                return httpx.Response(200, json={})
            if request.method == "POST":
                payload = json.loads(request.content)
                submissions.append((request.url.path, payload))
                if fail_post:
                    raise httpx.ReadTimeout("sensitive provider details must not be logged")
                run_id = f"run-{len(submissions)}"
                if request.url.path == "/api/run":
                    case_id = payload["sample_case_id"]
                    if case_id not in cases:
                        source = next(c for c in ordinary_cases() if c["prompt"] == payload["user_msg"])
                        cases[case_id] = {**source, "case_id": case_id}
                    case = cases[case_id]
                    ids = {"run_id": run_id, "conversation_id": "conversation-" + run_id}
                    stage = "request"
                else:
                    original = runs[payload["approval_id"].removeprefix("approval-")]
                    case, original_ids, _ = original
                    ids = {"run_id": run_id, "conversation_id": original_ids["conversation_id"]}
                    statuses[payload["approval_id"]] = {"approve": "approved", "deny": "rejected", "conditional": "held"}[payload["decision"]]
                    stage = "resume"
                runs[run_id] = (case, ids, stage)
                return httpx.Response(200, json=ids)
            if request.url.path.startswith("/api/approvals/"):
                return httpx.Response(200, json={"status": statuses[request.url.path.split("/")[3]]})
            raise AssertionError("Unexpected endpoint")

        def events(_client, run_id):
            case, ids, stage = runs[run_id]
            result = [{"event": "response_delta", "data": {}}]
            if case.get("subagent_type"):
                result.append({"event": "agent_call_started", "data": {"subagent_type": case["subagent_type"]}})
            if case.get("decision") and stage == "request":
                if missing_pause and case["case_id"] == plan["approvals"][0]["case_id"]:
                    return result
                result.append({"event": "approval_required", "data": {
                    **{k: case[k] for k in ("tenant_id", "user_id", "account_id")},
                    "operation": "save_account_brief", "conversation_id": ids["conversation_id"],
                    "agent_id": f"agent:{case['tenant_id']}/{case.get('subagent_type', 'customer-ops')}",
                    "approval_id": "approval-" + run_id, "expected_version": 1}})
                if bad_scope:
                    result[-1]["data"]["tenant_id"] = "foreign"
                return result
            event = case["expected_event"] if stage == "request" else (
                "tool_rejected" if case["decision"] == "deny" else "tool_completed")
            category = case["expected_category"] if stage == "request" else "approval"
            result.append({"event": event, "data": {"tool_name": case["tool_name"],
                "call_id": "call-" + run_id, "rejection_category": category}})
            return result

        def client(**options):
            client_options.append(options)
            return real_client(transport=httpx.MockTransport(handle), **options)

        with patch.object(batch.httpx, "Client", side_effect=client), patch.object(batch, "events_for", side_effect=events), \
                redirect_stdout(io.StringIO()):
            try:
                summary = batch.run_batch("http://127.0.0.1:8000", path, count=count, include_hitl=hitl, workers=1)
            except (httpx.ReadTimeout, ValueError):
                summary = None
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertTrue(all(o["trust_env"] is False and o["follow_redirects"] is False for o in client_options))
        return summary, rows, submissions, client_options

    def test_ordinary_batch_and_session_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows, submissions, options = self.simulate(Path(tmp) / "batch.jsonl", count=40)
        self.assertEqual(summary["total_roots"], 40)
        self.assertEqual(summary["review_roots"], 0)
        self.assertEqual(summary["unexpected_cases"], [])
        self.assertEqual(len(submissions), 40)
        self.assertEqual(len(options), 4)  # health + three sessions of <=18 turns
        self.assertEqual(len([r for r in rows if r["kind"] == "started"]), 40)

    def test_hundred_roots_and_all_three_review_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows, _, _ = self.simulate(Path(tmp) / "batch.jsonl", count=100, hitl=True)
        self.assertEqual((summary["agent_roots"], summary["review_roots"]), (90, 10))
        self.assertEqual(summary["unexpected_cases"], [])
        self.assertEqual({r["status"] for r in rows if r["kind"] == "review_outcome"}, {"approved", "rejected", "held"})

    def test_missing_pause_kept_and_unused_slots_filled_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows, _, _ = self.simulate(Path(tmp) / "batch.jsonl", hitl=True, missing_pause=True)
        self.assertEqual((summary["agent_roots"], summary["review_roots"]), (18, 2))
        self.assertEqual(len(summary["unexpected_cases"]), 1)
        fills = [r for r in rows if r["kind"] == "result" and r["case_id"].startswith("fill-")]
        self.assertEqual(len(fills), 2)
        self.assertTrue(all(r["expected_category"] is None and r["tool_name"].startswith("get_") for r in fills))

    def test_no_post_retry_and_partial_manifest_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows, submissions, _ = self.simulate(Path(tmp) / "batch.jsonl", fail_post=True)
        self.assertIsNone(summary)
        self.assertEqual(len(submissions), 1)
        self.assertEqual(rows[-1]["kind"], "interrupted")
        self.assertEqual(rows[-1]["error_type"], "ReadTimeout")
        self.assertNotIn("sensitive", json.dumps(rows))

    def test_mismatched_proposal_is_never_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, rows, submissions, _ = self.simulate(Path(tmp) / "batch.jsonl", hitl=True, bad_scope=True)
        self.assertIsNone(summary)
        self.assertEqual(len(submissions), 1)
        self.assertFalse(any(r["kind"] == "review_submitted" for r in rows))

    def test_generic_count_verification_and_no_empty_reviewer_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch.jsonl"
            _, rows, _, _ = self.simulate(path, count=1)
            case = next(r for r in rows if r["kind"] == "started")
            root = SimpleNamespace(id=case["run_id"], end_time=True, parent_run_id=None,
                name="customer_operations.turn", tags=[], child_runs=[], extra={"metadata": {
                    "sample_batch_id": rows[0]["batch_id"], "thread_id": case["conversation_id"],
                    "conversation_id": case["conversation_id"]}})
            client = Mock()
            client.read_run.return_value = root
            with patch("langsmith.Client", return_value=client), patch("dotenv.load_dotenv"), redirect_stdout(io.StringIO()):
                report = batch.verify_batch(path)
                self.assertEqual(report["verified_roots"], 1)
                client.list_runs.assert_not_called()
                with self.assertRaises(FileExistsError):
                    batch.verify_batch(path)


if __name__ == "__main__":
    unittest.main()
