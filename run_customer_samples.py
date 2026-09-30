"""Run live samples through the same HTTP/main-agent boundary as the chat UI.

Usage: python run_customer_samples.py [--base-url http://127.0.0.1:8000]
       python run_customer_samples.py --case workflow --include-write
"""

import argparse
import json
import time
from collections import Counter
from urllib.parse import urlparse

import httpx

CASES = {
    "skill-loading": ({}, "Read the sql-analysis skill instructions, then briefly explain how you will handle invoice amounts and daily active seats. Do not run a query or save anything.", "skill_instructions_loaded"),
    "assessment": ({}, "Ask the renewal analyst to assess Meridian Retail. Explain the blockers and cite sources.", "agent_call_completed"),
    "restricted-note": ({"user_id": "user:northstar/support"}, "Read document:northstar/100-commercial for Meridian Retail.", "tool_denied"),
    "cross-tenant": ({}, "Look up account:beacon/AC-100 and show me its renewal details.", "request_denied"),
    "wrong-tenant-name": ({"tenant_id": "beacon", "user_id": "user:beacon/csm"}, "Tell me about Meridian Retail.", "request_denied"),
    "greeting": ({}, "Hello, what can you help me with?", "tenant_verification_completed"),
    "delegation-denied": ({"profile": "support"}, "Ask the renewal analyst to assess Meridian Retail.", "agent_call_denied"),
    "handoff": ({"invocation_mode": "handoff", "handoff_account": "account:northstar/AC-100"}, "Assess the selected account.", "agent_call_completed"),
    "healthy": ({}, "Ask the renewal analyst to assess Westhaven Energy.", "agent_call_completed"),
    "incomplete": ({}, "Ask the renewal analyst to assess Orchard Travel.", "agent_call_completed"),
    "sql-adoption": ({}, "Ask the SQL analyst to compare Meridian Retail's average active seats and API error rates for September 1–14 versus 15–28, 2026. Show the SQL and sources.", "sql_query_completed"),
    "sql-incidents": ({"user_id": "user:northstar/support", "profile": "support"}, "For Meridian Retail, use query_customer_analytics with dataset service_incidents: SELECT id, service, status, impact_minutes FROM service_incidents WHERE status != 'resolved'", "sql_query_completed"),
    "sql-billing-denied": ({"user_id": "user:northstar/support"}, "For Meridian Retail, use query_customer_analytics with dataset invoices: SELECT SUM(amount_cents) FROM invoices WHERE status = 'overdue'", "tool_denied"),
    "sql-delegation-denied": ({"profile": "support"}, "Ask the SQL analyst to analyze Meridian Retail's September adoption using usage_daily.", "agent_call_denied"),
}


def events_for(client, run_id):
    events, event_name = [], ""
    started = time.monotonic()
    with client.stream("GET", "/api/stream/" + run_id) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                event = {"event": event_name, "data": json.loads(line[6:]),
                         "received_ms": round((time.monotonic() - started) * 1000)}
                events.append(event)
                if event_name == "error":
                    raise RuntimeError(event["data"]["message"])
    return events


def turn(client, prompt, **context):
    response = client.post("/api/run", json={"user_msg": prompt, **context})
    response.raise_for_status()
    ids = response.json()
    events = events_for(client, ids["run_id"])
    replies = [e["data"]["content"] for e in events if e["event"] == "agent_response"]
    deltas = [e for e in events if e["event"] == "response_delta"]
    print(json.dumps({**ids, "event_counts": dict(Counter(e["event"] for e in events)),
                      "streaming": {"first_text_ms": deltas[0]["received_ms"] if deltas else None,
                                    "last_text_ms": deltas[-1]["received_ms"] if deltas else None,
                                    "finished_ms": events[-1]["received_ms"] if events else None},
                      "response": "\n".join(replies)[:900]}, indent=2))
    return ids, events


def workflow(client):
    context = {"tenant_id": "beacon", "user_id": "user:beacon/csm"}
    ids, events = turn(client, "Prepare me for Juniper Manufacturing's renewal. Find its account details and open issues.", **context)
    context["conversation_id"] = ids["conversation_id"]
    _, events = turn(client, "Ask the renewal analyst to assess this same account and summarize its blockers.", **context)
    assert any(e["event"] == "agent_call_completed" for e in events), "Specialist did not run"
    _, events = turn(client, "Save that assessment as the account brief, including sources and next steps.", **context)
    proposal = next(e["data"] for e in events if e["event"] == "approval_required")
    reviewed = client.post("/api/resume/" + ids["conversation_id"], json={
        "approval_id": proposal["approval_id"], "reviewer": "user:beacon/lead", "decision": "approve",
        "comment": "Approved for this fictional demo smoke test.",
    })
    reviewed.raise_for_status()
    events = events_for(client, reviewed.json()["run_id"])
    assert any(e["event"] == "tool_completed" and e["data"].get("status") == "saved" for e in events)
    turn(client, "Show me the saved brief and its current version for that account.", **context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--case", choices=[*CASES, "workflow"], action="append")
    parser.add_argument("--include-write", action="store_true", help="Allow the workflow sample to approve a fictional Beacon brief.")
    args = parser.parse_args()
    url = urlparse(args.base_url)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}:
        parser.error("Samples target only the local demo server")
    selected = args.case or ["assessment", "restricted-note", "cross-tenant", "wrong-tenant-name", "delegation-denied", "handoff"]
    if "workflow" in selected and not args.include_write:
        parser.error("The workflow sample requires --include-write")
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        catalog = client.get("/api/catalog")
        catalog.raise_for_status()
        for name in selected:
            print(f"\nSample: {name}", flush=True)
            if name == "workflow":
                workflow(client)
                continue
            context, prompt, expected = CASES[name]
            _, events = turn(client, prompt, **context)
            assert any(e["event"] == expected for e in events), f"Missing expected event: {expected}"
            if name == "skill-loading":
                names = [e["event"] for e in events]
                assert names.index("skills_discovered") < names.index("skill_instructions_loaded")
                assert any(e["event"] == "tool_completed" and e["data"].get("tool_name") == "read_file" for e in events)
            if expected == "request_denied":
                assert not any(e["event"] in {"tool_intent", "response_delta"} for e in events), "Denied request reached execution"
            elif context.get("invocation_mode") != "handoff":
                assert any(e["event"] == "response_delta" for e in events), "No model text was streamed"
    print("Live samples passed.")


if __name__ == "__main__":
    main()
