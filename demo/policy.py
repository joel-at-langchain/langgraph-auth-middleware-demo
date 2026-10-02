"""Immutable mock policy, deterministic observations, and per-execution lifecycle.

No policy object is placed in model input, graph state, or checkpoint config.
The detector works without callbacks; tracing is a best-effort projection.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import re
from threading import Lock
import uuid

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import get_callback_manager_for_config

from demo.paths import REPO_ROOT

RULE_TAGS = {
    "MON-001": "security-signal:repeated-authorization-denials",
    "MON-002": "security-signal:multiple-denied-paths",
    "MON-003": "security-signal:cross-tenant-attempt",
}
MAX_ACTIONS = 512
MAX_EVIDENCE = 50
_execution = ContextVar("governance_execution", default=None)


@dataclass(frozen=True)
class Rule:
    control_id: str
    tag: str
    threshold: int


@dataclass(frozen=True)
class Policy:
    policy_id: str
    version: str
    bundle_hash: str
    rules: tuple[Rule, ...]

    def metadata(self):
        return {"policy_id": self.policy_id, "policy_version": self.version,
                "policy_bundle_sha256": self.bundle_hash, "policy_evaluator_version": "1"}


def load_bundle(directory):
    """Strict configuration, fixed filenames, no executable expressions or paths."""
    manifest = (directory / "demo-governance.json").read_bytes()
    document = (directory / "demo-governance.md").read_bytes()
    data = json.loads(manifest)
    if (set(data) != {"policy_id", "policy_version", "evaluator_version", "scope", "rules"}
            or data["policy_id"] != "customer-operations-governance"
            or not re.fullmatch(r"\d+\.\d+\.\d+", data["policy_version"])
            or data["evaluator_version"] != "1" or data["scope"] != "agent-turn"
            or f'Policy version: **{data["policy_version"]}**.' not in document.decode()
            or f'Policy ID: `{data["policy_id"]}`.' not in document.decode()):
        raise ValueError("Invalid governance policy identity or document revision")
    rules = []
    for item in data["rules"]:
        if (set(item) != {"control_id", "tag", "threshold", "mode"}
                or item["control_id"] not in RULE_TAGS or item["mode"] != "observe"
                or item["tag"] != RULE_TAGS[item["control_id"]]
                or type(item["threshold"]) is not int or not 1 <= item["threshold"] <= MAX_EVIDENCE
                or item["control_id"] == "MON-002" and item["threshold"] < 2
                or item["control_id"] == "MON-003" and item["threshold"] != 1):
            raise ValueError("Invalid governance observation rule")
        rules.append(Rule(item["control_id"], item["tag"], item["threshold"]))
    if len(rules) != len(RULE_TAGS) or {rule.control_id for rule in rules} != set(RULE_TAGS):
        raise ValueError("Governance rules must be unique and complete")
    digest = hashlib.sha256(b"manifest\0" + manifest + b"\0document\0" + document).hexdigest()
    return Policy(data["policy_id"], data["policy_version"], digest, tuple(rules))


@lru_cache(maxsize=1)
def active_policy():
    return load_bundle(REPO_ROOT / "policies")


def policy_metadata():
    state = _execution.get()
    return (state.policy if state else active_policy()).metadata()


def policy_config(config):
    """Trusted policy fields take precedence over caller metadata."""
    return {**(config or {}), "metadata": {**(config or {}).get("metadata", {}), **policy_metadata()}}


def resource_evidence(store, actor, request):
    """Resolve audit-only fixture keys, never relax the tenant-scoped resolver.

    Short references prefer the current tenant; foreign names must be unique.
    Unknown/untrusted text is never copied into evidence.
    """
    keys, foreign = set(), False
    references = request.get("references", [request.get("resource", "")])
    if request.get("record_id"):
        references = [request["record_id"]]
    for reference in references:
        if not isinstance(reference, str):
            continue
        text = reference.strip().casefold()
        record = store.records.get(reference)
        matches = [record] if record else [a for a in store.accounts.values()
            if text in {a["id"].casefold(), a["name"].casefold(), a["reference"].casefold()}]
        local = [a for a in matches if a["tenant_id"] == actor.tenant_id]
        resolved = local if local else matches
        if len(resolved) == 1:
            item = resolved[0]
            keys.add(item["id"])
            foreign |= item["tenant_id"] != actor.tenant_id
    return sorted(keys), foreign


class PolicyExecution:
    """Thread-safe, bounded detector; no LangSmith types or lookups in observe."""

    def __init__(self, policy):
        self.policy = policy
        self.execution_id = uuid.uuid4().hex
        self.actions = {}
        self.lock = Lock()
        self.incomplete = False
        self.actions_truncated = False
        self.telemetry_error = False
        self.interrupted = False
        self.root_id = None

    def observe(self, event):
        with self.lock:
            key = event["action_id"]
            previous = self.actions.get(key)
            if previous is None:
                if len(self.actions) >= MAX_ACTIONS:
                    self.incomplete = True
                    self.actions_truncated = True
                    return []
                self.actions[key] = dict(event)
            elif previous["decision"] == "deny" or event["decision"] != "deny":
                return []  # Internal rechecks cannot inflate an action's count.
            else:
                self.actions[key] = dict(event)
            return self._findings() if event["decision"] == "deny" else []

    def _findings(self):
        denied = [a for a in self.actions.values() if a["decision"] == "deny"]
        findings = []
        for rule in self.policy.rules:
            groups = [(None, denied)]
            if rule.control_id == "MON-002":
                targets = sorted({key for a in denied for key in a["resource_keys"]})
                groups = [(key, [a for a in denied if key in a["resource_keys"]]) for key in targets]
            elif rule.control_id == "MON-003":
                groups = [(None, [a for a in denied if a["cross_tenant"]])]
            for target, evidence in groups:
                count = len({a["path"] for a in evidence}) if rule.control_id == "MON-002" else len(evidence)
                if count < rule.threshold:
                    continue
                retained = evidence
                if len(evidence) > MAX_EVIDENCE:
                    candidates = (list({a["path"]: a for a in evidence}.values())
                                  if rule.control_id == "MON-002" else evidence)
                    # Keep witnesses for distinct paths plus the latest action,
                    # so truncation never prevents tagging the current trigger.
                    retained = candidates[:MAX_EVIDENCE - 1]
                    if evidence[-1] not in retained:
                        retained = [*retained, evidence[-1]]
                findings.append({"control_id": rule.control_id, "tag": rule.tag, "mode": "observe",
                    "threshold": rule.threshold, "count": count, "resource_key": target,
                    "evidence": [dict(a) for a in retained],
                    "evidence_truncated": len(evidence) > MAX_EVIDENCE})
        return findings

    def summary(self, status):
        with self.lock:
            return {"policy_execution_id": self.execution_id, "policy_evaluation_scope": "agent-turn",
                    "policy_evaluation_status": "incomplete" if self.incomplete else status,
                    "policy_evaluated_action_count": len(self.actions),
                    "policy_denied_action_count": sum(a["decision"] == "deny" for a in self.actions.values()),
                    "policy_actions_truncated": self.actions_truncated,
                    "policy_trace_projection_failed": self.telemetry_error,
                    "security_signals": self._findings()}


class _PolicyLifecycle(BaseCallbackHandler):
    # Finalize root annotations before normal tracer end callbacks persist them.
    run_inline = True

    def __init__(self, state):
        self.state = state
        self.config = None
        self.pending = {}
        self.lock = Lock()

    def queue_policy_findings(self, tracer, run_id, findings):
        with self.lock:
            _, previous = self.pending.setdefault((tracer, run_id), (tracer, {}))
            for finding in findings:
                key = (finding["control_id"], finding["resource_key"])
                if key not in previous or previous[key]["count"] <= finding["count"]:
                    previous[key] = finding

    def flush(self, run_id):
        from demo.tracing import persist_policy_findings
        with self.lock:
            matches = [key for key in self.pending if key[1] == run_id]
            pending = [self.pending.pop(key) for key in matches]
        for tracer, findings in pending:
            try:
                persist_policy_findings(tracer, run_id, list(findings.values()))
            except Exception:
                self.state.telemetry_error = True

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        if self.state.root_id is None:
            self.state.root_id = run_id

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.flush(run_id)
        if run_id == self.state.root_id:
            self.finish("interrupted" if self.state.interrupted else "completed")

    def on_chain_error(self, error, *, run_id, **kwargs):
        self.flush(run_id)
        from langgraph.errors import GraphInterrupt
        if isinstance(error, GraphInterrupt):
            self.state.interrupted = True
        if run_id == self.state.root_id:
            self.finish("error")

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.flush(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.flush(run_id)

    def finish(self, status):
        from demo.tracing import tag_policy_summary
        try:
            tag_policy_summary(self.config, self.state.root_id, self.state.summary(status))
        except Exception:
            self.state.telemetry_error = True


@contextmanager
def policy_scope(state):
    """Activate only while advancing execution, not while yielding to a client."""
    token = _execution.set(state)
    try:
        yield
    finally:
        _execution.reset(token)


def prepare_policy_execution(config, *, policy=None):
    state = PolicyExecution(policy or active_policy())
    with policy_scope(state):
        prepared = policy_config(config)
    manager = get_callback_manager_for_config(prepared)
    callback = _PolicyLifecycle(state)
    # Detach lists: do not mutate the caller's callback manager or siblings.
    manager.handlers = [callback, *manager.handlers]
    manager.inheritable_handlers = [callback, *manager.inheritable_handlers]
    prepared = {**prepared, "callbacks": manager}
    callback.config = prepared
    return prepared, state


@contextmanager
def policy_execution(config, *, policy=None):
    prepared, state = prepare_policy_execution(config, policy=policy)
    with policy_scope(state):
        yield prepared, state


def observe_authorization(store, actor, request, result, config, *, scope):
    """Consume final decisions, not raw FGA events or arbitrary model arguments."""
    state = _execution.get()
    if state is None or request.get("phase") == "discover":
        return
    from demo.tracing import policy_run_reference, tag_policy_findings
    try:
        context = (config or {}).get("configurable", {})
        action_id = context.get("governance_action_id")
        if not action_id:
            return  # Not a public-entry-point logical action (e.g. catalog lookup).
        keys, cross_tenant = resource_evidence(store, actor, request)
        kind = context.get("subagent_type") or request.get("subagent_type", "")
        # Specialist names come from validated dispatch, never task prose.
        from demo.subagents import SUBAGENTS
        kind = kind if kind in SUBAGENTS else ""
        operation = context.get("governance_operation", request["operation"])
        event = {"action_id": action_id, "call_id": context.get("call_id", ""),
                 "tenant_id": actor.tenant_id, "user_id": actor.user_id,
                 "agent_id": context.get("agent_id", actor.agent_id),
                 "parent_agent_id": context.get("parent_agent_id", ""),
                 "operation": operation, "subagent_type": kind, "auth_scope": scope,
                 "decision": result["decision"], "reason_code": result["reason_code"],
                 "resource_keys": keys, "cross_tenant": cross_tenant,
                 "path": (context.get("agent_id", actor.agent_id), context.get("parent_agent_id", ""),
                          operation, kind)}
        try:
            event.update(policy_run_reference(config))
        except Exception:
            state.telemetry_error = True
        # Evaluation precedes telemetry and remains valid even if tracing fails.
        findings = state.observe(event)
        if not findings:
            return
        # Only findings supported by this action belong on the current span.
        findings = [f for f in findings if any(a["action_id"] == action_id for a in f["evidence"])]
        if not findings:
            return
        def evaluate(_input, config):
            tag_policy_findings(config, findings)
            return {"status": "matched", "control_ids": sorted({f["control_id"] for f in findings})}
        try:
            child = policy_config(config)
            child.pop("run_id", None)
            child["run_name"] = "governance.evaluate_signals"
            RunnableLambda(evaluate, name="governance.evaluate_signals").invoke(
                {"action_id": action_id, "decision": result["decision"]}, config=child)
        except Exception:
            state.telemetry_error = True
    except Exception:
        # Observation is not authorization. Preserve the actual enforcement result.
        state.incomplete = True
