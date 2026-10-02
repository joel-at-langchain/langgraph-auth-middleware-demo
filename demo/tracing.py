"""Canonical decisions/outcomes and backward-compatible failure aggregates."""

from threading import Lock

from langchain_core.tracers.base import BaseTracer
from langsmith.run_helpers import get_current_run_tree


TRACE_SCHEMA_VERSION = "10"


REJECTION_CATEGORIES = {
    "tool_denied": "authorization",
    "agent_call_denied": "authorization",
    "tool_rejected": "sql-validation",
    "tool_failed": "execution",
}
MAX_REJECTION_DETAILS = 100
_TAG_LOCK = Lock()


def _add_tags(run, *tags):
    run.tags = list(run.tags or [])
    run.add_tags([tag for tag in tags if tag not in (run.tags or [])])


def _metadata(run, **fields):
    run.extra = {**run.extra, "metadata": {**run.extra.get("metadata", {}), **fields}}


def _outcome(run, scope, value):
    prefix = scope + "-outcome:"
    run.tags = [tag for tag in (run.tags or []) if not tag.startswith(prefix)]
    _add_tags(run, prefix + value)
    _metadata(run, **{scope.replace("-", "_") + "_outcome": value})


def _active_runs(config, current_run=None):
    callbacks = (config or {}).get("callbacks")
    handlers = callbacks if isinstance(callbacks, (list, tuple)) else getattr(callbacks, "handlers", [])
    current_run = current_run or get_current_run_tree()
    parent_id = getattr(callbacks, "parent_run_id", None)
    current_id = current_run.id if current_run is not None else parent_id
    for tracer in handlers:
        if isinstance(tracer, BaseTracer):
            # SDK middleware can add a raw LangSmith @traceable span that is not
            # managed by this LangChain tracer. Its callback parent is still live.
            run = tracer.run_map.get(str(current_id)) or tracer.run_map.get(str(parent_id))
            if run is not None:
                yield tracer, run


def policy_run_reference(config):
    """Optional evidence linkage; absence of tracing never prevents evaluation."""
    for _, run in _active_runs(config):
        return {"run_id": str(run.id), "trace_id": str(run.trace_id)}
    return {}


def tag_policy_summary(config, root_id, summary):
    callbacks = (config or {}).get("callbacks")
    handlers = callbacks if isinstance(callbacks, (list, tuple)) else getattr(callbacks, "handlers", [])
    with _TAG_LOCK:
        for tracer in handlers:
            if isinstance(tracer, BaseTracer):
                root = tracer.run_map.get(str(root_id))
                if root is not None:
                    _metadata(root, **summary)
                    _add_tags(root, *(finding["tag"] for finding in summary["security_signals"]))


def tag_policy_findings(config, findings):
    """Queue owner annotations until end: SDK children inherit active metadata."""
    callbacks = (config or {}).get("callbacks")
    handlers = callbacks if isinstance(callbacks, (list, tuple)) else getattr(callbacks, "handlers", [])
    lifecycle = next((h for h in handlers if hasattr(h, "queue_policy_findings")), None)
    if lifecycle is None:
        return
    with _TAG_LOCK:
        for tracer, run in _active_runs(config):
            ancestors = list(_ancestors(tracer, run))
            source = ancestors[1] if len(ancestors) > 1 else run
            tool = next((item for item in ancestors if item.run_type == "tool"), None)
            root = tracer.run_map.get(str(run.trace_id))
            targets = {item.id: item for item in (run, source, tool, root) if item is not None}
            targets.update({item.id: item for item in ancestors if _subagent_boundary(tracer, item)})
            for target in targets.values():
                lifecycle.queue_policy_findings(tracer, target.id, findings)


def persist_policy_findings(tracer, run_id, findings):
    with _TAG_LOCK:
        run = tracer.run_map.get(str(run_id))
        if run is not None:
            _metadata(run, security_signals=findings)
            _add_tags(run, *(item["tag"] for item in findings))


def tag_authorization(config, actor, result):
    """Record the transaction's final decision, never intermediate FGA checks."""
    with _TAG_LOCK:
        for _, run in _active_runs(config):
            run.tags = [tag for tag in (run.tags or [])
                        if tag not in {"fga-allow", "fga-deny"} and not tag.startswith("auth-decision:")]
            _add_tags(run, "fga-" + result["decision"], "auth-decision:" + result["decision"])
            _metadata(run, **actor, auth_decision=result["decision"], auth_reason_code=result["reason_code"])


def tag_request(config, decision, reason_code="granted"):
    with _TAG_LOCK:
        for tracer, run in _active_runs(config):
            root = tracer.run_map.get(str(run.trace_id))
            for target in {run.id: run, **({root.id: root} if root else {})}.values():
                _outcome(target, "request", decision)
                _metadata(target, request_reason_code=reason_code)


def tag_review(config, outcome, decision=None):
    with _TAG_LOCK:
        for _, run in _active_runs(config):
            _outcome(run, "review", outcome)
            if decision is not None:
                _metadata(run, approval_decision=decision)
                _outcome(run, "approval", {"approve": "approved", "deny": "denied", "conditional": "held"}[decision])


def _ancestors(tracer, run):
    while run is not None:
        yield run
        run = tracer.run_map.get(str(run.parent_run_id))


def _subagent_boundary(tracer, run):
    metadata = run.extra.get("metadata", {})
    if metadata.get("subagent_boundary"):
        return True
    # The SDK stamps its child graph with lc_agent_name. Descendants inherit it;
    # only the transition is the owning graph, independent of display names.
    name = metadata.get("lc_agent_name")
    parent = tracer.run_map.get(str(run.parent_run_id))
    return bool(name and name == metadata.get("subagent_type") and
                (parent is None or parent.extra.get("metadata", {}).get("lc_agent_name") != name))


def mark_subagent(config, kind):
    with _TAG_LOCK:
        for _, run in _active_runs(config):
            _metadata(run, subagent_boundary=True, subagent_type=kind, tool_name="task")


def tag_subagent_outcome(config, outcome):
    with _TAG_LOCK:
        for tracer, run in _active_runs(config):
            for ancestor in _ancestors(tracer, run):
                if _subagent_boundary(tracer, ancestor):
                    _outcome(ancestor, "subagent", outcome)
                    if ancestor.run_type == "tool":
                        _outcome(ancestor, "tool", outcome)


def tag_execution_event(config, current_run, event_name, data):
    """Success and review states are independent of unsuccessful-call aggregates."""
    if event_name not in {"tool_completed", "hitl_request", "hitl_response"}:
        return
    with _TAG_LOCK:
        for tracer, current in _active_runs(config, current_run):
            tool = next((r for r in _ancestors(tracer, current) if r.run_type == "tool"), None)
            if tool is None:
                continue
            if event_name == "tool_completed":
                status = data.get("status")
                _outcome(tool, "tool", {"rejected": "rejected", "pending_secondary_approval": "held"}.get(status, "completed"))
                _metadata(tool, tool_name=data["tool_name"])
                continue
            approval = "pending" if event_name == "hitl_request" else {
                "approve": "approved", "deny": "denied", "conditional": "held",
            }[data["decision"]]
            root = tracer.run_map.get(str(tool.trace_id))
            targets = {tool.id: tool, **({root.id: root} if root else {})}
            for ancestor in _ancestors(tracer, tool):
                if _subagent_boundary(tracer, ancestor):
                    targets[ancestor.id] = ancestor
            for target in targets.values():
                _outcome(target, "approval", approval)
                _metadata(target, approval_id=data["approval_id"])
            if event_name == "hitl_request":
                _outcome(tool, "tool", "waiting")


def tag_tool_rejection(config, current_run, event_name, data):
    """Annotate active tool/root runs; their normal end callbacks persist this.

    The original rejection tags are backward-compatible unsuccessful-tool
    filters. Execution failures also get explicit failure tags, not FGA labels.

    Modern RunTree no longer populates parent_run. Resolve live ancestors through
    BaseTracer.run_map, as LangChain's own RunnableConfig context binding does.
    This also works with the offline collector, without creating a remote client.
    No config tags/metadata are mutated: later siblings must not inherit outcomes.
    """
    category = REJECTION_CATEGORIES.get(event_name)
    if category is None:
        return
    if event_name == "tool_rejected":
        category = data.get("rejection_category", category)
        if category not in {"sql-validation", "input-validation", "approval"}:
            return
    # Rejection events can originate in worker threads. Only updates to in-memory
    # trace annotations are serialized; no network or tool work occurs here.
    with _TAG_LOCK:
        for tracer, current in _active_runs(config, current_run):
            run = next((r for r in _ancestors(tracer, current) if r.run_type == "tool"), None)
            if run is None:
                continue
            metadata = run.extra.get("metadata", {})
            if metadata.get("tool_rejected"):
                continue  # One rejected call, even if more than one event reports it.
            detail = {
                "tool_name": data["tool_name"],
                "call_id": data.get("call_id", ""),
                "reason_code": data.get("reason_code", "resource_unavailable"),
                "rejection_category": category,
            }
            kind = data.get("subagent_type") or metadata.get("subagent_type")
            if kind:
                detail["subagent_type"] = kind
            # Tracers can initially share an input metadata dictionary. Detach
            # before writing outcomes so another collector/config is untouched.
            run.extra = {**run.extra, "metadata": dict(metadata)}
            _add_tags(run, "tool-rejected", "rejection-" + category)
            run.add_metadata({"tool_rejected": True, **detail})
            _outcome(run, "tool", "error" if category == "execution" else "denied" if category == "authorization" else "rejected")
            if category == "execution":
                _add_tags(run, "tool-failed", "failure-execution")
                run.add_metadata({"tool_failed": True})

            # Nested failures belong to the leaf call. Mark the owning native
            # task/subagent as containing that rejection, without counting it
            # again as another rejected tool call.
            subagent_rejected = False
            for ancestor in _ancestors(tracer, run):
                if _subagent_boundary(tracer, ancestor):
                    ancestor.extra = {**ancestor.extra, "metadata": dict(ancestor.extra.get("metadata", {}))}
                    _add_tags(ancestor, "subagent-rejected", "contains-tool-rejection", "rejection-" + category)
                    ancestor.add_metadata({"contains_tool_rejection": True})
                    if category == "execution":
                        _add_tags(ancestor, "subagent-failed", "contains-tool-failure", "failure-execution")
                        _metadata(ancestor, contains_tool_failure=True)
                    subagent_rejected = True

            root = tracer.run_map.get(str(run.trace_id))
            if root is None or root.id == run.id:
                continue
            summary = root.extra.get("metadata", {})
            details = [*summary.get("tool_rejections", [])]
            if len(details) < MAX_REJECTION_DETAILS:
                details.append(detail)
            count = summary.get("rejected_tool_call_count", 0) + 1
            root.extra = {**root.extra, "metadata": dict(summary)}
            _add_tags(root, "contains-tool-rejection", "rejection-" + category)
            if subagent_rejected:
                _add_tags(root, "contains-subagent-rejection")
            if category == "execution":
                _add_tags(root, "contains-tool-failure", "failure-execution")
                _metadata(root, contains_tool_failure=True)
            root.add_metadata({
                "contains_tool_rejection": True,
                "rejected_tool_call_count": count,
                "rejected_tools": sorted(set(summary.get("rejected_tools", [])) | {detail["tool_name"]}),
                "rejection_categories": sorted(set(summary.get("rejection_categories", [])) | {category}),
                "tool_rejections": details,
                "rejection_details_truncated": count > MAX_REJECTION_DETAILS,
            })
