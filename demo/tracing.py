"""Shared outcome tags for failed/rejected tools, preserving reason categories."""

from threading import Lock

from langchain_core.tracers.base import BaseTracer


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
    callbacks = config.get("callbacks")
    handlers = callbacks if isinstance(callbacks, (list, tuple)) else getattr(callbacks, "handlers", [])
    current_id = current_run.id if current_run is not None else getattr(callbacks, "parent_run_id", None)
    if current_id is None:
        return
    # Rejection events can originate in worker threads. Only updates to in-memory
    # trace annotations are serialized; no network or tool work occurs here.
    with _TAG_LOCK:
        for tracer in handlers:
            if not isinstance(tracer, BaseTracer):
                continue
            run = tracer.run_map.get(str(current_id))
            while run is not None and run.run_type != "tool":
                run = tracer.run_map.get(str(run.parent_run_id))
            if run is None:
                continue
            metadata = run.extra.get("metadata", {})
            if metadata.get("tool_rejected"):
                continue  # One rejected call, even if more than one event reports it.
            detail = {
                "tool_name": run.name.removeprefix("tools."),
                "call_id": data.get("call_id", ""),
                "reason_code": data.get("reason_code", "resource_unavailable"),
                "rejection_category": category,
            }
            # Tracers can initially share an input metadata dictionary. Detach
            # before writing outcomes so another collector/config is untouched.
            run.extra = {**run.extra, "metadata": dict(metadata)}
            _add_tags(run, "tool-rejected", "rejection-" + category)
            run.add_metadata({"tool_rejected": True, **detail})
            if category == "execution":
                _add_tags(run, "tool-failed", "failure-execution")
                run.add_metadata({"tool_failed": True})

            # Nested failures belong to the leaf call. Mark the owning native
            # task/subagent as containing that rejection, without counting it
            # again as another rejected tool call.
            ancestor = run
            subagent_rejected = False
            while ancestor is not None:
                if ancestor.name.startswith("tools.task.") or ancestor.name in {
                    "billing-review", "support-escalation", "renewal-planning", "subagents.dispatch_task",
                }:
                    ancestor.extra = {**ancestor.extra, "metadata": dict(ancestor.extra.get("metadata", {}))}
                    _add_tags(ancestor, "subagent-rejected", "contains-tool-rejection", "rejection-" + category)
                    ancestor.add_metadata({"contains_tool_rejection": True, "subagent_outcome": "rejected"})
                    if category == "execution":
                        _add_tags(ancestor, "subagent-failed", "contains-tool-failure", "failure-execution")
                    subagent_rejected = True
                ancestor = tracer.run_map.get(str(ancestor.parent_run_id))

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
            root.add_metadata({
                "contains_tool_rejection": True,
                "rejected_tool_call_count": count,
                "rejected_tools": sorted(set(summary.get("rejected_tools", [])) | {detail["tool_name"]}),
                "rejection_categories": sorted(set(summary.get("rejection_categories", [])) | {category}),
                "tool_rejections": details,
                "rejection_details_truncated": count > MAX_REJECTION_DETAILS,
            })
