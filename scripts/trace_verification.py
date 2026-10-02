"""Shared review-trace checks, including historical schema-9 batches."""

import json


def review_filter(finished_at):
    return ('and(or(eq(name,"approvals.review"),'
            'and(eq(name,"authorization.authorize_transaction"),eq(metadata_key,"auth_phase"),'
            'eq(metadata_value,"inbox_decision"))),'
            f'lt(start_time,{json.dumps(finished_at)}))')


def verify_review(root, case, batch_id):
    assert root.end_time and root.parent_run_id is None
    assert root.outputs["decision"] == "allow"
    reviewer = case.get("reviewer") or f"user:{case['tenant_id']}/lead"
    assert root.inputs["actor"]["user_id"] == reviewer
    if root.name != "approvals.review":
        assert root.name == "authorization.authorize_transaction"
        return  # Old reviews only exposed the reviewer-authorization result.
    metadata = root.extra["metadata"]
    assert metadata["trace_schema_version"] == "10"
    assert metadata["interaction_phase"] == "review"
    assert metadata["conversation_id"] == metadata["thread_id"] == case["conversation_id"]
    assert metadata["approval_id"] == root.inputs["approval_id"] == case["approval_id"]
    assert metadata["sample_batch_id"] == batch_id
    assert metadata["sample_case_id"] == case["case_id"]
    assert metadata["reviewer_id"] == reviewer
    assert metadata["review_outcome"] == root.outputs["review_outcome"] == "accepted"
    assert metadata["approval_decision"] == root.outputs["approval_decision"] == case["decision"]
    assert metadata["request_run_id"]
    assert "review-outcome:accepted" in root.tags
    assert "approval-outcome:" + {"approve": "approved", "deny": "denied", "conditional": "held"}[case["decision"]] in root.tags
