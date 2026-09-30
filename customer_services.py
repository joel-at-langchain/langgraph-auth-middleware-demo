"""Read-only fictional services with fixed, clearly labeled failure fixtures."""

from langchain_core.runnables import RunnableConfig, RunnableLambda


SERVICE_TOOLS = {
    "get_billing_snapshot": {
        "service": "billing", "restricted": True,
        "description": "Read a fictional billing-service snapshot for an account. Requires billing-service reader access; does not charge or change anything.",
    },
    "get_support_sla_report": {
        "service": "support-sla", "restricted": False,
        "description": "Read a fictional support SLA report. The mock upstream can time out; report the actual tool result without inventing metrics.",
    },
    "get_crm_sync_status": {
        "service": "crm-sync", "restricted": False,
        "description": "Read the fictional CRM synchronization status. The mock upstream can be unavailable; this never initiates a sync.",
    },
    "get_usage_export_status": {
        "service": "usage-export", "restricted": False,
        "description": "Read a fictional usage-export manifest status. The mock upstream can return an invalid response. This does not create or download an export.",
    },
    "get_renewal_forecast": {
        "service": "renewal-forecast", "restricted": True,
        "description": "Read a fictional, precomputed renewal forecast. Requires forecast-service reader access; not a prediction about real customers.",
    },
}
FAULTS = {
    "get_support_sla_report": "service_timeout",
    "get_crm_sync_status": "service_unavailable",
    "get_usage_export_status": "invalid_service_response",
}


class MockServiceFailure(RuntimeError):
    def __init__(self, code):
        if code not in FAULTS.values():
            raise ValueError("Unknown mock service failure")
        self.code = code
        super().__init__(f"Fictional service fixture failed: {code}")


class CustomerServices:
    def __init__(self, store):
        self.store = store

    def read(self, actor, operation, reference, *, emit=None, config=None):
        def fetch(_request, config: RunnableConfig):
            scope = self.store.authorization.authorize(actor, operation, reference, emit=emit, config=config)
            if emit:
                emit("service_call_started", resource=scope["service_id"], mock_service=True)
            # Faults are server-owned fixture data, never model-controlled toggles.
            # Authorization always precedes either a result or a simulated fault.
            if scope["account_id"].endswith("/AC-100") and operation in FAULTS:
                if emit:
                    emit("service_call_failed", resource=scope["service_id"], reason_code=FAULTS[operation], mock_service=True)
                raise MockServiceFailure(FAULTS[operation])
            payloads = {
                "get_billing_snapshot": {"currency": "USD", "open_balance_cents": 900_000, "overdue_balance_cents": 0},
                "get_support_sla_report": {"period": "2026-09", "sla_met_pct": 99.8, "breached_cases": 0},
                "get_crm_sync_status": {"sync_status": "up_to_date", "last_sync_at": "2026-09-28T08:00:00Z"},
                "get_usage_export_status": {"export_status": "ready", "row_count": 28, "period": "2026-09-01/2026-09-28"},
                "get_renewal_forecast": {"forecast_category": "commit", "confidence": "fixture_only", "renewal_value_cents": 10_800_000},
            }
            result = {"account_id": scope["account_id"], "mock_service": True, "as_of": "2026-09-28",
                      "source_ids": [scope["service_id"] + "/" + scope["account_id"].split("/")[-1]], **payloads[operation]}
            if emit:
                emit("service_call_completed", resource=scope["service_id"], mock_service=True)
            return result

        name = "customer_services." + operation
        child = {**(config or {}), "run_name": name}
        child.pop("run_id", None)
        return RunnableLambda(fetch, name=name).invoke({"account_reference": reference}, config=child)
