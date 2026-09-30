"""Fictional analytics service backed by short-lived, scoped SQLite databases.

The model never queries a shared tenant database. Only authorized tables for one
account are copied into each connection; SQL predicates are not an auth boundary.
"""

from __future__ import annotations

from contextlib import closing
from datetime import date
import json
import re
import sqlite3

from langchain_core.runnables import RunnableConfig, RunnableLambda


SCHEMAS = {
    "usage_daily": {
        "description": "Daily product adoption and API reliability, September 1–28, 2026. Seats are daily snapshots, not additive.",
        "columns": {"id": "TEXT", "tenant_id": "TEXT", "account_id": "TEXT", "usage_date": "TEXT",
                    "active_seats": "INTEGER", "licensed_seats": "INTEGER",
                    "api_requests": "INTEGER", "error_requests": "INTEGER"},
    },
    "invoices": {
        "description": "Fictional July–September invoices. Amounts are integer cents in USD. Overdue means unpaid and due before the demo date.",
        "columns": {"id": "TEXT", "tenant_id": "TEXT", "account_id": "TEXT", "issued_date": "TEXT",
                    "due_date": "TEXT", "amount_cents": "INTEGER", "currency": "TEXT", "status": "TEXT"},
    },
    "service_incidents": {
        "description": "Account-specific service incidents; impact_minutes is observed impact, not an SLA credit entitlement.",
        "columns": {"id": "TEXT", "tenant_id": "TEXT", "account_id": "TEXT", "started_at": "TEXT",
                    "service": "TEXT", "severity": "TEXT", "status": "TEXT",
                    "impact_minutes": "INTEGER", "summary": "TEXT"},
    },
}
ANALYTICS_TOOLS = ("get_analytics_schema", "query_customer_analytics", "analyze_customer_data")
MAX_ROWS = 100
MAX_RESULT_BYTES = 32_000
SAFE_FUNCTIONS = frozenset({
    "abs", "avg", "coalesce", "count", "date", "ifnull", "julianday", "length", "like",
    "lower", "max", "min", "nullif", "round", "strftime", "substr", "substring", "sum",
    "total", "trim", "upper",
})


def dataset_id(account_id, name):
    return account_id.replace("account:", "dataset:") + "/" + name


class QueryRejected(ValueError):
    """Deliberately generic: no SQLite errors, paths, or hidden schema names."""

    def __init__(self):
        super().__init__("Query rejected. Use one bounded SELECT over the authorized schema; no data was changed.")


class CustomerAnalytics:
    def __init__(self, store):
        self.store = store
        self._rows = {name: [] for name in SCHEMAS}
        self._seed()

    @property
    def row_count(self):
        return sum(len(rows) for rows in self._rows.values())

    def _seed(self):
        for account in self.store.accounts.values():
            aid, tenant = account["id"], account["tenant_id"]
            index = int(account["reference"].split("-")[-1]) - 100
            tenant_index = list(self.store.tenants).index(tenant)
            for day in range(1, 29):
                falling = index == 0 and day > 14
                active = (590 if falling else 820) if index == 0 else (300 + day * 4 if index == 1 else 110 + day * 2)
                requests = (93_000 if falling else 120_000) if index == 0 else (45_000 + day * 300)
                self._rows["usage_daily"].append((
                    f"usage:{tenant}/{index + 100}-202609{day:02}", tenant, aid, str(date(2026, 9, day)),
                    active + tenant_index * 10, (1000, 500, 750)[index], requests,
                    1800 if falling else 120,
                ))
            for month in (7, 8, 9):
                status = "overdue" if month == 9 and index == 0 else "open" if month == 9 and index == 2 else "paid"
                self._rows["invoices"].append((
                    f"invoice:{tenant}/{index + 100}-2026{month:02}", tenant, aid,
                    f"2026-{month:02}-01", "2026-10-05" if status == "open" else f"2026-{month:02}-18",
                    (2_400_000, 900_000, 1_600_000)[index] + tenant_index * 100_000, "USD", status,
                ))
            for number in (1, 2):
                self._rows["service_incidents"].append((
                    f"incident:{tenant}/{index + 100}-{number}", tenant, aid,
                    f"2026-09-{16 if number == 1 else 22}T09:00:00Z",
                    "Identity gateway" if number == 1 else "Regional failover",
                    "critical" if index == 0 and number == 1 else "medium" if index == 0 else "low",
                    "monitoring" if index == 0 and number == 1 else "investigating" if index == 0 else "resolved",
                    (47 if number == 1 else 18) if index == 0 else 0,
                    ("Intermittent SSO errors; mitigation active, permanent fix pending." if number == 1
                     else "Failover validation incomplete; service owners investigating.") if index == 0
                    else "Scheduled validation completed; no customer impact recorded.",
                ))

    def schema(self, actor, reference, *, datasets=None, emit=None, config=None):
        scope = self.store.authorization.authorize(
            actor, "get_analytics_schema", reference, datasets=datasets, emit=emit, config=config,
        )
        return {"account_id": scope["account_id"], "dialect": "SQLite",
                "demo_reference_date": "2026-09-28", "max_rows": MAX_ROWS,
                "tables": {name: {**SCHEMAS[name], "source_id": dataset_id(scope["account_id"], name)}
                           for name in scope["datasets"]}}

    def query(self, actor, reference, datasets, sql, *, emit=None, config=None):
        def execute(_request, config: RunnableConfig):
            scope = self.store.authorization.authorize(
                actor, "query_customer_analytics", reference, datasets=datasets, emit=emit, config=config,
            )
            try:
                result = self._execute(scope["account_id"], scope["datasets"], sql)
            except QueryRejected:
                if emit:
                    emit("sql_query_rejected", resource=scope["account_id"], reason_code="unsafe_or_invalid_sql")
                raise
            if emit:
                emit("sql_query_completed", resource=scope["account_id"], row_count=result["row_count"],
                     truncated=result["truncated"], source_ids=result["source_ids"])
            return result

        child_config = {**(config or {}), "run_name": "analytics.execute_read_only_query"}
        child_config.pop("run_id", None)
        return RunnableLambda(execute, name="analytics.execute_read_only_query").invoke(
            {"account_reference": reference, "datasets": datasets, "sql": sql}, config=child_config,
        )

    def _execute(self, account_id, datasets, sql):
        if not isinstance(sql, str) or not 1 <= len(sql) <= 6000 or not re.match(r"^\s*(SELECT|WITH)\b", sql, re.I):
            raise QueryRejected()
        with closing(sqlite3.connect(":memory:")) as db:
            # All identifiers below come from the server-owned schema allowlist.
            for name in datasets:
                columns = SCHEMAS[name]["columns"]
                db.execute(f"CREATE TABLE {name} (" + ", ".join(f"{key} {kind}" for key, kind in columns.items()) + ")")
                db.executemany(f"INSERT INTO {name} VALUES (" + ",".join("?" for _ in columns) + ")",
                               [row for row in self._rows[name] if row[2] == account_id])
            db.commit()
            db.execute("PRAGMA query_only = ON")
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 64_000)
            db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 6000)
            db.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 40)
            db.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 40)
            db.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 10)
            db.setlimit(sqlite3.SQLITE_LIMIT_VDBE_OP, 25_000)
            reads = set()

            def authorize(action, arg1, arg2, database, trigger):
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                # SQLite reports database=None, column="" for COUNT(*) and
                # other table reads that do not access a particular column.
                if (action == sqlite3.SQLITE_READ and arg1 in datasets
                        and (database == "main" or database is None and arg2 == "")):
                    reads.add(arg1)
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in SAFE_FUNCTIONS:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            ticks = 0

            def budget():
                nonlocal ticks
                ticks += 1
                return int(ticks > 100)  # Bound even aggregate/cross-join queries before their first row.

            db.set_authorizer(authorize)
            db.set_progress_handler(budget, 1000)
            try:
                cursor = db.execute(sql)  # One statement; generated SQL is intentionally untrusted.
                columns = [item[0] for item in cursor.description]
                rows = cursor.fetchmany(MAX_ROWS + 1)
                if not reads or len(json.dumps([columns, rows])) > MAX_RESULT_BYTES:
                    raise QueryRejected()
            except (sqlite3.Error, MemoryError, TypeError, OverflowError):
                raise QueryRejected() from None
            return {"account_id": account_id, "sql": sql, "columns": columns,
                    "rows": rows[:MAX_ROWS], "row_count": min(len(rows), MAX_ROWS),
                    "truncated": len(rows) > MAX_ROWS, "max_rows": MAX_ROWS,
                    "source_ids": [dataset_id(account_id, name) for name in sorted(reads)],
                    "demo_reference_date": "2026-09-28", "read_only": True}
