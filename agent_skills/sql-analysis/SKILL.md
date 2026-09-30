---
name: sql-analysis
description: Analyze customer adoption, API reliability, invoices, and service incidents using scoped SQL. Use for analytics questions, SQL generation, billing totals, or usage and incident queries.
---

# Scoped SQL analysis

Use for product usage, invoices, and service-incident questions. Call
get_analytics_schema for the requested account before choosing tables or columns.
For natural-language analysis, delegate to analyze_customer_data with the exact
question and required dataset names. For supplied SQL or an explicit request to
query directly, use query_customer_analytics. Never change a supplied statement
to conceal an unsafe operation; the read-only service must validate it.

The database contains only the selected account and authorized datasets. Never
invent rows, query another account to bypass a denial, or treat a zero result as
proof about an inaccessible dataset. Dates use the fixed demo reference date,
not today's date. Amounts are integer cents; report currency and divide by 100
for money. Daily active seats are snapshots: use averages or compare periods,
not a sum of seats. Compute error rates as SUM(error_requests) / SUM(api_requests)
with floating-point arithmetic and a zero-denominator guard. Joining daily usage
to invoices can multiply amounts: aggregate each source before joining.

Cite returned dataset/row IDs, state the time window and any truncation, and show
the executed SQL when asked. A skill is guidance, not an authorization grant.
