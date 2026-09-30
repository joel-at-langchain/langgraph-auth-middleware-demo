"""
Simulated OpenFGA Store
=======================

Mirrors OpenFGA semantics — types, relations, relationship tuples, and a
``check()`` method that resolves implied relation hierarchies.

Types:    user, agent, document, tool, system
Relations: reader, writer, admin, executor, restricted_executor, delegate
Implied:   admin -> writer -> reader
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FGATuple:
    """A single relationship tuple in the FGA store."""

    user: str  # "user:alice" or "agent:research-bot"
    relation: str  # "reader", "writer", "executor", etc.
    object: str  # "document:report-2024", "tool:fetch_user_profile"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FGATuple):
            return NotImplemented
        return (
            self.user == other.user
            and self.relation == other.relation
            and self.object == other.object
        )

    def __hash__(self) -> int:
        return hash((self.user, self.relation, self.object))


@dataclass
class FGACheckResult:
    """Result of an authorization check."""

    allowed: bool
    tuple_matched: FGATuple | None
    reasoning: str
    requires_elevation: bool = False  # True for write/admin on sensitive resources
    elevation_level: str | None = None  # "writer" or "admin"
    grant_path: list[FGATuple] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FGA Store
# ---------------------------------------------------------------------------


class FGAStore:
    """In-memory authorization store with OpenFGA-style semantics."""

    # admin can do everything writer can; writer can do everything reader can.
    RELATION_HIERARCHY: dict[str, set[str]] = {
        "admin": {"writer", "reader"},
        "writer": {"reader"},
        "reader": set(),
        "executor": set(),
        "restricted_executor": set(),
    }

    def __init__(self) -> None:
        self._tuples: set[FGATuple] = set()
        # object_type -> set of relations that require HITL elevation
        self._elevation_rules: dict[str, set[str]] = {}

    # -- Tuple management ---------------------------------------------------

    def write_tuple(self, user: str, relation: str, object: str) -> None:
        """Add a relationship tuple to the store."""
        self._tuples.add(FGATuple(user=user, relation=relation, object=object))

    def delete_tuple(self, user: str, relation: str, object: str) -> None:
        """Remove a relationship tuple from the store."""
        self._tuples.discard(FGATuple(user=user, relation=relation, object=object))

    def list_tuples(
        self,
        user: str | None = None,
        relation: str | None = None,
        object: str | None = None,
    ) -> list[FGATuple]:
        """List tuples, optionally filtered by user, relation, or object."""
        results: list[FGATuple] = []
        for t in self._tuples:
            if user is not None and t.user != user:
                continue
            if relation is not None and t.relation != relation:
                continue
            if object is not None and t.object != object:
                continue
            results.append(t)
        return results

    # -- Elevation rules ----------------------------------------------------

    def add_elevation_rule(self, object_type: str, relations: set[str]) -> None:
        """Mark certain relations on an object type as requiring HITL elevation."""
        self._elevation_rules[object_type] = relations

    # -- Authorization check ------------------------------------------------

    def _implied_relations(self, relation: str) -> set[str]:
        """Return the set of relations implied by *relation* (including itself)."""
        implied = {relation}
        for r in self.RELATION_HIERARCHY.get(relation, set()):
            implied |= self._implied_relations(r)
        return implied

    def check(self, user: str, relation: str, object: str) -> FGACheckResult:
        """Check whether *user* holds *relation* on *object*.

        Handles implied hierarchies: if ``user`` has ``admin`` on the object
        and ``reader`` is requested, the check succeeds because admin implies
        writer implies reader.
        """
        path = self._grant_path(user, relation, object, set())
        if path:
            matched = path[-1]

            # Check elevation requirement
            object_type = object.split(":")[0] if ":" in object else object
            elevation_relations = self._elevation_rules.get(object_type, set())
            requires_elevation = relation in elevation_relations

            return FGACheckResult(
                allowed=True,
                tuple_matched=matched,
                reasoning=(
                    f"{user} has '{relation}' access on {object}"
                    + (f" (implied by '{matched.relation}')" if matched and matched.relation != relation else "")
                ),
                requires_elevation=requires_elevation,
                elevation_level=relation if requires_elevation else None,
                grant_path=path,
            )

        return FGACheckResult(
            allowed=False,
            tuple_matched=None,
            reasoning=f"{user} does not have '{relation}' access on {object}",
        )

    def _grant_path(self, user, relation, object, visited):
        """Resolve direct grants and team#member grants, with bounded traversal.

        This is deliberately a small FGA subset, not an OpenFGA interpreter.
        Return the actual membership/grant path for auditable explanations.
        """
        key = (user, relation, object)
        if key in visited or len(visited) >= 8:
            return []
        visited = visited | {key}
        candidates = sorted(
            self.list_tuples(object=object),
            key=lambda t: (t.relation != relation, t.user, t.relation),
        )
        for grant in candidates:
            if relation not in self._implied_relations(grant.relation):
                continue
            if grant.user == user:
                return [grant]
            if grant.user.startswith("team:") and grant.user.endswith("#member"):
                team = grant.user.removesuffix("#member")
                membership = self._grant_path(user, "member", team, visited)
                if membership:
                    return [*membership, grant]
        return []


# ---------------------------------------------------------------------------
# Tool-to-resource mapping
# ---------------------------------------------------------------------------

TOOL_RESOURCE_MAP: dict[str, dict[str, str]] = {
    "fetch_user_profile": {
        "object": "tool:fetch_user_profile",
        "relation": "executor",
        "sensitivity": "standard",
    },
    "search_documents": {
        "object": "tool:search_documents",
        "relation": "executor",
        "sensitivity": "standard",
    },
    "write_document": {
        "object": "document:report-2024",
        "relation": "writer",
        "sensitivity": "elevated",
        "resource_type": "document",
    },
    "delete_document": {
        "object": "document:report-2024",
        "relation": "admin",
        "sensitivity": "elevated",
        "resource_type": "document",
    },
}

# Agent capabilities use the same FGA tuple store as tools.  Keeping the
# mapping explicit makes it possible to audit parent -> child delegation
# without granting a parent arbitrary graph execution.
TOOL_RESOURCE_MAP["delegate_to_summary_agent"] = {
    "object": "agent:summary-bot",
    "relation": "delegate",
    "sensitivity": "standard",
    "resource_type": "agent",
}


# ---------------------------------------------------------------------------
# Factory: create and seed a store with demo data
# ---------------------------------------------------------------------------


def create_demo_store() -> FGAStore:
    """Return an :class:`FGAStore` pre-loaded with demo tuples and rules."""
    store = FGAStore()

    # -- alice --------------------------------------------------------------
    store.write_tuple("user:alice", "executor", "tool:fetch_user_profile")
    store.write_tuple("user:alice", "executor", "tool:search_documents")
    store.write_tuple("user:alice", "reader", "document:report-2024")
    store.write_tuple("user:alice", "writer", "document:report-2024")

    # -- bob ----------------------------------------------------------------
    store.write_tuple("user:bob", "reader", "document:report-2024")
    # bob has NO tool executor permissions

    # -- agent: research-bot ------------------------------------------------
    store.write_tuple("agent:research-bot", "executor", "tool:search_documents")
    store.write_tuple("agent:research-bot", "reader", "document:report-2024")
    store.write_tuple("agent:research-bot", "delegate", "agent:summary-bot")

    # -- agent: admin-bot ---------------------------------------------------
    store.write_tuple("agent:admin-bot", "admin", "document:report-2024")
    store.write_tuple("agent:admin-bot", "executor", "tool:fetch_user_profile")
    store.write_tuple("agent:admin-bot", "executor", "tool:search_documents")
    store.write_tuple("agent:admin-bot", "delegate", "agent:summary-bot")

    # -- elevation rules: writer and admin on documents require HITL --------
    store.add_elevation_rule("document", {"writer", "admin"})

    return store
