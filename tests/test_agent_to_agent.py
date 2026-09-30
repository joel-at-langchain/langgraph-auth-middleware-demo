"""Graph-level tests for governed parent -> child-agent interactions."""

from __future__ import annotations

import asyncio
import unittest

from tests.legacy_support import ScriptedParentModel
from demo.fga import create_demo_store
from examples.legacy.langgraph_fga_governance import ALL_TOOLS, build_fga_graph, invoke_main_agent


class AgentToAgentTests(unittest.TestCase):
    def run_agent(
        self,
        message: str,
        *,
        agent_id: str = "agent:research-bot",
        invocation_mode: str = "tool",
    ) -> dict:
        graph = build_fga_graph(
            ALL_TOOLS,
            ScriptedParentModel(),
            create_demo_store(),
        )
        return asyncio.run(
            invoke_main_agent(
                graph,
                message,
                user_id="user:alice",
                agent_id=agent_id,
                invocation_mode=invocation_mode,
            )
        )

    def test_parent_agent_delegates_via_governed_tool(self) -> None:
        result = self.run_agent(
            "Summarize: Revenue grew 12% in Q3. Enterprise expansion led the increase."
        )

        self.assertIn("Summary:", result["response"])
        self.assertTrue(result["thread_id"].startswith("agent-"))
        self.assertEqual(
            [event["event"] for event in result["events"]],
            [
                "tool_intent",
                "fga_allow",
                "agent_call_started",
                "agent_call_completed",
            ],
        )

    def test_parent_agent_surfaces_governed_tool_denial(self) -> None:
        result = self.run_agent(
            "Summarize this restricted request.",
            agent_id="agent:restricted-bot",
        )

        self.assertIn("could not complete delegation", result["response"])
        self.assertEqual(
            [event["event"] for event in result["events"]],
            ["tool_intent", "fga_deny", "agent_call_denied"],
        )

    def test_parent_agent_handoff_uses_same_governance_boundary(self) -> None:
        result = self.run_agent(
            "The service is healthy. Latency improved after the rollout.",
            invocation_mode="handoff",
        )

        self.assertIn("Summary:", result["response"])
        self.assertEqual(
            [event["event"] for event in result["events"]],
            ["agent_call_started", "agent_call_completed"],
        )

    def test_parent_agent_handoff_denial_is_visible_to_caller(self) -> None:
        result = self.run_agent(
            "This handoff should be rejected.",
            agent_id="agent:restricted-bot",
            invocation_mode="handoff",
        )

        self.assertIn("Access denied", result["response"])
        self.assertEqual(
            [event["event"] for event in result["events"]],
            ["agent_call_denied"],
        )

    def test_parent_agent_rejects_unknown_invocation_mode(self) -> None:
        graph = build_fga_graph(
            ALL_TOOLS,
            ScriptedParentModel(),
            create_demo_store(),
        )
        with self.assertRaises(ValueError):
            asyncio.run(
                invoke_main_agent(
                    graph,
                    "Summarize this.",
                    user_id="user:alice",
                    agent_id="agent:research-bot",
                    invocation_mode="arbitrary",
                )
            )


if __name__ == "__main__":
    unittest.main()
