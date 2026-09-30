"""Native Deep Agents hooks, progressive disclosure, and FGA read boundaries."""

import asyncio
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import langsmith as ls
from deepagents.middleware.skills import SkillsMiddleware
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from pydantic import ValidationError

from customer_agent import SPECS, SkillReadRequest, build_customer_graph, response_text, stream_turn
from customer_skills import AuthorizedSkillsBackend, SKILLS
from customer_store import AccessDenied, CustomerStore
from test_customer_observability import StreamingTestModel, all_runs
from test_customer_operations import ScriptedCustomerModel

SQL_PATH = "/skills/sql-analysis/SKILL.md"
SQL_BODY_MARKER = "Amounts are integer cents"


class SkillBackendTests(unittest.TestCase):
    def setUp(self):
        self.tracing = ls.tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.store = CustomerStore()
        self.actor = self.store.actor("northstar", "user:northstar/csm")
        self.backend = AuthorizedSkillsBackend(self.store.skills, self.actor)

    def test_exact_virtual_paths_only_and_no_request_time_disk_reads(self):
        paths = ["/etc/passwd", "/skills/../sql-analysis/SKILL.md", "agent_skills/sql-analysis/SKILL.md",
                 "https://example.invalid/SKILL.md", "/skills/unknown/SKILL.md", "/skills/sql-analysis/SKILL.md/extra",
                 "/skills/sql-analysis/%2e%2e/SKILL.md", "/skills/beacon/sql-analysis/SKILL.md"]
        with patch.object(Path, "read_text", side_effect=AssertionError("Unexpected OS read")):
            self.assertIn(SQL_BODY_MARKER, self.store.skills.read(self.actor, SQL_PATH)["content"])
            for path in paths:
                with self.subTest(path=path), self.assertRaises(AccessDenied):
                    self.store.skills.read(self.actor, path)
            self.assertTrue(all(item.error == "permission_denied" for item in self.backend.download_files(paths)))
            self.assertEqual(self.backend.ls("/").error, "permission_denied")

    def test_user_agent_loader_and_reader_executor_revocations_block_full_read(self):
        grants = [
            ("team:northstar/success#member", "reader", "skill:northstar/sql-analysis"),
            (self.actor.agent_id, "reader", "skill:northstar/sql-analysis"),
            (self.actor.agent_id, "executor", "tool:northstar/load_agent_skills"),
            (self.actor.agent_id, "executor", "tool:northstar/read_file"),
            (self.actor.user_id, "member", "tenant:northstar"),
            (self.actor.agent_id, "member", "tenant:northstar"),
        ]
        for grant in grants:
            with self.subTest(grant=grant):
                self.store.fga.delete_tuple(*grant)
                try:
                    with self.assertRaises(AccessDenied):
                        self.store.skills.read(self.actor, SQL_PATH)
                finally:
                    self.store.fga.write_tuple(*grant)

    def test_reads_are_bounded_and_paginated(self):
        first = self.store.skills.read(self.actor, SQL_PATH, limit=2)
        self.assertEqual((first["start_line"], first["end_line"], first["next_offset"]), (1, 2, 2))
        second = self.store.skills.read(self.actor, SQL_PATH, offset=2)
        self.assertEqual(second["start_line"], 3)
        self.assertIsNone(second["next_offset"])
        self.assertEqual(first["content"] + "\n" + second["content"], self.store.skills.read(self.actor, SQL_PATH)["content"])
        for window in ({"offset": -1}, {"limit": 0}, {"limit": 1001}, {"offset": 10001}):
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.store.skills.read(self.actor, SQL_PATH, **window)

    def test_no_general_filesystem_shell_or_identity_override_tools(self):
        self.assertFalse(set(SPECS) & {"execute", "write_file", "edit_file", "ls", "glob", "grep"})
        with self.assertRaises(NotImplementedError):
            self.backend.write(SQL_PATH, "replacement")
        with self.assertRaises(ValidationError):
            SkillReadRequest(file_path=SQL_PATH, tenant_id="beacon")


class SkillsMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tracing = ls.tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.store = CustomerStore()
        self.actor = self.store.actor("northstar", "user:northstar/csm")
        self.collector = RunCollectorCallbackHandler()
        self.config = {"run_name": "customer_operations.turn", "callbacks": [self.collector], "configurable": {
            "thread_id": uuid.uuid4().hex, "turn_id": uuid.uuid4().hex,
            "tenant_id": "northstar", "user_id": self.actor.user_id,
            "agent_id": self.actor.agent_id, "invocation_mode": "tool",
        }}

    async def run_turn(self, model=None, *, graph=None, extra=None):
        graph = graph or build_customer_graph(self.store, model)
        data = {"messages": [HumanMessage(content="Explain the SQL skill instructions")], **(extra or {})}
        return [event async for event in stream_turn(graph, data, self.config)]

    async def test_real_sdk_discovery_and_model_hooks_build_metadata_only_prompt(self):
        self.assertIsInstance(self.store.skills.middleware(self.actor), SkillsMiddleware)
        hooks = []
        discover = SkillsMiddleware.abefore_agent
        wrap_model = SkillsMiddleware.awrap_model_call

        async def observed_discovery(middleware, *args):
            hooks.append("discovery")
            return await discover(middleware, *args)

        async def observed_model(middleware, *args):
            hooks.append("model")
            return await wrap_model(middleware, *args)

        model = ScriptedCustomerModel()
        with patch.object(SkillsMiddleware, "abefore_agent", observed_discovery), \
             patch.object(SkillsMiddleware, "awrap_model_call", observed_model):
            events = await self.run_turn(model)
        self.assertEqual(hooks, ["discovery", "model"])
        system = response_text(model.seen[0][0].content)
        self.assertIn(SQL_PATH, system)
        self.assertIn("Analyze customer adoption", system)  # Parsed frontmatter description.
        self.assertNotIn(SQL_BODY_MARKER, system)
        self.assertNotIn("# Scoped SQL analysis", system)
        discovered, = [e["data"] for e in events if e["event"] == "skills_discovered"]
        self.assertEqual(set(discovered["skill_ids"]), set(SKILLS))
        self.assertEqual(discovered["middleware"], "deepagents.SkillsMiddleware")
        self.assertNotIn("skill_instructions_loaded", [e["event"] for e in events])

    async def test_full_body_enters_context_only_after_governed_read_tool(self):
        model = ScriptedCustomerModel([("read_file", {"file_path": SQL_PATH})])
        events = await self.run_turn(model)
        self.assertEqual(len(model.seen), 2)
        self.assertNotIn(SQL_BODY_MARKER, response_text(model.seen[0][0].content))
        tool_result, = [message for message in model.seen[1] if isinstance(message, ToolMessage)]
        self.assertIn(SQL_BODY_MARKER, json.loads(tool_result.content)["content"])
        names = [e["event"] for e in events]
        self.assertLess(names.index("skills_discovered"), names.index("skill_instructions_loaded"))
        self.assertEqual(names.count("skills_discovered"), 2)

    async def test_metadata_refreshes_after_revocation_in_same_conversation(self):
        model = ScriptedCustomerModel([], [("read_file", {"file_path": SQL_PATH})])
        graph = build_customer_graph(self.store, model)
        await self.run_turn(graph=graph)
        self.store.fga.delete_tuple(self.actor.agent_id, "reader", "skill:northstar/sql-analysis")
        events = await self.run_turn(graph=graph, extra={"skills_metadata": [
            {"name": "sql-analysis", "description": "SPOOFED INSTRUCTIONS", "path": SQL_PATH},
        ]})
        self.assertIn(SQL_PATH, response_text(model.seen[0][0].content))
        for invocation in model.seen[1:]:
            system = response_text(invocation[0].content)
            self.assertNotIn(SQL_PATH, system)
            self.assertNotIn("SPOOFED INSTRUCTIONS", system)
        self.assertIn("tool_denied", [e["event"] for e in events])
        self.assertNotIn("skill_instructions_loaded", [e["event"] for e in events])
        self.assertNotIn(SQL_BODY_MARKER, json.dumps(events))

    async def test_removing_loader_grant_omits_all_skills_without_stopping_chat(self):
        self.store.fga.delete_tuple(self.actor.agent_id, "executor", "tool:northstar/load_agent_skills")
        model = ScriptedCustomerModel()
        events = await self.run_turn(model)
        discovered, = [e["data"] for e in events if e["event"] == "skills_discovered"]
        self.assertEqual(discovered["skill_ids"], [])
        self.assertNotIn(SQL_PATH, response_text(model.seen[0][0].content))
        self.assertIn("agent_response", [e["event"] for e in events])

    async def test_reader_revocation_between_discovery_and_read_is_enforced(self):
        await self.store.skills.discover(self.actor)
        self.store.fga.delete_tuple(self.actor.agent_id, "executor", "tool:northstar/read_file")
        events = await self.run_turn(ScriptedCustomerModel([("read_file", {"file_path": SQL_PATH})]))
        self.assertIn("tool_denied", [e["event"] for e in events])
        self.assertNotIn("skill_instructions_loaded", [e["event"] for e in events])
        self.assertNotIn(SQL_BODY_MARKER, json.dumps(events))

    async def test_concurrent_tenant_and_persona_discovery_are_isolated(self):
        actors = [self.actor, self.store.actor("northstar", "user:northstar/support"),
                  self.store.actor("beacon", "user:beacon/csm")]
        self.store.fga.delete_tuple("team:northstar/success#member", "reader", "skill:northstar/sql-analysis")
        self.store.fga.delete_tuple("team:northstar/support#member", "reader", "skill:northstar/approval-workflow")
        backends = []

        def make_backend(*args, **kwargs):
            backend = AuthorizedSkillsBackend(*args, **kwargs)
            backends.append(backend)
            return backend

        with patch("customer_skills.AuthorizedSkillsBackend", side_effect=make_backend):
            states = await asyncio.gather(*(self.store.skills.discover(actor) for actor in actors))
        self.assertEqual([{skill["name"] for skill in state["skills_metadata"]} for state in states], [
            set(SKILLS) - {"sql-analysis"}, set(SKILLS) - {"approval-workflow"}, set(SKILLS),
        ])
        self.assertEqual(len({id(backend) for backend in backends}), 3)

    async def test_descriptive_trace_nesting_and_authorization_phases(self):
        await self.run_turn(ScriptedCustomerModel([("read_file", {"file_path": SQL_PATH})]))
        runs = {run.id: run for run in all_runs(self.collector.traced_runs)}
        names = {run.name for run in runs.values()}
        self.assertTrue({"skills.discover_metadata", "skills.apply_model_middleware", "tools.read_file",
                         "skills.read_instructions", "authorization.verify_tenant"} <= names)
        self.assertFalse(names & {"Unnamed", "RunnableLambda", "skills.load_defaults"})
        read, = [run for run in runs.values() if run.name == "skills.read_instructions"]
        self.assertEqual(runs[read.parent_run_id].name, "tools.read_file")
        auth, = [run for run in read.child_runs if run.name == "authorization.authorize_transaction"]
        self.assertEqual(auth.extra["metadata"]["auth_scope"], "skill")
        self.assertEqual(auth.extra["metadata"]["auth_phase"], "read")
        self.assertEqual(auth.outputs["decision"], "allow")
        self.assertIn("authorization.verify_tenant", [run.name for run in auth.child_runs])
        discoveries = [run for run in runs.values() if run.name == "skills.discover_metadata"]
        for run in discoveries:
            self.assertEqual(runs[run.parent_run_id].name, "customer_operations.respond")
            self.assertTrue(all(child.extra["metadata"]["auth_phase"] == "discover" for child in run.child_runs))

    async def test_real_model_stream_stays_incremental_after_skill_read(self):
        gate = asyncio.Event()
        model = StreamingTestModel(gate=gate, batches=[
            [AIMessageChunk(content="", tool_call_chunks=[{"name": "read_file", "args": json.dumps({"file_path": SQL_PATH}),
                                                           "id": "skill-call", "index": 0}])],
            [AIMessageChunk(content="Loaded "), AIMessageChunk(content="the SQL instructions.")],
        ])
        graph = build_customer_graph(self.store, model)
        events = []

        async def consume():
            async for event in stream_turn(graph, {"messages": [HumanMessage(content="Read the SQL skill")]}, self.config):
                events.append(event)
                if event["event"] == "response_delta" and not gate.is_set():
                    self.assertEqual(model.completed, 1)  # Final model response has not completed.
                    gate.set()

        try:
            await asyncio.wait_for(consume(), timeout=10)
        finally:
            gate.set()
        deltas = [e["data"]["content"] for e in events if e["event"] == "response_delta"]
        self.assertEqual(deltas, ["Loaded ", "the SQL instructions."])
        self.assertIn("skill_instructions_loaded", [e["event"] for e in events])
        generations = [run for run in all_runs(self.collector.traced_runs) if run.name == "customer_operations.generate_response"]
        self.assertEqual(len(generations), 2)
        runs = {run.id: run for run in all_runs(self.collector.traced_runs)}
        self.assertTrue(all(runs[run.parent_run_id].name == "skills.apply_model_middleware" for run in generations))


if __name__ == "__main__":
    unittest.main()
