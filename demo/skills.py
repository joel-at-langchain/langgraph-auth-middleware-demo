"""Native Deep Agents skills middleware over an FGA-filtered read-only backend.

The existing LangGraph owns routing, streaming, and approval checkpoints. This
adapter invokes the SDK's discovery/model hooks without adding the rest of the
Deep Agents tool stack or copying its frontmatter/prompt implementation.
"""

from demo.paths import SKILLS_DIR

from deepagents.backends.protocol import BackendProtocol, FileDownloadResponse, LsResult, ReadResult
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.runtime import Runtime


SKILLS = {
    "account-briefing": "Evidence-backed account briefing",
    "sql-analysis": "Scoped SQL analysis",
    "approval-workflow": "Human-reviewed account changes",
}
SKILL_FILES = {f"/skills/{key}/SKILL.md": key for key in SKILLS}
SKILLS_PROMPT = """## Skills library
{skills_locations}{skills_load_warnings}

{skills_list}

These are summaries, not the full instructions. When a skill matches the task,
read its listed SKILL.md with read_file(file_path=..., limit=1000), then follow
the instructions. Already-read instructions remain in conversation history.
This library is read-only and contains only the listed files; do not attempt to
create skills, run scripts, or access other paths. Skills never grant tool or
data access. If a skill is unavailable, do not try alternate paths or identities.
"""


def trace_config(config, name):
    child = {**(config or {}), "run_name": name}
    child.pop("run_id", None)
    return child


class AuthorizedSkillsBackend(BackendProtocol):
    """An invocation-local backend. No model-controlled paths reach the OS."""

    def __init__(self, skills, actor, *, emit=None, config=None):
        self.skills, self.actor = skills, actor
        self.emit, self.config = emit, config

    def ls(self, path):
        if path != "/skills/":
            return LsResult(error="permission_denied")
        scope = self.skills.store.authorization.authorize_skills(self.actor, emit=self.emit, config=self.config)
        return LsResult(entries=[{"path": f"/skills/{key}", "is_dir": True} for key in scope["skills"]])

    def download_files(self, paths):
        from demo.store import AccessDenied

        responses = []
        for path in paths:
            try:
                scope = self.skills.store.authorization.authorize_skill_file(
                    self.actor, path, operation="load_agent_skills", emit=self.emit, config=self.config,
                )
                responses.append(FileDownloadResponse(path=path, content=self.skills._content[scope["skill_id"]].encode("utf-8")))
            except AccessDenied:
                responses.append(FileDownloadResponse(path=path, error="permission_denied"))
        return responses

    def read(self, file_path, offset=0, limit=100):
        if not isinstance(offset, int) or not isinstance(limit, int) or not 0 <= offset <= 10_000 or not 1 <= limit <= 1000:
            raise ValueError("Invalid skill read window")
        scope = self.skills.store.authorization.authorize_skill_file(
            self.actor, file_path, operation="read_file", emit=self.emit, config=self.config,
        )
        lines = self.skills._content[scope["skill_id"]].splitlines()
        window = lines[offset:offset + limit]
        if not window:
            return ReadResult(error="No lines in the requested range")
        end = offset + len(window)
        return ReadResult(file_data={"content": "\n".join(window), "encoding": "utf-8"},
                          start_line=offset + 1, end_line=end, total_lines=len(lines),
                          next_offset=end if end < len(lines) else None)

    # BackendProtocol defaults reject unsupported writes, edits, and uploads.
    # No filesystem, shell, write, or discovery tools are exposed to the model.


class DefaultSkills:
    def __init__(self, store):
        self.store = store
        # No request, model output, directory scan, or URL can choose a file.
        root = SKILLS_DIR
        self._content = {key: (root / key / "SKILL.md").read_text(encoding="utf-8") for key in SKILLS}

    def catalog(self, actor):
        return [{"id": key, "name": title} for key, title in SKILLS.items()
                if self.store.fga.check(actor.user_id, "reader", f"skill:{actor.tenant_id}/{key}").allowed
                and self.store.fga.check(actor.agent_id, "reader", f"skill:{actor.tenant_id}/{key}").allowed
                and self.store.fga.check(actor.agent_id, "executor", f"tool:{actor.tenant_id}/load_agent_skills").allowed]

    def middleware(self, actor, *, emit=None, config=None):
        return SkillsMiddleware(backend=AuthorizedSkillsBackend(self, actor, emit=emit, config=config),
                                sources=[("/skills/", "Customer operations")], system_prompt=SKILLS_PROMPT)

    async def discover(self, actor, *, emit=None, config=None):
        async def discover_metadata(_request, config: RunnableConfig):
            middleware = self.middleware(actor, emit=emit, config=config)
            # Fresh state is intentional: don't trust caller-supplied or cached
            # metadata after a tenant, persona, or permission change.
            result = await middleware.abefore_agent({"messages": [], "skills_metadata": None},
                                                   Runtime(context=actor), config)
            if emit:
                emit("skills_discovered", resource=f"tenant:{actor.tenant_id}",
                     skill_ids=[skill["name"] for skill in result["skills_metadata"]],
                     middleware="deepagents.SkillsMiddleware")
            return result

        return await RunnableLambda(discover_metadata, name="skills.discover_metadata").ainvoke(
            {"tenant_id": actor.tenant_id}, config=trace_config(config, "skills.discover_metadata"),
        )

    async def invoke_model(self, actor, model, bound_model, tools, system, messages, *, emit=None, config=None, subagents=None):
        state = await self.discover(actor, emit=emit, config=config)

        async def wrap_model(_request, config: RunnableConfig):
            middleware = self.middleware(actor, config=config)
            request = ModelRequest(model=model, system_message=system, messages=messages, tools=tools,
                                   state={**state, "messages": messages}, runtime=Runtime(context=actor))

            async def handler(modified_request):
                response = await bound_model.ainvoke([modified_request.system_message, *modified_request.messages],
                                                    config=trace_config(config, "customer_operations.generate_response"))
                return ModelResponse(result=[response])

            async def with_subagents(modified_request):
                if subagents is not None:
                    return await subagents.awrap_model_call(modified_request, handler)
                return await handler(modified_request)

            result = await middleware.awrap_model_call(request, with_subagents)
            return result.result[0]

        return await RunnableLambda(wrap_model, name="skills.apply_model_middleware").ainvoke(
            {"skill_ids": [skill["name"] for skill in state["skills_metadata"]]},
            config=trace_config(config, "skills.apply_model_middleware"),
        )

    def read(self, actor, file_path, *, offset=0, limit=1000, emit=None, config=None):
        def read_instructions(_request, config: RunnableConfig):
            result = AuthorizedSkillsBackend(self, actor, emit=emit, config=config).read(file_path, offset, limit)
            if result.error:
                raise ValueError(result.error)
            if emit:
                emit("skill_instructions_loaded", resource=f"skill:{actor.tenant_id}/{SKILL_FILES[file_path]}",
                     file_path=file_path, start_line=result.start_line, end_line=result.end_line)
            return {"file_path": file_path, "content": result.file_data["content"],
                    "start_line": result.start_line, "end_line": result.end_line,
                    "total_lines": result.total_lines, "next_offset": result.next_offset}

        return RunnableLambda(read_instructions, name="skills.read_instructions").invoke(
            {"file_path": file_path, "offset": offset, "limit": limit},
            config=trace_config(config, "skills.read_instructions"),
        )
