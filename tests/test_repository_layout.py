"""Keep launchers and trusted asset paths working after repository reorganization."""

import ast
from contextlib import chdir, redirect_stdout
import io
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import httpx

from demo.paths import PACKAGE_DIR, REPO_ROOT, SKILLS_DIR, TRACE_DIR, WEB_DIR
from demo.server import create_app
from demo.skills import SKILLS
from demo.store import CustomerStore
from scripts import trace_batch
from tests.test_customer_operations import ScriptedCustomerModel


class RepositoryLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_assets_and_skills_resolve_from_foreign_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp, chdir(tmp):
            store = CustomerStore()
            self.assertEqual(set(store.skills._content), set(SKILLS))
            self.assertTrue(all(store.skills._content.values()))
            self.assertEqual(SKILLS_DIR, PACKAGE_DIR / "playbooks")
            self.assertTrue(all((SKILLS_DIR / key / "SKILL.md").is_file() for key in SKILLS))
            app = create_app(model=ScriptedCustomerModel(), store=store)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                response = await client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])
            self.assertEqual(response.text, (WEB_DIR / "index.html").read_text())

    def test_root_launchers_delegate_without_path_manipulation(self):
        with tempfile.TemporaryDirectory() as tmp, chdir(tmp):
            with patch("demo.server.main") as serve:
                runpy.run_path(str(REPO_ROOT / "server.py"), run_name="__main__")
                serve.assert_called_once_with()
            with patch("scripts.trace_batch.main", return_value=0) as generate:
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(REPO_ROOT / "generate_traces.py"), run_name="__main__")
                self.assertEqual(result.exception.code, 0)
                generate.assert_called_once_with()

    def test_default_artifact_directory_is_repo_relative(self):
        with tempfile.TemporaryDirectory() as tmp, chdir(tmp), redirect_stdout(io.StringIO()), \
                patch.object(trace_batch, "run_batch", return_value={"unexpected_cases": []}) as run:
            self.assertEqual(trace_batch.main(["--count", "1"]), 0)
            self.assertEqual(run.call_args.args[1].parent, TRACE_DIR)
            self.assertEqual(TRACE_DIR, REPO_ROOT / "trace_batches")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_runtime_does_not_import_experiments_scripts_or_tests(self):
        for source in PACKAGE_DIR.glob("*.py"):
            for node in ast.walk(ast.parse(source.read_text())):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                           [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    with self.subTest(file=source.name, module=module):
                        self.assertNotIn((module or "").split(".")[0], {"examples", "local", "scripts", "tests"})


if __name__ == "__main__":
    unittest.main()
