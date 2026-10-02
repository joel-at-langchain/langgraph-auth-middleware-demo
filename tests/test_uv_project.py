"""Check the uv project without resolving dependencies or importing application code."""

import ast
from pathlib import Path
import re
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UvProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
        cls.project = cls.manifest["project"]
        cls.lock = tomllib.loads((ROOT / "uv.lock").read_text())

    def test_repo_runs_as_non_package_with_pinned_python(self):
        self.assertIs(self.manifest["tool"]["uv"]["package"], False)
        self.assertEqual((ROOT / ".python-version").read_text().strip(), "3.12")
        self.assertEqual(self.project["requires-python"], ">=3.12")
        self.assertEqual(self.lock["requires-python"], self.project["requires-python"])

    def test_lock_records_current_project_dependencies(self):
        project = next(p for p in self.lock["package"] if p["name"] == self.project["name"])
        self.assertEqual(project["source"], {"virtual": "."})
        self.assertEqual(project["version"], self.project["version"])
        requirements = {
            item["name"] + item.get("specifier", "")
            for item in project["metadata"]["requires-dist"]
        }
        self.assertEqual(requirements, set(self.project["dependencies"]))

    def test_third_party_imports_are_declared_directly(self):
        declared = {re.split(r"[<>=!~\[; ]", item, maxsplit=1)[0]
                    for item in self.project["dependencies"]}
        local = {"demo", "scripts", "tests", "examples", "server", "generate_traces"}
        sources = [ROOT / "server.py", ROOT / "generate_traces.py"]
        for directory in ("demo", "scripts", "tests", "examples"):
            sources.extend((ROOT / directory).rglob("*.py"))
        for source in sources:
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    name = module.split(".")[0]
                    if not name or name in sys.stdlib_module_names or name in local:
                        continue
                    distribution = "python-dotenv" if name == "dotenv" else name.replace("_", "-")
                    with self.subTest(file=str(source.relative_to(ROOT)), module=module):
                        self.assertIn(distribution, declared)


if __name__ == "__main__":
    unittest.main()
