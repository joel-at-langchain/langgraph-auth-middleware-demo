"""Trusted repository paths, independent of the caller's working directory."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
WEB_DIR = REPO_ROOT / "web"
SKILLS_DIR = PACKAGE_DIR / "playbooks"
TRACE_DIR = REPO_ROOT / "trace_batches"
