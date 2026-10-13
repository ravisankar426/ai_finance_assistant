"""Repository-level deliverables: layout, setup files, quality gates, docs (REQ-DEL, REQ-NFR)."""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib

import yaml

from src.core.config import PROJECT_ROOT

ROOT = PROJECT_ROOT


def _text(rel: str) -> str:
    return (ROOT / rel).read_text()


def test_prescribed_layout() -> None:
    """REQ-DEL-01: src/{agents,core,data,rag,web_app,utils,workflow}, tests, root files."""
    for d in ("agents", "core", "data", "rag", "web_app", "utils", "workflow"):
        assert (ROOT / "src" / d / "__init__.py").is_file(), d
    for f in ("config.yaml", "requirements.txt", "README.md", "pyproject.toml"):
        assert (ROOT / f).is_file(), f
    assert (ROOT / "tests").is_dir()


def test_sample_data_shipped() -> None:
    """REQ-DEL-02: sample portfolios, recorded market data, and sample conversations."""
    assert len(list((ROOT / "src/data/sample_portfolios").glob("*.csv"))) >= 3
    assert len(list((ROOT / "src/data/fixtures/market").rglob("*.json"))) >= 20
    assert (ROOT / "tests/evals/golden_qa.yaml").is_file()


def test_environment_setup_files() -> None:
    """REQ-DEL-03, REQ-NFR-04: .env.example names every key; lock and requirements exist."""
    example = _text(".env.example")
    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ALPHAVANTAGE_API_KEY"):
        assert re.search(rf"^{key}=\s*$", example, re.M), f"{key} missing or has a value"
    assert (ROOT / "uv.lock").is_file()
    reqs = _text("requirements.txt")
    for pkg in ("langgraph", "faiss-cpu", "fastapi", "streamlit", "plotly", "mcp"):
        assert re.search(rf"^{pkg}==", reqs, re.M), pkg


def test_no_secrets_tracked() -> None:
    """REQ-NFR-04: .env is ignored, never tracked, and excluded from Docker builds."""
    assert ".env" in _text(".gitignore").splitlines()
    assert ".env" in _text(".dockerignore").splitlines()
    git = shutil.which("git")
    assert git is not None
    tracked = subprocess.run(  # noqa: S603 - fixed arguments, no user input
        [git, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.splitlines()
    assert ".env" not in tracked
    leaked = [f for f in tracked if f.endswith((".py", ".yaml", ".toml", ".md"))]
    pattern = re.compile(r"sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}")
    assert not [f for f in leaked if pattern.search((ROOT / f).read_text(errors="ignore"))]


def test_coverage_gate_enforced_in_ci() -> None:
    """REQ-NFR-01: coverage >= 80% is configured and CI runs the gated test command."""
    cfg = tomllib.loads(_text("pyproject.toml"))
    assert cfg["tool"]["coverage"]["report"]["fail_under"] >= 80
    ci = _text(".github/workflows/ci.yml")
    assert "--cov" in ci and "trace_matrix.py" in ci


def test_docstring_rules_enforced() -> None:
    """REQ-NFR-10: ruff pydocstyle (D) rules are enabled for src."""
    lint = tomllib.loads(_text("pyproject.toml"))["tool"]["ruff"]["lint"]
    assert "D" in lint["select"]
    per_file = lint.get("per-file-ignores", {})
    assert not any(k.startswith("src") and "D" in v for k, v in per_file.items())


def test_docker_compose_starts_api_and_ui() -> None:
    """REQ-NFR-06: one `docker compose up` starts the API and the UI (after the index)."""
    compose = yaml.safe_load(_text("docker-compose.yml"))
    services = compose["services"]
    assert {"api", "ui", "index"} <= set(services)
    for name in ("api", "ui"):
        assert services[name]["depends_on"]["index"]["condition"] == (
            "service_completed_successfully"
        )
        assert "healthcheck" in services[name]
    dockerfile = _text("Dockerfile")
    assert "USER app" in dockerfile and "uv sync --frozen" in dockerfile
    assert "API_KEY" not in dockerfile  # keys only at run time


def test_benchmark_report_sections() -> None:
    """REQ-NFR-08: docs/benchmarks.md reports every required metric."""
    report = _text("docs/benchmarks.md")
    for needle in (
        "TTFT p95",
        "p95 (ms)",
        "Cost / query",
        "cache hit rate",
        "Router accuracy",
        "recall@5",
        "BEGIN concurrency",
        "BEGIN judge",
    ):
        assert needle in report, needle


def test_mcp_documented_for_claude_desktop() -> None:
    """REQ-MCP-02: README documents Claude Desktop config and the protocol design."""
    readme = _text("README.md")
    assert '"mcpServers"' in readme and "src.web_app.mcp_server" in readme
    assert "stdio" in readme and "structured" in readme.lower()
