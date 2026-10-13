"""Write generated sections into docs/benchmarks.md without touching the rest of the file."""

from __future__ import annotations

from pathlib import Path


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"


def replace_section(markdown: str, name: str, body: str) -> str:
    """Replace the text between a section's markers, or append the section if it's missing."""
    begin, end = _markers(name)
    block = f"{begin}\n{body.strip()}\n{end}"
    if begin in markdown and end in markdown:
        head, rest = markdown.split(begin, 1)
        _, tail = rest.split(end, 1)
        return head + block + tail
    return markdown.rstrip() + "\n\n" + block + "\n"


def update_report(path: Path, name: str, body: str) -> None:
    """Update one generated section of the report file in place."""
    text = path.read_text() if path.exists() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replace_section(text, name, body))


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def fmt(x: float | int | None, spec: str = ".0f", suffix: str = "") -> str:
    """Format a number for a table cell ('n/a' when missing)."""
    return "n/a" if x is None else f"{format(x, spec)}{suffix}"
