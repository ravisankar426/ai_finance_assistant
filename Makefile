.PHONY: install lint format typecheck test check trace models requirements

install:        ## install deps + git hooks
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run mypy

test:           ## offline tests with coverage gate
	uv run pytest -m "not live and not judge" --cov --cov-report=term-missing

check: lint typecheck test trace   ## everything CI runs

trace:          ## requirement -> test traceability report
	uv run python scripts/trace_matrix.py

models:         ## live: list models + smoke-test every role (needs keys in .env)
	uv run python scripts/check_models.py --list

requirements:   ## regenerate requirements.txt from uv.lock
	uv export --format requirements-txt --no-hashes --no-dev --no-emit-project --no-header -o requirements.txt
