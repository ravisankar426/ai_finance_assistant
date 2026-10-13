.PHONY: install lint format typecheck test check trace models requirements ui chat index eval live api mcp bench judge docker

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

ui:             ## run the Streamlit app
	uv run streamlit run src/web_app/ui/app.py

chat:           ## live: chat with the assistant in the terminal (needs keys in .env)
	uv run python scripts/chat_cli.py

index:          ## rebuild the knowledge-base FAISS index (needs OPENAI_API_KEY)
	uv run python scripts/build_index.py

eval:           ## live: retrieval recall@5 / MRR for hybrid vs vector vs BM25
	uv run python scripts/eval_retrieval.py --sweep

live:           ## live: tests that call real APIs (router accuracy, retrieval recall)
	uv run pytest -m live -q

api:            ## run the REST API (docs at http://localhost:8000/docs)
	uv run uvicorn src.web_app.api.app:app --reload --port 8000

mcp:            ## run the MCP server over stdio (for Claude Desktop / MCP clients)
	uv run python -m src.web_app.mcp_server

bench:          ## live: latency/cost/cache/router/retrieval/concurrency -> docs/benchmarks.md
	uv run python -m scripts.benchmark

judge:          ## live: LLM-as-judge on the golden set (Gemini) -> docs/benchmarks.md
	uv run python -m scripts.judge_eval

docker:         ## build and start API (:8000) + UI (:8501) in containers
	docker compose up --build
