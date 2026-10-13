# AI Finance Assistant

Multi-agent conversational assistant for personal-finance **education**: six specialist agents
orchestrated with LangGraph, RAG over a curated knowledge base (FAISS + BM25), live market data,
and a Streamlit UI. Educational only — not financial advice.

> Work in progress (build window Oct 5–16, 2026). Full README lands on Day 10 (REQ-NFR-07).
> Specs: [`specs/`](specs/) — start with [`04-traceability.md`](specs/04-traceability.md).

## Quick start

```bash
make install                  # uv sync + pre-commit hooks
cp .env.example .env          # add OPENAI_API_KEY, GOOGLE_API_KEY, ALPHAVANTAGE_API_KEY
make models                   # live: list models your keys can use + smoke-test every role
make check                    # lint + types + tests (coverage gate 80%) + traceability
make ui                       # Streamlit app at http://localhost:8501 (Chat, Portfolio, Market, Goals, Learn)
make chat                     # chat in the terminal (live models)
make api                      # REST API at http://localhost:8000/docs (SSE chat + tools)
```

**Offline demo (no market APIs needed):** `MARKET__PROVIDERS='["fixture"]' make ui` serves recorded
market data, clearly labeled “recorded <date>”. Refresh it with `uv run python scripts/record_market_fixtures.py`.
**Check market data:** `uv run python scripts/market_check.py AAPL SPY` (add `--yahoo-down` to simulate an outage).

## Docker

```bash
cp .env.example .env            # add your keys
docker compose up --build       # API http://localhost:8000/docs · UI http://localhost:8501
API_PORT=8080 UI_PORT=8502 docker compose up   # if the default ports are taken
```

One image runs three services: `index` builds the knowledge-base index once into a volume
(skipped on later starts when nothing changed), then `api` and `ui` start. Keys are read from
`.env` at run time and never baked into the image; containers run as a non-root user.

## MCP server (Claude Desktop and other MCP clients)

`make mcp` runs the server over **stdio**. Tools: `get_quote`, `market_overview`,
`analyze_portfolio`, `plan_goal`, `search_knowledge_base`. To use it from Claude Desktop, add
this to `claude_desktop_config.json` (Settings → Developer → Edit Config) and restart:

```json
{
  "mcpServers": {
    "ai-finance-assistant": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/ai_finance_assistant",
               "run", "python", "-m", "src.web_app.mcp_server"]
    }
  }
}
```

Protocol design:
- **stdio transport**: the client launches the server as a subprocess; stdout carries
  JSON-RPC, so all logs go to stderr.
- **Thin tools over the shared service**: the same `AssistantService` as the API and UI, so
  numbers match across interfaces. Tools are deterministic (no LLM calls inside); the client's
  model does the reasoning.
- **Typed results**: tools return Pydantic models, so clients receive structured content
  (with a JSON output schema) as well as text. Inputs are validated against the schema
  before any tool runs.
- **Safe errors**: an unknown ticker or unavailable data becomes a tool error with a
  user-safe message; internal details stay in the server logs.
- **Lazy start**: the handshake is instant; the market cache and index load on the first
  tool call (the index must exist; `make index` builds it).

## Benchmarks and evaluation

`make bench` measures latency (p50/p95 per agent, time to first token), cost per query,
cache hit rate, router accuracy, retrieval recall, and 20 concurrent sessions; `make judge`
scores a 30-question golden set with Gemini as the judge. Results: [`docs/benchmarks.md`](docs/benchmarks.md).

## Layout

```
src/{agents,core,data,rag,web_app,utils,workflow}   tests/   specs/   scripts/   config.yaml
```
