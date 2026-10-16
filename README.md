# AI Finance Assistant

A multi-agent assistant that teaches personal finance to beginners. Six specialist agents,
orchestrated with LangGraph, explain concepts, analyze a portfolio, report live market data,
plan savings goals, summarize news, and explain US tax rules — with every answer grounded in a
curated knowledge base or computed by deterministic code, and every source cited.

> **Educational only — not financial, investment, or tax advice.** The assistant never tells
> users to buy or sell; this is enforced in code, not just in prompts.

| | |
|---|---|
| **Interfaces** | Streamlit app (5 tabs) · REST API with streaming · MCP server for Claude Desktop |
| **Quality** | 428 tests, 97% coverage, every requirement traced to a test · live evals and an LLM judge |
| **Performance** | first token in ~1.1 s (p50), full answer 2.5–6 s, under $0.001 per question |
| **Docs** | [Technical design](docs/technical_design.md) · [Benchmarks](docs/benchmarks.md) · [Sample conversations](docs/sample_conversations.md) · [Specs](specs/) |

**Contents:** [Features](#features) · [Architecture](#architecture-overview) ·
[Setup](#setup) · [Usage](#usage-examples) · [API](#api-documentation) ·
[MCP server](#mcp-server-claude-desktop) · [Configuration](#configuration) ·
[Testing](#testing-and-quality) · [Troubleshooting](#troubleshooting) ·
[Layout](#project-layout)

---

## Features

| Agent | Ask it… | What it does |
|---|---|---|
| **Finance Q&A** | "What is an index fund?" | Answers from 60 curated articles (SEC, IRS, FINRA, CFPB…) with numbered citations, adapted to beginner/intermediate/advanced, and suggests what to learn next |
| **Portfolio Analysis** | "Analyze 20 VTI, 15 AAPL, 10 NVDA, 30 BND" | Computes weights, asset classes, sectors, concentration (HHI), volatility, max drawdown, Sharpe, beta vs SPY, gains (from the price you paid, or the real closing price on the date you bought); flags concentration; remembers the holdings for follow-ups |
| **Market Analysis** | "How is the S&P 500 doing?" | Live quotes and history → moving averages, returns, 52-week range, volatility, trend; index and sector overview; always shows source and freshness |
| **Goal Planning** | "$60k for a house in 5 years" | Inflation-adjusted target, required monthly saving under pessimistic/base/optimistic returns, ways to close a gap, illustrative allocation; asks for missing inputs and your risk tolerance |
| **News Synthesizer** | "Any news on Nvidia?" | Recent headlines, ranked and de-duplicated, summarized with links; no predictions |
| **Tax Education** | "Roth vs traditional IRA?" | US rules with the tax year stated, citations to IRS pages, and a referral to a professional |

Questions that span agents ("What's NVDA trading at, and any news on it?") run the agents
**in parallel** and merge their answers with one numbered source list. Conversations keep
context ("what about its fees?"), and long ones are summarized automatically.

**How this maps to the brief's evaluation criteria**

| Criterion | Where to look |
|---|---|
| Multi-agent architecture | Six agents with one contract (`AgentResult`), no agent calls another — [design §4–5](docs/technical_design.md#4-agent-communication-protocol) |
| LangGraph workflow | Guarded graph with parallel `Send` fan-out, reducers, memory, checkpointing — `src/workflow/` |
| RAG | Hybrid FAISS + BM25 with rank fusion, recall@5 0.972 — [design §6](docs/technical_design.md#6-rag-implementation) |
| Real-time data | yfinance → Alpha Vantage fallback, cache, rate limits, circuit breaker — [design §8](docs/technical_design.md#8-market-data-integration-and-resilience) |
| MCP server | 5 tools for Claude Desktop — [below](#mcp-server-claude-desktop) |
| Streamlit app, visualization | Chat, Portfolio, Market, Goals, Learn tabs with Plotly charts — [usage](#usage-examples) |
| Conversational flow | Follow-ups, clarifying questions, memory — [sample conversations](docs/sample_conversations.md) |
| Domain knowledge | 60 sourced articles; deterministic finance engines; compliance guard |
| Code quality, docs, tests | Typed (strict mypy), linted, 97% coverage, specs with traceability |

---

## Architecture overview

```
  Streamlit UI            REST API (FastAPI)          MCP server
  (in-process)            HTTP + Server-Sent Events   stdio (Claude Desktop)
        └─────────────────────────┼──────────────────────────┘
                                  ▼
                 AssistantService  (one facade for every interface)
                                  │
                       LangGraph workflow
  input guard → router → [parallel agents] → synthesizer → output guard → memory
                                  │
     ┌──────────────────┬─────────┴──────────┬───────────────────┐
  Market data       Finance engines        RAG               LLM gateway
  yfinance →        portfolio, goals,      FAISS + BM25,     roles → models (config)
  Alpha Vantage,    market analytics       rank fusion,      OpenAI primary,
  cache, breaker    (pure Python)          60 articles       Gemini fallback & judge
```

One chat turn: the **input guard** redacts personal data and removes prompt-injection
sentences → the **router** (a small model plus deterministic rules) picks the agents → each
agent **computes** numbers with Python, **retrieves** relevant passages, and makes **one LLM
call** to explain → the **synthesizer** merges answers and citations → the **output guard**
rewrites any buy/sell or guarantee language and adds the disclaimer → state is saved per
conversation. Details and trade-offs: [technical design](docs/technical_design.md).

---

## Setup

**Prerequisites:** Python 3.12, [uv](https://docs.astral.sh/uv/), and API keys:

| Key | Needed for | Required? |
|---|---|---|
| `OPENAI_API_KEY` | All agents, embeddings | Yes |
| `GOOGLE_API_KEY` | Fallback when OpenAI fails; LLM judge ([free key](https://aistudio.google.com/apikey)) | Recommended |
| `ALPHAVANTAGE_API_KEY` | Backup market data when Yahoo fails ([free key](https://www.alphavantage.co/support/#api-key)) | Recommended |

```bash
git clone https://github.com/ravisankar426/ai_finance_assistant.git
cd ai_finance_assistant
make install            # uv sync + git hooks
cp .env.example .env    # then paste your keys into .env
make models             # checks every key and model with one tiny call each
make index              # builds the knowledge-base index (seconds; well under $0.01)
make ui                 # http://localhost:8501
```

The index is also built automatically on first start if it's missing.

**With Docker** (API and UI in containers):

```bash
cp .env.example .env              # add your keys
docker compose up --build         # API http://localhost:8000/docs · UI http://localhost:8501
API_PORT=8080 UI_PORT=8502 docker compose up   # if those ports are taken
```

A one-shot `index` service builds the index into a volume first (and skips the work on later
starts); then `api` and `ui` start with healthchecks. Keys are read from `.env` at run time,
never baked into the image; containers run as a non-root user.

**Offline demo** (no market-data APIs; OpenAI still needed for answers):

```bash
MARKET__PROVIDERS='["fixture"]' make ui    # recorded prices, labeled "recorded <date>"
```

---

## Usage examples

### Streamlit app (`make ui`)

| Tab | Try this |
|---|---|
| 💬 **Chat** | "What is an index fund?" → "What about its fees?" → "Can I hold one in a Roth IRA?" |
| 📊 **Portfolio** | Pick the `concentrated_tech` sample → **Analyze**: metric cards, concentration warnings, allocation donut, sector bars, growth vs SPY. Then ask Chat "is it diversified?" — it uses these holdings. |
| 📈 **Market** | Index prices and a sector heatmap load on open; type `AAPL` for a price chart with 50/200-day averages |
| 🎯 **Goals** | Take the 5-question risk quiz, then project "$60,000 in 5 years, $10,000 saved, $600/month" to see the scenario band and how to close the gap |
| 📚 **Learn** | Browse the 60 articles by category, or search titles |

The sidebar sets your knowledge level (answers adapt) and risk tolerance, and starts a new
conversation. Full multi-turn transcripts recorded from the live system:
[docs/sample_conversations.md](docs/sample_conversations.md).

### Terminal

```bash
make chat     # interactive chat in the terminal, with sources and agent names
```

### More commands

| Command | What it does |
|---|---|
| `make api` | REST API at http://localhost:8000 (interactive docs at `/docs`) |
| `make mcp` | MCP server over stdio (for Claude Desktop) |
| `make check` | Lint, type-check, tests with coverage gate, requirement traceability |
| `make live` | Tests against real APIs: router accuracy, retrieval recall, provider contracts |
| `make bench` / `make judge` | Benchmarks / LLM-as-judge → `docs/benchmarks.md` |
| `make eval` | Retrieval quality: hybrid vs vector vs BM25 |
| `make models` | List available models and smoke-test every configured role |
| `uv run python scripts/market_check.py AAPL --yahoo-down` | Market data path, with a simulated Yahoo outage |

---

## API documentation

Start with `make api`. Interactive OpenAPI docs: **http://localhost:8000/docs**.

| Method | Path | Body / params | Returns |
|---|---|---|---|
| `POST` | `/chat` | `ChatRequest` | **Server-Sent Events**: `token` events, then `final` (a `ChatResponse`) |
| `POST` | `/chat/sync` | `ChatRequest` | `ChatResponse` |
| `POST` | `/portfolio/analyze` | `{"holdings": [{"ticker", "quantity", "cost_basis"?, "purchase_date"?}]}` | Metrics, allocations, flags, performance series |
| `GET` | `/market/quote/{ticker}` | — | Price, change, source, freshness |
| `GET` | `/market/trend/{ticker}` | — | Moving averages, returns, volatility, trend + daily bars |
| `GET` | `/market/overview` | — | 4 index ETFs + 11 sector ETFs |
| `POST` | `/goals/plan` | `target_amount`, `years`, `risk_profile`, `current_savings`?, `monthly_contribution`?, `goal`? | Scenarios, required monthly saving, levers, allocation |
| `GET` | `/goals/questionnaire` | — | The 5 risk questions |
| `POST` | `/goals/risk-profile` | `{"answers": {question_id: option}}` | `conservative` / `moderate` / `aggressive` |
| `GET` | `/learn/articles` | `?category=tax` | Knowledge-base articles |
| `GET` | `/health` | — | Status, market providers, article count |

`ChatRequest`: `message` (required, 1–4000 chars), `thread_id` (reuse it to continue a
conversation), `profile` (`knowledge_level`, `risk_tolerance`), `portfolio` (holdings).
`ChatResponse`: `answer`, `agents`, `citations`, `data` (per-agent chart payloads),
`learn_next`, `thread_id`, `request_id`, `llm_calls`, `cost_usd`, `latency_ms`.

**Examples**

```bash
# Chat (non-streaming); pass the returned thread_id to continue the conversation
curl -s localhost:8000/chat/sync -H 'content-type: application/json' \
  -d '{"message": "What is an expense ratio?"}' | jq '.answer, .agents, .citations[0]'

# Chat, streamed (Server-Sent Events)
curl -N localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message": "How is AAPL doing?", "profile": {"knowledge_level": "beginner"}}'
# event: token / data: {"text": "Apple"} ... event: final / data: {"answer": ..., "agents": [...]}

# Portfolio analysis
curl -s localhost:8000/portfolio/analyze -H 'content-type: application/json' \
  -d '{"holdings": [{"ticker": "VTI", "quantity": 20}, {"ticker": "AAPL", "quantity": 15, "cost_basis": 150}]}' \
  | jq '.total_value, .asset_classes, .flags[].message'

# Market data and goal planning
curl -s localhost:8000/market/quote/MSFT | jq
curl -s localhost:8000/goals/plan -H 'content-type: application/json' \
  -d '{"target_amount": 60000, "years": 5, "current_savings": 10000, "monthly_contribution": 600, "risk_profile": "moderate"}' \
  | jq '.on_track, .levers'
```

**Errors.** Every response carries an `X-Request-ID` header (send your own to correlate);
the same id appears in every log line for the request.

| Status | When | Body |
|---|---|---|
| 422 | Invalid input | `{"error": "Invalid request.", "details": ["message: String should have at least 1 character"], "request_id"}` |
| 404 | Unknown ticker | `{"error": "I couldn't find market data for “ZZQXW”. Check the symbol.", "request_id"}` |
| 503 | Market data, AI service, or knowledge base unavailable | User-safe message + `request_id` |
| 500 | Anything unexpected | `{"error": "Internal error.", "request_id"}` — no internal details |

Once a stream has started its status is already sent, so streaming failures arrive as an
`event: error` message.

---

## MCP server (Claude Desktop)

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

Then ask Claude, for example, *"Use the finance tools to analyze 10 AAPL and 20 VTI"* or
*"Search the finance knowledge base for wash sale rules."*

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

---

## Configuration

Non-secret settings live in [`config.yaml`](config.yaml) (commented); secrets live in `.env`.
Any setting can be overridden with an environment variable, using `__` for nesting:

```bash
LLM__ROLES__AGENT__MODEL=gpt-4.1 make ui            # a bigger model for explanations
WORKFLOW__CHECKPOINTER=sqlite make api              # conversations survive restarts
MARKET__PROVIDERS='["fixture"]' make ui             # offline, recorded market data
APP__LOG_FORMAT=json make api                       # structured JSON logs
```

| Section | Controls |
|---|---|
| `llm` | Which provider and model serve each role (router, guard, agent, synthesizer, judge), fallback model, timeouts, retries, prices for cost tracking. Switching providers is a config change. |
| `workflow` | Memory window, summarization threshold, checkpointer, combined overview |
| `rag` | Chunking, top-k, rank-fusion constant, similarity gate |
| `market` | Providers and order, cache TTLs, rate limits, breaker, retries |
| `portfolio` / `planning` | Concentration thresholds, benchmark, risk-free rate, return assumptions, inflation, allocations |

---

## Testing and quality

```bash
make check    # ruff (lint + docstrings), strict mypy, 428 offline tests, coverage ≥ 80%, traceability
make live     # 12 tests against real APIs (needs keys)
```

- **Offline by design:** tests use scripted fake models, an offline index, and recorded
  market data, so CI never calls a paid API. Coverage: 97%.
- **Traceability:** every test cites requirement IDs from [`specs/01-requirements.md`](specs/01-requirements.md);
  `scripts/trace_matrix.py --strict` fails CI if a must-have requirement has no test.
- **Measured** ([docs/benchmarks.md](docs/benchmarks.md)): p95 first token 1.5–2.4 s, p95 full
  answer 7–9.4 s, $0.0009 per question, router accuracy 100%, retrieval recall@5 0.972,
  20 concurrent sessions with no errors or cross-session leaks.
- **LLM judge** (Gemini grading OpenAI on 30 golden questions): groundedness 4.80,
  citation faithfulness 4.73, compliance 5.00, clarity 5.00 out of 5.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ConfigurationError: No API key configured for provider 'openai'` at startup | Missing or blank key | Add it to `.env` (next to `config.yaml`); run `make models` |
| "The AI service is misconfigured or unavailable" and an ERROR log with 401/403/404 | Wrong key, no access to the model, or a mistyped model name | `make models` lists the models your key can use; fix `config.yaml`. Auth errors deliberately **don't** fall back to Gemini, so they're visible. |
| Answers slow or show the Gemini model in logs | OpenAI rate limit or outage → automatic fallback | Expected behavior; check `llm_error` log lines for the cause |
| Gemini `429 RESOURCE_EXHAUSTED` | Free-tier daily quota (20 requests/day on 3.5 Flash) | Wait for the reset, or use `gemini-3.5-flash-lite` (the default judge) |
| "Live market data is unavailable right now" | Yahoo outage and no/exhausted Alpha Vantage key (25 requests/day) | Add `ALPHAVANTAGE_API_KEY`, wait, or run offline: `MARKET__PROVIDERS='["fixture"]'`. Diagnose with `uv run python scripts/market_check.py AAPL` |
| "I couldn't find market data for “XYZ”" | Every provider says the symbol doesn't exist | Check the ticker (US symbols; `BRK.B` and `BRK-B` both work) |
| Prices labeled *stale* or *recorded* | Providers failed and a cached copy was used / fixture mode | Informational; data refreshes when providers recover |
| "Knowledge base unavailable" / index errors | Index missing or embeddings key invalid | `make index`; if embeddings are temporarily down, answers continue in keyword-only mode |
| `Address already in use` on 8501 or 8000 | Another app (or a previous run) holds the port | Stop it, or `uv run streamlit run src/web_app/ui/app.py --server.port 8502`; Docker: `UI_PORT=8502 API_PORT=8080` |
| `docker compose up` stops at `index` | `.env` missing or without `OPENAI_API_KEY` | `docker compose logs index`; add the key, rerun |
| Claude Desktop shows no finance tools | Wrong path, or `uv` not on Claude Desktop's PATH | Use the absolute repo path, and the full path to `uv` from `which uv` as `command`; check `~/Library/Logs/Claude/mcp*.log`; run `make index` once |
| Settings change has no effect in the UI | The app builds its service once per process | Restart `make ui` |
| `PydanticSerializationUnexpectedValue` warnings | A harmless LangChain serialization notice for structured outputs | Ignore; answers are unaffected |

Logs are structured (`APP__LOG_FORMAT=json` for JSON) and include `request_id`, node, latency,
tokens, and cost for every LLM call; personal data is redacted before logging.

---

## Project layout

```
ai_finance_assistant/
├── src/
│   ├── agents/      six agents + shared grounded-agent base, ticker extraction
│   ├── core/        config, LLM gateway, service facade, guards, finance engines, models, errors
│   ├── data/        market providers + resilience, knowledge base (60 articles), samples, fixtures
│   ├── rag/         loading, chunking, embeddings cache, FAISS store, hybrid retriever
│   ├── web_app/     api/ (FastAPI), ui/ (Streamlit + charts), mcp_server/
│   ├── utils/       structured logging, PII redaction
│   ├── workflow/    LangGraph state, router, graph, memory
│   └── evals/       benchmark and LLM-judge logic
├── tests/           unit/, integration/, evals/ (eval sets), fixtures/
├── scripts/         CLI tools: chat, index, eval, benchmark, judge, market checks, recorders
├── docs/            technical design, benchmarks, sample conversations
├── specs/           constitution, requirements, design, task plan, traceability
├── config.yaml  requirements.txt  pyproject.toml  uv.lock  .env.example
└── Dockerfile  docker-compose.yml
```

Sample data: portfolios in `src/data/sample_portfolios/` (shares and purchase dates; the cost
basis is each date's real closing price; one file has deliberate errors), recorded
market data for 26 tickers in `src/data/fixtures/market/`, and recorded conversations in
[`docs/sample_conversations.md`](docs/sample_conversations.md).

---

*Built Oct 5–16, 2026 with spec-driven development: requirements, design, and a task plan in
[`specs/`](specs/) came first, and every requirement is traced to the tests that verify it.*
