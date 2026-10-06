# Requirements — AI Finance Assistant

Notation: EARS (Easy Approach to Requirements Syntax).
- **Ubiquitous:** The system shall …
- **Event:** WHEN <trigger>, the system shall …
- **Unwanted:** IF <bad condition>, THEN the system shall …
- **State:** WHILE <state>, the system shall …

Priority: **M** = must (graded / required), **S** = should, **C** = could (cut first).
Every ID below must be referenced by at least one test (constitution P5).

---

## 1. Workflow & Routing (WF)

| ID | Pri | Requirement |
|---|---|---|
| REQ-WF-01 | M | WHEN a user submits a message, the system shall classify it into one or more intents from {`qa`, `portfolio`, `market`, `goals`, `news`, `tax`, `out_of_scope`}. |
| REQ-WF-02 | M | WHEN more than one intent is detected, the system shall run the matching agents in parallel and return one synthesized response. |
| REQ-WF-03 | M | IF the routing LLM call fails or times out, THEN the system shall classify using a deterministic keyword router. |
| REQ-WF-04 | M | The system shall keep conversation history per session (`thread_id`) so follow-up questions resolve against earlier turns (e.g. "what about its dividend?"). |
| REQ-WF-05 | M | IF one agent fails, THEN the system shall return the other agents' results plus a notice naming the unavailable capability. |
| REQ-WF-06 | S | The system shall stream the response to the client as it is generated. |
| REQ-WF-07 | S | WHEN history exceeds the configured message limit, the system shall summarize older turns instead of dropping them. |
| REQ-WF-08 | M | The system shall persist a user profile (knowledge level, risk tolerance, goals, horizon) per session and pass it to agents. |
| REQ-WF-09 | M | Each specialist agent (portfolio, market, goals, news) shall ground its explanation with relevant knowledge-base content when available and cite it, linking live data to the concept it illustrates (problem statement: data flow "Agent(s) → RAG Retrieval → LLM"; "learn concepts while seeing real-world applications"). |
| REQ-WF-10 | M | IF inputs an agent needs are missing (e.g. goal amount, horizon, holdings), THEN the agent shall ask one targeted follow-up question instead of assuming values. |

## 2. Guardrails & Compliance (GR)

| ID | Pri | Requirement |
|---|---|---|
| REQ-GR-01 | M | WHEN a message is out of scope (not personal finance/investing/tax), the system shall decline politely and say what it can help with. |
| REQ-GR-02 | M | IF a message contains a prompt-injection attempt (e.g. "ignore previous instructions"), THEN the system shall not follow it and shall answer only the legitimate part, if any. |
| REQ-GR-03 | M | The system shall append an educational disclaimer to every response with investment or tax content. |
| REQ-GR-04 | M | IF a draft response contains a personalized directive (e.g. "you should buy TSLA"), THEN the system shall rewrite it into educational framing before returning it. |
| REQ-GR-05 | M | The system shall redact PII (SSN, account/card numbers, emails, phone numbers) before text is sent to the LLM or written to logs. |
| REQ-GR-06 | M | The system shall adapt explanation depth and jargon to the profile's knowledge level (beginner / intermediate / advanced); for beginners, every technical term is defined on first use. |
| REQ-GR-07 | M | The system shall frame guidance conservatively: state risks alongside potential returns, never promise or imply guaranteed returns, and favor diversification and long-term principles. |

## 3. Finance Q&A Agent (QA)

| ID | Pri | Requirement |
|---|---|---|
| REQ-QA-01 | M | WHEN asked a general finance question, the agent shall answer from the knowledge base and cite each source used (title + URL). |
| REQ-QA-02 | M | IF no retrieved chunk scores above the relevance threshold, THEN the agent shall state that its knowledge base does not cover the topic and shall not cite sources. |
| REQ-QA-03 | S | The agent shall suggest 1–3 "learn next" articles matched to the user's knowledge level and recent topics (personalized learning path). |

## 4. Portfolio Analysis Agent (PF)

| ID | Pri | Requirement |
|---|---|---|
| REQ-PF-01 | M | The system shall accept holdings as (ticker, quantity, optional cost basis) via chat text, form entry, or CSV upload. |
| REQ-PF-02 | M | The system shall compute total value, position weights, asset-class and sector allocation, concentration (HHI), annualized volatility, beta vs SPY, Sharpe ratio, max drawdown, and unrealized gain/loss (when cost basis given). |
| REQ-PF-03 | M | WHEN any single holding exceeds the configured weight limit (default 25%) or any sector exceeds its limit (default 40%), the system shall flag a concentration risk. |
| REQ-PF-04 | M | The agent shall explain the metrics in plain language and give educational, non-security-specific recommendations (e.g. "your tech sector weight is 62%; diversified portfolios typically spread risk across sectors — consider learning about broad index funds and rebalancing"). It shall never instruct the user to buy, sell, or hold a specific security. |
| REQ-PF-05 | M | IF some tickers are invalid or have no data, THEN the system shall analyze the valid holdings and list the ones it skipped. |

## 5. Market Analysis Agent (MK)

| ID | Pri | Requirement |
|---|---|---|
| REQ-MK-01 | M | WHEN asked for a quote, the system shall return price, change, % change, volume, as-of timestamp, and data source. |
| REQ-MK-02 | M | The system shall compute trend indicators for a ticker: 50/200-day SMA, 1M/3M/1Y return, 52-week range, annualized volatility. |
| REQ-MK-03 | M | The system shall provide a market overview of major index ETFs (SPY, QQQ, DIA, IWM) and the 11 SPDR sector ETFs. |
| REQ-MK-04 | M | The system shall show data freshness and label stale data. |

## 6. Goal Planning Agent (GP)

| ID | Pri | Requirement |
|---|---|---|
| REQ-GP-01 | M | The system shall derive a risk profile (conservative / moderate / aggressive) from a short questionnaire or an explicit user statement. |
| REQ-GP-02 | M | WHEN given a target amount, horizon, current savings, and risk profile, the system shall compute the required monthly contribution and a projection under pessimistic / base / optimistic return assumptions. |
| REQ-GP-03 | M | The system shall express targets in both nominal and inflation-adjusted terms (inflation rate configurable). |
| REQ-GP-04 | M | The system shall show an illustrative model allocation for the risk profile and horizon, labeled as educational. |
| REQ-GP-05 | M | IF the required contribution exceeds what the user says they can save, THEN the system shall flag the goal as at risk and show which levers (time, amount, contribution) would close the gap. |

## 7. News Synthesizer Agent (NW)

| ID | Pri | Requirement |
|---|---|---|
| REQ-NW-01 | M | WHEN asked about news for a ticker or topic, the system shall fetch recent articles (configurable lookback). |
| REQ-NW-02 | M | The agent shall summarize, de-duplicate, and link every article it mentions. |
| REQ-NW-03 | M | The agent shall explain what the news could mean for a beginner, without predicting prices. |
| REQ-NW-04 | M | IF no news is found, THEN the agent shall say so. |

## 8. Tax Education Agent (TX) — United States only

| ID | Pri | Requirement |
|---|---|---|
| REQ-TX-01 | M | The agent shall explain US tax concepts — 401(k), Traditional/Roth IRA, HSA, 529, taxable brokerage, short- vs long-term capital gains, qualified dividends, tax-loss harvesting, wash-sale rule — with citations. |
| REQ-TX-02 | M | WHEN stating a year-specific figure (contribution limit, bracket threshold), the agent shall state the tax year and its source. |
| REQ-TX-03 | M | The agent shall recommend a qualified tax professional for situation-specific questions. |

## 9. Market Data Layer (MD)

| ID | Pri | Requirement |
|---|---|---|
| REQ-MD-01 | M | The system shall fetch market data from yfinance first and Alpha Vantage as fallback, behind one provider interface. |
| REQ-MD-02 | M | The system shall cache responses with per-type TTLs (configurable; defaults: quote 60 s, history 6 h, news 15 min). |
| REQ-MD-03 | M | The system shall enforce a client-side rate limit per provider (configurable). |
| REQ-MD-04 | M | IF a provider call fails transiently, THEN the system shall retry with exponential backoff and jitter (default 3 attempts). |
| REQ-MD-05 | M | IF a provider fails N consecutive times (default 5), THEN the system shall open a circuit breaker for a cool-down period (default 60 s) and skip that provider. |
| REQ-MD-06 | M | IF all providers fail and stale cached data exists, THEN the system shall return the stale data flagged as stale. |
| REQ-MD-07 | M | The system shall validate ticker format before calling providers. |

## 9a. LLM Gateway (LLM)

| ID | Pri | Requirement |
|---|---|---|
| REQ-LLM-01 | M | Application code shall get chat and embedding models only through the LLM gateway, by role (`router`, `guard`, `agent`, `synthesizer`, `judge`). No module outside the gateway shall import a vendor SDK or vendor LangChain package. |
| REQ-LLM-02 | M | The provider, model, and parameters for each role shall be set in `config.yaml`; switching a role to another supported provider shall need only config and environment changes. |
| REQ-LLM-03 | M | IF a primary model call fails with a transient or provider-side error (rate limit, 5xx, timeout, connection, invalid request, context overflow, unparseable output) after retries, THEN the gateway shall retry the call on the configured fallback model (default: Google Gemini) and log that the fallback was used. |
| REQ-LLM-07 | M | IF a model call fails with a configuration error (authentication 401, permission 403, model not found 404) or an error type not on the fallback allowlist, THEN the gateway shall NOT fall back, shall log it at ERROR level with the corrective action, and shall raise it. |
| REQ-LLM-04 | M | The gateway shall record tokens and estimated cost for every call, whatever the provider, using a pricing table in config. |
| REQ-LLM-05 | M | IF the embeddings provider is unavailable, THEN retrieval shall fall back to BM25 only and flag the result as degraded. |
| REQ-LLM-06 | C | The LLM-as-judge shall use a different provider from the generating model (default: Gemini judges OpenAI output). |

## 10. Knowledge Base & RAG (RAG)

| ID | Pri | Requirement |
|---|---|---|
| REQ-RAG-01 | M | The knowledge base shall contain 50–100 Markdown articles, each with front-matter: id, title, category, difficulty, source_url, tax_year (if applicable), last_reviewed. |
| REQ-RAG-02 | M | Ingestion shall be idempotent: unchanged articles (same content hash) are not re-embedded. |
| REQ-RAG-03 | M | Retrieval shall be hybrid (BM25 + FAISS vector search) fused with Reciprocal Rank Fusion. The FAISS index shall persist to disk and load on startup without re-embedding. |
| REQ-RAG-04 | M | Retrieval shall support filtering by category. |
| REQ-RAG-05 | M | Every retrieved chunk shall carry source attribution (article title, URL, category). |
| REQ-RAG-06 | S | On the retrieval eval set, recall@5 shall be ≥ 0.85. |

## 11. API (API)

| ID | Pri | Requirement |
|---|---|---|
| REQ-API-01 | M | The system shall expose a REST API: `POST /chat` (SSE stream), `POST /portfolio/analyze`, `GET /market/quote/{ticker}`, `GET /market/overview`, `POST /goals/plan`, `GET /health`. |
| REQ-API-02 | M | The API shall publish an OpenAPI schema with request/response models. |
| REQ-API-03 | M | IF a request fails validation, THEN the API shall return HTTP 422 with a readable message; unexpected errors return 500 with a `request_id` and no internals. |

## 12. User Interface (UI)

| ID | Pri | Requirement |
|---|---|---|
| REQ-UI-01 | M | The Streamlit app shall have tabs: Chat, Portfolio, Market, Goals, Learn. |
| REQ-UI-02 | M | The Chat tab shall stream responses, show which agents answered, and show citations in an expandable section. |
| REQ-UI-03 | M | The Portfolio tab shall accept holdings (table entry or CSV) and show metric cards, an allocation donut, a sector bar chart, and performance vs SPY. |
| REQ-UI-04 | M | The Market tab shall show index overview, sector heatmap, and a ticker chart with SMAs, with an as-of timestamp and a refresh control (data re-fetched when the cache TTL has expired). |
| REQ-UI-05 | M | The Goals tab shall take goal inputs and show a projection chart with pessimistic/base/optimistic bands. |
| REQ-UI-06 | S | The Learn tab shall let users browse knowledge-base articles by category. |
| REQ-UI-07 | M | The layout shall remain usable at narrow (mobile) widths, and a disclaimer shall be visible on every tab. |

## 13. MCP Server (MCP) — bonus

| ID | Pri | Requirement |
|---|---|---|
| REQ-MCP-01 | C | The system shall expose MCP tools: `get_quote`, `market_overview`, `analyze_portfolio`, `plan_goal`, `search_knowledge_base`. |
| REQ-MCP-02 | C | The README shall document Claude Desktop configuration and the protocol design. |

## 14. Non-Functional (NFR)

| ID | Pri | Requirement |
|---|---|---|
| REQ-NFR-01 | M | Test line coverage shall be ≥ 80%, enforced in CI. |
| REQ-NFR-02 | S | p95 time-to-first-token ≤ 3 s and p95 full response ≤ 12 s for single-agent queries (measured by benchmark script). |
| REQ-NFR-03 | S | Average LLM cost per query shall be ≤ $0.01, reported by the benchmark script. |
| REQ-NFR-04 | M | No secret shall be committed; `.env.example` documents required variables. |
| REQ-NFR-05 | M | Logs shall be structured JSON with `request_id`, node, latency, and (for LLM calls) tokens. |
| REQ-NFR-06 | S | `docker compose up` shall start the API and UI with one command. |
| REQ-NFR-07 | M | The repo shall include a README (architecture overview, setup instructions, API documentation, usage examples, troubleshooting guide) and a technical design document covering system architecture decisions, agent communication protocols, RAG implementation details, and performance considerations. |
| REQ-NFR-08 | M | The repo shall include a performance benchmark report (`docs/benchmarks.md`): latency p50/p95 per agent, cost per query, cache hit rate, router accuracy, retrieval recall@5. |
| REQ-NFR-09 | S | The API shall be async and stateless apart from the configured checkpointer/cache; the benchmark shall run 20 concurrent sessions with no errors and no cross-session state leakage. |
| REQ-NFR-10 | M | Public modules, classes, and functions shall have docstrings (enforced by ruff pydocstyle rules). |
| REQ-NFR-11 | C | An opt-in LLM-as-judge eval shall score a golden set of ~30 Q&As from 1 to 5 on groundedness, citation faithfulness, compliance, and clarity, and report the averages in `docs/benchmarks.md`. |

## 15. Submission Deliverables (DEL)

| ID | Pri | Requirement |
|---|---|---|
| REQ-DEL-01 | M | The codebase shall follow the prescribed layout: `src/{agents,core,data,rag,web_app,utils,workflow}`, `tests/`, `config.yaml`, `requirements.txt`, `README.md` at the repo root. |
| REQ-DEL-02 | M | The repo shall ship sample data for testing: sample portfolios (CSV), recorded market-data fixtures, and sample conversations. |
| REQ-DEL-03 | M | The repo shall ship environment setup files: `.env.example`, `pyproject.toml` + `uv.lock`, exported `requirements.txt`. |
| REQ-DEL-04 | M | A 5–10 minute demo video shall show: multi-turn conversations with different agents, a portfolio analysis, market data integration, and a goal-planning example. A demo script in `docs/demo_script.md` covers each item. |
