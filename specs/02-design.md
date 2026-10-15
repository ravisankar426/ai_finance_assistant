# Design — AI Finance Assistant

Traces to: `01-requirements.md`. Governed by: `00-constitution.md`.

## 1. Architecture

```
  Streamlit UI (web_app/ui)     REST API (web_app/api)      MCP server (web_app/mcp_server)
     in-process calls            HTTP + SSE                  stdio (JSON-RPC)
            └───────────────────────┼───────────────────────────┘
                                    ▼
                 Service layer: AssistantService (core/services.py)
                                    │
                         LangGraph workflow (workflow/)
   input_guard → router → [Send fan-out] → agent nodes → synthesizer → output_guard → compact
                                    │
      ┌────────────────────┬────────┴────────────┬─────────────────────┐
  Market data (data/)  Finance engines (core/)  RAG (rag/)        LLM gateway (core/llm.py)
  providers, cache,    portfolio, goals, news,  FAISS + BM25      role → provider/model from config
  rate limit, breaker  analytics (no LLM)       + RRF             OpenAI primary, Gemini fallback/judge
```

**Why a service layer:** the UI, REST API, and MCP server all call the same functions, so
each piece of business logic is written and tested once (REQ-API-01, REQ-MCP-01).

## 2. Directory layout (follows the prescribed structure)

```
src/
  agents/      base.py (run_safely), grounded.py (RAG agent base + citation numbering), entities.py
               (ticker extraction), qa.py, tax.py, portfolio.py, market.py, goals.py, news.py
  core/        config.py, llm.py, errors.py, models.py, services.py, guards.py, holdings_io.py,
               portfolio_engine.py, goal_engine.py, market_analytics.py, news.py
  data/        providers/{base,yfinance_provider,alphavantage_provider,fixture_provider}.py,
               market_service.py, resilience.py (cache, retry, rate limit, breaker), overview.py,
               knowledge_base/<category>/*.md, sample_portfolios/*.csv, fixtures/market/,
               index/ and embedding_cache/ (git-ignored, rebuilt)
  rag/         knowledge_base.py, chunker.py, text.py, embedding_cache.py, vector_store.py,
               index.py (load_or_build), retriever.py (hybrid + RRF)
  evals/       benchmark.py, judge.py, report.py
  web_app/     api/ (FastAPI), ui/ (Streamlit app, tabs, charts), mcp_server/ (MCP stdio server)
  utils/       logging.py, redaction.py
  workflow/    state.py, router.py, graph.py, memory.py
tests/         unit/, integration/, evals/, fixtures/
config.yaml  requirements.txt  README.md          ← required by the brief
scripts/  specs/  docs/  .env.example  pyproject.toml  uv.lock  Dockerfile  docker-compose.yml  ← additions
```

The seven `src/` folders of the brief are all present (REQ-DEL-01); `evals/` is an addition. The MCP server goes under `web_app/`
because it's another interface layer, like the API and UI; a new top-level folder would break
the prescribed structure. The repo root folder is named `ai_finance_assistant/`, as in the brief.
`requirements.txt` is exported from `uv.lock`, because the brief asks for one.

## 3. Key decisions (ADR summary)

| ID | Decision | Alternatives rejected → why |
|---|---|---|
| ADR-01 | LangGraph for orchestration | CrewAI → routing is implicit and hard to test; rubric scores LangGraph |
| ADR-02 | Router = small-model structured output + keyword fallback; parallel fan-out via `Send` | Supervisor ReAct agent → slower, costlier, nondeterministic |
| ADR-03 | All math in deterministic Python tools | LLM math → unreliable (constitution P2) |
| ADR-04 | Provider-agnostic LLM gateway: code asks for a **role** (`router`, `agent`, `guard`, `synthesizer`, `judge`); `config.yaml` maps each role to provider + model. **OpenAI is primary; Google Gemini is the cross-provider fallback.** Built on LangChain `init_chat_model` + `.with_fallbacks()` | Same-provider fallback → a provider outage takes everything down. Direct vendor SDKs → each provider has a different API, so switching means rewriting code. Custom HTTP adapters → reinvents what LangChain already standardizes |
| ADR-05 | Embeddings: OpenAI `text-embedding-3-small`, **no cross-provider fallback**; if embeddings are down, retrieval degrades to BM25 only | Gemini embeddings as fallback → vectors from different models live in different spaces, so they can't query the same index. A second, parallel index is possible but doubles ingest work for little gain. Local sentence-transformers → ~2 GB torch dependency |
| ADR-06 | **FAISS** (`IndexFlatIP` on normalized vectors = exact cosine search) used **directly**: index saved with `faiss.write_index`, chunk metadata as JSON (no pickle). Category filtering = mask over chunk metadata | LangChain's FAISS wrapper → saves its docstore with pickle (`allow_dangerous_deserialization`), a code-execution risk if an index file is ever tampered with; direct FAISS + JSON removes the risk and a dependency. Chroma → heavier dependency and its own storage format. Pinecone → network + account for ~300 chunks. IVF/HNSW → approximate search only pays off around 100k+ vectors; exact search here is sub-millisecond |
| ADR-14 | LLM-as-judge runs on **Gemini** (a different provider from the OpenAI generator) | Same model as judge → self-preference bias (models rate their own outputs higher). Human grading → too slow to repeat on every change |
| ADR-07 | Hybrid BM25 + vector with RRF (k=60) | Vector-only → misses exact terms like "401(k)", "wash sale" |
| ADR-08 | yfinance primary, Alpha Vantage fallback, behind one `MarketDataProvider` interface. Measured 2026-10-07: AV free tier = 1 req/s + 25/day, throttling returned as **HTTP 200** + `Information`, OVERVIEW and full history are premium (fallback covers quotes + last 100 trading days, unadjusted) | Either alone → yfinance is unofficial and breaks without notice; AV free tier is tiny |
| ADR-15 | Retry, rate limiter, circuit breaker, and TTL cache are small in-house classes (`data/resilience.py`) with injectable clocks | `tenacity`/`cachetools`/`pybreaker` → three dependencies for ~150 lines, and harder to make deterministic in tests; the in-house versions run instantly under a fake clock |
| ADR-09 | In-process TTL cache (`data/resilience.py`, LRU-bounded, stale copies kept for fallback) | Redis → extra infrastructure for one process; the cache sits behind a small interface so it can be swapped in later |
| ADR-10 | One `AssistantService` facade; FastAPI and Streamlit both call it (the UI in-process, so `make ui` is one process) | Streamlit-only → no API to document, logic duplicated for MCP; UI over HTTP → two processes to run and test for no functional gain |
| ADR-11 | LangGraph `InMemorySaver` by default, `SqliteSaver` by config (`workflow.checkpointer`) | Postgres → overkill for MVP; `PostgresSaver` is a config-level swap later |
| ADR-12 | Guards as graph nodes (regex + small-model classifier) | Prompt-only rules → not enforceable or testable |
| ADR-13 | Lightweight Markdown specs (this folder) | Spec Kit / Kiro tooling → extra tool setup on a 2-week schedule |

## 4. Core data models (Pydantic, `core/models.py`)

```python
class Holding(BaseModel):  ticker: str; quantity: float > 0; cost_basis: float | None
class Quote(BaseModel):    ticker, price, change, change_pct, volume, as_of: datetime, source: str, stale: bool
class Citation(BaseModel): title, url, category, chunk_id
class AgentResult(BaseModel):
    agent: AgentName; answer: str; citations: list[Citation]
    data: dict          # structured payload for UI charts (metrics, projections…)
    error: str | None   # set when the agent degraded (REQ-WF-05)
    follow_up_question: str | None   # set when inputs are missing (REQ-WF-10)
    state_updates: dict              # remembered across turns; only "portfolio"/"profile" applied
class UserProfile(BaseModel):
    knowledge_level: Literal["beginner","intermediate","advanced"] = "beginner"
    risk_tolerance: Literal["conservative","moderate","aggressive"] | None
    goals: list[Goal]; horizon_years: int | None
```

## 5. Graph state (`workflow/state.py`)

```python
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]          # conversation (REQ-WF-04)
    profile: UserProfile                                          # REQ-WF-08
    portfolio: list[Holding] | None
    intents: list[Intent]                                         # router output
    agent_results: Annotated[list[AgentResult], operator.add]     # fan-in reducer (REQ-WF-02)
    final_answer: str
    blocked: bool                                                 # set by input_guard
    request_id: str
```

**Reducers are the key idea:** parallel agent nodes each return `{"agent_results": [result]}`
and LangGraph concatenates them with `operator.add`. Without a reducer, parallel writes to the
same key raise an error.

## 6. Graph

```
START → input_guard ─(blocked)→ guard_refusal ──────────────┐
             └→ router ─(Send per intent)→ agent_{qa|portfolio|market|goals|news|tax}
                                                     └→ synthesizer → output_guard ─┤
                                         (history long) → compact_history → END ◄──┘
```
- The router node writes `intents` and a **standalone question** (follow-ups like "what about its
  fees?" rewritten as self-contained questions, so retrieval works across turns). A conditional edge
  (`dispatch`) turns the intents into `list[Send]`, one per intent; this is LangGraph's dynamic fan-out.
  Agent nodes declare `input_schema=AgentInput`: each receives only its `Send` payload.
- Every agent runs inside `run_safely()`: exceptions become `AgentResult(error=...)` (REQ-WF-05), but
  configuration errors are re-raised (REQ-LLM-07). Per-call timeouts come from the model config.
- `input_guard` redacts PII **in place** (same message id → `add_messages` replaces it), so neither the
  LLMs nor the checkpoint store ever hold the raw value (REQ-GR-05). It also clears last turn's
  `agent_results` (custom `add_or_reset` reducer: `None` = reset).
- `input_guard` also removes prompt-injection sentences ("ignore previous instructions", "reveal
  your system prompt", "you are now DAN") with a regex, *before* any LLM sees them (REQ-GR-02). If
  nothing legitimate is left, `blocked=True` routes to `guard_refusal` and no model is called;
  otherwise the cleaned question continues ("Ignore… What is an ETF?" → only the ETF part).
- Citations (REQ-RAG-05): the user sees one numbered source per article/headline, in reading
  order. `renumber_citations` rewrites each agent's `[n]` (which referred to the prompt's chunk
  list) to that numbering and drops markers pointing at nothing; the synthesizer then maps every
  agent's numbers onto one global list, so `[2]` always means the second entry under Sources.
  Found by the LLM judge: answers citing chunks [1]-[4] of one article showed "Sources (1)".
- Router guards (deterministic, after the LLM): tax terms → `tax`; follow-ups about remembered
  holdings → `portfolio` (and drop a redundant `qa`); `[qa, goals]` with no personal goal signal (amount, time frame, "I want",
  "can I", "retire at") → `qa` only, so rule-of-thumb questions aren't answered with a follow-up.
- Multi-intent scope: the `Send` payload carries `others` (the other intents this turn), and
  `user_turn()` tells each agent what the others cover, so Market doesn't say "I have no
  news" beside the News agent. Found while recording the sample conversations.
- Synthesizer: one result → as is (no overview when only follow-up questions came back); several → one headed section per agent in a fixed order, plus
  failure notices (REQ-WF-05). With `workflow.combined_overview`, an LLM writes a 2–3 sentence
  "**In short:**" overview on top. It is **rejected if it contains a number the agents didn't
  produce** (cheap groundedness check), so the overview can only connect, never invent.
- `output_guard` (REQ-GR-03/04/07): `compliance_issues()` flags directives ("you should buy",
  "strong buy", "I recommend selling") and guarantees ("guaranteed returns", "risk-free way";
  "risk-free rate", negations, and warnings *about* guarantees — "a red flag", "scam" — are exempt,
  so scam education isn't deleted). `ComplianceRewriter` asks the guard model to rephrase
  as education, **re-checks** the rewrite, and strips any sentence still flagged; with no model or
  an outage it strips directly. The guarantee therefore never depends on the LLM. Disclaimer is
  added except for out-of-scope and guard refusals.
- Memory (`workflow/memory.py`, REQ-WF-04/07/08): profile and portfolio live in graph state, so the
  checkpointer remembers them per thread; agents write them back via `AgentResult.state_updates`
  (allowlisted keys only). When history exceeds `summarize_after_messages`, `compact_history`
  replaces it with `[RemoveMessage(REMOVE_ALL_MESSAGES), SystemMessage(summary), *last N]`.
  Rebuilding (not appending) keeps the summary first, which is where models weigh it.
- Checkpointer: `InMemorySaver` (default) or `SqliteSaver` (`workflow.checkpointer: sqlite`, survives
  restarts), both with a serializer that allowlists our Pydantic types; LangGraph refuses to
  deserialize unregistered classes from saved state.
- Day 2 retrieval: `KeywordRetriever` (IDF-weighted term coverage, title bonus for ranking) behind
  the `Retriever` protocol; Day 3 replaces it with FAISS + BM25 without changing any agent.

## 7. Agent contract (`agents/base.py`)

Every agent = system prompt + tools + `run(state) -> AgentResult`. The pipeline follows the brief's
data flow: **check inputs → compute (tools) → retrieve (RAG) → explain (LLM)**.
1. Check inputs: if something required is missing, return a clarifying question (REQ-WF-10).
2. Compute: call deterministic tools (engines, market service) to get the numbers (P2).
3. Retrieve: pull 1–3 knowledge-base chunks for the concepts involved, using a category filter (REQ-WF-09).
4. Explain: one LLM call with profile + tool output + chunks → answer with citations.

Tool-calling loops are kept to ≤ 3 iterations.

| Agent | Compute (tools) | Retrieve (KB categories) | LLM role |
|---|---|---|---|
| QA | – | all except tax | Answer from chunks, cite, suggest what to learn next |
| Portfolio | holdings from the message (structured output, regex fallback), the remembered portfolio, or a CSV → `portfolio_engine` (weights, asset classes, GICS sectors, HHI, volatility, max drawdown, Sharpe, beta vs SPY, unrealized P&L, threshold flags) | portfolio | Explain the fact block; educational, non-security-specific observations only |
| Market | ticker extraction → `market_service` quotes/history → `market_analytics.trend_snapshot`; overview of 4 index + 11 sector ETFs (plus SPY's trend) | markets, portfolio | Explain the computed fact block; a single day's move is never called a trend; no predictions; deterministic data-source/staleness footer |
| Goals | goal parameters via structured output over recent turns; risk profile from the message, the remembered profile, or the questionnaire → `goal_engine` (inflation-adjusted target, required monthly saving under 3 scenarios, projections, gap levers, horizon-adjusted model allocation) | retirement, portfolio, basics | Explain the plan; allocation is illustrative; asks for missing amount/horizon/risk |
| News | ticker extraction → `market_service.get_news` → `core.news.select_headlines` (lookback, relevance, focus ranking, dedupe) | markets | Summarize **headlines only** (free sources give no article text) with each headline linked; no predictions |
| Tax | – | tax | Explain, cite, state tax year, refer to a professional |

**Ticker extraction** (`agents/entities.py`): the router-tier model maps names to symbols
("Apple" → AAPL) with structured output; a regex fallback catches explicit symbols if the call
fails; config errors fail loud. A question that names no stock ("what's the stock price?") returns no
tickers, and the Market agent asks which one (REQ-WF-10) instead of guessing.

**Recorded market data** (`data/providers/fixture_provider.py`, REQ-DEL-02): `scripts/record_market_fixtures.py`
saves normalized quotes, ~400 days of history, profiles (26 tickers), and headlines into
`src/data/fixtures/market/`. Tests run on it, and `MARKET__PROVIDERS='["fixture"]'` gives an offline
demo. Every value is labeled `source="recorded <date>"`, so recorded prices are never shown as live.

## 7a. Agent communication protocol

Agents never call each other directly. They communicate only through the shared graph state:
- **Router → agent:** `Send(node, {"messages", "profile", "portfolio", "intent", "request_id"})`, where each
  agent gets only the slice of state it needs.
- **Agent → synthesizer:** each agent appends one `AgentResult` to `agent_results` (the reducer
  combines the parallel writes). `AgentResult` is the single contract: `answer`, `citations`, `data`
  (structured payload for charts), `error`, `follow_up_question`.
- **Synthesizer → guard → client:** `final_answer` + merged citations + per-agent `data` → API response.

Why this design: agents stay separate (clean separation of concerns), each can be tested with a
hand-built state dict, and adding a seventh agent means adding one node and one router label.

**Memory beyond messages (REQ-WF-04, REQ-WF-08):** agents return `state_updates`; the agent-node
wrapper applies only whitelisted keys (`portfolio`, `profile`), so the Portfolio agent remembers
holdings and the Goals agent remembers a stated risk tolerance and goals. The router adds two
deterministic guards on top of the LLM: precise tax terms → Tax; a reference ("it", "my") plus a
portfolio topic, when holdings are remembered → Portfolio.

## 7b. Scalability

The API is async and keeps no per-request state in memory. Session state lives in the checkpointer
(keyed by `thread_id`), and market data lives in the cache. Both are in-process for the MVP. Scaling
to multiple workers means changing config to use `PostgresSaver` and Redis, with no code change,
because both sit behind interfaces (P6). For the MVP, the concurrency benchmark (REQ-NFR-09)
confirms that sessions stay isolated.

## 8. Market data and resilience (`data/`)

```
get_quote("$aapl")
  └─ normalize_ticker  → "AAPL" (format check before any network call; REQ-MD-07)
  └─ cache fresh?      → return (TTL: quote 60 s, history 6 h, profile 24 h; REQ-MD-02)
  └─ for provider in [yfinance, alphavantage]:
        breaker open?      → skip (5 consecutive failures → 60 s cool-down → half-open probe; REQ-MD-05)
        local rate limit?  → wait if a slot frees within 2 s (AV 1 call/1.1 s), else skip (25/day used up; REQ-MD-03)
        call with retry    → transient errors only, exponential backoff + full jitter, 3 attempts (REQ-MD-04)
          ok                   → cache, close breaker, return
          transient (final)    → breaker failure, next provider
          no data / unsupported → next provider, no breaker penalty
          ConfigurationError   → raise (bad key fails loud)
  └─ all failed: stale cache → return flagged `stale=True` (REQ-MD-06, REQ-MK-04)
                 every answering provider said "no data" → InvalidTickerError
                 otherwise → MarketDataUnavailableError
```

**Provider error categories** (`providers/base.py`) decide everything above: `TransientProviderError`
(retry, counts toward the breaker), `RateLimitedError` (a transient subclass), `NoDataError` and
`UnsupportedOperationError` (move on, no penalty), `ConfigurationError` (fail loud).

**Normalization quirks handled by the adapters (all observed live):**
- yfinance returns an *empty* frame for unknown tickers, and also during Yahoo outages → `NoDataError`,
  never "invalid ticker" on its own; the ticker is only called invalid when **every** provider that
  answered says "no data".
- Yahoo sector names → GICS names ("Technology" → "Information Technology"); ETFs have no sector.
- Alpha Vantage throttling messages *also* mention "premium plans": classify rate-limit wording first,
  then the specific "premium endpoint/feature" phrases (a recorded-fixture test caught the mis-ordering).
- Alpha Vantage history is unadjusted (`adjusted=False`) and ≤ 100 trading days.

All resilience classes take an injectable clock/sleep, so the tests (breaker cool-downs, daily quotas,
backoff) run instantly. Limits are per process: with several workers, the cache and quota counters would
move to Redis (design 7b).

## 8a. LLM gateway (`core/llm.py`)

The only module that knows LLM vendors exist (REQ-LLM-01). Everything else calls:

```python
llm = get_chat_model("agent")             # BaseChatModel with fallback + retries already attached
emb = get_embeddings()                    # Embeddings interface
judge = get_chat_model("judge")
```

`config.yaml` maps roles to models (model IDs are confirmed against your keys on Day 1):
```yaml
llm:
  roles:
    router:      {provider: openai,       model: <small>,  temperature: 0,   timeout_s: 10}
    guard:       {provider: openai,       model: <small>,  temperature: 0,   timeout_s: 10}
    agent:       {provider: openai,       model: <mid>,    temperature: 0.2, timeout_s: 30}
    synthesizer: {provider: openai,       model: <mid>,    temperature: 0.2, timeout_s: 30}
    judge:       {provider: google_genai, model: <gemini>, temperature: 0,   timeout_s: 60}
  fallback:      {provider: google_genai, model: <gemini-flash>}   # used by every role except judge
  embeddings:    {provider: openai,       model: text-embedding-3-small}
  pricing_usd_per_1m_tokens: {<model>: {input: x, output: y}, ...}
```

How a call flows: `get_chat_model(role)` → `init_chat_model(provider, model, …)` with the provider SDK's
own retries (`max_retries`) → `.with_fallbacks([fallback_model], exceptions_to_handle=FALLBACK_ERRORS)` → a usage
callback records tokens + cost (read from LangChain's standard `usage_metadata` field, which is the same for every provider).

**Fallback policy (REQ-LLM-03 / REQ-LLM-07)**, using LangChain's provider-neutral error classes:

| Error class | Policy | Why |
|---|---|---|
| `ModelAuthenticationError` (401), `ModelPermissionDeniedError` (403), `ModelNotFoundError` (404) | **Fail loud**: no fallback, ERROR log with corrective action, raise | Configuration bugs. Switching provider would hide them (a bad key once silently fell back during testing on 2026-10-06) |
| `ModelRateLimitError`, `ModelAPIError` (5xx), `ModelConnectionError`, `ModelTimeoutError`, `ModelInvalidRequestError`, `ContextOverflowError`, `OutputParserException` | Fall back | Provider-side or provider-specific; another provider may succeed |
| Anything else | **Fail loud** | `with_fallbacks` only accepts an allowlist of types; unknown errors surface instead of being masked |

A unit test enumerates every error class in both integrations and asserts each one is in exactly one bucket, so a
library upgrade that adds an error type breaks the build instead of silently changing behavior.

**To switch providers later:** install the LangChain integration package (e.g. `langchain-anthropic`),
add the API key to `.env`, and change `provider`/`model` in `config.yaml`. No code changes. A unit test
stops vendor imports from appearing outside `core/llm.py`.

**What's not fully portable, and how we handle it:**
| Difference between providers | Mitigation |
|---|---|
| Reply shape: OpenAI `content` is a string; Gemini 3 `content` is a list of blocks (text + thought signatures) — observed 2026-10-06 | Callers always read `message.text` (provider-neutral), never `message.content` |
| Thinking models: Gemini 3.x thinks by default (more output tokens, more latency); it recommends temperature 1.0 | Fallback uses a low-thinking model (3.5 Flash-Lite, ~1 s); temperature left unset for Gemini; provider knobs go in the `extra` field in config |
| Capacity: newest models can return 503/504 under load (gemini-3.8-flash on 2026-10-06) | Use a stable, slightly older model for the fallback; the judge gets more retries |
| Structured-output quirks (JSON-schema support differs) | All structured output goes through `.with_structured_output(PydanticModel)`; a contract test runs each schema against both providers (live, opt-in) |
| Tool-calling format | LangChain normalizes it; we keep tool schemas simple (flat arguments) |
| Prompt sensitivity | Prompts avoid provider-specific tricks; live evals run on both providers |
| Embedding spaces | Not interchangeable; changing the embedding model means a full re-index (see ADR-05) |

## 9. RAG

- **Knowledge base (REQ-RAG-01):** 60 Markdown articles in 6 categories under
  `data/knowledge_base/<category>/<slug>.md`, front-matter validated by Pydantic on load (bad YAML,
  missing fields, folder/category mismatch, duplicate ids all fail loudly). Sources are U.S. government
  pages (SEC/Investor.gov, IRS, SSA, CFPB, FINRA, FDIC, BLS, BEA, Federal Reserve) plus CFA Institute,
  MSCI, and Fidelity where no government page exists. Year-specific figures carry `tax_year` and are
  cross-checked by a human reviewer against the cited sources (REQ-TX-02); the review checklist is
  kept outside the repository as a private working document.
- **Chunking:** one chunk per `##` section (sections are topical units); sections over 250 words split
  into overlapping 40-word windows. The *embedding text* is prefixed with "Article title — Section" so
  short chunks keep their context. Result: 297 chunks, median 37 words. Chunk ids `<article_id>#<n>`.
- **Embeddings & cache (REQ-RAG-02):** `text-embedding-3-small` via the gateway; vectors cached in
  `data/embedding_cache/<model>.npz` keyed by sha256(model + text). Only new/changed chunks call the API.
- **Index lifecycle:** `load_or_build()` at startup compares a fingerprint (articles + model + chunking
  settings) with the saved manifest → load (no API calls) or rebuild (cache makes it cheap). Built index:
  `data/index/{index.faiss, chunks.json, manifest.json}` (git-ignored). `make index` forces a rebuild.
- **Retrieval (REQ-RAG-03/04, REQ-QA-02):** category mask → FAISS ranking (meaning) + BM25 ranking
  (exact terms, `rank_bm25`) → Reciprocal Rank Fusion (k=60; ranks only, so no score calibration) →
  **cosine gate** (`min_cosine` 0.3): chunks below it are dropped; if none remain the agent answers
  "not covered". The gate value was chosen from a sweep: recall is flat from 0.20 to 0.40, so 0.3 sits
  mid-plateau; off-topic queries score ≤ 0.14 and on-topic ≥ 0.4 with real embeddings.
- **Degraded mode (REQ-LLM-05):** a transient embeddings failure → BM25-only ranking gated by
  IDF-weighted keyword coverage (≥ 0.5), results flagged `degraded=True` and surfaced in the agent's
  `data`. Auth/permission/not-found errors (LangChain-mapped or raw HTTP 401/403/404) are re-raised.
- **Citations:** the agent sees "[n] Title — Section (url)"; the user sees one numbered source
  per article, and the answer's markers are renumbered to match (see §6).
- **Evaluation (REQ-RAG-06):** `tests/evals/retrieval_set.yaml` — 40 cases (keyword, paraphrase,
  jargon, off-topic). Results (2026-10-07, `make eval`):

  | Method | recall@5 | MRR | paraphrase recall@5 | off-topic rejected |
  |---|---|---|---|---|
  | Hybrid (BM25 + FAISS, RRF) | **0.972** | 0.885 | **0.933** | **100%** |
  | FAISS only | 0.944 | 0.911 | 0.867 | – |
  | BM25 only (degraded) | 0.611 | 0.597 | 0.200 | 100% |

  Hybrid wins on paraphrases; BM25 alone is a fallback, not a design. Known miss: the metaphor "how
  bumpy is the ride?" doesn't reach the volatility article (articles were not tuned to the eval set).

## 9a. Service layer and REST API (`core/services.py`, `web_app/api/app.py`)

`AssistantService` is the single facade (chat, portfolio, quote, trend, overview, goal plan,
questionnaire, articles) used by the API, the Streamlit tabs and the MCP server, so business
logic exists once. `build_service()` shares one market cache and one index between the graph
and the endpoints.

| Endpoint | Purpose |
|---|---|
| `POST /chat` | SSE: `token` events while agent nodes write, then `final` (full `ChatResponse`) |
| `POST /chat/sync` | Same turn, plain JSON |
| `POST /portfolio/analyze` · `GET /market/quote/{t}` · `/market/trend/{t}` · `/market/overview` | Deterministic tools |
| `POST /goals/plan` · `GET /goals/questionnaire` · `POST /goals/risk-profile` · `GET /learn/articles` | Planning and learning |
| `GET /health` | Liveness plus providers and article count |

Error contract (REQ-API-03): validation → 422 with `field: message` details; `AppError` → its
status (invalid ticker 404, unavailable data/LLM/KB 503) with the user-safe message; LangChain
`ModelError` → 503; anything else → 500 `"Internal error."` with no internals. Once an SSE stream
has started the status is already sent, so failures arrive in-band as an `error` event. Every
response carries `X-Request-ID` (generated or propagated), bound into the logs.

## 9b. Streamlit UI (`web_app/ui/`)

`app.py` is the shell (title, disclaimer above the tabs so it shows on every tab, sidebar
profile, Chat tab); `tabs.py` holds Portfolio, Market, Goals, Learn; `charts.py` holds pure
Plotly builders (data in, figure out) so every chart is unit-tested without Streamlit.

| Tab | Shows | Service calls |
|---|---|---|
| Chat | streamed tokens, "Answered by" badges, expandable sources, learn-next links | `chat_events` (sync twin of the SSE stream) |
| Portfolio | editable holdings table / CSV upload / samples → 8 metric cards, flags, asset-class donut, sector bars, growth-of-100 vs SPY, holdings table | `analyze_portfolio` |
| Market | index metrics, sector heatmap (diverging around 0), ticker chart with 50/200-day SMAs, as-of + source, refresh | `overview`, `trend`, `history` |
| Goals | risk quiz → form → pessimistic–optimistic band, base path, target line, gap levers, illustrative mix | `risk_profile`, `plan_goal` |
| Learn | category filter + title search over the 60 articles | `categories`, `list_articles` |

Analyzed holdings are sent with chat requests, so "how diversified is my portfolio?" works
without retyping. The engine adds `performance` (growth of 100) to `PortfolioAnalysis` for the
chart; the Portfolio agent excludes it from its checkpointed data. Refresh relies on the market
cache: data is re-fetched only after its TTL expires (REQ-UI-04). Streamlit columns stack on
narrow screens (REQ-UI-07). Tests drive the real page with `AppTest` and a fake service.

## 9c. MCP server (`web_app/mcp_server/`)

`MCPServer` (MCP Python SDK 2.x, formerly FastMCP) over stdio with five tools that wrap
`AssistantService`: `get_quote`, `market_overview`, `analyze_portfolio`, `plan_goal`,
`search_knowledge_base`. The service is built with `chat=False` (no LLM workflow: MCP clients
bring their own model) on the first tool call, so the handshake is instant. Tools return
Pydantic models → `structuredContent` plus an output schema. `AppError` → tool error with the
user-safe message; anything else → "Internal error". Logs go to stderr because stdout is the
JSON-RPC channel (`configure_logging(stream=sys.stderr)`). Tested end to end with the SDK's
in-memory client; verified live over stdio.

## 9d. Deployment (Docker)

One image (python:3.12-slim + uv, `uv sync --frozen`, non-root `app` user) runs three compose
services: `index` (one-shot `build_index.py --if-needed`; reuses a fingerprint-matching index)
→ `api` and `ui` (`depends_on: service_completed_successfully`, so they never race to write
the index). Named volumes hold the index, the embedding cache, and SQLite state. Keys come
from `.env` at run time, never from the image; `.dockerignore` excludes `.env`, tests, specs.
Healthchecks hit `/health` and `/_stcore/health`. Host ports via `API_PORT` / `UI_PORT`.

## 9e. Benchmarks and evaluation (`src/evals/`, `scripts/benchmark.py`, `scripts/judge_eval.py`)

- Benchmark: per-agent queries through `AssistantService.chat_events` on fresh threads,
  sequential, after one warm-up query; TTFT, latency p50/p95 (nearest rank), LLM calls and
  cost from the usage callback; market `TTLCache` hit rate (hit/miss counters); router
  accuracy on the labeled cases; hybrid retrieval recall@5/MRR; 20 concurrent two-turn
  sessions with a marker-based leakage check on the saved checkpoints.
- Structured-output contract tests (live) send every LLM schema to the OpenAI primary and the
  Gemini fallback directly, so a provider change can't silently break routing after a
  fallback.
- LLM-as-judge: Gemini grades OpenAI answers (cross-family) on groundedness, citation
  faithfulness, compliance, clarity (1-5) against reference key points **and the cited
  passages' text**; 30 golden questions (grounded, data, safety/advice-bait).
- Scripts rewrite only their marked sections of `docs/benchmarks.md`; the logic is in
  `src/evals/` so it is typed and tested offline with the fake service.

## 10. Configuration

`config.yaml` holds non-secret settings (LLM roles, TTLs, thresholds, limits). `.env` holds
`OPENAI_API_KEY`, `GOOGLE_API_KEY` (Gemini), `ALPHAVANTAGE_API_KEY`. Loaded by `pydantic-settings` into a typed
`Settings` object; env vars override YAML (12-factor).

## 11. Testing strategy

| Layer | Approach |
|---|---|
| Engines | Pure unit tests + `hypothesis` property tests (weights sum to 1, required saving lands on the target, SMA definition) |
| Data layer | Fake providers; recorded Alpha Vantage/yfinance payloads; injectable clocks for TTLs, breaker cool-downs, quotas, backoff |
| Agents/graph | `ScriptedChatModel` (tests/fakes.py) with scripted replies and failures; offline index with hashing embeddings; assert routing, fan-in, degradation, guards, memory |
| API / MCP / UI | FastAPI `TestClient`, the MCP SDK's in-memory client, and Streamlit `AppTest`, all against a real `AssistantService` built on fakes and recorded market data |
| LLM gateway | Role → model resolution from config; fallback fires when the primary fake raises; cost callback math; no vendor imports outside `core/llm.py` |
| Repository | Layout, setup files, no tracked secrets, coverage gate in CI, docstring rules, compose wiring, required docs sections |
| Live (opt-in, `-m live`) | Real models: router accuracy, retrieval recall, structured-output contract on OpenAI and Gemini |
| Benchmarks and judge (scripts) | `make bench`, `make judge` → `docs/benchmarks.md` (see §9e) |

Every test docstring starts with the requirement ID(s) it verifies. `scripts/trace_matrix.py`
lists requirements that have no test.
