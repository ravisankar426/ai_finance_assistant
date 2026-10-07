# Design — AI Finance Assistant

Traces to: `01-requirements.md`. Governed by: `00-constitution.md`.

## 1. Architecture

```
  Streamlit UI (web_app/ui)            MCP server (mcp_server)
          │ HTTP + SSE                         │ stdio
          ▼                                    ▼
  FastAPI (web_app/api) ──────► Service layer (core/services.py) ◄──────┘
                                       │
                         LangGraph workflow (workflow/)
   input_guard → router → [Send fan-out] → agent nodes → synthesizer → output_guard
                                       │
      ┌────────────────────┬───────────┴─────────┬─────────────────────┐
  Market data (data/)  Finance engines (core/)  RAG (rag/)        LLM gateway (core/llm.py)
  providers, cache,    portfolio, goals, risk   FAISS + BM25      role → provider/model from config
  rate limit, breaker  (pure Python, no LLM)    + RRF             OpenAI primary, Gemini fallback/judge
```

**Why a service layer:** the UI, REST API, and MCP server all call the same functions, so
each piece of business logic is written and tested once (REQ-API-01, REQ-MCP-01).

## 2. Directory layout (follows the prescribed structure)

```
src/
  agents/      base.py, qa.py, portfolio.py, market.py, goals.py, news.py, tax.py
  core/        config.py, llm.py, errors.py, models.py, services.py,
               portfolio_engine.py, goal_engine.py, risk_profile.py, guards.py
  data/        providers/{base,yfinance_provider,alphavantage_provider}.py,
               market_service.py, cache.py, resilience.py (retry, rate limit, breaker),
               knowledge_base/*.md, sample_portfolios/*.csv,
               index/ (FAISS index + docstore, git-ignored), embedding_cache/ (git-ignored)
  rag/         ingest.py, chunker.py, store.py, retriever.py (hybrid + RRF)
  web_app/     api/ (FastAPI), ui/ (Streamlit tabs + charts), mcp_server/ (MCP stdio server)
  utils/       logging.py, redaction.py, timing.py
  workflow/    state.py, router.py, graph.py, synthesizer.py, memory.py
tests/         unit/, integration/, evals/, fixtures/
config.yaml  requirements.txt  README.md          ← required by the brief
scripts/  specs/  docs/  .env.example  pyproject.toml  uv.lock  Dockerfile  docker-compose.yml  ← additions
```

The seven `src/` folders match the brief exactly (REQ-DEL-01). The MCP server goes under `web_app/`
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
| ADR-09 | In-process TTL cache (`cachetools`) behind a `Cache` protocol | Redis → extra infrastructure; can be swapped in later |
| ADR-10 | FastAPI service + Streamlit client | Streamlit-only → no API to document, logic duplicated for MCP |
| ADR-11 | LangGraph `MemorySaver` in dev, `SqliteSaver` by config | Postgres → overkill for MVP |
| ADR-12 | Guards as graph nodes (regex + small-model classifier) | Prompt-only rules → not enforceable or testable |
| ADR-13 | Lightweight Markdown specs (this folder) | Spec Kit / Kiro tooling → tooling learning curve on a 2-week schedule |

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
START → input_guard ─(blocked)→ refuse → END
             └→ router ─(Send per intent)→ {qa|portfolio|market|goals|news|tax}_agent
                                                     └→ synthesizer → output_guard → END
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
- The synthesizer is deterministic today (one result → as is; several → one headed section per agent,
  plus failure notices). The LLM merge comes on Day 7.
- `output_guard`: (Day 7) directive detector → rewrite (REQ-GR-04); adds the disclaimer except to
  pure out-of-scope refusals (REQ-GR-03). History stores the answer *without* the disclaimer.
- Checkpointer: `InMemorySaver` with a serializer that allowlists our Pydantic types; LangGraph refuses
  to deserialize unregistered classes from saved state.
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
| Portfolio | market history → `portfolio_engine` | portfolio | Explain metrics, flags, educational recommendations |
| Market | `market_service.quote/trend/overview` | markets | Explain moves and indicators, no predictions |
| Goals | `risk_profile`, `goal_engine` | retirement, portfolio | Explain plan, risk fit, levers |
| News | `market_service.news` | markets | Summarize, dedupe, contextualize |
| Tax | – | tax | Explain, cite, state tax year, refer to a professional |

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
- **Citations:** the agent sees "[n] Title — Section (url)"; returned citations are deduplicated per
  article.
- **Evaluation (REQ-RAG-06):** `tests/evals/retrieval_set.yaml` — 40 cases (keyword, paraphrase,
  jargon, off-topic). Results (2026-10-07, `make eval`):

  | Method | recall@5 | MRR | paraphrase recall@5 | off-topic rejected |
  |---|---|---|---|---|
  | Hybrid (BM25 + FAISS, RRF) | **0.972** | 0.885 | **0.933** | **100%** |
  | FAISS only | 0.944 | 0.911 | 0.867 | – |
  | BM25 only (degraded) | 0.611 | 0.597 | 0.200 | 100% |

  Hybrid wins on paraphrases; BM25 alone is a fallback, not a design. Known miss: the metaphor "how
  bumpy is the ride?" doesn't reach the volatility article (articles were not tuned to the eval set).

## 10. Configuration

`config.yaml` holds non-secret settings (LLM roles, TTLs, thresholds, limits). `.env` holds
`OPENAI_API_KEY`, `GOOGLE_API_KEY` (Gemini), `ALPHAVANTAGE_API_KEY`. Loaded by `pydantic-settings` into a typed
`Settings` object; env vars override YAML (12-factor).

## 11. Testing strategy

| Layer | Approach |
|---|---|
| Engines | Pure unit tests + `hypothesis` property tests (e.g. weights sum to 1) |
| Data layer | Fake providers; `respx`/monkeypatch for HTTP; time frozen with `freezegun` |
| Agents/graph | `GenericFakeChatModel` scripted responses; assert routing, fan-in, degradation |
| API | `httpx.AsyncClient` against the FastAPI app with services faked |
| LLM gateway | Role → model resolution from config; fallback fires when the primary fake raises; cost callback math; no vendor imports outside `core/llm.py` |
| Evals (opt-in, `-m live`) | Real models: router accuracy, retrieval recall, disclaimer presence, structured-output contract on OpenAI and Gemini |
| LLM-as-judge (opt-in, `-m judge`) | Gemini scores ~30 golden Q&As from 1 to 5 on groundedness (claims supported by the cited chunks), citation faithfulness, compliance (no directives, disclaimer present), and clarity for the profile's level. Scores go to `docs/benchmarks.md` |

Every test docstring starts with the requirement ID(s) it verifies. `scripts/trace_matrix.py`
lists requirements that have no test.
