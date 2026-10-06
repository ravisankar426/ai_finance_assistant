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
| ADR-06 | **FAISS** (`IndexFlatIP` on normalized vectors = exact cosine search), saved to disk, via LangChain's FAISS wrapper. Category filtering = over-fetch (`fetch_k`) then filter on chunk metadata | Chroma → filters metadata inside the database, but adds a heavier dependency and its own storage format. Pinecone → network + account for ~1k chunks. IVF/HNSW indexes → approximate search only pays off above roughly 100k vectors; with ~1k, exact search takes under 1 ms |
| ADR-14 | LLM-as-judge runs on **Gemini** (a different provider from the OpenAI generator) | Same model as judge → self-preference bias (models rate their own outputs higher). Human grading → too slow to repeat on every change |
| ADR-07 | Hybrid BM25 + vector with RRF (k=60) | Vector-only → misses exact terms like "401(k)", "wash sale" |
| ADR-08 | yfinance primary, Alpha Vantage fallback | Either alone → yfinance is unofficial and can break; AV free tier is tiny |
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
- The router returns `list[Send]`, one per intent. This is how LangGraph runs a dynamic fan-out.
- Every agent node is wrapped by `safe_node()`: it applies a timeout, catches exceptions, and returns
  `AgentResult(error=...)` instead of raising (REQ-WF-05).
- The synthesizer skips the LLM when there's one result with no errors (saves latency and cost);
  otherwise it merges the results and de-duplicates citations.
- `output_guard`: directive detector → rewrite (REQ-GR-04), then add the disclaimer (REQ-GR-03).

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

## 8. Resilience (`data/resilience.py`)

Order of a provider call: validate ticker → cache hit? → circuit open? skip → rate-limit
acquire → call with retry (`tenacity`, exponential backoff + jitter) → on success cache + reset breaker
→ on final failure record breaker failure → try next provider → all failed → stale cache
(flag `stale=True`) → else raise `MarketDataUnavailable`.

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

- Articles: `data/knowledge_base/<category>/<slug>.md` with YAML front-matter (REQ-RAG-01).
- Chunking: split by Markdown headers, then recursive split at ~500 tokens with 75 overlap; each chunk is
  prefixed with the article title (adds context for embedding).
- Store: FAISS `IndexFlatIP` over L2-normalized vectors (inner product = cosine similarity). The index and
  the docstore (chunk text + metadata) are saved to `data/index/` with `save_local`.
- Ingest (REQ-RAG-02): embeddings are cached on disk, keyed by the chunk's content hash (LangChain
  `CacheBackedEmbeddings`). On each ingest the flat index is **rebuilt** from cached vectors: only new or changed chunks
  call the embeddings API. Rebuilding a ~1k-vector flat index takes milliseconds, which is simpler
  and safer than deleting vectors in place.
- Category filter (REQ-RAG-04): vector search with `fetch_k=50`, filtered on `metadata.category`, then top 20.
  The BM25 index applies the same filter.
- Retrieve: vector top-20 + BM25 top-20 → RRF → top-k (default 5) → threshold (REQ-QA-02).
- Degraded mode (REQ-LLM-05): if the embeddings API fails, return BM25 results with `degraded=True`.
- Security: LangChain's FAISS loader uses pickle (it requires `allow_dangerous_deserialization=True`). We only load
  the index our own ingest job produced, from a path in config — never a file from a user.
- Eval: `tests/evals/retrieval_set.yaml` (question → expected article ids), recall@5 and MRR.

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
