# AI Finance Assistant — Technical Design

**Audience:** reviewers and engineers who want to understand how the system works and why it
is built this way. **Scope:** the delivered MVP. The working specification (requirements with
IDs, task plan, traceability) lives in [`specs/`](../specs/); measured results live in
[`benchmarks.md`](benchmarks.md).

**Contents**
1. [Goals and constraints](#1-goals-and-constraints)
2. [System architecture](#2-system-architecture)
3. [Architecture decisions](#3-architecture-decisions)
4. [Agent communication protocol](#4-agent-communication-protocol)
5. [The six agents](#5-the-six-agents)
6. [RAG implementation](#6-rag-implementation)
7. [LLM gateway and provider strategy](#7-llm-gateway-and-provider-strategy)
8. [Market data integration and resilience](#8-market-data-integration-and-resilience)
9. [Guardrails and compliance](#9-guardrails-and-compliance)
10. [Interfaces: UI, REST API, MCP](#10-interfaces-ui-rest-api-mcp)
11. [Performance considerations](#11-performance-considerations)
12. [Quality: testing and evaluation](#12-quality-testing-and-evaluation)
13. [Security and privacy](#13-security-and-privacy)
14. [Known limitations](#14-known-limitations)

---

## 1. Goals and constraints

The assistant teaches personal finance to beginners: it explains concepts, analyzes a
portfolio the user describes, reports market data, plans savings goals, summarizes news, and
explains US tax rules. Four constraints shaped every decision:

| Constraint | Consequence in the design |
|---|---|
| **Education, not advice** (regulatory) | No buy/sell/hold directives or guarantees, enforced in code after generation, not only in prompts (§9). A disclaimer on every finance answer. |
| **Numbers must be right** | LLMs never do math. Every metric (returns, beta, required saving) comes from deterministic Python engines; the LLM only explains a fact block (§5). |
| **External services fail** | Two LLM providers, two market-data providers, caches with stale fallback, and a keyword router and BM25 retrieval as degraded modes (§7, §8). Configuration errors fail loudly instead. |
| **Small team, two weeks** | One process, in-process state, no extra infrastructure; every component behind an interface so Redis/Postgres can replace in-memory parts later (§11). |

---

## 2. System architecture

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

**Layers.** Three interfaces (UI, API, MCP) translate their protocol into calls on one
facade, `AssistantService`, so each operation — chat, analyze a portfolio, quote a ticker,
plan a goal, search the knowledge base — is implemented and tested once. Chat goes through the
LangGraph workflow; the tool operations call the engines directly. Below that, four
subsystems have no knowledge of each other's internals: market data, finance engines,
retrieval, and the LLM gateway (the only module that imports vendor SDKs).

**Lifecycle of one chat turn** ("What's NVDA trading at, and any news on it?"):

1. **Input guard** redacts PII in place (emails, phone numbers, SSNs, account numbers) and
   removes prompt-injection sentences. If nothing legitimate remains, the turn ends with a
   refusal and no model is called.
2. **Router** (`gpt-4.1-nano`, structured output) returns intents `[market, news]` plus a
   standalone rewrite of the question; deterministic guards then correct known LLM mistakes
   (tax terms → Tax; follow-ups about remembered holdings → Portfolio; rule-of-thumb
   questions → Q&A, not Goals). If the router call fails, a keyword router takes over.
3. **Fan-out:** a conditional edge returns one `Send` per intent, so Market and News run **in
   parallel**, each receiving only the slice of state it needs.
4. **Agents** follow one pipeline: check inputs → compute with tools → retrieve concepts →
   one LLM call to explain. Market fetches the quote and history, computes the trend snapshot
   in Python, retrieves 3 concept passages, then explains the computed facts.
5. **Synthesizer** merges results in a fixed order with headings, maps every agent's citation
   numbers onto one source list, adds failure notices for agents that degraded, and (for
   multi-agent answers) an "In short" overview that is rejected if it introduces numbers.
6. **Output guard** detects directive or guarantee language, asks the guard model to rewrite
   it as education, re-checks, strips anything still non-compliant, and appends the disclaimer.
7. **Compaction:** if the thread exceeds 16 messages, older turns become one summary.
8. The **checkpointer** saves state per `thread_id`; the API streams tokens over SSE while
   agents write, then a final event with answer, agents, citations, chart data, and cost.

---

## 3. Architecture decisions

Each decision lists the alternatives considered and why they lost.

| # | Decision | Alternatives → why rejected |
|---|---|---|
| 1 | **LangGraph** for orchestration | CrewAI: routing is implicit in agent conversation, harder to test and to make deterministic. A hand-written loop: would re-implement state reducers, checkpointing, and streaming. |
| 2 | **Router = one small-model structured-output call + deterministic guards + keyword fallback; parallel fan-out with `Send`** | A supervisor ReAct agent: several LLM round trips per turn, slower, costlier, nondeterministic. Pure keyword routing: brittle on paraphrases (the keyword baseline is the fallback, not the design). |
| 3 | **All math in Python engines; the LLM explains a fact block** | LLM arithmetic: unreliable and untestable. Tool-calling loops: more latency and failure modes; the agent always needs the same computation, so it runs before the LLM call. |
| 4 | **Provider-agnostic LLM gateway**: code asks for a *role* (`router`, `guard`, `agent`, `synthesizer`, `judge`); `config.yaml` maps roles to provider and model. OpenAI primary, Gemini fallback. | Vendor SDKs directly: different APIs, switching means rewriting. Same-provider fallback: an outage takes everything down. |
| 5 | **Fail loud on configuration errors** (401/403/404, unknown error types) — never fall back | Falling back on a bad key hid the misconfiguration during testing: the app "worked" on the backup provider at different cost and quality. |
| 6 | **FAISS `IndexFlatIP` used directly**, metadata as JSON | LangChain's FAISS wrapper pickles its docstore (code-execution risk on load). Chroma/Pinecone: extra storage engine or network account for 297 chunks. Approximate indexes (IVF/HNSW) pay off around 100k+ vectors; exact search here is sub-millisecond. |
| 7 | **Hybrid retrieval: BM25 + vectors fused with RRF** | Vector-only misses exact terms ("401(k)", "wash sale"); BM25-only misses paraphrases (measured in §6). |
| 8 | **yfinance primary, Alpha Vantage fallback**, behind one provider interface, plus recorded fixtures | Either alone: yfinance is unofficial and breaks without notice; Alpha Vantage's free tier is 25 requests/day. |
| 9 | **In-house resilience primitives** (TTL cache, rate limiter, breaker, retry) with injectable clocks | `tenacity`/`cachetools`/`pybreaker`: three dependencies for ~150 lines, harder to test deterministically. |
| 10 | **One service facade**; the Streamlit UI calls it in-process | UI over HTTP: two processes to run and test for no functional gain; the API exposes the same facade for other clients. |
| 11 | **Guards as graph nodes in code** (regex + rewrite + re-check) | Prompt-only rules can't be enforced or tested. |
| 12 | **LLM-as-judge on a different model family** (Gemini grades OpenAI) | Same-family judging is biased toward its own outputs; human grading can't be repeated on every change. |

---

## 4. Agent communication protocol

Agents never call each other. All communication goes through the graph state, with one
contract per hop:

```python
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation memory
    profile: UserProfile  # knowledge level, risk, goals
    portfolio: list[Holding]  # remembered holdings
    intents: list[Intent]  # router output
    question: str  # standalone rewrite
    agent_results: Annotated[list[AgentResult], add_or_reset]  # fan-in
    final_answer: str
    citations: list[Citation]
    blocked: bool  # set by the input guard
    request_id: str
```

| Hop | Mechanism | Payload |
|---|---|---|
| Router → agent | `Send("agent_<name>", payload)`; agent nodes declare `input_schema=AgentInput` | `messages`, `question`, `profile`, `portfolio`, `request_id` — only what an agent needs |
| Agent → synthesizer | Each agent returns `{"agent_results": [AgentResult]}`; the **reducer** concatenates parallel writes | `AgentResult`: `answer`, `citations`, `data` (chart payload), `error`, `follow_up_question`, `state_updates` |
| Agent → memory | `state_updates` applied by the node wrapper, **allowlisted keys only** (`portfolio`, `profile`) | e.g. holdings parsed from "I own 10 AAPL", a stated risk tolerance |
| Synthesizer → client | `final_answer`, merged `citations`, per-agent `data` | rendered by UI/API; `data` drives charts |

**Why reducers matter.** Two agents running in parallel both write `agent_results`. Without
a reducer LangGraph rejects concurrent writes to one key; with it, the lists are merged. The
custom `add_or_reset` reducer also lets the input guard clear the previous turn's results.

**Failure isolation.** Each agent runs inside `run_safely()`: an exception becomes
`AgentResult(error=...)`, the other agents' answers still return, and the synthesizer adds a
notice ("Market data is unavailable right now"). Configuration errors are re-raised.

**Scope in multi-agent turns.** Each `Send` payload also lists the *other* agents answering
the turn, and the agent's prompt says what they cover ("Other specialists are covering: news
headlines"). Without this, the Market agent answering "NVDA price and news?" would add "I
don't have news", while the News agent beside it said it had no price.

**Clarifying questions.** When inputs are missing ("What's the stock price?" with no ticker,
a goal without amount or horizon), the agent returns `follow_up_question` instead of guessing;
the answer to that question arrives as the next turn and is resolved from memory.

**Memory.** The checkpointer (`InMemorySaver`, or `SqliteSaver` by config) stores state per
`thread_id`, so follow-ups ("what about its fees?") work: the router rewrites them into
standalone questions using recent history. Long threads are compacted into
`[summary, last 6 messages]` by removing all messages and rebuilding, so the summary comes
first. A serializer allowlist means only our own Pydantic types can be deserialized from
saved state.

**Extensibility.** A seventh agent is one class implementing `run(AgentInput) -> AgentResult`,
one node, and one router label; nothing else changes.

---

## 5. The six agents

Every agent follows **check inputs → compute → retrieve → explain** (one LLM call).

| Agent | Compute (deterministic) | Retrieve (KB categories) | Explains |
|---|---|---|---|
| **Finance Q&A** | — | all except tax | Concepts, with citations and "learn next" articles suited to the reader's level |
| **Portfolio Analysis** | Holdings from the message (structured output + regex fallback), CSV, or memory → weights, asset classes, GICS sectors, HHI and effective holdings, volatility, max drawdown, Sharpe, beta vs SPY, unrealized P&L, concentration flags | portfolio | What the metrics mean; observations framed as education, not trades |
| **Market Analysis** | Ticker extraction → quote and ~400 days of history → 50/200-day averages, 1m/3m/1y returns, 52-week range, volatility, trend label; index + 11-sector overview | markets, portfolio | The computed facts; a one-day move is never called a trend; no predictions; data source and freshness footer |
| **Goal Planning** | Goal parameters from recent turns + risk profile (message, memory, or 5-question quiz) → inflation-adjusted target, required monthly saving under 3 return scenarios, yearly projections, gap levers, illustrative allocation | retirement, portfolio, basics | The plan, assumptions, and trade-offs; asks for whatever is missing |
| **News Synthesizer** | Headlines (lookback window, relevance and focus ranking, de-duplication) | markets | A summary of **headlines only**, each linked; no article text, no predictions |
| **Tax Education** | — | tax | US rules with the tax year stated; refers to a professional |

"Recommendations" in the brief's sense are delivered as **educational observations**: the
Portfolio agent flags a holding above 25% or a missing bond allocation and explains why
diversification matters, without telling the user what to buy or sell.

---

## 6. RAG implementation

**Knowledge base.** 60 Markdown articles in 6 categories (basics, investing, markets,
portfolio, retirement, tax), each with validated front matter: id, category, difficulty,
source name and URL, review date, and `tax_year` for year-specific figures. Sources are US
government pages (SEC/Investor.gov, IRS, SSA, CFPB, FINRA, FDIC, Federal Reserve) plus CFA
Institute, MSCI, and Fidelity where no government page exists. Tax figures were checked by
hand against IRS pages.

**Chunking.** One chunk per `##` section (sections are topical units); sections over 250
words are split into overlapping 40-word windows. The text that gets embedded is prefixed
with "Article title — Section", so short chunks keep their context. Result: 297 chunks,
median 37 words, ids `<article_id>#<n>`.

**Embeddings and index lifecycle.** `text-embedding-3-small` through the gateway; vectors are
cached on disk keyed by sha256(model + text), so only new or changed chunks call the API. At
startup `load_or_build()` compares a fingerprint (articles, model, chunking settings) with the
saved manifest: match → load with no API calls; mismatch → rebuild (cheap thanks to the cache).
The index is FAISS `IndexFlatIP` over normalized vectors (exact cosine); metadata is JSON.

**Retrieval.**
1. Category mask (agents restrict to relevant categories; Tax content is excluded from Q&A).
2. Two rankings over the candidates: FAISS (meaning) and BM25 (exact terms).
3. **Reciprocal Rank Fusion** (k = 60): uses ranks only, so the two score scales never need
   calibrating against each other.
4. **Cosine gate** (`min_cosine` = 0.3): chunks below it are dropped; if none remain, the agent
   says the topic isn't covered instead of improvising. The value came from a sweep — recall
   is flat from 0.20 to 0.40; off-topic queries score ≤ 0.14 and on-topic ≥ 0.4.
5. Top 4 chunks go to the agent; the remaining ranked articles feed "learn next".

**Degraded mode.** If the embeddings API has a transient failure, retrieval falls back to
BM25 gated by keyword coverage, results are flagged `degraded`, and the answer still cites.
Authentication errors are raised, not degraded.

**Source attribution.** The prompt numbers passages `[1]…[n]`. The user sees one numbered
source per article in reading order, and each agent's markers are rewritten to that
numbering; the synthesizer then maps every agent onto one global list, so `[2]` always means
the second source shown. (The LLM judge found the earlier mismatch; §12.)

**Measured quality** (40-case eval set: keyword, paraphrase, jargon, off-topic):

| Method | recall@5 | MRR | paraphrase recall@5 | off-topic rejected |
|---|---|---|---|---|
| **Hybrid (BM25 + FAISS, RRF)** | **0.972** | 0.885 | **0.933** | **100%** |
| FAISS only | 0.944 | 0.911 | 0.867 | – |
| BM25 only (degraded mode) | 0.611 | 0.597 | 0.200 | 100% |

---

## 7. LLM gateway and provider strategy

`core/llm.py` is the only module that knows vendors exist (a test enforces it). Callers ask
for a role:

| Role | Model (config) | Why this size |
|---|---|---|
| router, guard | `gpt-4.1-nano` | Classification and short rewrites; latency matters most |
| agent, synthesizer | `gpt-4.1-mini` | Explanations for beginners; quality matters, cost stays < $0.001/turn |
| judge | `gemini-3.5-flash-lite` | Different family from the generator; free-tier quota fits a 30-question run |
| fallback (all roles but judge) | `gemini-3.5-flash-lite` | Cross-provider; ~1 s, close to the primary's latency |
| embeddings | `text-embedding-3-small` | No cross-provider fallback: different models' vectors can't share an index; BM25 is the fallback |

Each model is built with `init_chat_model`, wrapped with `.with_fallbacks()` that handles an
**allowlist** of transient error classes (rate limit, 5xx, timeouts, connection errors,
parser errors). Authentication, permission, and not-found errors — and anything not on the
list — propagate, so misconfiguration is visible immediately. A usage callback records
tokens and cost from LangChain's provider-neutral `usage_metadata`. Provider differences are
contained: callers read `message.text` (Gemini returns content blocks, OpenAI a string), all
structured output goes through `.with_structured_output(PydanticModel)`, and a live contract
test sends every schema to both providers.

Switching providers is a config change: install the LangChain integration, add the key,
change `provider`/`model` in `config.yaml`.

---

## 8. Market data integration and resilience

```
get_quote("$aapl")
  └─ normalize_ticker → "AAPL" (format check before any network call)
  └─ cache fresh?     → return (TTL: quote 60 s, history 6 h, profile 24 h, news 15 min)
  └─ for provider in [yfinance, alphavantage]:
        circuit open?     → skip (5 consecutive failures → 60 s cool-down → half-open probe)
        rate limit?       → wait if a slot frees within 2 s, else skip (daily quota used up)
        call with retry   → transient errors only, exponential backoff + full jitter, 3 attempts
          ok              → cache, close breaker, return
          no data         → next provider, no breaker penalty
          config error    → raise (bad key fails loudly)
  └─ all failed: stale cached copy → return marked stale
                 every provider said "no data" → InvalidTickerError ("check the symbol")
                 otherwise → MarketDataUnavailableError
```

Provider quirks observed live are normalized in the adapters: yfinance returns empty frames
both for unknown tickers *and* during outages (so a ticker is only "invalid" when every
provider says no data); Alpha Vantage reports throttling as HTTP 200 with a message, its free
history is unadjusted and limited to 100 days, and sector names are mapped to GICS. Every
value carries its `source` and `as_of`, and stale or recorded data is labeled as such in
answers and charts. Recorded fixtures for 26 tickers power the tests and an offline demo
mode (`MARKET__PROVIDERS='["fixture"]'`).

---

## 9. Guardrails and compliance

| Risk | Control | Where |
|---|---|---|
| Personal data reaching models or logs | PII redaction in place (same message id), before the LLM and the checkpoint store; log processor redacts again | `input_guard`, `utils/redaction.py` |
| Prompt injection | Regex removes override sentences before any model call; a pure injection is refused with zero LLM calls | `input_guard` |
| Personalized advice / guarantees | Detector → guard-model rewrite → **re-check** → strip remaining sentences; works with no model at all | `output_guard`, `core/guards.py` |
| Missing disclaimer | Appended in code to every finance answer | `output_guard` |
| Out-of-scope questions | Router intent `out_of_scope` → polite refusal, no disclaimer | router, synthesizer |
| Invented numbers in summaries | Overview rejected if it contains numbers the agents didn't produce | synthesizer |
| Wrong tax year | Tax answers state the tax year from article metadata; referral to a professional | Tax agent |

The detector exempts warnings *about* guarantees ("promises of guaranteed returns are a red
flag"): an early version deleted exactly that sentence from scam-education answers, which
recording the sample conversations revealed. The compliance guarantee doesn't depend on the
LLM behaving: if the rewrite still contains a
directive, or the guard model is down, the offending sentences are removed deterministically.
The LLM judge scored compliance **5.00/5** across 30 golden questions, including four
advice-bait questions.

---

## 10. Interfaces: UI, REST API, MCP

**Streamlit UI** — five tabs; the disclaimer sits above them so it shows on every tab;
columns stack on narrow screens. Tabs are lazy: only the open tab's code runs, so chatting
never triggers the Market tab's data fetches.
- *Chat*: streamed answer, "Answered by" badges, numbered sources, learn-next links.
- *Portfolio*: editable table / CSV / samples → 8 metric cards, flags, asset-class donut,
  sector bars, growth of 100 vs SPY, holdings table. Analyzed holdings are shared with Chat.
- *Market*: index metrics, sector heatmap, ticker chart with 50/200-day averages, as-of time,
  source, refresh (re-fetches only when the cache has expired).
- *Goals*: risk quiz, goal form, pessimistic–optimistic band with base path and target line,
  gap levers, illustrative allocation.
- *Learn*: browse the 60 articles by category or title.

**REST API** (FastAPI, OpenAPI at `/docs`): `POST /chat` (SSE: `token` events, then `final`),
`POST /chat/sync`, `POST /portfolio/analyze`, `GET /market/quote/{t}`, `GET /market/trend/{t}`,
`GET /market/overview`, `POST /goals/plan`, `GET /goals/questionnaire`,
`POST /goals/risk-profile`, `GET /learn/articles`, `GET /health`. Error contract: 422 with
field-level details; expected failures map to 404/503 with a user-safe message; anything
else is a 500 with only a request id. Every response carries `X-Request-ID`, which appears in
every log line for that request.

**MCP server** (stdio, MCP SDK 2.x): `get_quote`, `market_overview`, `analyze_portfolio`,
`plan_goal`, `search_knowledge_base`. Tools are deterministic (the client's model reasons),
return typed structured content with output schemas, map expected failures to tool errors
with safe messages, log to stderr (stdout carries JSON-RPC), and build the service lazily so
the handshake is instant. Claude Desktop configuration is in the README.

---

## 11. Performance considerations

**Latency budget.** A single-agent turn makes 2–3 LLM calls: router (nano, ~0.5–1 s),
optional extraction (nano), and the agent's explanation (mini, streamed). Streaming is what
makes it feel fast: the first token arrives in ~1.1 s (p50) even when the full answer takes
3–6 s. Measured over 32 queries (full table in `benchmarks.md`):

| Metric | Target | Measured (two runs) |
|---|---|---|
| p95 time to first token, single-agent | ≤ 3 s | 1.47 s / 2.43 s |
| p95 full response, single-agent | ≤ 12 s | 6.97 s / 9.42 s |
| Mean LLM cost per query | ≤ $0.01 | $0.00086 / $0.00089 |
| 20 concurrent two-turn sessions | no errors or leaks | 0 errors, 0 leaks |

Design choices behind these numbers:
- **Small models where quality allows** (routing and guards on nano) and **one explanation
  call per agent** (math precomputed, no tool loops).
- **Parallel agents** via `Send`: a two-agent question costs about one agent's latency, not two.
- **Guard rewrite only when needed**: the output guard calls a model only if the regex
  detector flags something, so clean answers pay nothing.
- **Caching at every expensive boundary**: market data TTL cache (43% hit rate in the
  benchmark, higher in real sessions that revisit tickers), embedding cache (re-indexing
  costs nothing unless articles change), fingerprinted index (startup loads, doesn't rebuild).
- **Exact vector search** is sub-millisecond at 297 chunks; retrieval is never the bottleneck.
- **Bounded context**: history compaction keeps prompts small on long threads.
- **Rate-limit aware fetching**: short waits for per-second limits, skip on daily quotas, so a
  slow fallback never blocks a turn for long.

**Scaling path.** The API is async and keeps no per-request state in memory; session state is
in the checkpointer and market data in the cache, both behind interfaces. Moving to several
workers means `PostgresSaver` for sessions and Redis for the cache and rate-limit counters —
configuration and adapter changes, not changes to agents or the graph. The concurrency
benchmark shows sessions stay isolated today.

---

## 12. Quality: testing and evaluation

- **428 offline tests, 97% line and branch coverage** (CI gate: 80%), plus 12 opt-in live
  tests. Every test docstring cites the requirement IDs it verifies; `scripts/trace_matrix.py
  --strict` fails CI if any MUST requirement lacks a test.
- **Offline by design:** scripted fake models (including failures), hashing embeddings for an
  offline index, recorded market data, injectable clocks. CI never calls a paid API.
- **Property-based tests** (Hypothesis) for the math: weights sum to 1, the required monthly
  saving lands on the target, moving averages match their definition. Hypothesis found a
  floating-point drift in the chart's running-sum moving average, now summed exactly.
- **Interfaces tested end to end:** FastAPI `TestClient`, the MCP SDK's in-memory client, and
  Streamlit `AppTest`, all over a real `AssistantService` built on fakes.
- **Live evaluations:** router accuracy 100% on 39 labeled cases; retrieval recall@5 0.972;
  structured-output contract on both providers.
- **LLM-as-judge** (30 golden questions; Gemini grading OpenAI with the cited passages' text):

  | Criterion (1–5) | First run | After fixes | Latest run |
  |---|---|---|---|
  | Groundedness | 3.93 | 4.73 | 4.80 |
  | Citation faithfulness | 3.07 | 4.73 | 4.73 |
  | Compliance | 4.97 | 5.00 | 5.00 |
  | Clarity | 4.80 | 4.97 | 5.00 |

  The first run exposed two real defects — citation numbers that didn't match the visible
  source list, and rule-of-thumb questions over-routed to Goal Planning — both fixed with
  regression tests. This is the main argument for running a judge: it found bugs that unit
  tests, written by the same author as the code, did not.

---

## 13. Security and privacy

- Secrets only in `.env` (git- and docker-ignored); a test fails if a key pattern appears in
  any tracked file; pre-commit blocks committing `.env` and private keys.
- PII redacted before it reaches an LLM, the checkpoint store, or logs.
- No pickle anywhere: FAISS metadata is JSON; the checkpoint serializer only loads allowlisted
  types.
- API errors never expose internals (a test plants a fake password in an exception and checks
  it doesn't leak); MCP tool errors follow the same rule.
- Docker image runs as a non-root user; keys are injected at run time, never baked in.

---

## 14. Known limitations

- **Model knowledge can leak into answers.** The judge noted a Roth vs. traditional IRA answer
  with contribution limits not present in the retrieved passages. Agents are instructed to use
  only the sources, but this is enforced by prompt, not code.
- **News is headlines only:** free sources give no article text, so summaries are shallow by
  design and say so.
- **Market data is delayed and unofficial:** yfinance can change without notice; the free Alpha
  Vantage fallback covers quotes and 100 days of unadjusted history.
- **Single process:** cache, rate limits, and sessions are in-memory (SQLite optional); see the
  scaling path in §11.
- **US only:** tax content and market data assume the United States.
- **Performance varies with providers:** two runs an hour apart differed by ~35% at p95 with
  identical code; budgets are set with that headroom.
