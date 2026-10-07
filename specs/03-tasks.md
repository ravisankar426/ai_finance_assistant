# Tasks & Schedule — Oct 5 → Oct 16, 2026

Claude writes the code; you review it and learn from it. Budget per day: about 1.5 h of concept walkthrough + about 1 h of
your review and "explain it back" + build time. The Oct 10–11 weekend is **buffer only**.

**Definition of Done (every task):** code + tests citing REQ IDs, CI green, coverage not lower than before,
and you can answer that day's checkpoint questions.

**Cut order if we slip:** T9.3 MCP → T9.2b LLM-as-judge evals → T8.4 Learn tab → T9.4 docker-compose (keep the Dockerfile).

---

## Day 1 — Mon Oct 5 · Foundations + specs sign-off
| ID | Task | REQs |
|---|---|---|
| T1.1 | You review and sign off on specs 00–03 (decisions can change today, cheaply) | — |
| T1.2 | Scaffold: uv project in `ai_finance_assistant/`, prescribed layout, ruff (incl. docstring rules), mypy, pytest, pre-commit, git | NFR-04, NFR-10, DEL-01 |
| T1.3 | `config.yaml` + `Settings` (pydantic-settings), `.env.example`, `requirements.txt` export | NFR-04, DEL-03 |
| T1.4 | JSON logging with request_id, PII redaction util, error hierarchy | NFR-05, GR-05 |
| T1.5 | LLM gateway: role → provider/model from config, retries, OpenAI → Gemini fallback, token/cost callback, no-vendor-import test; confirm the model IDs your OpenAI and Gemini keys can use | LLM-01..04 |
| T1.6 | GitHub Actions CI with coverage gate | NFR-01 |
**Learn:** spec-driven development, 12-factor config, structured logging, the LangChain chat model interface.

## Day 2 — Tue Oct 6 · End-to-end slice
| T2.1 | Graph state + reducers, input_guard → router → qa_agent → output_guard | WF-01, WF-04 |
| T2.2 | Router: structured output + keyword fallback | WF-01, WF-03 |
| T2.3 | Minimal Streamlit chat calling the graph in-process | UI-02 (partial) |
**Learn:** LangGraph StateGraph, reducers, conditional edges, `Send`, checkpointers, thread_id.
**Milestone:** you can chat with a one-agent assistant in the browser.

## Day 3 — Wed Oct 7 · Knowledge base + RAG
| T3.1 | ✅ 60 articles across 6 categories from public sources (SEC, IRS, SSA, CFPB, FINRA, FDIC…); figures checklist for human review | RAG-01, TX-02 |
| T3.2 | Chunker, embedding cache keyed by content hash, FAISS index build/save/load | RAG-02, RAG-03, RAG-05 |
| T3.3 | Hybrid retriever (BM25 + FAISS + RRF), category filter (over-fetch + metadata filter), threshold, BM25-only degraded mode | RAG-03, RAG-04, QA-02, LLM-05 |
| T3.4 | Retrieval eval set (~40 questions) + recall@5 / MRR script | RAG-06 |
**Learn:** embeddings, cosine vs inner product, FAISS index types (Flat vs IVF vs HNSW), chunking trade-offs, BM25, RRF, how to evaluate retrieval.
**Your job today:** spot-check the tax articles' figures against IRS.gov (≈45 min).

## Day 4 — Thu Oct 8 · Q&A + Tax agents, market data layer
| T4.1 | ✅ Base agent contract (check inputs → compute → retrieve → explain), `safe_node` wrapper, QA agent with citations + "learn next" | QA-01..03, WF-05, WF-09, WF-10 |
| T4.2 | ✅ Tax agent (category-filtered RAG, tax-year + pro referral) | TX-01..03 |
| T4.3 | ✅ Provider interface, yfinance + Alpha Vantage providers, Pydantic normalization | MD-01, MD-07 |
| T4.4 | ✅ TTL cache, rate limiter, retry, circuit breaker, stale fallback | MD-02..06 |
**Learn:** resilience patterns (retry, backoff, jitter, circuit breaker, bulkhead), caching TTL strategy.

## Day 5 — Fri Oct 9 · Market + News agents
| T5.1 | Trend analytics (SMA, returns, 52-wk, volatility), market overview | MK-01..04 |
| T5.2 | Market agent | MK-01..04 |
| T5.3 | News fetch + dedupe + News agent | NW-01..04 |
| T5.4 | Recorded fixtures so tests and the demo don't depend on live APIs | DEL-02 |
**Learn (finance primer I):** quotes, indices, ETFs, sectors, SMA, volatility, how to read news without predicting.

## Weekend Oct 10–11 · Buffer / catch-up / re-read code

## Day 6 — Mon Oct 12 · Portfolio + Goals
| T6.1 | Portfolio engine: weights, allocation, HHI, vol, beta, Sharpe, drawdown, P&L, flags | PF-02, PF-03, PF-05 |
| T6.2 | Holding parsing from chat / CSV; sample portfolio CSVs; Portfolio agent | PF-01, PF-04, DEL-02 |
| T6.3 | Risk questionnaire, goal engine (FV, PMT, inflation, scenarios, gap levers), model allocations | GP-01..05 |
| T6.4 | Goals agent; property-based tests for the math | GP-01..05 |
**Learn (finance primer II):** diversification, beta, Sharpe, drawdown, time value of money, risk tolerance.

## Day 7 — Tue Oct 13 · Full orchestration + guardrails + API
| T7.1 | Multi-intent fan-out with `Send`, synthesizer, partial-failure notices | WF-02, WF-05 |
| T7.2 | User profile memory, history summarization, SqliteSaver option | WF-04, WF-07, WF-08 |
| T7.3 | Guards: scope, injection, directive rewrite, disclaimer, knowledge-level adaptation, conservative framing | GR-01..07 |
| T7.4 | Service layer + FastAPI (SSE chat, endpoints, error handling, OpenAPI) | API-01..03, WF-06 |
**Learn:** the advice-vs-education line (SEC/FINRA), prompt injection, SSE streaming, FastAPI dependency injection.

## Day 8 — Wed Oct 14 · Streamlit UI
| T8.1 | App shell, tabs, sidebar profile, disclaimer, API client | UI-01, UI-07 |
| T8.2 | Chat tab: streaming, agent badges, citations | UI-02 |
| T8.3 | Portfolio dashboard + Market tab + Goals tab (Plotly) | UI-03..05 |
| T8.4 | Learn tab | UI-06 |

## Day 9 — Thu Oct 15 · Quality + bonus
| T9.1 | Fill coverage gaps to ≥ 80%, edge cases, traceability check | NFR-01 |
| T9.2a | Benchmark script (latency, cost, cache hit, router accuracy, 20 concurrent sessions) + structured-output contract tests on OpenAI and Gemini → `docs/benchmarks.md` | NFR-02, NFR-03, NFR-08, NFR-09 |
| T9.2b | LLM-as-judge on Gemini: golden set of ~30 Q&As, rubric scoring, averages in `docs/benchmarks.md` | NFR-11, LLM-06 |
| T9.3 | MCP server (FastMCP) + Claude Desktop config | MCP-01, MCP-02 |
| T9.4 | Dockerfile + docker-compose | NFR-06 |

## Day 10 — Fri Oct 16 · Ship
| T10.1 | README (architecture, setup, API, usage, troubleshooting) | NFR-07 |
| T10.2 | Technical design doc: architecture decisions, agent communication protocol, RAG details, performance considerations (from specs/02 + benchmarks) | NFR-07 |
| T10.3 | `docs/demo_script.md` + sample conversations; **you record the 5–10 min video** | DEL-02, DEL-04 |
| T10.4 | Final pass: fresh-clone setup test, interview Q&A rehearsal | — |
