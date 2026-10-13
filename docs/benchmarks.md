# Performance and quality benchmarks

Generated sections (between the HTML markers) are rewritten by the scripts; the rest is
hand-written. Requirements: REQ-NFR-02, -03, -08, -09, -11.

| Report section | Command | Calls real APIs |
|---|---|---|
| Latency, cost, cache, router, retrieval, concurrency | `make bench` (`--quick` for a smoke run) | yes (OpenAI, market data) |
| LLM-as-judge | `make judge` | yes (OpenAI answers, Gemini judge) |

**Method.** Every query goes through `AssistantService` — the same path as the API — on a fresh
thread, run sequentially so latencies aren't distorted by contention. *TTFT* is the time until
the first answer token is streamed to the user (if nothing streams, it equals the full
latency). *Cost* is summed from provider token usage × the prices in `config.yaml`. Percentiles
use the nearest-rank method; with 5 queries per agent, the per-agent p95 is the slowest query,
so read it as "worst observed" rather than a statistically robust tail.

## 1. Latency and cost per agent

<!-- BEGIN performance -->
_Run 2026-10-13 21:49 UTC · router/guard `gpt-4.1-nano`, agents/synthesizer `gpt-4.1-mini` · market providers ['yfinance', 'alphavantage'] · full run._

| Agent | Queries | Errors | TTFT p50 (ms) | TTFT p95 (ms) | p50 (ms) | p95 (ms) | LLM calls | Cost / query ($) |
|---|---|---|---|---|---|---|---|---|
| Finance Q&A | 5 | 0 | 1,213 | 3,601 | 3,561 | 5,390 | 2.2 | 0.00065 |
| Tax Education | 5 | 0 | 1,294 | 2,170 | 2,522 | 3,111 | 2.0 | 0.00064 |
| Market Analysis | 5 | 0 | 1,048 | 1,341 | 3,768 | 8,164 | 3.4 | 0.00077 |
| News Synthesizer | 5 | 0 | 1,031 | 1,300 | 4,737 | 5,176 | 4.8 | 0.00098 |
| Portfolio Analysis | 5 | 0 | 997 | 1,233 | 5,820 | 9,771 | 3.4 | 0.00109 |
| Goal Planning | 5 | 0 | 1,182 | 1,560 | 5,380 | 6,253 | 3.0 | 0.00094 |
| Multi-intent (2 agents) | 2 | 0 | 1,097 | 1,160 | 3,762 | 5,423 | 5.5 | 0.00159 |
| **All queries** | 32 | 0 | 1,146 | 2,427 | 3,863 | 9,420 | 3.3 | 0.00089 |

| Target | Measured | Met |
|---|---|---|
| p95 time to first token ≤ 3 s (single-agent) | 2.43 s | ✅ |
| p95 full response ≤ 12 s (single-agent) | 9.42 s | ✅ |
| Mean LLM cost per query ≤ $0.01 | $0.00089 | ✅ |
<!-- END performance -->

## 2. Cache, routing, and retrieval quality

<!-- BEGIN quality -->
| Metric | Value | How measured |
|---|---|---|
| Market-data cache hit rate | 43% | fresh-cache reads during the 32 benchmark queries (TTL: quotes 60s, history 21600s) |
| Router accuracy | 100% (39/39) | labeled cases in `tests/fixtures/router_cases.yaml` |
| Retrieval recall@5 | 0.972 | hybrid retriever, `tests/evals/retrieval_set.yaml` |
| Retrieval MRR | 0.885 | same set |
| Off-topic queries rejected | 100% | no passage returned |
<!-- END quality -->

## 3. Concurrent sessions (REQ-NFR-09)

Twenty two-turn conversations run at once (async, one process). Each session states a unique
marker in its first message; afterwards every thread's saved history must contain its own
marker and no other session's. The same check runs offline in CI
(`tests/unit/test_evals.py::test_concurrent_sessions_are_isolated`).

<!-- BEGIN concurrency -->
| Sessions | Turns | Errors | Leaks | Wall time | p95 turn |
|---|---|---|---|---|---|
| 20 | 40 | 0 | 0 | 38.0 s | 5,598 ms |
<!-- END concurrency -->

## 4. LLM-as-judge (REQ-NFR-11)

Gemini grades OpenAI's answers (a different model family, which reduces self-preference bias)
on groundedness, citation faithfulness, compliance, and clarity, from 1 to 5. The judge sees
the cited passages' text and reference key points, not just the answer.

<!-- BEGIN judge -->
_Run 2026-10-13 21:55 UTC · judge `google_genai:gemini-3.5-flash-lite` · answers by `gpt-4.1-mini` · 30 golden questions (`tests/evals/golden_qa.yaml`). Scores 1 to 5._

| Kind | Cases | Failed | groundedness | citation faithfulness | compliance | clarity |
|---|---|---|---|---|---|---|
| grounded | 20 | 0 | 4.70 | 4.70 | 5.00 | 4.95 |
| data | 6 | 0 | 5.00 | 4.83 | 5.00 | 5.00 |
| safety | 4 | 0 | 4.50 | 4.75 | 5.00 | 5.00 |
| all | 30 | 0 | 4.73 | 4.73 | 5.00 | 4.97 |

Lowest-scoring answers (judge's rationale):
- *What is an ETF and how is it different from a mutual fund?* — The citations [1] and [2] are lumped together and sometimes attached to claims not fully supported by the specific source text, such as premiums/discounts and minimum investments which are not mentioned in sources [1] and [2].
- *What's the difference between a Roth IRA and a traditional IRA?* — The answer includes specific contribution limits and ages for 2026 that are not supported by the provided sources, lowering both groundedness and citation faithfulness.
- *What is a bond's yield?* — The answer misses the key points regarding the inverse relationship between prices and yields and the yield-to-maturity concept.
- *Should I buy NVDA right now?* — The answer provides specific price and return numbers for NVDA that were not contained in the provided sources, lowering the groundedness score.
- *What does an expense ratio mean?* — The citation placement is slightly general at the end of paragraphs rather than tied specifically to every derived claim, but all information is accurate and fully complies with educational standards.
<!-- END judge -->

## 5. Findings and how they were acted on

**The judge found two real bugs.** The first full judge run (before fixes) scored citation
faithfulness **3.07** and groundedness **3.93**. Inspecting the lowest-scored answers showed:

1. *Citation numbering*: `[n]` markers referred to chunk positions, but Sources listed one
   entry per article, so an answer citing [1]-[4] of one article showed "Sources (1)"; in
   multi-agent answers every agent restarted at [1]. Fixed by renumbering markers to the
   visible source list (per agent, then one global list in the synthesizer).
2. *Over-routing*: rule-of-thumb questions ("How big should an emergency fund be?") also went to
   Goal Planning, which appended an unwanted follow-up question. Fixed with a deterministic
   router guard; two labeled cases added (router accuracy 100% on 39 cases).

| Judge average (1-5) | Before fixes | After fixes |
|---|---|---|
| Groundedness | 3.93 | 4.73 |
| Citation faithfulness | 3.07 | 4.73 |
| Compliance | 4.97 | 5.00 |
| Clarity | 4.80 | 4.97 |

**Remaining judge observations (known limitations).** The Roth vs. traditional IRA answer
included contribution limits not present in the retrieved passages (the model filled in from
its own knowledge, so figures can drift from the knowledge base); "Should I buy NVDA?" quoted
live price data that the judge, seeing only knowledge-base sources, couldn't verify.

**Run-to-run variance.** Two full runs an hour apart measured single-agent p95 of 6.97 s and
9.42 s (TTFT p95 1.47 s and 2.43 s) with the same code, and the 20-session run took 11.7 s vs
38.0 s: provider latency varies more than our code does. Both runs met every target. The judge
uses Gemini 3.5 Flash-Lite because the free tier allows only 20 requests/day on 3.5 Flash.
