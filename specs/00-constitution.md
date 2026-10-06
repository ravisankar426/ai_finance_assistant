# Constitution — AI Finance Assistant

Non-negotiable principles. Every spec, design choice, and line of code must comply.
If a requirement conflicts with this document, this document wins and the requirement is revised.

## P1. Education, not advice
The system educates; it never tells a specific user to buy, sell, or hold a specific security.
Every response that touches investing or taxes carries a disclaimer. Specific tax or legal
situations are referred to a licensed professional.

## P2. Numbers come from code
LLMs never perform arithmetic that is shown to the user. Portfolio metrics, projections,
returns, and tax-bracket math are computed by deterministic, unit-tested Python. The LLM
only explains numbers it receives from tools.

## P3. Grounded or honest
Knowledge-base answers cite their sources (title + URL). If retrieval finds nothing above
the relevance threshold, the system says so instead of inventing an answer.

## P4. Degrade gracefully
Failure of any external dependency (LLM, market data, news) yields a partial answer plus a
plain-language notice — never a stack trace, never a silent wrong answer. Stale data is
labeled as stale with its timestamp.

## P5. Spec → test → code
Every requirement has an ID (e.g. `REQ-PF-02`). Every requirement has at least one test whose
docstring cites that ID. Code with no requirement behind it is not written.

## P6. Interfaces at the boundaries
The LLM, vector store, market-data source, and cache sit behind small interfaces so they can
be swapped or faked in tests without touching agent logic. No code outside the LLM gateway
knows which LLM vendor is in use; changing provider is a configuration change, not a code change.

## P7. Configuration over code; secrets only in the environment
Model names, TTLs, thresholds, and limits live in `config.yaml`. API keys live only in
environment variables / `.env` (git-ignored). No secret is ever logged.

## P8. Observable by default
Every request gets a `request_id`. Each graph node logs latency; each LLM call logs
model, tokens, and estimated cost. Logs are structured (JSON) and PII-redacted.

## P9. Boring under deadline
Prefer the simplest option that meets the requirement. Anything optional (MCP, LLM-as-judge
evals, tracing SaaS) is cut before anything required.
