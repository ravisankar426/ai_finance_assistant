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
```

## Layout

```
src/{agents,core,data,rag,web_app,utils,workflow}   tests/   specs/   scripts/   config.yaml
```
