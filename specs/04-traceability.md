# Problem Statement → Requirements Traceability

Every item in the problem statement (sections 1–6) mapped to the requirement(s) that cover it.
Section 7 (Future Directions) is out of scope by agreement.

## §1 Business Use Case

| Problem statement item | Covered by |
|---|---|
| Democratized access — clear, jargon-free explanations | GR-06 |
| Personalized learning paths — adapts to knowledge level, goals, risk tolerance | WF-08, GR-06, QA-03, GP-01 |
| Real-time market integration — learn concepts while seeing real-world data | MK-01..04, WF-09 |
| Scalable guidance — thousands of users simultaneously | NFR-09, design §7b |
| Specialized expertise per agent | Agents QA, PF, MK, GP, NW, TX; design §7 |
| Context preservation — conversation history and user profiles | WF-04, WF-07, WF-08 |
| RAG enhancement — grounded in verified knowledge | RAG-01..06, QA-01..02, WF-09 |
| Regulatory compliance — education vs advice, disclaimers | Constitution P1, GR-03, GR-04, PF-04, TX-03 |
| Accuracy & reliability — accurate, up-to-date, robust API error handling | Constitution P2, MD-01..07, LLM-03, LLM-05, MK-04, RAG-01 (last_reviewed), TX-02, NFR-11 |
| User trust — transparency, citations, conservative recommendations | QA-01, RAG-05, GR-07 |

## §2 Technical Architecture

| Problem statement item | Choice | Covered by |
|---|---|---|
| Multi-agent system: LangChain/LangGraph/CrewAI | LangChain + LangGraph | ADR-01, WF-01..10 |
| Language model: Gemini/GPT/Claude | OpenAI GPT primary, Gemini fallback and judge, behind a provider-agnostic gateway | ADR-04, ADR-14, LLM-01..07 |
| Vector DB: FAISS/Chroma/Pinecone | FAISS | ADR-06, RAG-03, RAG-04 |
| Market data: Alpha Vantage/yFinance | Both (primary + fallback) | ADR-08, MD-01 |
| Web interface: Streamlit/Gradio/React | Streamlit (+ FastAPI) | ADR-10, UI-01..07 |
| State management: LangGraph/CrewAI | LangGraph checkpointer | ADR-11, WF-04 |
| Six agents (Q&A, Portfolio, Market, Goal, News, Tax) | All six | QA, PF, MK, GP, NW, TX sections |
| Data flow: Query → Router → Agent(s) → RAG → LLM → Response → UI | Same, plus guard nodes | WF-01, WF-02, WF-09, design §6–7 |

## §3 Deliverables / Objectives

| Problem statement item | Covered by |
|---|---|
| 1. Six agents with distinct capabilities | QA, PF, MK, GP, NW, TX |
| 1. Robust workflow orchestration (LangGraph) | WF-01..10 |
| 1. Test suite with 80%+ coverage | NFR-01 |
| 1. Error handling and fallback mechanisms | WF-03, WF-05, MD-04..06, LLM-03, LLM-05, LLM-07, API-03, P4 |
| 2. Conversational interface | UI-02, WF-06 |
| 2. Portfolio dashboard with visualizations | UI-03 |
| 2. Market overview with real-time data | UI-04, MK-03 |
| 3. 50–100 curated articles | RAG-01 |
| 3. Vector indexing | RAG-02, RAG-03 |
| 3. Category-based filtering | RAG-04 |
| 3. Source attribution | RAG-05, QA-01 |
| 4. Alpha Vantage/yFinance live quotes | MD-01, MK-01 |
| 4. Caching strategy | MD-02 |
| 4. Rate limits and API failures | MD-03..06 |
| 4. Market trend analysis and insights | MK-02, MK-03 |
| 5. [Optional] MCP tools / Claude Desktop / protocol docs | MCP-01, MCP-02 |

## §4 Learning Goals
Not system requirements. Covered by the daily concept walkthroughs in `03-tasks.md`.

## §5 Submission Guidelines

| Problem statement item | Covered by |
|---|---|
| All six working agents | QA, PF, MK, GP, NW, TX |
| Web interface for conversation | UI-01, UI-02 |
| Portfolio analysis of user input | PF-01..05 |
| Real-time market data lookup | MK-01, MD-01 |
| Plan financial goals keeping risk appetite in mind | GP-01..05, WF-10 |
| Demo video (5–10 min): multi-turn, portfolio, market, goals | DEL-04 |
| Prescribed codebase structure | DEL-01, design §2 |
| [Optional] unit + integration tests | NFR-01, design §11 |
| [Optional] YAML / env configuration | P7, NFR-04, design §10 |
| Error handling and logging throughout | P4, P8, NFR-05 |
| README: architecture, setup, API docs, usage examples, troubleshooting | NFR-07 |
| Technical design doc: architecture decisions, agent communication protocols, RAG details, performance | NFR-07, design §3, §7a, §9 |
| Docker configuration (optional) | NFR-06 |
| Environment setup files | DEL-03, NFR-04 |
| Sample data for testing | DEL-02 |
| Performance benchmarks | NFR-08, NFR-02, NFR-03 |

## §6 Evaluation Criteria

| Criterion | Weight | Covered by |
|---|---|---|
| Multi-agent architecture, clean separation | 10% | design §7, §7a |
| LangGraph workflow: orchestration, state, routing | 10% | WF-01..10, design §5–6 |
| RAG implementation | 8% | RAG-01..06, LLM-05, NFR-11 |
| Real-time data integration with error handling | 7% | MD-01..07, MK-01..04 |
| MCP server (bonus) | 5% | MCP-01..02 |
| Streamlit multi-tab, responsive | 10% | UI-01, UI-07 |
| Conversational flow, context preservation | 8% | WF-04, WF-06, WF-07, WF-10 |
| Data visualization | 7% | UI-03..05 |
| Educational content | 8% | RAG-01, QA-01..03, TX-01..03 |
| Portfolio analysis: metrics **and recommendations** | 7% | PF-02..04 (educational recommendations) |
| Market intelligence | 5% | MK-01..04, NW-01..04 |
| Code organization | 5% | DEL-01, P6, LLM-01..02 |
| Documentation (README + inline) | 5% | NFR-07, NFR-10 |
| Testing with edge cases | 5% | NFR-01, P5, design §11 |
