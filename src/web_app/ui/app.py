"""Streamlit UI — run with ``make ui`` (``uv run streamlit run src/web_app/ui/app.py``).

Day 2: the Chat tab talks to the LangGraph workflow in-process and streams the answer.
Other tabs are placeholders until their features land (Days 6-8).
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st
from langchain_core.exceptions import ModelError
from langchain_core.messages import AIMessageChunk, HumanMessage

from src.core.config import get_settings
from src.core.errors import AppError
from src.core.guards import DISCLAIMER
from src.core.models import AGENT_LABELS, UserProfile
from src.utils.logging import configure_logging, request_context
from src.workflow import graph as workflow_graph

AGENT_NODES = {workflow_graph.agent_node_name(name) for name in AGENT_LABELS}


@st.cache_resource(show_spinner="Starting the assistant…")
def get_graph() -> Any:
    """Build the workflow once per server process (models are created at startup)."""
    settings = get_settings()
    configure_logging(level=settings.app.log_level, fmt=settings.app.log_format)
    return workflow_graph.build_default_graph(settings)


def init_session() -> None:
    """Per-browser-session state: a conversation thread and the rendered history."""
    st.session_state.setdefault("thread_id", uuid.uuid4().hex)
    st.session_state.setdefault("history", [])


def render_turn(turn: dict[str, Any]) -> None:
    """Render one stored chat turn, including agent badges and citations."""
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("agents"):
            st.caption("Answered by: " + " · ".join(turn["agents"]))
        if turn.get("citations"):
            with st.expander(f"Sources ({len(turn['citations'])})"):
                for c in turn["citations"]:
                    st.markdown(f"- [{c['title']}]({c['url']}) · _{c['category']}_")
        if turn.get("learn_next"):
            links = " · ".join(f"[{a['title']}]({a['url']})" for a in turn["learn_next"])
            st.caption(f"📚 Learn next: {links}")


def ask(graph: Any, question: str, profile: UserProfile) -> dict[str, Any]:
    """Stream one turn through the graph; return the assistant turn to store."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {"messages": [HumanMessage(question)], "profile": profile}
    placeholder = st.empty()
    streamed = ""
    final: dict[str, Any] = {}
    with request_context(thread_id=st.session_state.thread_id) as request_id:
        inputs["request_id"] = request_id
        for mode, chunk in graph.stream(inputs, config, stream_mode=["messages", "values"]):
            if mode == "messages":
                message, meta = chunk
                # Only stream agent tokens (not the router's JSON) to the user.
                if (
                    isinstance(message, AIMessageChunk)
                    and meta.get("langgraph_node") in AGENT_NODES
                ):
                    streamed += message.text
                    placeholder.markdown(streamed + "▌")
            else:
                final = chunk
    placeholder.empty()
    agents = [
        AGENT_LABELS.get(r.agent, r.agent)
        for r in final.get("agent_results", [])
        if r.agent != "out_of_scope"
    ]
    return {
        "role": "assistant",
        "content": final.get("final_answer", "Sorry, something went wrong."),
        "agents": agents,
        "citations": [c.model_dump() for c in final.get("citations", [])],
        "learn_next": [
            a for r in final.get("agent_results", []) for a in r.data.get("learn_next", [])
        ][:3],
    }


def sidebar() -> UserProfile:
    """Profile controls and conversation reset."""
    with st.sidebar:
        st.header("Your profile")
        level = st.radio(
            "How familiar are you with investing?",
            ["beginner", "intermediate", "advanced"],
            format_func=str.capitalize,
        )
        if st.button("New conversation", use_container_width=True):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.history = []
            st.rerun()
        st.caption(f"Session: `{st.session_state.thread_id[:8]}`")
    return UserProfile(knowledge_level=level)


def chat_tab(graph: Any, profile: UserProfile) -> None:
    """Conversational interface (REQ-UI-02)."""
    if not st.session_state.history:
        st.info(
            "Ask me about investing basics — for example *“What is an index fund?”* or "
            "*“How does compound interest work?”*"
        )
    for turn in st.session_state.history:
        render_turn(turn)

    question = st.chat_input("Ask a personal-finance question…")
    if not question:
        return
    user_turn = {"role": "user", "content": question}
    st.session_state.history.append(user_turn)
    render_turn(user_turn)
    with st.chat_message("assistant"):
        try:
            turn = ask(graph, question, profile)
        except AppError as exc:
            st.error(exc.user_message)
            return
        except ModelError:
            # Config errors (bad key, unknown model) fail loud in logs; show a safe message.
            st.error(
                "The AI service is misconfigured or unavailable. Details are in the server logs."
            )
            return
    st.session_state.history.append(turn)
    st.rerun()


def load_graph_or_stop() -> Any:
    """Build the workflow on page load so misconfiguration shows up immediately."""
    try:
        return get_graph()
    except AppError as exc:
        st.error(exc.user_message)
    except Exception:
        st.error("The assistant failed to start. Details are in the server logs.")
    st.stop()


def main() -> None:
    """Page layout."""
    st.set_page_config(page_title="AI Finance Assistant", page_icon="💹", layout="wide")
    init_session()
    st.title("💹 AI Finance Assistant")
    profile = sidebar()
    graph = load_graph_or_stop()
    chat, portfolio, market, goals, learn = st.tabs(
        ["💬 Chat", "📊 Portfolio", "📈 Market", "🎯 Goals", "📚 Learn"]
    )
    with chat:
        chat_tab(graph, profile)
    for tab, label in (
        (portfolio, "Portfolio analysis"),
        (market, "Market overview"),
        (goals, "Goal planning"),
        (learn, "Knowledge-base browser"),
    ):
        with tab:
            st.info(f"{label} is coming soon.")
    st.caption(DISCLAIMER)  # visible on every tab (REQ-UI-07)


main()
