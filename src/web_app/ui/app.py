"""Streamlit UI — run with ``make ui`` (``uv run streamlit run src/web_app/ui/app.py``).

Tabs: Chat, Portfolio, Market, Goals, Learn (REQ-UI-01). Every tab calls the shared
:class:`AssistantService` in-process — the same facade behind the REST API — so the UI holds
presentation only. The disclaimer sits above the tabs, so it is visible on each of them, and
Streamlit columns stack on narrow screens (REQ-UI-07).
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st
from langchain_core.exceptions import ModelError

from src.core import services
from src.core.config import get_settings
from src.core.errors import AppError
from src.core.guards import DISCLAIMER
from src.core.models import UserProfile
from src.core.services import AssistantService, ChatRequest, ChatResponse
from src.utils.logging import configure_logging, request_context
from src.web_app.ui import tabs

LEVELS = ["beginner", "intermediate", "advanced"]
RISK_CHOICES = ["not set", *tabs.RISK_PROFILES]


@st.cache_resource(show_spinner="Starting the assistant…")
def get_service() -> AssistantService:
    """Build models, index, market cache, and graph once per server process."""
    settings = get_settings()
    configure_logging(level=settings.app.log_level, fmt=settings.app.log_format)
    return services.build_service(settings)


def init_session() -> None:
    """Per-browser-session state."""
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


def ask(service: AssistantService, question: str, profile: UserProfile) -> dict[str, Any]:
    """Stream one turn (tokens appear as agents write); return the assistant turn to store."""
    req = ChatRequest(
        message=question,
        thread_id=st.session_state.thread_id,
        profile=profile,
        portfolio=st.session_state.get("portfolio"),  # holdings analyzed in the Portfolio tab
    )
    placeholder = st.empty()
    streamed = ""
    final: ChatResponse | None = None
    with request_context(thread_id=st.session_state.thread_id) as request_id:
        for kind, payload in service.chat_events(req, request_id):
            if kind == "token":
                streamed += payload
                placeholder.markdown(streamed + "▌")
            else:
                final = payload
    placeholder.empty()
    if final is None:  # pragma: no cover - chat_events always ends with "final"
        return {"role": "assistant", "content": "Sorry, something went wrong."}
    return {
        "role": "assistant",
        "content": final.answer or "Sorry, something went wrong.",
        "agents": final.agents,
        "citations": [c.model_dump() for c in final.citations],
        "learn_next": final.learn_next,
    }


def sidebar() -> tuple[UserProfile, Any]:
    """Profile controls, conversation reset, and a status slot filled after the tabs run."""
    with st.sidebar:
        st.header("Your profile")
        level = st.radio("How familiar are you with investing?", LEVELS, format_func=str.capitalize)
        remembered = st.session_state.get("risk_profile", "not set")
        risk = st.selectbox(
            "Risk tolerance",
            RISK_CHOICES,
            index=RISK_CHOICES.index(remembered),
            format_func=str.capitalize,
            help="Take the quiz in the Goals tab if you're unsure.",
        )
        if st.button("New conversation", width="stretch"):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.history = []
            st.rerun()
        status = st.empty()
        st.caption(f"Session: `{st.session_state.thread_id[:8]}`")
    profile = UserProfile(knowledge_level=level, risk_tolerance=None if risk == "not set" else risk)
    return profile, status


def chat_tab(service: AssistantService, profile: UserProfile) -> None:
    """Conversational interface (REQ-UI-02)."""
    if not st.session_state.history:
        st.info(
            "Ask me about investing basics — for example *“What is an index fund?”*, "
            "*“How is AAPL doing?”*, or *“Can I save $50k in 5 years?”*"
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
            turn = ask(service, question, profile)
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


def load_service_or_stop() -> AssistantService:
    """Build everything on page load so misconfiguration shows up immediately."""
    try:
        return get_service()
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
    st.caption(DISCLAIMER)  # above the tabs -> visible on every tab (REQ-UI-07)
    profile, status = sidebar()
    service = load_service_or_stop()
    chat, portfolio, market, goals, learn = st.tabs(
        ["💬 Chat", "📊 Portfolio", "📈 Market", "🎯 Goals", "📚 Learn"]
    )
    with chat:
        chat_tab(service, profile)
    with portfolio:
        tabs.portfolio_tab(service)
    with market:
        tabs.market_tab(service)
    with goals:
        tabs.goals_tab(service)
    with learn:
        tabs.learn_tab(service)
    if st.session_state.get("portfolio"):  # may have just been set by the Portfolio tab
        n = len(st.session_state.portfolio)
        status.caption(f"Portfolio shared with chat: {n} holdings")


main()
