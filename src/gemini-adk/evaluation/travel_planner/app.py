"""Travel Planner Agent - Streamlit chat interface (Google ADK version)."""

import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

logging.getLogger("google_adk").setLevel(logging.CRITICAL)

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent import root_agent

APP_NAME = "adk-travel-planner-agent"

st.set_page_config(
    page_title="Travel Planner Agent (ADK)",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _check_env() -> list[str]:
    required = ["GOOGLE_API_KEY", "OPENWEATHER_API_KEY", "EXCHANGERATE_API_KEY"]
    return [key for key in required if not os.getenv(key)]


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "user_id" not in st.session_state:
        st.session_state.user_id = "streamlit_user"

    if "env_error" not in st.session_state:
        missing = _check_env()
        st.session_state.env_error = (
            f"Missing API keys: {', '.join(missing)}. "
            "Copy `.env.example` to `.env` and add your keys."
            if missing
            else None
        )

    if "adk_runner" not in st.session_state and not st.session_state.env_error:
        session_service = InMemorySessionService()
        st.session_state.adk_session_service = session_service
        st.session_state.adk_runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )


async def _run_agent_async(message: str) -> str:
    runner: Runner = st.session_state.adk_runner
    session_service: InMemorySessionService = st.session_state.adk_session_service
    session_id: str = st.session_state.session_id
    user_id: str = st.session_state.user_id

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if session is None:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

    content = types.Content(role="user", parts=[types.Part(text=message)])

    final_text = ""
    agen = runner.run_async(
        session_id=session_id,
        user_id=user_id,
        new_message=content,
    )
    try:
        async for event in agen:
            if getattr(event, "error_code", None):
                raise RuntimeError(event.error_message or event.error_code)
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "\n".join(
                    part.text for part in event.content.parts if getattr(part, "text", None)
                )
                break
    finally:
        await agen.aclose()

    return final_text or "Sorry, I could not generate a response."


def get_agent_reply(message: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_run_agent_async(message))
    except RuntimeError:
        return asyncio.run(_run_agent_async(message))


def main() -> None:
    _init_session()

    with st.sidebar:
        st.title("✈️ Travel Planner")
        st.markdown(
            "Plan your trip through conversation - weather, budget, "
            "highlights, and packing in one place."
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

        st.markdown("---")
        st.markdown(
            "### Powered by\n"
            "- 🤖 Google ADK + Gemini\n"
            "- 🌤️ OpenWeatherMap\n"
            "- 💱 ExchangeRate-API\n"
            "- 🔎 DuckDuckGo Search"
        )

    st.title("Travel Planner Agent")
    st.caption("Describe your trip in natural language - I'll research and build your plan.")

    if st.session_state.get("env_error"):
        st.error(st.session_state.env_error)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Where would you like to go?"):
        if st.session_state.get("env_error"):
            st.error(st.session_state.env_error)
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Researching your trip..."):
                try:
                    answer = get_agent_reply(prompt)
                except Exception as exc:
                    answer = f"Something went wrong: {exc}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
