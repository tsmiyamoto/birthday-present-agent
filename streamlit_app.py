"""Streamlit interface for the birthday present agent."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent

from birthday_present_agent import root_agent

load_dotenv()

APP_NAME = os.getenv("ADK_APP_NAME", "birthday-present-agent")


async def _send_message(runner: InMemoryRunner, session, text: str) -> tuple[str, List[Dict[str, Any]]]:
    """Send a message to the agent and gather the final response and tool logs."""
    tool_logs: List[Dict[str, Any]] = []
    final_text = ""

    user_content = UserContent(parts=[Part(text=text)])

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=user_content,
    ):
        content = getattr(event, "content", None)
        if not content:
            continue

        for part in getattr(content, "parts", []) or []:
            if getattr(part, "function_call", None):
                args = getattr(part.function_call, "args", {}) or {}
                tool_logs.append(
                    {
                        "type": "call",
                        "name": part.function_call.name,
                        "payload": json.dumps(args, ensure_ascii=False, indent=2),
                    }
                )
            elif getattr(part, "function_response", None):
                response_payload = getattr(part.function_response, "response", {}) or {}
                if isinstance(response_payload, (str, bytes)):
                    payload_text = response_payload if isinstance(response_payload, str) else response_payload.decode()
                else:
                    payload_text = json.dumps(response_payload, ensure_ascii=False, indent=2)
                tool_logs.append(
                    {
                        "type": "response",
                        "name": part.function_response.name,
                        "payload": payload_text,
                    }
                )

        if event.turn_complete and getattr(content, "role", None) == "model":
            final_text = "".join(
                part.text for part in getattr(content, "parts", []) if getattr(part, "text", None)
            ).strip()

    return final_text, tool_logs


def _ensure_runner_and_session() -> tuple[InMemoryRunner, Any]:
    if "runner" not in st.session_state:
        st.session_state.runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    runner: InMemoryRunner = st.session_state.runner

    if "adk_session" not in st.session_state:
        session = asyncio.run(
            runner.session_service.create_session(
                app_name=APP_NAME,
                user_id=str(uuid.uuid4()),
            )
        )
        st.session_state.adk_session = session
    return runner, st.session_state.adk_session


def _initialize_conversation(runner: InMemoryRunner, session) -> None:
    if st.session_state.get("initialized"):
        return

    st.session_state.messages = []
    try:
        initial_reply, tool_logs = asyncio.run(_send_message(runner, session, ""))
    except Exception:
        initial_reply, tool_logs = "", []
    if not initial_reply:
        initial_reply = "その人の職業や年齢、Xのリンクなどを教えてください。誕生日プレゼント選びをお手伝いします。"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": initial_reply,
            "tool_logs": tool_logs,
        }
    )
    st.session_state.initialized = True


def _render_messages() -> None:
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for log in message.get("tool_logs", []):
                label = "ツール呼び出し" if log["type"] == "call" else "ツール応答"
                with st.expander(f"{label}: {log['name']}", expanded=False):
                    st.code(log["payload"], language="json")


def main() -> None:
    st.set_page_config(page_title="Birthday Present Agent", page_icon="🎁", layout="wide")
    st.title("🎁 誕生日プレゼント提案エージェント")
    st.caption("Google ADK + Gemini + Grok + SerpApi を活用したギフトコンシェルジュ")

    runner, session = _ensure_runner_and_session()
    _initialize_conversation(runner, session)
    _render_messages()

    user_input = st.chat_input("メッセージを入力してください")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("候補を考えています..."):
            try:
                reply, tool_logs = asyncio.run(_send_message(runner, session, user_input))
            except Exception as error:
                reply = f"ツール呼び出し中にエラーが発生しました: {error}"
                tool_logs = []
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply or "申し訳ありません、返答を生成できませんでした。",
                "tool_logs": tool_logs,
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
