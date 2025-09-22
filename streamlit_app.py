"""Streamlit interface for the birthday present agent."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
import warnings
from html import escape
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent

from birthday_present_agent import root_agent

# Pydantic の警告を抑制
warnings.filterwarnings("ignore", message="Field name .* shadows an attribute in parent")
# aiohttp の警告を抑制
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")

load_dotenv()

APP_NAME = os.getenv("ADK_APP_NAME", "birthday-present-agent")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")


def _inject_custom_styles() -> None:
    """Inject CSS styles for custom card layout."""
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #518378;
            --secondary-background-color: #f2f7f4;
            --text-color: #0f172a;
            --font: 'Inter', 'Hiragino Sans', sans-serif;
        }
        .product-section {
            border: none;
            border-radius: 18px;
            padding: 16px 0;
            margin: 26px 0;
            background: transparent;
            position: relative;
        }
        .product-section::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 18px;
            border: 3px solid var(--primary-color);
            opacity: 0.75;
            pointer-events: none;
        }
        .product-section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 6px;
            color: #0f172a;
        }
        .product-section-summary {
            color: #475569;
            font-size: 0.95rem;
            margin-bottom: 8px;
            line-height: 1.6;
            white-space: pre-line;
        }
        .product-card-row {
            display: flex;
            gap: 16px;
            overflow-x: auto;
            padding-bottom: 8px;
            padding-top: 12px;
            padding-left: 4px;
            padding-right: 4px;
            border-radius: 18px;
            margin-bottom: 16px;
        }
        .product-card-row::-webkit-scrollbar {
            height: 8px;
        }
        .product-card-row::-webkit-scrollbar-thumb {
            background: rgba(81, 131, 120, 0.35);
            border-radius: 999px;
        }
        .product-card {
            width: 100%;
            border-radius: 18px;
            border: 1px solid rgba(0, 0, 0, 0.05);
            background: #edf0ef;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-height: 600px;
            box-sizing: border-box;
        }
        .product-card-body {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .product-card img {
            width: 100%;
            aspect-ratio: 1 / 1;
            height: auto;
            object-fit: contain;
            box-sizing: border-box;
            margin: 0 auto 12px;
            padding: 12px;
            border-radius: 12px;
            background: #fff;
            border: 1px solid rgba(0,0,0,0.05);
        }
        .product-card-title {
            font-weight: 600;
            font-size: 0.95rem;
        }
        .product-card-price {
            color: #518378;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .product-card-rating {
            color: #f59e0b;
            font-size: 0.9rem;
            letter-spacing: 1px;
        }
        .product-card-reason {
            color: #1f2937;
            font-size: 0.9rem;
            line-height: 1.5;
            min-height: 48px;
        }
        .product-card-meta {
            font-size: 0.85rem;
            color: #6b7280;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .product-card-footer {
            margin-top: auto;
            display: flex;
        }
        .product-card-footer:empty {
            display: none;
        }
        .product-card-button {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            padding: 8px 0;
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid rgba(81, 131, 120, 0.45);
            color: #518378 !important;
            font-weight: 600;
            text-decoration: none !important;
        }
        .product-card-button:visited,
        .product-card-button:focus {
            color: #518378 !important;
            text-decoration: none !important;
            outline: none;
        }
        .product-card-button:hover {
            background: rgba(81, 131, 120, 0.12);
            color: #518378 !important;
        }
        .sidebar-product-image {
            margin: 12px auto;
            padding: 12px;
            border-radius: 12px;
            background: #fff;
            border: 1px solid rgba(0,0,0,0.05);
            box-sizing: border-box;
            max-width: 324px;
            max-height: 324px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .sidebar-product-image img {
            width: 100%;
            height: auto;
            max-width: 300px;
            max-height: 300px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


async def _send_message(runner: InMemoryRunner, session, text: str) -> tuple[str, List[Dict[str, Any]]]:
    """Send a message to the agent and gather the final response and tool logs."""
    tool_logs: List[Dict[str, Any]] = []
    final_text = ""

    user_content = UserContent(parts=[Part(text=text)])

    try:
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=user_content,
        ):
            content = getattr(event, "content", None)
            if not content:
                continue

            text_parts: List[str] = []
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
                        payload_text = (
                            response_payload if isinstance(response_payload, str) else response_payload.decode()
                        )
                    else:
                        payload_text = json.dumps(response_payload, ensure_ascii=False, indent=2)
                    tool_logs.append(
                        {
                            "type": "response",
                            "name": part.function_response.name,
                            "payload": payload_text,
                        }
                    )
                elif getattr(part, "text", None):
                    text_parts.append(part.text)

            if text_parts:
                candidate = "".join(text_parts).strip()
                if candidate:
                    final_text = candidate

            if getattr(event, "error_message", None):
                final_text = event.error_message

            if hasattr(event, "is_final_response") and event.is_final_response():
                break

    except Exception as e:
        # エラーハンドリングを追加
        print(f"Error in _send_message: {e}")
        final_text = "エラーが発生しました。"
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


def _format_price(entry: Dict[str, Any]) -> str:
    price = entry.get("price")
    if price:
        return str(price)
    extracted = entry.get("extracted_price")
    if extracted is None:
        return "価格情報なし"
    try:
        number = float(extracted)
        return f"¥{number:,.0f}"
    except (TypeError, ValueError):
        return str(extracted)


def _rating_from_position(position: Optional[int]) -> float:
    if position is None:
        return 3.5
    rating = 5.0 - (position - 1) * 0.5
    return max(3.0, min(5.0, rating))


def _rating_to_stars(rating: float) -> str:
    rounded = int(round(rating))
    full = max(0, min(5, rounded))
    empty = 5 - full
    return "★" * full + "☆" * empty


def _extract_non_section_text(text: str) -> str:
    if not text:
        return ""
    section_heading = re.compile(r"(^|\n)###\s+", re.MULTILINE)
    match = section_heading.search(text)
    if not match:
        return text.strip()
    return text[: match.start()].strip()


def _build_product_card(entry: Dict[str, Any]) -> str:
    title = escape(entry.get("title") or "名称不明")
    price = escape(_format_price(entry))
    rating = _rating_to_stars(_rating_from_position(entry.get("position")))
    image_url = (
        entry.get("thumbnail")
        or "https://images.unsplash.com/photo-1707944145479-12755f0434d8?q=80&w=2237&auto=format&fit=crop"
    )
    image = escape(image_url)

    meta_lines = []
    if shipping := entry.get("shipping"):
        meta_lines.append(escape(str(shipping)))
    meta_html = "<div class='product-card-meta'>" + "<br/>".join(meta_lines) + "</div>" if meta_lines else ""

    reason_html = ""
    if reason := entry.get("reason"):
        reason_html = f"<div class='product-card-reason'>{escape(reason)}</div>"

    description_html = ""
    if description := entry.get("description"):
        description_html = f"<div class='product-card-meta'>{escape(description)}</div>"

    body_html = (
        "<div class='product-card-body'>"
        f"<img src='{image}' alt='{title}'/>"
        f"<div class='product-card-title'>{title}</div>"
        f"<div class='product-card-price'>{price}</div>"
        f"<div class='product-card-rating'>{rating}</div>"
        f"{reason_html}"
        f"{description_html}"
        f"{meta_html}"
        "</div>"
    )

    return f"<div class='product-card'>{body_html}</div>"


def _queue_related_query(prompt: str) -> None:
    st.session_state.prefill_message = prompt
    st.session_state.scroll_to_bottom = True


def _fetch_product_details(serpapi_url: str) -> Optional[Dict[str, Any]]:
    """Fetch product details from SerpApi."""
    if not SERPAPI_KEY:
        st.error("SERPAPI_KEY が設定されていません")
        return None

    try:
        # URLにapi_keyパラメータを追加
        separator = "&" if "?" in serpapi_url else "?"
        url_with_key = f"{serpapi_url}{separator}api_key={SERPAPI_KEY}"

        response = requests.get(url_with_key, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"商品詳細の取得に失敗しました: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"レスポンスの解析に失敗しました: {e}")
        return None


def _display_product_details_sidebar(product_data: Dict[str, Any]) -> None:
    """Display product details in sidebar."""
    st.sidebar.markdown("## 📱 商品詳細")

    # 基本情報
    product_results = product_data.get("product_results", {})
    if product_results:
        title = product_results.get("title", "商品名不明")
        st.sidebar.markdown(f"**{title}**")

        # 価格情報
        prices = product_results.get("prices", [])
        if prices:
            st.sidebar.markdown("### 💰 価格")
            for price in prices[:3]:  # 最初の3つの価格を表示
                st.sidebar.markdown(f"- {price}")

        # 評価
        rating = product_results.get("rating")
        reviews = product_results.get("reviews")
        if rating and reviews:
            st.sidebar.markdown(f"### ⭐ 評価: {rating}/5.0 ({reviews:,}件)")

        # 商品画像
        media = product_results.get("media", [])
        if media and media[0].get("type") == "image":
            image_url = media[0]["link"]
            st.sidebar.markdown(
                f"""
                <div class="sidebar-product-image">
                    <img src="{image_url}" alt="商品画像" style="width: 100%; height: auto; object-fit: contain;">
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 商品説明
        description = product_results.get("description")
        if description:
            st.sidebar.markdown("### 📝 商品説明")
            st.sidebar.markdown(description)

    # 販売店情報
    sellers_results = product_data.get("sellers_results", {})
    online_sellers = sellers_results.get("online_sellers", [])
    if online_sellers:
        st.sidebar.markdown("### 🏪 販売店")
        for seller in online_sellers[:5]:  # 最初の5つの販売店を表示
            name = seller.get("name", "販売店名不明")
            total_price = seller.get("total_price", "価格不明")
            direct_link = seller.get("direct_link")

            if direct_link:
                st.sidebar.markdown(f"**[{name}]({direct_link})** - {total_price}")
            else:
                st.sidebar.markdown(f"**{name}** - {total_price}")

            # 配送情報
            details = seller.get("details_and_offers", [])
            if details:
                for detail in details[:2]:  # 最初の2つの詳細情報
                    text = detail.get("text", "")
                    if text:
                        st.sidebar.markdown(f"  - {text}")

            st.sidebar.markdown("---")


def _handle_product_detail_click(serpapi_url: str, product_title: str) -> None:
    """Handle product detail button click."""
    # セッション状態に商品詳細データを保存
    st.session_state.current_product_title = product_title
    st.session_state.loading_product_details = True
    st.session_state.product_details_data = None

    # 商品詳細を取得
    product_data = _fetch_product_details(serpapi_url)

    st.session_state.loading_product_details = False
    if product_data:
        st.session_state.product_details_data = product_data

    st.rerun()


def _parse_agent_sections(text: str) -> List[Dict[str, Any]]:
    section_pattern = re.compile(r"^###\s+(.*)")
    item_pattern = re.compile(r"^(\d+)\.\s*(.*)")
    field_pattern = re.compile(r"^-\s*([^:：]+)[：:]+\s*(.*)")

    sections: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None
    current_item: Optional[Dict[str, Any]] = None
    last_field: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        section_match = section_pattern.match(stripped)
        if section_match:
            if current_section:
                if current_item:
                    current_item = None
                    last_field = None
                sections.append(current_section)
            current_section = {
                "title": section_match.group(1).strip(),
                "summary_lines": [],
                "items": [],
            }
            current_item = None
            last_field = None
            continue

        if current_section is None:
            continue

        item_match = item_pattern.match(stripped)
        if item_match:
            current_item = {
                "title": item_match.group(2).strip(),
                "fields": {},
            }
            current_section["items"].append(current_item)
            last_field = None
            continue

        field_match = field_pattern.match(stripped)
        if field_match and current_item is not None:
            label = field_match.group(1).strip()
            value = field_match.group(2).strip()
            current_item["fields"][label] = value
            last_field = label
            continue

        if current_item is not None and last_field:
            current_item["fields"][last_field] = (current_item["fields"].get(last_field, "") + "\n" + stripped).strip()
        elif current_item is None:
            current_section["summary_lines"].append(line)

    if current_section:
        sections.append(current_section)

    for section in sections:
        section["summary"] = "\n".join(section.get("summary_lines", [])).strip()
        section.pop("summary_lines", None)

    return sections


def _render_shopping_sections(
    message_index: int,
    message: Dict[str, Any],
    sections: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    if sections is None:
        sections = _parse_agent_sections(message.get("content", ""))
    if not sections:
        return False

    queries: List[str] = []
    for log in message.get("tool_logs", []):
        if log.get("name") != "shopping_search" or log.get("type") != "response":
            continue
        try:
            payload = json.loads(log.get("payload", "{}"))
        except json.JSONDecodeError:
            continue
        query = payload.get("query")
        if query:
            queries.append(query)

    for section_index, section in enumerate(sections):
        items = section.get("items", [])
        if not items:
            continue

        title = section.get("title") or f"提案 {section_index + 1}"
        summary = section.get("summary", "")

        section_container = st.container()
        with section_container:
            st.markdown("<div class='product-section'>", unsafe_allow_html=True)

            header_cols = st.columns([0.75, 0.25])
            with header_cols[0]:
                st.markdown(
                    f"<div class='product-section-title'>{escape(title)}</div>",
                    unsafe_allow_html=True,
                )
                if summary:
                    st.markdown(
                        f"<div class='product-section-summary'>{escape(summary)}</div>",
                        unsafe_allow_html=True,
                    )

            prompt_query = queries[section_index] if section_index < len(queries) else title

            with header_cols[1]:
                st.button(
                    "関連商品を見る",
                    type="primary",
                    icon=":material/linked_services:",
                    key=f"related_{message_index}_{section_index}",
                    on_click=_queue_related_query,
                    args=(f"{prompt_query}に関連する商品を探してください。",),
                )

            card_entries: List[Dict[str, Any]] = []
            for item_index, item in enumerate(items, start=1):
                fields = item.get("fields", {})
                entry: Dict[str, Any] = {
                    "title": item.get("title") or f"候補 {item_index}",
                    "price": fields.get("おおよその価格") or fields.get("価格"),
                    "position": item_index,
                    "thumbnail": fields.get("画像URL") or fields.get("画像リンク"),
                    "product_link": fields.get("商品ページURL") or fields.get("購入リンク"),
                    "serpapi_product_api": fields.get("serpapi_product_api")
                    or fields.get("SerpApi")
                    or fields.get("商品ID"),
                    "reason": fields.get("推薦理由"),
                    "description": fields.get("詳細") or fields.get("補足"),
                }
                card_entries.append(entry)

            # 商品カードとボタンを同じカラム内に表示
            cards_per_row = 3
            for start in range(0, len(card_entries), cards_per_row):
                row_entries = card_entries[start : start + cards_per_row]
                cols = st.columns(cards_per_row)
                for offset, entry in enumerate(row_entries):
                    card_position = start + offset
                    with cols[offset]:
                        card_html = _build_product_card(entry)
                        st.markdown(card_html, unsafe_allow_html=True)
                        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

                        serpapi_url = entry.get("serpapi_product_api")
                        product_link = entry.get("product_link")
                        button_key = f"detail_{message_index}_{section_index}_{card_position}"

                        if serpapi_url:
                            if st.button(
                                "詳しく見る",
                                key=button_key,
                                type="secondary",
                                use_container_width=True,
                            ):
                                _handle_product_detail_click(serpapi_url, entry.get("title", "商品"))
                        elif product_link:
                            st.markdown(
                                f"<a class='product-card-button' href='{escape(product_link)}' target='_blank' rel='noopener'>商品ページ</a>",
                                unsafe_allow_html=True,
                            )
            st.markdown("</div>", unsafe_allow_html=True)

    return True


def _initialize_conversation(runner: InMemoryRunner, session) -> None:
    if st.session_state.get("initialized"):
        return

    st.session_state.messages = []
    try:
        initial_reply, tool_logs = asyncio.run(_send_message(runner, session, ""))
    except Exception:
        initial_reply, tool_logs = "", []
    if (not initial_reply) or ("エラー" in initial_reply) or ("SERPAPI" in initial_reply):
        initial_reply = "その人の職業や年齢、Xのリンクなどを教えてください。誕生日プレゼント選びをお手伝いします。"
        tool_logs = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": initial_reply,
            "tool_logs": tool_logs,
        }
    )
    st.session_state.initialized = True


def _render_messages() -> None:
    for index, message in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(message["role"]):
            content = message.get("content", "")
            sections: List[Dict[str, Any]] = []
            if message["role"] == "assistant":
                sections = _parse_agent_sections(content)
                text_to_render = _extract_non_section_text(content) if sections else content
            else:
                text_to_render = content

            if text_to_render:
                st.markdown(text_to_render)

            if message["role"] == "assistant":
                _render_shopping_sections(index, message, sections)
            for log in message.get("tool_logs", []):
                label = "ツール呼び出し" if log["type"] == "call" else "ツール応答"
                with st.expander(f"{label}: {log['name']}", expanded=False):
                    st.code(log["payload"], language="json")


def _handle_user_turn(runner: InMemoryRunner, session, text: str) -> None:
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    with st.spinner("候補を考えています..."):
        try:
            reply, tool_logs = asyncio.run(_send_message(runner, session, text))
        except Exception as error:
            st.error(f"エラーが発生しました: {error}")
            reply = "申し訳ありません、処理中にエラーが発生しました。もう一度お試しください。"
            tool_logs = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply or "申し訳ありません、返答を生成できませんでした。",
            "tool_logs": tool_logs,
        }
    )
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Birthday Present Agent", page_icon="🎁", layout="wide")
    st.title("🎁 誕生日プレゼント提案エージェント")
    st.caption("Google ADK + Gemini + Grok + SerpApi を活用したギフトコンシェルジュ")

    _inject_custom_styles()
    runner, session = _ensure_runner_and_session()
    _initialize_conversation(runner, session)

    # サイドバーでローディング状態と商品詳細を表示
    if st.session_state.get("loading_product_details", False):
        st.sidebar.markdown("## 🔄 商品詳細を読み込み中...")
        st.sidebar.markdown("少々お待ちください...")
    elif st.session_state.get("product_details_data"):
        _display_product_details_sidebar(st.session_state.product_details_data)
    else:
        st.sidebar.markdown("## 💡 使い方")
        st.sidebar.markdown("商品カードの「詳しく見る」ボタンを押すと、詳細情報がここに表示されます。")

    _render_messages()

    if st.session_state.pop("scroll_to_bottom", False):
        components.html(
            """
            <script>
            const mainSection = window.parent.document.querySelector('section.main');
            if (mainSection) {
                mainSection.scrollTo({ top: mainSection.scrollHeight, behavior: 'smooth' });
            } else {
                window.parent.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
            }
            </script>
            """,
            height=0,
        )

    if "prefill_message" in st.session_state:
        st.session_state["chat_input"] = st.session_state.pop("prefill_message")

    user_input = st.chat_input("メッセージを入力してください", key="chat_input")
    if user_input:
        _handle_user_turn(runner, session, user_input)
        st.session_state["chat_input"] = ""


if __name__ == "__main__":
    main()
