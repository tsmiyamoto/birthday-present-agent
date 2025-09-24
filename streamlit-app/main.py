"""Streamlit interface for the birthd.ai [バースデイ]."""

from __future__ import annotations

import json
import os
import re
import uuid
import warnings
from html import escape
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components
import vertexai
from google.oauth2 import service_account
from vertexai import agent_engines

# Pydantic の警告を抑制
warnings.filterwarnings("ignore", message="Field name .* shadows an attribute in parent")
# aiohttp の警告を抑制
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

AGENT_ENGINE_ID = os.getenv("VERTEX_AI_AGENT_ENGINE_ID")
VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
_SA_REQUIRED_ENV_VARS: Dict[str, str] = {
    "type": "VERTEXAI_SERVICE_ACCOUNT_TYPE",
    "project_id": "VERTEXAI_SERVICE_ACCOUNT_PROJECT_ID",
    "private_key_id": "VERTEXAI_SERVICE_ACCOUNT_PRIVATE_KEY_ID",
    "private_key": "VERTEXAI_SERVICE_ACCOUNT_PRIVATE_KEY",
    "client_email": "VERTEXAI_SERVICE_ACCOUNT_CLIENT_EMAIL",
    "client_id": "VERTEXAI_SERVICE_ACCOUNT_CLIENT_ID",
    "auth_uri": "VERTEXAI_SERVICE_ACCOUNT_AUTH_URI",
    "token_uri": "VERTEXAI_SERVICE_ACCOUNT_TOKEN_URI",
    "auth_provider_x509_cert_url": "VERTEXAI_SERVICE_ACCOUNT_AUTH_PROVIDER_X509_CERT_URL",
    "client_x509_cert_url": "VERTEXAI_SERVICE_ACCOUNT_CLIENT_X509_CERT_URL",
}

_SA_OPTIONAL_ENV_VARS: Dict[str, str] = {
    "universe_domain": "VERTEXAI_SERVICE_ACCOUNT_UNIVERSE_DOMAIN",
}

if not AGENT_ENGINE_ID:
    raise RuntimeError("VERTEX_AI_AGENT_ENGINE_ID must be set for Agent Engine access.")

if not VERTEX_PROJECT or not VERTEX_LOCATION:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be configured.")

vertex_kwargs: Dict[str, Any] = {
    "project": VERTEX_PROJECT,
    "location": VERTEX_LOCATION,
}


def _build_service_account_credentials() -> Optional[service_account.Credentials]:
    service_account_info: Dict[str, str] = {}
    missing_env_vars: List[str] = []
    any_env_values = False

    for field, env_name in _SA_REQUIRED_ENV_VARS.items():
        value = os.getenv(env_name)
        if value:
            service_account_info[field] = value
            any_env_values = True
        else:
            missing_env_vars.append(env_name)

    if any_env_values:
        if missing_env_vars:
            missing_str = ", ".join(missing_env_vars)
            raise RuntimeError(f"Missing Vertex AI service account environment variables: {missing_str}.")

        service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

        for field, env_name in _SA_OPTIONAL_ENV_VARS.items():
            value = os.getenv(env_name)
            if value:
                service_account_info[field] = value

        return service_account.Credentials.from_service_account_info(service_account_info)

    if GOOGLE_APPLICATION_CREDENTIALS:
        return service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)

    return None


credentials = _build_service_account_credentials()
if credentials:
    vertex_kwargs["credentials"] = credentials

vertexai.init(**vertex_kwargs)
remote_app = agent_engines.get(AGENT_ENGINE_ID)


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


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iter_parts_from_event(event: Any) -> Iterable[Any]:
    content = _get_field(event, "content")
    if content:
        parts = _get_field(content, "parts", [])
        if parts:
            for part in parts:
                yield part

    result = _get_field(event, "result")
    if not result:
        return

    response = _get_field(result, "response")
    if response:
        for candidate in _get_field(response, "candidates", []) or []:
            candidate_content = _get_field(candidate, "content")
            if not candidate_content:
                continue
            for part in _get_field(candidate_content, "parts", []) or []:
                yield part

        output_text = _get_field(response, "output_text")
        if output_text:
            if isinstance(output_text, (list, tuple)):
                for text_value in output_text:
                    yield {"text": text_value}
            else:
                yield {"text": output_text}

    output = _get_field(result, "output")
    if output:
        for part in _get_field(output, "parts", []) or []:
            yield part
        text_value = _get_field(output, "text")
        if text_value:
            yield {"text": text_value}


def _format_payload(payload: Any) -> str:
    if isinstance(payload, bytes):
        return payload.decode(errors="replace")
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(payload)


def _stream_agent_query(
    user_id: str,
    session_id: str,
    message: str,
    on_text_update: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    tool_logs: List[Dict[str, Any]] = []
    final_text = ""
    query = message if message else " "

    try:
        for event in remote_app.stream_query(user_id=user_id, session_id=session_id, message=query):
            text_parts: List[str] = []
            for part in _iter_parts_from_event(event):
                function_call = _get_field(part, "function_call")
                if function_call:
                    args = _get_field(function_call, "args", {}) or {}
                    tool_logs.append(
                        {
                            "type": "call",
                            "name": _get_field(function_call, "name", ""),
                            "payload": _format_payload(args),
                        }
                    )
                    continue

                function_response = _get_field(part, "function_response")
                if function_response:
                    response_payload = _get_field(function_response, "response", {}) or {}
                    tool_logs.append(
                        {
                            "type": "response",
                            "name": _get_field(function_response, "name", ""),
                            "payload": _format_payload(response_payload),
                        }
                    )
                    continue

                text_value = _get_field(part, "text")
                if text_value:
                    text_parts.append(text_value)

            if text_parts:
                candidate = "".join(text_parts).strip()
                if candidate:
                    final_text = candidate
                    preview_segments, preview_text = _extract_structured_segments(candidate)
                    preview_sections = _sections_from_segments(preview_segments)
                    preview = preview_text or _summarize_sections(preview_sections) or candidate
                    if on_text_update:
                        on_text_update(preview)

            error_message = _get_field(event, "error_message")
            if error_message:
                final_text = error_message
                if on_text_update:
                    on_text_update(final_text)

            is_final = _get_field(event, "is_final_response")
            if callable(is_final):
                try:
                    is_final = is_final()
                except Exception:
                    is_final = False
            if is_final:
                break
    except Exception as exc:
        print(f"Error in _stream_agent_query: {exc}")
        final_text = "エラーが発生しました。"
        if on_text_update:
            on_text_update(final_text)

    final_text = final_text or "申し訳ありません、返答を生成できませんでした。"
    segments, display_text = _extract_structured_segments(final_text)
    sections = _sections_from_segments(segments)
    normalized_text = display_text or _summarize_sections(sections) or final_text

    return {
        "raw_text": final_text,
        "display_text": normalized_text,
        "segments": segments,
        "sections": sections,
        "tool_logs": tool_logs,
    }


def _get_session_id(session: Any) -> str:
    if session is None:
        raise RuntimeError("Agent Engine session could not be created.")
    if isinstance(session, dict):
        session_id = session.get("id") or session.get("session_id")
    else:
        session_id = getattr(session, "id", None) or getattr(session, "session_id", None)
    if not session_id:
        raise RuntimeError("Agent Engine session is missing an ID.")
    return session_id


def _ensure_agent_session() -> tuple[str, str]:
    if "agent_engine_user_id" not in st.session_state:
        st.session_state.agent_engine_user_id = str(uuid.uuid4())

    if "agent_engine_session_id" not in st.session_state:
        user_id = st.session_state.agent_engine_user_id
        session = remote_app.create_session(user_id=user_id)
        st.session_state.agent_engine_session = session
        st.session_state.agent_engine_session_id = _get_session_id(session)

    return st.session_state.agent_engine_user_id, st.session_state.agent_engine_session_id


def _send_message(
    user_id: str,
    session_id: str,
    text: str,
    on_text_update: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    return _stream_agent_query(user_id, session_id, text, on_text_update)


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


def _stringify_struct(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _decode_unicode_escapes(value: Any) -> Any:
    if isinstance(value, str):
        if "\\u" in value:
            try:
                return value.encode("utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                return value
        return value
    if isinstance(value, list):
        return [_decode_unicode_escapes(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_unicode_escapes(val) for key, val in value.items()}
    return value


def _extract_json_payload(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    trimmed = raw_text.strip()

    if trimmed.startswith("```"):
        trimmed = trimmed[3:]
        if trimmed.lower().startswith("json"):
            trimmed = trimmed[4:]
        trimmed = trimmed.strip()
        if trimmed.endswith("```"):
            trimmed = trimmed[:-3]
        trimmed = trimmed.strip()

    try:
        json.loads(trimmed)
        return trimmed
    except (TypeError, ValueError):
        pass

    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = trimmed[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except (TypeError, ValueError):
            return None

    return None


def _flatten_segments(payload: Any) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        for item in payload:
            segments.extend(_flatten_segments(item))
        return segments

    if isinstance(payload, dict):
        if isinstance(payload.get("segments"), list):
            for item in payload["segments"]:
                segments.extend(_flatten_segments(item))
            return segments

        if "type" in payload:
            segment: Dict[str, Any] = {"type": str(payload.get("type"))}
            if "content" in payload:
                segment["content"] = payload["content"]
            text_value = payload.get("text")
            if text_value is not None:
                segment["text"] = text_value
                if "content" not in segment:
                    segment["content"] = text_value
            if "section" in payload:
                segment["section"] = payload["section"]
            for meta_key in ("title", "summary", "label", "id", "metadata"):
                if meta_key in payload:
                    segment[meta_key] = payload[meta_key]
            segments.append(segment)
            return segments

        segments.append({"type": "object", "content": payload})
        return segments

    if payload is None:
        return []

    return [{"type": "text", "content": payload}]


def _extract_structured_segments(raw_text: str) -> Tuple[List[Dict[str, Any]], str]:
    if not raw_text:
        return [], ""

    payload_text = _extract_json_payload(raw_text)
    if payload_text is None:
        return [], raw_text

    try:
        payload = json.loads(payload_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], raw_text

    segments = _flatten_segments(payload)
    if not segments:
        return [], raw_text

    text_parts: List[str] = []
    for segment in segments:
        if segment.get("type") == "text":
            text_field = segment.get("text")
            if isinstance(text_field, str) and text_field.strip():
                text_parts.append(text_field)
                continue
        if segment.get("type") == "text" and isinstance(segment.get("content"), str):
            text_parts.append(segment["content"])

    display_text = "\n\n".join(part.strip() for part in text_parts if part) if text_parts else ""
    return segments, display_text.strip()


def _first_non_empty(
    data: Dict[str, Any],
    keys: Tuple[str, ...],
    fallback: Optional[Dict[str, Any]] = None,
):
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    if fallback:
        for key in keys:
            if key in fallback and fallback[key] not in (None, ""):
                return fallback[key]
    return None


def _normalize_card_entry(card_data: Dict[str, Any], position: int) -> Dict[str, Any]:
    fields = card_data.get("fields") if isinstance(card_data.get("fields"), dict) else {}

    entry: Dict[str, Any] = {}
    entry["title"] = (
        _first_non_empty(
            card_data,
            ("title", "name", "商品名", "label"),
            fields,
        )
        or f"候補 {position}"
    )

    entry["price"] = _first_non_empty(
        card_data,
        (
            "price",
            "price_text",
            "price_display",
            "price_range",
            "おおよその価格",
            "価格",
            "approx_price",
            "cost",
        ),
        fields,
    )

    entry["extracted_price"] = _first_non_empty(
        card_data,
        (
            "extracted_price",
            "price_value",
            "price_value_raw",
            "price_number",
            "price_numeric",
        ),
        fields,
    )

    entry["product_link"] = _first_non_empty(
        card_data,
        ("product_link", "url", "商品ページURL", "購入リンク", "link"),
        fields,
    )

    entry["thumbnail"] = _first_non_empty(
        card_data,
        ("thumbnail", "image", "image_url", "画像URL", "画像リンク", "thumbnail_url"),
        fields,
    )

    entry["serpapi_product_api"] = _first_non_empty(
        card_data,
        ("serpapi_product_api", "serpapi", "商品ID", "SerpApi", "serpapi_product_id"),
        fields,
    )

    entry["reason"] = _first_non_empty(card_data, ("reason", "推薦理由", "justification"), fields)
    entry["description"] = _first_non_empty(card_data, ("description", "詳細", "補足", "notes"), fields)
    entry["shipping"] = _first_non_empty(card_data, ("shipping", "送料情報"), fields)

    if not entry.get("product_link"):
        cta = card_data.get("cta") or card_data.get("button")
        if isinstance(cta, dict):
            entry["product_link"] = cta.get("url") or cta.get("href")

    links = card_data.get("links")
    if not entry.get("product_link") and isinstance(links, list):
        for link in links:
            if isinstance(link, dict) and link.get("url"):
                entry["product_link"] = link["url"]
                break

    position_raw = card_data.get("position")
    try:
        entry["position"] = int(position_raw) if position_raw is not None else position
    except (TypeError, ValueError):
        entry["position"] = position

    normalized = {k: v for k, v in entry.items() if v not in (None, "") or k == "position"}
    if "position" not in normalized:
        normalized["position"] = position

    if fields:
        normalized["fields"] = fields

    return normalized


def _normalize_section(section_data: Dict[str, Any], default_title: str = "") -> Dict[str, Any]:
    items_source = section_data.get("items") or section_data.get("cards") or section_data.get("products")
    items_list = _ensure_list(items_source)

    normalized_items: List[Dict[str, Any]] = []
    for idx, raw_item in enumerate(items_list, start=1):
        if isinstance(raw_item, dict):
            normalized = _normalize_card_entry(raw_item, idx)
            if normalized:
                normalized_items.append(normalized)

    title_value = _first_non_empty(section_data, ("title", "name", "カテゴリ名", "heading"))
    summary_value = _first_non_empty(section_data, ("summary", "description", "overview", "要約"))

    title = str(title_value) if title_value not in (None, "") else default_title
    summary = str(summary_value) if summary_value not in (None, "") else ""

    return {
        "title": title,
        "summary": summary,
        "items": normalized_items,
    }


def _sections_from_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []

    for segment in segments:
        seg_type = str(segment.get("type") or "").lower()
        section_payload = segment.get("section")
        content = segment.get("content")

        if section_payload and isinstance(section_payload, dict):
            normalized = _normalize_section(section_payload, default_title=segment.get("title", ""))
            if normalized["items"]:
                sections.append(normalized)
            continue

        if seg_type in {"product_section", "card_section", "section"} and isinstance(content, dict):
            normalized = _normalize_section(content, default_title=segment.get("title", ""))
            if normalized["items"]:
                sections.append(normalized)
        elif seg_type in {"section_list"} and isinstance(content, list):
            for entry in content:
                if isinstance(entry, dict):
                    normalized = _normalize_section(entry)
                    if normalized["items"]:
                        sections.append(normalized)
        elif seg_type in {"cards", "card"}:
            cards_list = content if isinstance(content, list) else [content]
            normalized_cards: List[Dict[str, Any]] = []
            for idx, card in enumerate(cards_list, start=1):
                if isinstance(card, dict):
                    normalized = _normalize_card_entry(card, idx)
                    if normalized:
                        normalized_cards.append(normalized)
            if normalized_cards:
                sections.append(
                    {
                        "title": segment.get("title") or f"提案 {len(sections) + 1}",
                        "summary": segment.get("summary", ""),
                        "items": normalized_cards,
                    }
                )
        else:
            continue

    return sections


def _summarize_sections(sections: List[Dict[str, Any]]) -> str:
    titles = [section.get("title", "").strip() for section in sections if section.get("title")]
    return "\n\n".join(titles)


def _render_structured_segments(
    message_index: int,
    message: Dict[str, Any],
    segments: List[Dict[str, Any]],
) -> None:
    for segment in segments:
        seg_type = str(segment.get("type") or "").lower()
        content = segment.get("content")
        text_value = segment.get("text") if isinstance(segment.get("text"), str) else None

        if seg_type in {"product_section", "card_section", "cards", "card", "section", "section_list"}:
            continue

        if seg_type in {"text", "message"}:
            display_text = text_value or (content if isinstance(content, str) else _stringify_struct(content))
            if display_text:
                st.markdown(display_text)
        elif seg_type == "markdown":
            display_text = text_value or (content if isinstance(content, str) else _stringify_struct(content))
            st.markdown(display_text)
        elif seg_type == "html":
            html_value = text_value or (content if isinstance(content, str) else _stringify_struct(content))
            st.markdown(html_value, unsafe_allow_html=True)
        elif seg_type in {"divider", "separator"}:
            st.divider()
        elif seg_type == "code":
            language = segment.get("language") or "text"
            st.code(content if isinstance(content, str) else _stringify_struct(content), language=language)
        elif seg_type in {"info", "warning", "success"}:
            text_value = content if isinstance(content, str) else _stringify_struct(content)
            if seg_type == "info":
                st.info(text_value)
            elif seg_type == "warning":
                st.warning(text_value)
            else:
                st.success(text_value)
        elif seg_type in {"object", "json"}:
            st.code(_stringify_struct(content), language="json")
        else:
            fallback = text_value or _stringify_struct(content)
            if fallback:
                st.markdown(fallback)


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
        data = response.json()
        return _decode_unicode_escapes(data)
    except requests.exceptions.RequestException as e:
        st.error(f"商品詳細の取得に失敗しました: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"レスポンスの解析に失敗しました: {e}")
        return None


def _display_product_details_sidebar(product_data: Dict[str, Any]) -> None:
    """Display product details in sidebar."""
    st.sidebar.markdown("## 📱 商品詳細")

    # エラーチェック
    if st.session_state.get("product_details_error"):
        st.sidebar.error(st.session_state.product_details_error)
        return

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
    st.session_state.product_details_error = None

    # 商品詳細を取得
    product_data = _fetch_product_details(serpapi_url)

    st.session_state.loading_product_details = False
    if product_data:
        st.session_state.product_details_data = product_data
    else:
        st.session_state.product_details_error = "取得に失敗しました。他の商品で試してみてください。"

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
            fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}

            def _coalesce(keys: Tuple[str, ...], default: Optional[Any] = None) -> Any:
                value = _first_non_empty(item, keys, fields)

                if isinstance(value, (list, tuple)):
                    resolved = None
                    for candidate in value:
                        if isinstance(candidate, str) and candidate.strip():
                            resolved = candidate
                            break
                        if isinstance(candidate, dict):
                            for link_key in ("url", "link", "href"):
                                link_value = candidate.get(link_key)
                                if isinstance(link_value, str) and link_value.strip():
                                    resolved = link_value
                                    break
                            if resolved:
                                break
                    value = resolved

                if isinstance(value, dict):
                    for link_key in ("url", "link", "href", "text"):
                        link_value = value.get(link_key)
                        if isinstance(link_value, str) and link_value.strip():
                            value = link_value
                            break

                if value in (None, ""):
                    return default
                return value

            entry: Dict[str, Any] = {
                "title": _coalesce(("title", "name", "label", "商品名"), f"候補 {item_index}"),
                "price": _coalesce(
                    (
                        "price",
                        "price_text",
                        "price_display",
                        "price_range",
                        "おおよその価格",
                        "価格",
                        "approx_price",
                        "cost",
                    )
                ),
                "extracted_price": _coalesce(
                    (
                        "extracted_price",
                        "price_value",
                        "price_value_raw",
                        "price_number",
                        "price_numeric",
                    )
                ),
                "position": item.get("position") or item_index,
                "thumbnail": _coalesce(
                    (
                        "thumbnail",
                        "image",
                        "image_url",
                        "画像URL",
                        "画像リンク",
                        "thumbnail_url",
                    )
                ),
                "product_link": _coalesce(
                    (
                        "product_link",
                        "url",
                        "link",
                        "productUrl",
                        "商品ページURL",
                        "購入リンク",
                        "purchase_url",
                    )
                ),
                "serpapi_product_api": _coalesce(
                    (
                        "serpapi_product_api",
                        "serpapi",
                        "SerpApi",
                        "商品ID",
                        "serpapi_product_id",
                    )
                ),
                "reason": _coalesce(("reason", "推薦理由", "justification", "why")),
                "description": _coalesce(("description", "詳細", "補足", "notes")),
                "shipping": _coalesce(("shipping", "送料情報", "delivery")),
            }

            if not entry.get("product_link"):
                cta_candidate = item.get("cta")
                if not isinstance(cta_candidate, dict):
                    fields_cta = fields.get("cta")
                    if not isinstance(fields_cta, dict):
                        fields_cta = None
                    cta_candidate = fields_cta
                if isinstance(cta_candidate, dict):
                    entry["product_link"] = cta_candidate.get("url") or cta_candidate.get("href")

            if not entry.get("product_link") and isinstance(item.get("links"), list):
                for link in item["links"]:
                    if isinstance(link, dict) and link.get("url"):
                        entry["product_link"] = link["url"]
                        break

            for key in (
                "title",
                "price",
                "thumbnail",
                "product_link",
                "reason",
                "description",
                "shipping",
                "serpapi_product_api",
            ):
                value = entry.get(key)
                if value not in (None, ""):
                    entry[key] = str(value)

            if entry.get("extracted_price") not in (None, ""):
                try:
                    entry["extracted_price"] = float(entry["extracted_price"])
                except (TypeError, ValueError):
                    entry["extracted_price"] = entry["extracted_price"]

            try:
                entry["position"] = int(entry.get("position", item_index))
            except (TypeError, ValueError):
                entry["position"] = item_index

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


def _initialize_conversation(user_id: str, session_id: str) -> None:
    if st.session_state.get("initialized"):
        return

    st.session_state.messages = []
    try:
        initial_response = _send_message(user_id, session_id, "")
    except Exception:
        initial_response = {
            "raw_text": "",
            "display_text": "",
            "segments": [],
            "sections": [],
            "tool_logs": [],
        }

    display_text = initial_response.get("display_text", "")
    if (not display_text) or ("エラー" in display_text) or ("SERPAPI" in display_text):
        display_text = "その人の職業や年齢、Xのリンクなどを教えてください。誕生日プレゼント選びをお手伝いします。"
        initial_response["segments"] = []
        initial_response["sections"] = []
        initial_response["tool_logs"] = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": display_text,
            "segments": initial_response.get("segments", []),
            "sections": initial_response.get("sections", []),
            "raw_response": initial_response.get("raw_text", display_text),
            "tool_logs": initial_response.get("tool_logs", []),
        }
    )
    st.session_state.initialized = True


def _render_messages() -> None:
    for index, message in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                segments: List[Dict[str, Any]] = message.get("segments", []) or []
                rendered_sections: List[Dict[str, Any]] = []

                if segments:
                    _render_structured_segments(index, message, segments)
                    rendered_sections = message.get("sections") or _sections_from_segments(segments)
                    if rendered_sections:
                        message.setdefault("sections", rendered_sections)
                        _render_shopping_sections(index, message, rendered_sections)
                else:
                    content = message.get("content", "")
                    sections = _parse_agent_sections(content)
                    text_to_render = _extract_non_section_text(content) if sections else content
                    if text_to_render:
                        st.markdown(text_to_render)
                    if sections:
                        message.setdefault("sections", sections)
                        _render_shopping_sections(index, message, sections)
            else:
                st.markdown(message.get("content", ""))
            for log in message.get("tool_logs", []):
                label = "ツール呼び出し" if log["type"] == "call" else "ツール応答"
                with st.expander(f"{label}: {log['name']}", expanded=False):
                    st.code(log["payload"], language="json")


def _handle_user_turn(user_id: str, session_id: str, text: str) -> None:
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    response_data: Dict[str, Any]
    with st.chat_message("assistant"):
        preview_placeholder = st.empty()
        try:
            with st.spinner("候補を考えています..."):
                response_data = _send_message(
                    user_id,
                    session_id,
                    text,
                    on_text_update=lambda preview: preview_placeholder.markdown(preview or ""),
                )
        except Exception as error:
            st.error(f"エラーが発生しました: {error}")
            preview_placeholder.markdown("申し訳ありません、処理中にエラーが発生しました。もう一度お試しください。")
            response_data = {
                "raw_text": "",
                "display_text": "申し訳ありません、処理中にエラーが発生しました。もう一度お試しください。",
                "segments": [],
                "sections": [],
                "tool_logs": [],
            }

    preview_placeholder.markdown(response_data.get("display_text") or response_data.get("raw_text", ""))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_data.get("display_text")
            or response_data.get("raw_text")
            or "申し訳ありません、返答を生成できませんでした。",
            "segments": response_data.get("segments", []),
            "sections": response_data.get("sections", []),
            "raw_response": response_data.get("raw_text"),
            "tool_logs": response_data.get("tool_logs", []),
        }
    )
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="birthd.ai [バースデイ]", page_icon="🎁", layout="wide")
    st.title("🎁 birthd.ai [バースデイ]")
    st.caption("Google ADK + Gemini + Grok + SerpApi を活用した誕生日プレゼント提案エージェント")

    _inject_custom_styles()

    loading_placeholder = st.empty()
    if not st.session_state.get("initialized"):
        with loading_placeholder.container():
            st.info("チャットを準備中です")
        with st.spinner("初期メッセージを準備しています..."):
            user_id, session_id = _ensure_agent_session()
            _initialize_conversation(user_id, session_id)
        loading_placeholder.empty()
    else:
        user_id, session_id = _ensure_agent_session()
        _initialize_conversation(user_id, session_id)

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
        _handle_user_turn(user_id, session_id, user_input)
        st.session_state["chat_input"] = ""


if __name__ == "__main__":
    main()
