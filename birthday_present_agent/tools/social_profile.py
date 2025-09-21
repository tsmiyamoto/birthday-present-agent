"""Tool that summarizes X profiles via the xAI SDK Live Search."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Optional

from google.adk.tools import ToolContext
from google.genai import types
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential
from xai_sdk import AsyncClient
from xai_sdk.chat import system, user
from xai_sdk.search import SearchParameters, x_source

_CLIENT_LOCK = asyncio.Lock()
_CLIENT: Optional[AsyncClient] = None


async def _get_client() -> AsyncClient:
    """Create or return a cached AsyncClient instance."""

    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    async with _CLIENT_LOCK:
        if _CLIENT is None:
            api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
            if not api_key:
                raise RuntimeError("XAI_API_KEY is not set. Add it to your .env file (fallback to GROK_API_KEY).")
            _CLIENT = AsyncClient(api_key=api_key)
    return _CLIENT


def _extract_handle(x_url: str) -> Optional[str]:
    """Return the X handle (without @) if present in the URL."""

    match = re.search(r"(?:x|twitter)\.com/([^/?#]+)", x_url)
    if not match:
        return None
    handle = match.group(1)
    handle = handle.strip("@")
    if handle:
        return handle
    return None


async def fetch_social_profile(x_url: str, tool_context: ToolContext) -> str:
    """Use xAI Grok Live Search to infer profile details from an X link."""

    client = await _get_client()

    handle = _extract_handle(x_url)
    search_sources = [
        x_source(included_x_handles=[handle]) if handle else x_source(),
    ]

    search_parameters = SearchParameters(
        mode="on",
        return_citations=True,
        max_search_results=10,
        sources=search_sources,
    )

    model = os.getenv("XAI_MODEL") or os.getenv("GROK_MODEL") or "grok-3-latest"

    chat = client.chat.create(
        model=model,
        messages=[
            system(
                "You extract public persona insights from X profiles.",
                "Respond in Japanese using JSON with keys: display_name, probable_roles, interests, style, notable_quotes, citations.",
                "Mark inferred attributes with a note such as '推測'.",
            ),
            user(f"対象URL: {x_url}\n" "公開情報だけを使い、その人物像や関係するヒントを簡潔にまとめてください。"),
        ],
        temperature=0.2,
        search_parameters=search_parameters,
        response_format="json_object",
    )

    async def _sample() -> str:
        response = await chat.sample()
        return response.content

    content: Optional[str] = None
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=6),
        ):
            with attempt:
                content = await _sample()
                break
    except RetryError as error:
        raise RuntimeError(f"Grok Live Search request failed after retries: {error}") from error

    if content is None:
        raise RuntimeError("Failed to retrieve response from Grok Live Search.")

    # chat.sample already returns JSON when response_format="json_object".
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    # Ensure pretty formatting for readability.
    try:
        parsed = json.loads(content)
        payload_text = json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        payload_text = content

    try:
        await tool_context.save_artifact(
            name="grok_profile",
            artifact=types.Part.from_text(payload_text),
        )
    except Exception:
        pass

    return payload_text
