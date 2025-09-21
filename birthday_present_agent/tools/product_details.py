"""Fetch detailed product information from SerpApi."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx
from google.adk.tools import ToolContext
from google.genai import types
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

PRODUCT_ENDPOINT = os.getenv("SERPAPI_PRODUCT_ENDPOINT", "https://serpapi.com/search.json")


async def _request(params: Dict[str, Any], url_override: Optional[str] = None) -> Dict[str, Any]:
    request_kwargs: Dict[str, Any] = {}
    if url_override:
        request_kwargs["url"] = url_override
    else:
        request_kwargs["url"] = PRODUCT_ENDPOINT
        request_kwargs["params"] = params

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6)
    ):
        with attempt:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(**request_kwargs)
                response.raise_for_status()
                return response.json()
    raise RuntimeError("Unreachable")


def _format_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    product = raw.get("product_results") or {}
    sellers = raw.get("sellers_results", {})
    online = sellers.get("online_sellers", []) if isinstance(sellers, dict) else []

    formatted: Dict[str, Any] = {
        "title": product.get("title"),
        "description": product.get("description"),
        "prices": product.get("prices"),
        "conditions": product.get("conditions"),
        "extensions": product.get("extensions"),
        "media": product.get("media"),
        "product_id": product.get("product_id"),
        "product_link": product.get("product_link") or raw.get("search_metadata", {}).get("google_product_url"),
        "sellers": online,
    }
    # Remove None entries for readability
    return {k: v for k, v in formatted.items() if v}


async def fetch_product_details(product_reference: str, tool_context: ToolContext) -> str:
    """Retrieve a rich product record for a SerpApi Google Shopping item.

    Args:
        product_reference: `serpapi_product_api` URL、商品の `product_id`、または `product_id:<ID>` 形式の文字列。
        tool_context: ADKツール実行コンテキスト。

    Returns:
        JSON文字列。商品概要と販売情報を含む。
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set. Add it to your .env file.")

    product_reference = product_reference.strip()
    params: Dict[str, Any] = {
        "engine": "google_product",
        "api_key": api_key,
    }
    url_override: Optional[str] = None

    if product_reference.startswith("http"):
        # Append API key without duplicating existing query parameters
        connector = "&" if "?" in product_reference else "?"
        url_override = f"{product_reference}{connector}api_key={api_key}"
    else:
        product_id = product_reference.split(":", 1)[1] if product_reference.startswith("product_id:") else product_reference
        params["product_id"] = product_id
        params.setdefault("gl", os.getenv("SERPAPI_GL", "jp"))
        params.setdefault("hl", os.getenv("SERPAPI_HL", "ja"))

    try:
        raw = await _request(params, url_override=url_override)
    except RetryError as error:
        raise RuntimeError(f"SerpApi product lookup failed after retries: {error}") from error
    except httpx.HTTPError as error:
        raise RuntimeError(f"SerpApi product lookup failed: {error}") from error

    formatted = _format_response(raw)
    payload = json.dumps(formatted, ensure_ascii=False, indent=2)

    try:
        await tool_context.save_artifact(
            name="product_details",
            artifact=types.Part.from_text(payload),
        )
    except Exception:
        pass

    return payload
