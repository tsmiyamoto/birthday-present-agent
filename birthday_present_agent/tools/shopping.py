"""SerpApi Google Shopping search tool."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import httpx
from google.adk.tools import ToolContext
from google.genai import types
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

SHOPPING_ENDPOINT = os.getenv("SERPAPI_SHOPPING_ENDPOINT", "https://serpapi.com/search.json")


async def _call_serpapi(params: Dict[str, Any]) -> Dict[str, Any]:
    async for attempt in AsyncRetrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6)):
        with attempt:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(SHOPPING_ENDPOINT, params=params)
                response.raise_for_status()
                return response.json()
    raise RuntimeError("Unreachable")


def _summarize_results(raw: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for item in raw.get("shopping_results", [])[:10]:
        entry = {
            "position": item.get("position"),
            "title": item.get("title"),
            "price": item.get("price") or item.get("extracted_price"),
            "extracted_price": item.get("extracted_price"),
            "source": item.get("source"),
            "product_link": item.get("product_link"),
            "thumbnail": item.get("thumbnail") or item.get("serpapi_thumbnail"),
            "serpapi_product_api": item.get("serpapi_product_api"),
            # "serpapi_immersive_product_api": item.get("serpapi_immersive_product_api"),
            "description": item.get("excerpt") or item.get("description"),
            "shipping": item.get("delivery"),
        }
        # Remove None values for readability
        entry = {k: v for k, v in entry.items() if v}
        results.append(entry)
    return {
        "results": results,
        "raw_metadata": {"total_results": raw.get("search_information", {}).get("total_results")},
    }


async def shopping_search(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Perform a Google Shopping search via SerpApi.

    Args:
        query: 商品検索キーワード。
        tool_context: ADKツール実行時のコンテキスト。

    Returns:
        辞書オブジェクト。最大10件の候補と関連メタデータを含む。
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set. Add it to your .env file.")

    params: Dict[str, Any] = {
        "engine": "google_shopping",
        "q": query,
        "gl": os.getenv("SERPAPI_GL", "jp"),
        "hl": os.getenv("SERPAPI_HL", "ja"),
        "num": 20,
        "api_key": api_key,
    }

    try:
        raw = await _call_serpapi(params)
    except RetryError as error:
        raise RuntimeError(f"SerpApi request failed after retries: {error}") from error
    except httpx.HTTPError as error:
        raise RuntimeError(f"SerpApi request failed: {error}") from error

    summary = {"query": query, **_summarize_results(raw)}

    try:
        await tool_context.save_artifact(
            name="shopping_results",
            artifact=types.Part.from_text(json.dumps(summary, ensure_ascii=False, indent=2)),
        )
    except Exception:
        # Artifact logging failures must not break the tool chain.
        pass

    return summary
