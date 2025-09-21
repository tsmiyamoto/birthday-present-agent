"""Tool that summarizes X profiles via Grok."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import httpx
from google.adk.tools import ToolContext
from google.genai import types
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

GROK_API_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")


async def _call_grok(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {os.getenv('GROK_API_KEY', '')}",
        "Content-Type": "application/json",
    }
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6)
    ):
        with attempt:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(GROK_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
    raise RuntimeError("Unreachable")


async def fetch_social_profile(x_url: str, tool_context: ToolContext) -> str:
    """Use Grok to infer profile details from an X link.

    Args:
        x_url: X(旧Twitter)のプロフィールまたは投稿URL。
        tool_context: ADKツール実行コンテキスト。

    Returns:
        JSON文字列。人物像の推測と引用文を含む。
    """
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("GROK_API_KEY is not set. Add it to your .env file.")

    prompt = (
        "次のXプロフィール/投稿URLから公開情報のみを用いて、人物の概要を推測してください。" \
        "属性は日本語で簡潔にまとめ、推測である場合はその旨を注記し、" \
        "JSONで出力してください。キーは display_name, probable_roles, interests, style, notable_quotes を含めてください。"
    )

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": "You extract public persona insights from X profiles."},
            {"role": "user", "content": f"URL: {x_url}\n\n{prompt}"},
        ],
        "temperature": 0.2,
    }

    try:
        raw = await _call_grok(payload)
    except RetryError as error:
        raise RuntimeError(f"Grok API request failed after retries: {error}") from error
    except httpx.HTTPError as error:
        raise RuntimeError(f"Grok API request failed: {error}") from error

    try:
        content = raw["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected Grok response structure: {raw}") from error

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
