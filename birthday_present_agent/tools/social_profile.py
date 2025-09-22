"""Tool that coordinates manual Grok checks for X profiles."""

from __future__ import annotations

import re
from typing import Optional

from google.adk.tools import ToolContext
from google.genai import types


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
    """Guide a human to run the Grok prompt manually to avoid quota limits."""

    handle = _extract_handle(x_url)
    profile_url = f"https://x.com/{handle}" if handle else x_url
    prompt = f"{profile_url} の最新100件の投稿を調べ、職業・趣味・欲しいものなどを調査してください"

    instructions = (
        "Grok API の直接呼び出しを避けるため、手動で確認してください。\n"
        "1. https://grok.com を開き、チャット画面にアクセスします。\n"
        "2. 次のプロンプトをそのまま入力して送信します。\n\n"
        f"{prompt}\n\n"
        "3. Grok の回答内容をコピーし、このエージェントにペーストして共有してください。"
    )

    try:
        await tool_context.save_artifact(
            name="grok_manual_prompt",
            artifact=types.Part.from_text(prompt),
        )
    except Exception:
        pass

    return instructions
