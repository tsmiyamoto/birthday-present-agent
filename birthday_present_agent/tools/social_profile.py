"""Tool that coordinates manual Grok checks for X profiles."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import quote_plus

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
    prompt = f"{profile_url} の投稿やプロフィールを元に、趣味・好きなもの・興味のあるもの・欲しているものなどを推測し、簡潔に回答してください"
    grok_url = f"https://grok.com/?q={quote_plus(prompt)}"

    instructions = (
        "ユーザーには次のメッセージを表示し、手動での確認を促してください。\n"
        "X API の制限を回避するため、手動で確認してください。\n"
        f"1. [Grokを開く]({grok_url}) をタップして、プロンプトが入力された状態の Grok チャットにアクセスします。\n"
        "2. Grok の回答内容をコピーします。\n"
        "3. このエージェントにペーストして共有してください。"
    )

    try:
        await tool_context.save_artifact(
            name="grok_manual_prompt",
            artifact=types.Part.from_text(prompt),
        )
    except Exception:
        pass

    return instructions
