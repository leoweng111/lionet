"""
LLM 调用工具模块。使用 anthropic SDK 格式调用 LLM。

所有 LLM 相关的调用代码集中在此文件中，支持多模型配置（通过 .env 的 LLM_PROFILES）。
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

# ─── 默认配置 ───
DEFAULT_BASE_URL = "https://api.minimaxi.com/anthropic"
DEFAULT_MODEL = "MiniMax-M2.7-highspeed"
DEFAULT_MAX_TOKENS = 16000


def _strip_inline_thinking(text: str) -> str:
    """
    清理 LLM 回复中可能内嵌的 <thinking>...</thinking> 或 <antThinking>...</antThinking> 标记。
    """
    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<antThinking>.*?</antThinking>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _find_env_path() -> Optional[Path]:
    """查找项目根目录下的 .env 文件。"""
    # 从本文件向上找项目根目录
    current = Path(__file__).resolve().parent
    for _ in range(5):
        env_path = current / '.env'
        if env_path.exists():
            return env_path
        current = current.parent
    return None


def load_llm_profiles() -> List[Dict[str, Any]]:
    """从 .env 文件中加载 LLM_PROFILES 配置列表。每次调用都重新读取文件，确保实时更新。"""
    env_path = _find_env_path()
    if env_path is None or not env_path.exists():
        return []

    content = env_path.read_text(encoding='utf-8')
    # 解析 LLM_PROFILES=[ ... ]
    match = re.search(r'LLM_PROFILES\s*=\s*(\[.*?\])', content, re.DOTALL)
    if not match:
        return []

    try:
        profiles = json.loads(match.group(1))
        if isinstance(profiles, list):
            return profiles
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def get_profile_by_name(name: str) -> Optional[Dict[str, Any]]:
    """根据 name 获取 LLM profile。每次调用都重新读取 .env。"""
    profiles = load_llm_profiles()
    for p in profiles:
        if p.get('name') == name:
            return p
    return None


def _get_client(base_url: Optional[str] = None, api_key: Optional[str] = None) -> "anthropic.Anthropic":
    """创建 anthropic 客户端实例。"""
    if anthropic is None:
        raise ImportError("请安装 anthropic: pip install anthropic")

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    resolved_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
    if not resolved_key:
        raise ValueError(
            "API Key 未配置。请在项目根目录 .env 文件的 LLM_PROFILES 中填入有效的 api_key。"
        )
    try:
        resolved_key.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(
            "API Key 包含非 ASCII 字符，请在 .env 中填入有效的 api_key。"
        )
    return anthropic.Anthropic(
        base_url=resolved_url,
        api_key=resolved_key,
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> str:
    """
    单轮 LLM 调用。

    :param system_prompt: 系统提示词
    :param user_prompt:   用户输入
    :param model:         模型名称
    :param max_tokens:    最大输出 token 数
    :param profile_name:  使用 .env 中的哪个 profile（优先级高于其他参数）
    :return:              模型回复的文本内容
    """
    if profile_name:
        profile = get_profile_by_name(profile_name)
        if profile:
            base_url = base_url or profile.get('base_url')
            api_key = api_key or profile.get('api_key')
            model = model or profile.get('model')
            max_tokens = profile.get('max_tokens', max_tokens)

    client = _get_client(base_url, api_key)
    resolved_model = model or os.environ.get("LLM_MODEL_NAME", DEFAULT_MODEL)
    message = client.messages.create(
        model=resolved_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ],
    )
    # 提取文本内容
    text_parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            text_parts.append(block.text)
    if not text_parts:
        for block in message.content:
            if hasattr(block, "text") and getattr(block, "type", None) != "thinking":
                text_parts.append(block.text)
    result = "\n".join(text_parts)
    result = _strip_inline_thinking(result)
    return result


def call_llm_streaming(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    profile_name: Optional[str] = None,
):
    """
    流式 LLM 调用，返回一个生成器，逐块产出文本。

    :yields: str - 每次产出的文本片段
    """
    if profile_name:
        profile = get_profile_by_name(profile_name)
        if profile:
            base_url = base_url or profile.get('base_url')
            api_key = api_key or profile.get('api_key')
            model = model or profile.get('model')
            max_tokens = profile.get('max_tokens', max_tokens)

    client = _get_client(base_url, api_key)
    resolved_model = model or os.environ.get("LLM_MODEL_NAME", DEFAULT_MODEL)

    with client.messages.stream(
        model=resolved_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ],
    ) as stream:
        for text in stream.text_stream:
            yield text


def extract_code_from_response(response_text: str) -> str:
    """
    从 LLM 回复中提取代码块。
    支持 ```python ... ``` 或 ```yaml ... ``` 等格式。
    如果没有代码块标记，则返回原文。
    """
    pattern = r"```(?:python|yaml|json|text)?\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return response_text.strip()

