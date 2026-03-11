"""LLM 调用模块：封装 OpenAI 兼容 API 调用 + MD5 缓存"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import LLMConfig

logger = logging.getLogger("regression_agent.llm")

# ── 缓存 ──────────────────────────────────────────────────────────────────────

_llm_cache: Dict[str, Dict[str, Any]] = {}
_MAX_CACHE_SIZE = 100


def _get_cache_key(system_prompt: str, user_content: str) -> str:
    combined = f"{system_prompt}|||{user_content}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def clear_llm_cache() -> None:
    """清空 LLM 调用缓存"""
    global _llm_cache
    _llm_cache.clear()
    logger.debug("LLM cache cleared")


def get_llm_cache_stats() -> Dict[str, Any]:
    """获取 LLM 缓存统计信息"""
    return {
        "cache_size": len(_llm_cache),
        "max_size": _MAX_CACHE_SIZE,
        "cache_usage_percent": round(len(_llm_cache) / _MAX_CACHE_SIZE * 100, 2),
    }


# ── LLM 调用 ──────────────────────────────────────────────────────────────────

def build_llm(config: LLMConfig) -> ChatOpenAI:
    """根据配置构建 ChatOpenAI 实例"""
    kwargs: Dict[str, Any] = {
        "api_key": config.api_key,
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    return ChatOpenAI(**kwargs)


def call_json(
    llm: ChatOpenAI,
    config: LLMConfig,
    system_prompt: str,
    user_content: str,
) -> Dict[str, Any]:
    """调用 LLM 并解析 JSON 响应（带缓存优化和重试机制）

    使用基于 prompt + content 哈希的缓存机制，避免重复调用相同内容的 LLM API。
    缓存大小限制为 _MAX_CACHE_SIZE 条目，采用 FIFO 策略。
    包含自动重试机制，处理网络连接错误。
    """
    cache_key = _get_cache_key(system_prompt, user_content)
    if cache_key in _llm_cache:
        logger.debug("LLM cache hit for key %s", cache_key[:8])
        return json.loads(json.dumps(_llm_cache[cache_key]))

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    last_error: Optional[Exception] = None
    max_attempts = config.max_retries + 1

    t0 = time.time()
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(messages)
            break
        except Exception as e:
            last_error = e
            error_type = type(e).__name__
            error_msg = str(e)

            # 模型不存在错误 → 不重试
            is_model_error = (
                "Model Not Exist" in error_msg
                or "model_not_found" in error_msg.lower()
                or "invalid_request_error" in error_msg.lower()
                or ("model" in error_msg.lower() and ("not exist" in error_msg.lower() or "not found" in error_msg.lower()))
            )
            if is_model_error:
                base_url = config.base_url or "https://api.openai.com/v1"
                model_name = config.model
                is_deepseek = "deepseek" in base_url.lower() or "deepseek" in model_name.lower()
                is_dashscope = "dashscope" in base_url.lower() or "aliyuncs.com" in base_url.lower()
                if is_deepseek:
                    raise ValueError(
                        f"DeepSeek 模型配置错误。当前: OPENAI_BASE_URL={base_url}, OPENAI_MODEL={model_name}。"
                        f"原始错误: {error_msg[:200]}"
                    )
                elif is_dashscope:
                    raise ValueError(
                        f"通义千问（DashScope）模型配置错误。当前: OPENAI_BASE_URL={base_url}, OPENAI_MODEL={model_name}。"
                        f"原始错误: {error_msg[:200]}"
                    )
                else:
                    raise ValueError(
                        f"模型不存在或配置错误。当前: OPENAI_MODEL={model_name}。原始错误: {error_msg[:200]}"
                    )

            # 连接错误 → 指数退避重试
            is_connection_error = (
                "Connection" in error_type
                or "connection" in error_msg.lower()
                or "timeout" in error_msg.lower()
                or "APIConnectionError" in error_type
            )
            if is_connection_error and attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 10)
                logger.warning(
                    "LLM connection error, retrying in %ds (attempt %d/%d): %s",
                    wait_time, attempt + 1, max_attempts, error_msg[:100],
                )
                time.sleep(wait_time)
                continue
            else:
                raise

    if last_error is not None:
        raise last_error

    elapsed = time.time() - t0
    logger.info("LLM call completed in %.2fs (model=%s)", elapsed, config.model)

    content_raw = response.content if hasattr(response, "content") else response
    if isinstance(content_raw, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content_raw
        )
    else:
        content = str(content_raw)

    # 解析 JSON
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(content[start: end + 1])
            except Exception:
                raise ValueError(f"Failed to parse JSON from LLM response: {content[:200]}")
        else:
            raise ValueError(f"Failed to parse JSON from LLM response: {content[:200]}")

    # 更新缓存（FIFO 策略）
    if len(_llm_cache) >= _MAX_CACHE_SIZE:
        oldest_key = next(iter(_llm_cache))
        del _llm_cache[oldest_key]
    _llm_cache[cache_key] = result

    return result
