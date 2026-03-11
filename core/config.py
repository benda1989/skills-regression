"""LLM 配置模块：从环境变量读取配置"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """LLM 配置类，用于配置 OpenAI 或兼容的 API 参数

    Attributes:
        api_key: OpenAI API 密钥，如果为 None 则从环境变量读取
        base_url: API 基础 URL，用于兼容其他 OpenAI 兼容服务
        model: 使用的模型名称，默认为 "gpt-4o-mini"
        temperature: 模型温度参数，控制输出的随机性，范围 0-2，默认 0.1
        max_tokens: 最大生成 token 数，None 表示使用模型默认值
        timeout: 请求超时时间（秒），默认 60
        max_retries: 最大重试次数，默认 3
    """

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    max_retries: int = 3

    @staticmethod
    def from_env() -> "LLMConfig":
        """从环境变量创建 LLMConfig 实例

        Environment Variables:
            OPENAI_API_KEY: API 密钥
            OPENAI_BASE_URL: API 基础 URL（可选）
            OPENAI_MODEL: 模型名称（默认: gpt-4o-mini）
            OPENAI_TEMPERATURE: 温度参数（默认: 0.1）
            OPENAI_MAX_TOKENS: 最大 token 数（可选）
            OPENAI_TIMEOUT: 请求超时时间（秒，默认: 60）
            OPENAI_MAX_RETRIES: 最大重试次数（默认: 3）
        """
        return LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "0")) or None,
            timeout=float(os.getenv("OPENAI_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
        )
