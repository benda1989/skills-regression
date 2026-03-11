"""Bearer token 鉴权中间件"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_secret() -> Optional[str]:
    return os.getenv("API_SECRET_KEY")


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> str:
    """依赖注入：验证 Bearer token，返回 token 字符串

    在没有配置 API_SECRET_KEY 时（开发模式）跳过验证。
    """
    secret = _get_secret()
    if not secret:
        # 未配置密钥时跳过验证（开发环境）
        return "dev-no-auth"

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials
