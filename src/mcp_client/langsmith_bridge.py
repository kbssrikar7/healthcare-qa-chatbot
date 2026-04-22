"""
LangSmith MCP bridge utilities.

Provides a thin adapter around the generic MCP client so FastAPI endpoints can
query LangSmith MCP tools without embedding MCP session logic in route handlers.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.mcp_client.agent import HealthcareMCPClient, execute_mcp_tool_oneshot


@dataclass
class LangSmithMCPSettings:
    enabled: bool
    server_cmd: str
    server_args: str
    api_key: Optional[str] = None
    workspace_id: Optional[str] = None
    endpoint: Optional[str] = None


def _langsmith_env(settings: LangSmithMCPSettings) -> Optional[Dict[str, str]]:
    env: Dict[str, str] = {}
    if settings.api_key:
        env["LANGSMITH_API_KEY"] = settings.api_key
    if settings.workspace_id:
        env["LANGSMITH_WORKSPACE_ID"] = settings.workspace_id
    if settings.endpoint:
        env["LANGSMITH_ENDPOINT"] = settings.endpoint
    return env or None


def _try_parse_json(text: str) -> Any:
    payload = (text or "").strip()
    if not payload:
        return payload
    try:
        return json.loads(payload)
    except Exception:
        return payload


def _looks_like_tool_error(payload: Any) -> bool:
    if isinstance(payload, str):
        lower = payload.lower()
        return (
            lower.startswith("[mcp error")
            or "validation error for call[" in lower
            or "error validating tool" in lower
            or "missing required argument" in lower
        )
    return False


async def call_langsmith_tool(
    settings: LangSmithMCPSettings,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a LangSmith MCP tool and return a structured result.
    """
    if not settings.enabled:
        return {"ok": False, "error": "LangSmith MCP is disabled"}

    args = shlex.split(settings.server_args or "")
    raw = await execute_mcp_tool_oneshot(
        server_cmd=settings.server_cmd,
        server_args=args,
        tool_name=tool_name,
        tool_args=tool_args or {},
        server_env=_langsmith_env(settings),
    )

    if isinstance(raw, str) and raw.startswith("[MCP Error"):
        return {"ok": False, "error": raw}

    parsed = _try_parse_json(raw)
    if _looks_like_tool_error(parsed):
        return {"ok": False, "error": parsed}
    return {"ok": True, "tool_name": tool_name, "data": parsed}


async def list_langsmith_tools(settings: LangSmithMCPSettings) -> Dict[str, Any]:
    """
    List tools exposed by the configured LangSmith MCP server.
    """
    if not settings.enabled:
        return {"ok": False, "error": "LangSmith MCP is disabled"}

    args = shlex.split(settings.server_args or "")
    try:
        async with HealthcareMCPClient(
            server_cmd=settings.server_cmd,
            server_args=args,
            server_env=_langsmith_env(settings),
        ) as client:
            tools = await client.list_tools()
        return {"ok": True, "tools": tools}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
