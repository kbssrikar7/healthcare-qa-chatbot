import os
import uvicorn
import asyncio
from typing import Any, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class HealthcareMCPClient:
    """
    Client for interacting with external Model Context Protocol (MCP) servers.
    This allows the QA pipeline to fetch real-time data if ChromaDB misses.
    """
    def __init__(self, server_cmd: str, server_args: List[str]):
        """
        Initialize the MCP Client.
        
        Args:
            server_cmd: the command used to run the MCP server (e.g. 'npx')
            server_args: the arguments for the command (e.g. ['-y', '@modelcontextprotocol/server-brave-search'])
        """
        self.server_cmd = server_cmd
        self.server_args = server_args
        self._session: Optional[ClientSession] = None
        self._exit_stack = None
        
    async def __aenter__(self):
        """Async context manager entry: start and connect to the MCP server."""
        from contextlib import AsyncExitStack
        self._exit_stack = AsyncExitStack()
        
        server_params = StdioServerParameters(
            command=self.server_cmd,
            args=self.server_args,
            env=None # Inherit from OS, where API keys should be
        )
        
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.read, self.write = stdio_transport
        
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )
        
        await self._session.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit: gracefully terminate the session."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools available on the connected MCP server."""
        if not self._session:
            raise RuntimeError("MCP Client session not initialized (use 'async with')")
            
        try:
            response = await self._session.list_tools()
            return [tool.model_dump() for tool in response.tools]
        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to list MCP tools: {e}")
            return []
            
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a specific tool on the connected MCP server.
        
        Args:
            tool_name: The precise name of the tool as listed by `list_tools()`
            arguments: The kwargs dictionary matching the tool's expected schema
        """
        if not self._session:
            raise RuntimeError("MCP Client session not initialized (use 'async with')")
            
        try:
            result = await self._session.call_tool(tool_name, arguments)
            
            # Extract plain text content from standard MCP response format
            text_outputs = []
            for item in getattr(result, "content", []):
                if item.type == "text":
                    text_outputs.append(item.text)
                    
            return "\n".join(text_outputs) if text_outputs else str(result)
        except Exception as e:
            return f"[MCP Error calling '{tool_name}']: {str(e)}"

# A global helper for quick one-off initialization and execution
async def execute_mcp_tool_oneshot(server_cmd: str, server_args: List[str], tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Convenience function to spin up an MCP server, run a single tool, and tear it down immediately."""
    async with HealthcareMCPClient(server_cmd, server_args) as client:
        return await client.call_tool(tool_name, tool_args)
