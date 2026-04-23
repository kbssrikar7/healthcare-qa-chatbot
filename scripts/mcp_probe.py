import argparse
import json

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


SERVERS = {
    "playwright": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest", "--headless"],
    },
    "selenium": {
        "command": "npx",
        "args": ["-y", "@angiejones/mcp-selenium@latest"],
    },
}


async def _main(server_name: str) -> None:
    config = SERVERS[server_name]
    params = StdioServerParameters(
        command=config["command"],
        args=config["args"],
        cwd="/home/kbs/Documents/final_project",
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            payload = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools.tools
            ]
            print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("server", choices=sorted(SERVERS))
    args = parser.parse_args()
    anyio.run(_main, args.server)
