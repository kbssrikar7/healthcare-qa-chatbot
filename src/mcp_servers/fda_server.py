"""
FDA OpenFDA MCP Server — exposes FDA drug database as MCP tools.

Provides real-time drug label information, recall data, and adverse event reports.
No API key required. Rate limit: 240 requests/minute.
Docs: https://open.fda.gov/apis/

Usage:
    python src/mcp_servers/fda_server.py
"""

import asyncio
from typing import List

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("fda-drug-information")
FDA_BASE = "https://api.fda.gov"


@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="fda_drug_search",
            description=(
                "Search FDA drug database for drug information, "
                "indications, warnings, and dosage."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Drug name (brand or generic, e.g., 'metformin', 'Lipitor')",
                    }
                },
                "required": ["drug_name"],
            },
        ),
        Tool(
            name="fda_drug_recalls",
            description="Check for recent FDA drug recalls and safety alerts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Drug name to check for recalls",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recall records (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["drug_name"],
            },
        ),
        Tool(
            name="fda_adverse_events",
            description="Search FDA adverse event reports (FAERS) for a drug.",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Drug name to check adverse events for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of adverse event records (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["drug_name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    if name == "fda_drug_search":
        return await _fda_drug_search(arguments["drug_name"])
    elif name == "fda_drug_recalls":
        return await _fda_drug_recalls(arguments["drug_name"], arguments.get("limit", 5))
    elif name == "fda_adverse_events":
        return await _fda_adverse_events(arguments["drug_name"], arguments.get("limit", 5))
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _fda_drug_search(drug_name: str) -> List[TextContent]:
    """Search FDA drug label database."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{FDA_BASE}/drug/label.json",
            params={
                "search": f"openfda.brand_name:{drug_name}+openfda.generic_name:{drug_name}",
                "limit": 1,
            },
        )
        if resp.status_code != 200:
            return [TextContent(type="text", text=f"FDA API returned status {resp.status_code}")]

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return [TextContent(type="text", text=f"No FDA label found for '{drug_name}'")]

        label = results[0]
        output = [f"Drug: {drug_name.upper()}"]

        # Extract key sections, truncating long text for context window efficiency
        sections = [
            ("Indications", "indications_and_usage"),
            ("Warnings", "warnings"),
            ("Dosage", "dosage_and_administration"),
            ("Contraindications", "contraindications"),
        ]
        for heading, key in sections:
            content = label.get(key)
            if content and isinstance(content, list) and content[0]:
                output.append(f"\n{heading}: {content[0][:400]}...")

        return [TextContent(type="text", text="\n".join(output))]


async def _fda_drug_recalls(drug_name: str, limit: int = 5) -> List[TextContent]:
    """Check for FDA drug recalls."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{FDA_BASE}/drug/enforcement.json",
            params={"search": f"product_description:{drug_name}", "limit": limit},
        )
        if resp.status_code != 200:
            return [TextContent(type="text", text="No recall data found or API error.")]

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return [TextContent(type="text", text=f"No recalls found for '{drug_name}'")]

        output = [f"FDA Recalls for {drug_name.upper()}:"]
        for r in results:
            output.append(
                f"\n- Date: {r.get('recall_initiation_date', 'N/A')}\n"
                f"  Reason: {r.get('reason_for_recall', 'N/A')[:200]}\n"
                f"  Classification: {r.get('classification', 'N/A')}"
            )
        return [TextContent(type="text", text="\n".join(output))]


async def _fda_adverse_events(drug_name: str, limit: int = 5) -> List[TextContent]:
    """Search FDA adverse event reports (FAERS)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{FDA_BASE}/drug/event.json",
            params={
                "search": f"patient.drug.medicinalproduct:{drug_name}",
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": limit,
            },
        )
        if resp.status_code != 200:
            return [TextContent(type="text", text="No adverse event data found.")]

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return [TextContent(type="text", text=f"No adverse events found for '{drug_name}'")]

        output = [f"Top Adverse Events for {drug_name.upper()} (FDA FAERS):"]
        for r in results[:limit]:
            output.append(f"  - {r.get('term', 'N/A')}: {r.get('count', 0)} reports")
        return [TextContent(type="text", text="\n".join(output))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
