"""
PubMed MCP Server — exposes NCBI E-utilities as MCP tools.

Provides live access to 36M+ peer-reviewed medical articles via PubMed.
No API key required for basic usage (3 req/sec). With NCBI_API_KEY: 10 req/sec.

Usage:
    python src/mcp_servers/pubmed_server.py
"""

import asyncio
import os
import xml.etree.ElementTree as ET
from typing import List

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("pubmed-medical-search")

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="pubmed_search",
            description=(
                "Search PubMed for peer-reviewed medical literature. "
                "Returns article titles, abstracts, and PMIDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results (default: 5, max: 20)",
                        "default": 5,
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Year range e.g. '2020:2025'",
                        "default": "",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="pubmed_get_abstract",
            description="Fetch the full abstract of a PubMed article by PMID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pmid": {"type": "string", "description": "PubMed article ID (PMID)"}
                },
                "required": ["pmid"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    if name == "pubmed_search":
        return await _pubmed_search(
            query=arguments["query"],
            max_results=arguments.get("max_results", 5),
            date_range=arguments.get("date_range", ""),
        )
    elif name == "pubmed_get_abstract":
        return await _pubmed_get_abstract(pmid=arguments["pmid"])
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _pubmed_search(
    query: str, max_results: int = 5, date_range: str = ""
) -> List[TextContent]:
    """Search PubMed using E-utilities esearch + efetch."""
    api_key = os.getenv("NCBI_API_KEY", "")

    # Step 1: Search for PMIDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": min(max_results, 20),
        "retmode": "json",
        "sort": "relevance",
    }
    if date_range and ":" in date_range:
        parts = date_range.split(":")
        search_params["datetype"] = "pdat"
        search_params["mindate"] = parts[0]
        search_params["maxdate"] = parts[-1]
    if api_key:
        search_params["api_key"] = api_key

    async with httpx.AsyncClient(timeout=15.0) as client:
        search_resp = await client.get(f"{NCBI_BASE}/esearch.fcgi", params=search_params)
        search_data = search_resp.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return [TextContent(type="text", text="No PubMed articles found for this query.")]

        # Step 2: Fetch article details via XML
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if api_key:
            fetch_params["api_key"] = api_key

        fetch_resp = await client.get(f"{NCBI_BASE}/efetch.fcgi", params=fetch_params)

        root = ET.fromstring(fetch_resp.text)

        results = []
        for article in root.findall(".//PubmedArticle")[:max_results]:
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//AbstractText")
            year_el = article.find(".//PubDate/Year")
            journal_el = article.find(".//Journal/Title")

            pmid_val = pmid_el.text if pmid_el is not None else "N/A"
            title_val = title_el.text if title_el is not None else "No title"
            abstract_val = abstract_el.text if abstract_el is not None else "No abstract available"
            year_val = year_el.text if year_el is not None else "N/A"
            journal_val = journal_el.text if journal_el is not None else "N/A"

            results.append(
                f"PMID: {pmid_val}\n"
                f"Title: {title_val}\n"
                f"Journal: {journal_val} ({year_val})\n"
                f"Abstract: {abstract_val[:500]}{'...' if len(abstract_val) > 500 else ''}\n"
                f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid_val}/\n"
                f"---"
            )

        return [TextContent(type="text", text="\n".join(results))]


async def _pubmed_get_abstract(pmid: str) -> List[TextContent]:
    """Fetch full abstract for a specific PMID."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
                "rettype": "abstract",
            },
        )
        root = ET.fromstring(resp.text)
        title_el = root.find(".//ArticleTitle")
        abstract_el = root.find(".//AbstractText")

        title = title_el.text if title_el is not None else "Unknown"
        abstract = abstract_el.text if abstract_el is not None else "No abstract available"

        return [
            TextContent(
                type="text",
                text=f"Title: {title}\n\nAbstract: {abstract}\n\nURL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
        ]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
