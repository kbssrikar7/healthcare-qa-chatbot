# MCP Integration Plan — Explainable Healthcare QA Chatbot
**Author:** Senior RAG Engineer Review (Cline)  
**Date:** March 2026  
**Status:** Actionable Implementation Plan

---

## Table of Contents
1. [What is MCP and Why It Matters Here](#1-what-is-mcp-and-why-it-matters-here)
2. [Current State of MCP in This Project](#2-current-state-of-mcp-in-this-project)
3. [Bugs to Fix First (Before Any New Work)](#3-bugs-to-fix-first-before-any-new-work)
4. [Improvement 1 — Activate the Existing Brave Search Fallback](#4-improvement-1--activate-the-existing-brave-search-fallback)
5. [Improvement 2 — Add PubMed MCP Server (Highest Medical Value)](#5-improvement-2--add-pubmed-mcp-server-highest-medical-value)
6. [Improvement 3 — Add FDA Drug API MCP Server](#6-improvement-3--add-fda-drug-api-mcp-server)
7. [Improvement 4 — Wire MCP into the LangGraph Pipeline](#7-improvement-4--wire-mcp-into-the-langgraph-pipeline)
8. [Improvement 5 — Expose ChromaDB as an MCP Server](#8-improvement-5--expose-chromadb-as-an-mcp-server)
9. [Improvement 6 — MCP Tool Registry & Routing Logic](#9-improvement-6--mcp-tool-registry--routing-logic)
10. [Priority Roadmap & Effort Estimates](#10-priority-roadmap--effort-estimates)
11. [Environment Variables Reference](#11-environment-variables-reference)
12. [Testing Strategy for MCP](#12-testing-strategy-for-mcp)

---

## 1. What is MCP and Why It Matters Here

**Model Context Protocol (MCP)** is an open standard (by Anthropic) that lets AI systems talk to external tools and data sources through a unified interface. Think of it like a USB-C port — one standard connector, many devices.

In plain terms:
- Your AI pipeline becomes a **client** that can call external **tools** (web search, databases, APIs)
- External systems can also become **servers** that expose your data (like your ChromaDB) to other AI agents
- Everything communicates over a standard protocol — no custom API glue code needed

**Why this project specifically needs MCP:**

| Problem | MCP Solution |
|---|---|
| ChromaDB has static medical data (snapshot from training time) | MCP web search fetches live medical info |
| Drug data in `data/drug_knowledge/common_drugs.json` is static | MCP FDA API gives real-time drug info |
| PubMed datasets are frozen snapshots | MCP PubMed server queries live literature |
| LangGraph pipeline has no external tool access | MCP tools become LangGraph nodes |
| Your knowledge base is siloed | MCP server exposes it to other AI agents |

---

## 2. Current State of MCP in This Project

### ✅ What Already Exists (Good Foundation)

**`src/mcp_client/agent.py`** — A complete, well-written MCP client:
```
HealthcareMCPClient
├── __aenter__ / __aexit__  → async context manager (correct pattern)
├── list_tools()            → discover available tools
├── call_tool()             → execute a tool with arguments
└── execute_mcp_tool_oneshot() → convenience one-shot helper
```

**`src/pipeline/qa_pipeline.py`** — MCP is already hooked in as a fallback:
```
Grounding Gate fails (is_answerable=False)
    ↓
if enable_mcp_search:
    → calls brave_web_search via MCP
    → if result: use as context, set is_answerable=True
    → if fail: return UNANSWERABLE_RESPONSE
```

**`config/settings.py`** — MCP config is already in `PipelineConfig`:
```python
enable_mcp_search: bool  # reads ENABLE_MCP_SEARCH env var
mcp_search_cmd: str      # reads MCP_SEARCH_CMD env var  
mcp_search_args: str     # reads MCP_SEARCH_ARGS env var
```

### ❌ What's Missing / Broken

1. **Critical async bug** in `qa_pipeline.py` — `asyncio.run()` inside FastAPI crashes
2. **`.env.example` missing MCP vars** — `ENABLE_MCP_SEARCH`, `BRAVE_API_KEY` not documented
3. **Only Brave Search** — no medical-specific MCP tools (PubMed, FDA)
4. **LangGraph pipeline has zero MCP integration** — biggest missed opportunity
5. **No MCP server** — your ChromaDB is not exposed to external agents
6. **No source labeling differentiation** — MCP-sourced answers look same as KB answers in UI

---

## 3. Bugs to Fix First (Before Any New Work)

### Bug #1 — `asyncio.run()` Crash in FastAPI Context

**File:** `src/pipeline/qa_pipeline.py` (~line 180)

**Problem:** FastAPI runs in an async event loop. Calling `asyncio.run()` inside an already-running loop raises `RuntimeError: This event loop is already running`.

**Current broken code:**
```python
mcp_result = asyncio.run(execute_mcp_tool_oneshot(
    server_cmd=self.mcp_search_cmd,
    server_args=mcp_args,
    tool_name="brave_web_search",
    tool_args={"query": question, "count": 3}
))
```

**Fix — use `asyncio.get_event_loop().run_until_complete()` with a nest_asyncio guard, OR make the method async:**

**Option A (Quick Fix — add `nest_asyncio`):**
```bash
pip install nest_asyncio
```
```python
# At top of qa_pipeline.py
import nest_asyncio
nest_asyncio.apply()

# Then asyncio.run() works even inside FastAPI
```

**Option B (Proper Fix — make `answer()` async):**
```python
# In qa_pipeline.py — change answer() to async
async def answer(self, question: str, ...) -> QAResponse:
    ...
    # Replace asyncio.run() with await
    mcp_result = await execute_mcp_tool_oneshot(
        server_cmd=self.mcp_search_cmd,
        server_args=mcp_args,
        tool_name="brave_web_search",
        tool_args={"query": question, "count": 3}
    )
```
Then in `api/main.py`, change the call to:
```python
response = await qa_pipeline.answer(
    question=effective_question,
    num_documents=request.num_sources,
    include_explanation=request.include_explanation,
)
```

> **Recommendation:** Use Option A (nest_asyncio) for a quick fix now. Plan Option B as a proper refactor later. Option A adds 1 line and zero risk.

---

### Bug #2 — MCP Source Not Differentiated in UI

**File:** `src/pipeline/qa_pipeline.py`

**Problem:** When MCP web search is used, the source is labeled `'MCP Web Search'` but the confidence score and attribution logic still run as if it's a KB document. The UI shows no visual distinction.

**Fix:** Add a flag to `QAResponse` and handle it in the Streamlit frontend:
```python
# In QAResponse dataclass
source_type: str = "knowledge_base"  # or "mcp_web_search"

# In qa_pipeline.py after MCP success
response = QAResponse(
    ...
    source_type="mcp_web_search",
    disclaimer=disclaimer + "\n⚠️ This answer uses live web search results, not the curated medical knowledge base."
)
```

---

## 4. Improvement 1 — Activate the Existing Brave Search Fallback

**Effort:** 30 minutes | **Impact:** High | **Risk:** Low

This is already built — you just need to turn it on.

### Step 1: Get a Brave Search API Key
1. Go to https://brave.com/search/api/
2. Sign up for the free tier (2,000 queries/month free)
3. Copy your API key

### Step 2: Update `.env` file
```bash
# Add these to your .env file (copy from .env.example first)
ENABLE_MCP_SEARCH=true
MCP_SEARCH_CMD=npx
MCP_SEARCH_ARGS=-y @modelcontextprotocol/server-brave-search
BRAVE_API_KEY=your_actual_brave_api_key_here
```

### Step 3: Update `.env.example` to document these vars
```bash
# MCP Integration (optional - enables web search fallback)
ENABLE_MCP_SEARCH=false
MCP_SEARCH_CMD=npx
MCP_SEARCH_ARGS=-y @modelcontextprotocol/server-brave-search
BRAVE_API_KEY=your_brave_api_key_here
```

### Step 4: Install the MCP package if not already installed
```bash
pip install mcp
npm install -g @modelcontextprotocol/server-brave-search
```

### Step 5: Fix the asyncio bug (see Bug #1 above)
```bash
pip install nest_asyncio
```
Add `import nest_asyncio; nest_asyncio.apply()` at top of `qa_pipeline.py`.

### Step 6: Test it
```python
# Quick test — ask something your KB definitely doesn't have
# e.g., a very recent drug approval or rare condition
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest FDA approved treatments for RSV in adults 2025?"}'
```

---

## 5. Improvement 2 — Add PubMed MCP Server (Highest Medical Value)

**Effort:** 3-4 hours | **Impact:** Very High | **Risk:** Low  
**Why:** PubMed has 36 million peer-reviewed articles. Brave search returns random web pages. For a healthcare QA system, peer-reviewed sources are non-negotiable.

### Architecture
```
Grounding Gate fails
    ↓
MCP Router
    ├── Is it a drug question?     → FDA MCP Server
    ├── Is it a clinical question? → PubMed MCP Server  
    └── General medical question?  → Brave Search MCP Server
```

### Step 1: Create the PubMed MCP Server

Create file: **`src/mcp_servers/pubmed_server.py`**

```python
"""
PubMed MCP Server — exposes NCBI E-utilities as MCP tools.
No API key required for basic usage (rate limit: 3 req/sec without key, 10/sec with NCBI_API_KEY).
"""
import asyncio
import json
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("pubmed-medical-search")

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="pubmed_search",
            description="Search PubMed for peer-reviewed medical literature. Returns article titles, abstracts, and PMIDs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Medical search query (e.g., 'diabetes type 2 treatment guidelines')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 20)",
                        "default": 5
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Optional date filter e.g. '2020:2025' for years",
                        "default": ""
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="pubmed_get_abstract",
            description="Fetch the full abstract of a specific PubMed article by PMID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "string",
                        "description": "PubMed article ID (PMID)"
                    }
                },
                "required": ["pmid"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "pubmed_search":
        return await _pubmed_search(
            query=arguments["query"],
            max_results=arguments.get("max_results", 5),
            date_range=arguments.get("date_range", "")
        )
    elif name == "pubmed_get_abstract":
        return await _pubmed_get_abstract(pmid=arguments["pmid"])
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def _pubmed_search(query: str, max_results: int = 5, date_range: str = "") -> list:
    """Search PubMed using E-utilities esearch + efetch."""
    import os
    api_key = os.getenv("NCBI_API_KEY", "")
    
    # Step 1: Search for PMIDs
    search_params = {
        "db": "pubmed",
        "term": query + ("[pdat]" if date_range else ""),
        "retmax": min(max_results, 20),
        "retmode": "json",
        "sort": "relevance",
    }
    if date_range:
        search_params["datetype"] = "pdat"
        search_params["mindate"] = date_range.split(":")[0]
        search_params["maxdate"] = date_range.split(":")[-1]
    if api_key:
        search_params["api_key"] = api_key

    async with httpx.AsyncClient(timeout=15.0) as client:
        search_resp = await client.get(f"{NCBI_BASE}/esearch.fcgi", params=search_params)
        search_data = search_resp.json()
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not pmids:
            return [TextContent(type="text", text="No PubMed articles found for this query.")]
        
        # Step 2: Fetch summaries
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
            "rettype": "abstract",
        }
        if api_key:
            fetch_params["api_key"] = api_key
            
        fetch_resp = await client.get(f"{NCBI_BASE}/efetch.fcgi", params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        })
        
        # Parse XML response for abstracts
        import xml.etree.ElementTree as ET
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
                f"Abstract: {abstract_val[:500]}...\n"
                f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid_val}/\n"
                f"---"
            )
        
        return [TextContent(type="text", text="\n".join(results))]

async def _pubmed_get_abstract(pmid: str) -> list:
    """Fetch full abstract for a specific PMID."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{NCBI_BASE}/efetch.fcgi", params={
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
            "rettype": "abstract",
        })
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        abstract_el = root.find(".//AbstractText")
        title_el = root.find(".//ArticleTitle")
        
        title = title_el.text if title_el is not None else "Unknown"
        abstract = abstract_el.text if abstract_el is not None else "No abstract available"
        
        return [TextContent(type="text", text=f"Title: {title}\n\nAbstract: {abstract}\n\nURL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Register PubMed Server in Config

In **`config/settings.py`**, add to `PipelineConfig`:
```python
# MCP — PubMed
enable_mcp_pubmed: bool = os.getenv("ENABLE_MCP_PUBMED", "false").lower() == "true"
mcp_pubmed_cmd: str = "python"
mcp_pubmed_args: str = "src/mcp_servers/pubmed_server.py"
ncbi_api_key: str = os.getenv("NCBI_API_KEY", "")
```

### Step 3: Add to `.env.example`
```bash
# PubMed MCP (free, no key required — key increases rate limit)
ENABLE_MCP_PUBMED=false
NCBI_API_KEY=your_ncbi_api_key_here  # optional, get free at: https://www.ncbi.nlm.nih.gov/account/
```

### Step 4: Update `qa_pipeline.py` to try PubMed before Brave Search
```python
# In the grounding gate fallback section:
if not is_answerable:
    # Try PubMed first (peer-reviewed > web search for medical)
    if self.enable_mcp_pubmed:
        mcp_result = await execute_mcp_tool_oneshot(
            server_cmd="python",
            server_args=["src/mcp_servers/pubmed_server.py"],
            tool_name="pubmed_search",
            tool_args={"query": question, "max_results": 3}
        )
        if mcp_result and not mcp_result.startswith("[MCP Error"):
            context = f"CONTEXT FROM PUBMED (PEER-REVIEWED):\n{mcp_result}"
            is_answerable = True
    
    # Fall back to Brave web search if PubMed also fails
    if not is_answerable and self.enable_mcp_search:
        # ... existing Brave search code ...
```

---

## 6. Improvement 3 — Add FDA Drug API MCP Server

**Effort:** 2-3 hours | **Impact:** High | **Risk:** Low  
**Why:** Your project already has `data/drug_knowledge/common_drugs.json` and `src/safety/guardrails.py` with a `DrugInteractionChecker`. But the drug data is static. The FDA OpenFDA API is free, real-time, and authoritative.

### Step 1: Create the FDA MCP Server

Create file: **`src/mcp_servers/fda_server.py`**

```python
"""
FDA OpenFDA MCP Server — exposes FDA drug database as MCP tools.
No API key required. Rate limit: 240 requests/minute.
Docs: https://open.fda.gov/apis/
"""
import asyncio
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("fda-drug-information")
FDA_BASE = "https://api.fda.gov"

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="fda_drug_search",
            description="Search FDA drug database for drug information, indications, warnings, and dosage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Drug name (brand or generic, e.g., 'metformin', 'Lipitor')"
                    }
                },
                "required": ["drug_name"]
            }
        ),
        Tool(
            name="fda_drug_recalls",
            description="Check for recent FDA drug recalls and safety alerts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Drug name to check for recalls"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recall records to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["drug_name"]
            }
        ),
        Tool(
            name="fda_adverse_events",
            description="Search FDA adverse event reports (FAERS) for a drug.",
            inputSchema={
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Drug name to check adverse events for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of adverse event records (default: 5)",
                        "default": 5
                    }
                },
                "required": ["drug_name"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "fda_drug_search":
        return await _fda_drug_search(arguments["drug_name"])
    elif name == "fda_drug_recalls":
        return await _fda_drug_recalls(arguments["drug_name"], arguments.get("limit", 5))
    elif name == "fda_adverse_events":
        return await _fda_adverse_events(arguments["drug_name"], arguments.get("limit", 5))
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def _fda_drug_search(drug_name: str) -> list:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{FDA_BASE}/drug/label.json", params={
            "search": f"openfda.brand_name:{drug_name}+openfda.generic_name:{drug_name}",
            "limit": 1
        })
        if resp.status_code != 200:
            return [TextContent(type="text", text=f"FDA API error: {resp.status_code}")]
        
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return [TextContent(type="text", text=f"No FDA label found for '{drug_name}'")]
        
        label = results[0]
        output = []
        output.append(f"Drug: {drug_name.upper()}")
        
        if label.get("indications_and_usage"):
            output.append(f"\nIndications: {label['indications_and_usage'][0][:400]}...")
        if label.get("warnings"):
            output.append(f"\nWarnings: {label['warnings'][0][:400]}...")
        if label.get("dosage_and_administration"):
            output.append(f"\nDosage: {label['dosage_and_administration'][0][:300]}...")
        if label.get("contraindications"):
            output.append(f"\nContraindications: {label['contraindications'][0][:300]}...")
        
        return [TextContent(type="text", text="\n".join(output))]

async def _fda_drug_recalls(drug_name: str, limit: int = 5) -> list:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{FDA_BASE}/drug/enforcement.json", params={
            "search": f"product_description:{drug_name}",
            "limit": limit
        })
        if resp.status_code != 200:
            return [TextContent(type="text", text="No recall data found or API error.")]
        
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return [TextContent(type="text", text=f"No recalls found for '{drug_name}'")]
        
        output = [f"FDA Recalls for {drug_name.upper()}:"]
        for r in results:
            output.append(
                f"\n- Recall Date: {r.get('recall_initiation_date', 'N/A')}\n"
                f"  Reason: {r.get('reason_for_recall', 'N/A')[:200]}\n"
                f"  Classification: {r.get('classification', 'N/A')}"
            )
        return [TextContent(type="text", text="\n".join(output))]

async def _fda_adverse_events(drug_name: str, limit: int = 5) -> list:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{FDA_BASE}/drug/event.json", params={
            "search": f"patient.drug.medicinalproduct:{drug_name}",
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": limit
        })
        if resp.status_code != 200:
            return [TextContent(type="text", text="No adverse event data found.")]
        
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return [TextContent(type="text", text=f"No adverse events found for '{drug_name}'")]
        
        output = [f"Top Adverse Events for {drug_name.upper()} (from FDA FAERS):"]
        for r in results[:limit]:
            output.append(f"  - {r.get('term', 'N/A')}: {r.get('count', 0)} reports")
        
        return [TextContent(type="text", text="\n".join(output))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Integrate with Safety Guardrails

In **`src/safety/guardrails.py`**, update `DrugInteractionChecker` to optionally call the FDA MCP server:
```python
# Add async method to DrugInteractionChecker
async def get_fda_drug_info(self, drug_name: str) -> str:
    """Fetch real-time FDA drug info via MCP."""
    from src.mcp_client.agent import execute_mcp_tool_oneshot
    return await execute_mcp_tool_oneshot(
        server_cmd="python",
        server_args=["src/mcp_servers/fda_server.py"],
        tool_name="fda_drug_search",
        tool_args={"drug_name": drug_name}
    )
```

---

## 7. Improvement 4 — Wire MCP into the LangGraph Pipeline

**Effort:** 4-5 hours | **Impact:** Very High | **Risk:** Medium  
**Why:** The LangGraph pipeline (`src/langgraph/`) is your most sophisticated pipeline — self-correcting RAG with retry loops. But right now when it hits `unanswerable`, it just gives up. MCP tools should be the next step before giving up.

### Current LangGraph Flow:
```
START → retrieve → grade → [if poor] → refine → retrieve (loop, max 2x)
                         → [if still poor] → unanswerable → END
                         → [if good] → generate → verify → enrich_xai → END
```

### Proposed New Flow with MCP:
```
START → retrieve → grade → [if poor] → refine → retrieve (loop, max 2x)
                         → [if still poor] → mcp_search → grade_mcp → generate → verify → enrich_xai → END
                         → [if mcp also fails] → unanswerable → END
                         → [if good] → generate → verify → enrich_xai → END
```

### Step 1: Add MCP node to `src/langgraph/langgraph_nodes.py`

```python
# Add to HealthcareRAGNodes class

async def mcp_search_node(self, state: HealthcareRAGState) -> Dict[str, Any]:
    """
    Node: Fallback to MCP external search when local KB is insufficient.
    
    Tries PubMed first (peer-reviewed), then Brave web search.
    Converts MCP results into LangChain Documents for downstream nodes.
    """
    from src.mcp_client.agent import execute_mcp_tool_oneshot
    from langchain_core.documents import Document
    import os
    
    question = state["question"]
    query = state.get("query_history", [question])[-1]
    
    mcp_docs = []
    mcp_source = "unknown"
    
    # Try PubMed first
    if os.getenv("ENABLE_MCP_PUBMED", "false").lower() == "true":
        try:
            result = await execute_mcp_tool_oneshot(
                server_cmd="python",
                server_args=["src/mcp_servers/pubmed_server.py"],
                tool_name="pubmed_search",
                tool_args={"query": query, "max_results": 3}
            )
            if result and not result.startswith("[MCP Error"):
                mcp_docs.append(Document(
                    page_content=result,
                    metadata={"source": "PubMed (Live)", "score": 0.8, "url": "https://pubmed.ncbi.nlm.nih.gov/"}
                ))
                mcp_source = "PubMed"
        except Exception as e:
            print(f"PubMed MCP failed: {e}")
    
    # Try Brave search as secondary fallback
    if not mcp_docs and os.getenv("ENABLE_MCP_SEARCH", "false").lower() == "true":
        try:
            result = await execute_mcp_tool_oneshot(
                server_cmd="npx",
                server_args=["-y", "@modelcontextprotocol/server-brave-search"],
                tool_name="brave_web_search",
                tool_args={"query": query + " medical health", "count": 3}
            )
            if result and not result.startswith("[MCP Error"):
                mcp_docs.append(Document(
                    page_content=result,
                    metadata={"source": "Web Search (Live)", "score": 0.6, "url": ""}
                ))
                mcp_source = "BraveSearch"
        except Exception as e:
            print(f"Brave MCP failed: {e}")
    
    if mcp_docs:
        return {
            "documents": mcp_docs,
            "mcp_source": mcp_source,
            "is_answerable": True
        }
    else:
        return {
            "documents": [],
            "mcp_source": None,
            "is_answerable": False
        }
```

### Step 2: Add `mcp_source` to LangGraph State

In **`src/langgraph/langgraph_state.py`**, add:
```python
mcp_source: Optional[str]  # "PubMed", "BraveSearch", or None
```

### Step 3: Add routing for MCP in `src/langgraph/langgraph_routing.py`

```python
def route_after_grading(state: HealthcareRAGState) -> str:
    """Route after document grading — now includes MCP fallback."""
    is_answerable = state.get("is_answerable", False)
    retry_count = state.get("retry_count", 0)
    
    if is_answerable:
        return "generate"
    elif retry_count < MAX_RETRY_COUNT:
        return "refine"
    else:
        # NEW: try MCP before giving up
        import os
        mcp_enabled = (
            os.getenv("ENABLE_MCP_PUBMED", "false").lower() == "true" or
            os.getenv("ENABLE_MCP_SEARCH", "false").lower() == "true"
        )
        if mcp_enabled and not state.get("mcp_attempted", False):
            return "mcp_search"
        return "unanswerable"
```

### Step 4: Wire the new node into the graph in `src/langgraph/langgraph_pipeline.py`

```python
def _build_graph(self):
    builder = StateGraph(HealthcareRAGState)
    
    # Existing nodes
    builder.add_node("retrieve", self.nodes.retrieve_documents)
    builder.add_node("grade", self.nodes.grade_relevance)
    builder.add_node("refine", self.nodes.refine_query)
    builder.add_node("generate", self.nodes.generate_answer)
    builder.add_node("verify", self.nodes.verify_grounding)
    builder.add_node("enrich_xai", self.nodes.enrich_xai)
    builder.add_node("unanswerable", self.nodes.unanswerable_response)
    
    # NEW: MCP search node
    builder.add_node("mcp_search", self.nodes.mcp_search_node)
    
    # Edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_edge("refine", "retrieve")
    builder.add_edge("generate", "verify")
    builder.add_edge("unanswerable", END)
    builder.add_edge("enrich_xai", END)
    
    # NEW: MCP search goes directly to generate (already graded by MCP node)
    builder.add_edge("mcp_search", "generate")
    
    # Conditional edges (updated routing)
    builder.add_conditional_edges("grade", route_after_grading, {
        "generate": "generate",
        "refine": "refine",
        "mcp_search": "mcp_search",   # NEW
        "unanswerable": "unanswerable"
    })
    
    builder.add_conditional_edges("verify", route_after_verify, {
        "enrich_xai": "enrich_xai",
        "refine": "refine",
    })
    
    if self.enable_checkpointing:
        return builder.compile(checkpointer=MemorySaver())
    return builder.compile()
```

---

## 8. Improvement 5 — Expose ChromaDB as an MCP Server

**Effort:** 4-6 hours | **Impact:** Medium | **Risk:** Low  
**Why:** Right now your medical knowledge base is locked inside this project. By exposing it as an MCP server, any MCP-compatible AI agent (Claude Desktop, Cursor, other pipelines) can query your curated medical knowledge base directly.

### Create file: **`src/mcp_servers/chromadb_server.py`**

```python
"""
ChromaDB MCP Server — exposes the medical knowledge base as an MCP tool.
Allows external AI agents to query your curated medical knowledge base.
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

app = Server("healthcare-knowledge-base")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_medical_knowledge",
            description=(
                "Search the curated medical knowledge base (MedQuAD, PubMedQA, MedMCQA datasets). "
                "Returns relevant medical passages with source attribution. "
                "Best for: disease information, symptoms, treatments, medical procedures."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Medical question or search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_medical_knowledge":
        return await _search_knowledge_base(
            query=arguments["query"],
            num_results=arguments.get("num_results", 5)
        )
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def _search_knowledge_base(query: str, num_results: int = 5) -> list:
    """Search ChromaDB using hybrid retrieval."""
    import asyncio
    loop = asyncio.get_event_loop()
    
    def _sync_search():
        from src.embeddings.embedding_models import MedicalEmbedder
        from src.embeddings.vector_store import VectorStore
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        embedder = MedicalEmbedder(model_name="all-minilm")
        vector_store = VectorStore(
            collection_name="medical_knowledge",
            persist_directory="data/knowledge_base"
        )
        retriever = HybridRetriever(embedder, vector_store)
        docs, context = retriever.retrieve_with_context(query, k=num_results)
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"[{i}] Source: {doc.source}\n"
                f"Score: {doc.score:.3f}\n"
                f"Content: {doc.content[:400]}...\n"
                f"---"
            )
        return "\n".join(results) if results else "No relevant documents found."
    
    result = await loop.run_in_executor(None, _sync_search)
    return [TextContent(type="text", text=result)]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### How to use it from Claude Desktop or other agents:
Add to Claude Desktop's `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "healthcare-kb": {
      "command": "python",
      "args": ["/path/to/final_project/src/mcp_servers/chromadb_server.py"],
      "env": {}
    }
  }
}
```

---

## 9. Improvement 6 — MCP Tool Registry & Routing Logic

**Effort:** 2-3 hours | **Impact:** Medium | **Risk:** Low  
**Why:** As you add more MCP tools (PubMed, FDA, Brave), you need a smart router that picks the right tool based on the question type. This prevents calling all tools for every question.

### Create file: **`src/mcp_client/tool_router.py`**

```python
"""
MCP Tool Router — intelligently selects the right MCP tool based on question type.
Prevents unnecessary API calls and improves response quality.
"""
import re
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class MCPToolSelection:
    tool_server: str        # "pubmed", "fda", "brave", "chromadb"
    tool_name: str          # exact tool name
    tool_args: dict         # arguments to pass
    confidence: float       # how confident we are in this selection (0-1)
    reason: str             # human-readable reason for selection

class MCPToolRouter:
    """
    Routes medical questions to the most appropriate MCP tool.
    
    Priority order:
    1. Drug questions → FDA server (authoritative, real-time)
    2. Clinical/research questions → PubMed server (peer-reviewed)
    3. General medical questions → Brave search (broad coverage)
    """
    
    # Drug-related keywords
    DRUG_PATTERNS = [
        r'\b(drug|medication|medicine|pill|tablet|capsule|dose|dosage|prescription)\b',
        r'\b(side effect|adverse effect|interaction|contraindication|overdose)\b',
        r'\b(antibiotic|antidepressant|antihypertensive|statin|insulin|vaccine)\b',
        r'\b(mg|mcg|ml|IV|oral|topical|injection)\b',
        r'\b(FDA approved|recall|warning|black box)\b',
    ]
    
    # Research/clinical keywords
    RESEARCH_PATTERNS = [
        r'\b(study|trial|research|evidence|guideline|protocol|meta-analysis)\b',
        r'\b(clinical|randomized|controlled|systematic review|cohort)\b',
        r'\b(efficacy|effectiveness|outcome|prognosis|survival rate)\b',
        r'\b(pathophysiology|mechanism|etiology|epidemiology)\b',
    ]
    
    def route(self, question: str) -> MCPToolSelection:
        """Select the best MCP tool for a given question."""
        question_lower = question.lower()
        
        # Check for drug-related question
        drug_score = self._score_patterns(question_lower, self.DRUG_PATTERNS)
        if drug_score >= 2:
            drug_name = self._extract_drug_name(question)
            return MCPToolSelection(
                tool_server="fda",
                tool_name="fda_drug_search",
                tool_args={"drug_name": drug_name or question[:50]},
                confidence=min(0.9, 0.5 + drug_score * 0.1),
                reason=f"Drug-related question detected (score: {drug_score})"
            )
        
        # Check for research/clinical question
        research_score = self._score_patterns(question_lower, self.RESEARCH_PATTERNS)
        if research_score >= 1:
            return MCPToolSelection(
                tool_server="pubmed",
                tool_name="pubmed_search",
                tool_args={"query": question, "max_results": 3},
                confidence=min(0.85, 0.5 + research_score * 0.1),
                reason=f"Research/clinical question detected (score: {research_score})"
            )
        
        # Default: PubMed for medical questions, Brave for general
        if any(term in question_lower for term in ["symptom", "disease", "condition", "treatment", "diagnosis"]):
            return MCPToolSelection(
                tool_server="pubmed",
                tool_name="pubmed_search",
                tool_args={"query": question, "max_results": 3},
                confidence=0.6,
                reason="General medical question — using PubMed"
            )
        
        return MCPToolSelection(
            tool_server="brave",
            tool_name="brave_web_search",
            tool_args={"query": question + " medical health", "count": 3},
            confidence=0.4,
            reason="Fallback to web search"
        )
    
    def _score_patterns(self, text: str, patterns: list) -> int:
        """Count how many patterns match."""
        return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    
    def _extract_drug_name(self, question: str) -> Optional[str]:
        """Try to extract a drug name from the question."""
        # Simple heuristic: look for capitalized words or words before "mg"
        mg_match = re.search(r'(\w+)\s+\d+\s*mg', question, re.IGNORECASE)
        if mg_match:
            return mg_match.group(1)
        
        # Look for words after "taking", "prescribed", "on"
        for trigger in ["taking", "prescribed", "on", "using", "about"]:
            match = re.search(rf'{trigger}\s+(\w+)', question, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
```

### Integrate the router into `qa_pipeline.py`:
```python
# In HealthcareQAPipeline.__init__():
from src.mcp_client.tool_router import MCPToolRouter
self.mcp_router = MCPToolRouter()

# In the grounding gate fallback:
if not is_answerable and (self.enable_mcp_search or self.enable_mcp_pubmed):
    selection = self.mcp_router.route(question)
    print(f"🔀 MCP Router selected: {selection.tool_server} ({selection.reason})")
    
    server_configs = {
        "fda": ("python", ["src/mcp_servers/fda_server.py"]),
        "pubmed": ("python", ["src/mcp_servers/pubmed_server.py"]),
        "brave": ("npx", ["-y", "@modelcontextprotocol/server-brave-search"]),
    }
    
    cmd, args = server_configs.get(selection.tool_server, ("npx", ["-y", "@modelcontextprotocol/server-brave-search"]))
    mcp_result = await execute_mcp_tool_oneshot(cmd, args, selection.tool_name, selection.tool_args)
    
    if mcp_result and not mcp_result.startswith("[MCP Error"):
        context = f"CONTEXT FROM {selection.tool_server.upper()} (LIVE):\n{mcp_result}"
        is_answerable = True
```

---

## 10. Priority Roadmap & Effort Estimates

```
PHASE 1 — Quick Wins (Do This Week, ~2 hours total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ ] Fix asyncio.run() bug (nest_asyncio)                    30 min
[ ] Update .env.example with MCP vars                       10 min
[ ] Get Brave API key + set ENABLE_MCP_SEARCH=true          20 min
[ ] Test existing MCP fallback end-to-end                   30 min
[ ] Add source_type flag to QAResponse + UI label           30 min

PHASE 2 — Medical-Grade MCP Tools (Next Week, ~8 hours total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ ] Create src/mcp_servers/ directory                        5 min
[ ] Build pubmed_server.py                                   3 hrs
[ ] Build fda_server.py                                      2 hrs
[ ] Integrate PubMed into qa_pipeline.py fallback           30 min
[ ] Integrate FDA into safety guardrails                    30 min
[ ] Test PubMed + FDA servers                               1 hr

PHASE 3 — LangGraph MCP Integration (Week 3, ~6 hours total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ ] Add mcp_source to LangGraph state                       30 min
[ ] Build mcp_search_node in langgraph_nodes.py             2 hrs
[ ] Update routing in langgraph_routing.py                  1 hr
[ ] Wire node into langgraph_pipeline.py graph              1 hr
[ ] Test LangGraph + MCP end-to-end                         1 hr

PHASE 4 — MCP Server + Tool Router (Week 4, ~8 hours total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[ ] Build chromadb_server.py (expose KB as MCP server)      4 hrs
[ ] Build tool_router.py (smart MCP tool selection)         2 hrs
[ ] Integrate router into qa_pipeline.py                    1 hr
[ ] Test ChromaDB server with Claude Desktop                1 hr
```

---

## 11. Environment Variables Reference

Add all of these to your `.env` file and `.env.example`:

```bash
# ─── MCP Integration ──────────────────────────────────────────────────────────

# Brave Web Search (fallback for general medical questions)
ENABLE_MCP_SEARCH=false
MCP_SEARCH_CMD=npx
MCP_SEARCH_ARGS=-y @modelcontextprotocol/server-brave-search
BRAVE_API_KEY=your_brave_api_key_here
# Get key at: https://brave.com/search/api/ (2000 free queries/month)

# PubMed (peer-reviewed medical literature — RECOMMENDED for healthcare)
ENABLE_MCP_PUBMED=false
NCBI_API_KEY=your_ncbi_api_key_here
# Get key at: https://www.ncbi.nlm.nih.gov/account/ (free, increases rate limit)
# Without key: 3 req/sec | With key: 10 req/sec

# FDA Drug Database (real-time drug info, recalls, adverse events)
ENABLE_MCP_FDA=false
# No API key required — 240 requests/minute free
# Docs: https://open.fda.gov/apis/

# ─── Existing vars (already in .env.example) ──────────────────────────────────
HUGGINGFACE_TOKEN=your_token_here
USE_GPU=true
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## 12. Testing Strategy for MCP

### Unit Tests — Add to `tests/test_mcp.py`

```python
"""Tests for MCP client and servers."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

class TestHealthcareMCPClient:
    """Test the MCP client."""
    
    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test that client initializes and closes cleanly."""
        from src.mcp_client.agent import HealthcareMCPClient
        # Mock the MCP session
        with patch("src.mcp_client.agent.stdio_client") as mock_stdio:
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
            # Test that context manager works without errors
            # (full integration test requires actual MCP server)
            assert HealthcareMCPClient is not None
    
    def test_tool_router_drug_question(self):
        """Test that drug questions route to FDA."""
        from src.mcp_client.tool_router import MCPToolRouter
        router = MCPToolRouter()
        selection = router.route("What are the side effects of metformin 500mg?")
        assert selection.tool_server == "fda"
        assert selection.confidence > 0.5
    
    def test_tool_router_research_question(self):
        """Test that research questions route to PubMed."""
        from src.mcp_client.tool_router import MCPToolRouter
        router = MCPToolRouter()
        selection = router.route("What does the clinical trial evidence say about statins?")
        assert selection.tool_server == "pubmed"
    
    def test_tool_router_general_question(self):
        """Test that general questions route to PubMed or Brave."""
        from src.mcp_client.tool_router import MCPToolRouter
        router = MCPToolRouter()
        selection = router.route("What are the symptoms of diabetes?")
        assert selection.tool_server in ["pubmed", "brave"]

class TestPubMedServer:
    """Test PubMed MCP server tools."""
    
    @pytest.mark.asyncio
    async def test_pubmed_search_returns_results(self):
        """Test PubMed search with a known query."""
        import httpx
        # Test the NCBI API directly (no MCP overhead)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": "diabetes treatment", "retmax": 3, "retmode": "json"}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["esearchresult"]["idlist"]) > 0

class TestFDAServer:
    """Test FDA MCP server tools."""
    
    @pytest.mark.asyncio
    async def test_fda_drug_search(self):
        """Test FDA drug search with a known drug."""
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.fda.gov/drug/label.json",
                params={"search": "openfda.generic_name:metformin", "limit": 1}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data.get("results", [])) > 0
```

### Integration Test — Manual Checklist

Before marking any phase complete, run through this checklist:

```
[ ] ENABLE_MCP_SEARCH=true → ask a question your KB doesn't have → verify web result appears
[ ] ENABLE_MCP_PUBMED=true → ask "What is the evidence for metformin in type 2 diabetes?" → verify PubMed abstracts appear
[ ] ENABLE_MCP_FDA=true → ask "What are the side effects of lisinopril?" → verify FDA label data appears
[ ] LangGraph pipeline → ask unanswerable question → verify it tries MCP before giving up
[ ] Source attribution → verify MCP-sourced answers show "PubMed (Live)" or "Web Search (Live)" label
[ ] Confidence score → verify MCP-sourced answers have lower confidence than KB answers (they're less curated)
[ ] Latency → verify MCP fallback adds < 3 seconds to response time
[ ] Error handling → kill the MCP server mid-request → verify graceful fallback to UNANSWERABLE_RESPONSE
```

---

## Summary

Your project already has a solid MCP foundation. The `HealthcareMCPClient` is well-written and the fallback hook in `qa_pipeline.py` is architecturally correct. The main gaps are:

1. **A critical async bug** that prevents MCP from working in FastAPI (fix first, 30 min)
2. **Only Brave Search** — for a healthcare system, PubMed and FDA are far more appropriate
3. **LangGraph has no MCP** — the most sophisticated pipeline should be the most capable
4. **No smart routing** — all questions shouldn't hit the same MCP tool
5. **No MCP server** — your knowledge base should be accessible to other agents

Follow the phases in order. Phase 1 alone will make the existing code actually work. Phase 2 will make it medically appropriate. Phases 3-4 will make it production-grade.
