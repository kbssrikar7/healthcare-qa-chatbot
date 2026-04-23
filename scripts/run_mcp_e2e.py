import json
import os
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


ROOT = Path("/home/kbs/Documents/final_project")
PYTHON = ROOT / "venv" / "bin" / "python"
ARTIFACTS = ROOT / "tests" / "artifacts" / "mcp"
HISTORY_PATH = ROOT / "data" / "question_history.json"


SERVER_CONFIGS = {
    "playwright": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest", "--headless"],
    },
    "selenium": {
        "command": "npx",
        "args": ["-y", "@angiejones/mcp-selenium@latest"],
    },
}


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_http(url: str, timeout: float = 45.0) -> None:
    import urllib.request

    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


class MockApiHandler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, fmt, *args):
        return

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if not content_length:
            return {}
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def _send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {})

    def do_GET(self):
        if self.path == "/models":
            self.__class__.requests.append({"method": "GET", "url": "/models"})
            self._send_json(
                200,
                {
                    "tinyllama": {
                        "display_name": "TinyLlama 1.1B",
                        "description": "Fast local test model",
                        "parameters": "1.1B",
                        "requires_gpu": False,
                        "loaded": True,
                    }
                },
            )
            return
        self._send_json(404, {"detail": "Not found"})

    def do_POST(self):
        body = self._read_json()
        self.__class__.requests.append({"method": "POST", "url": self.path, "body": body})

        if self.path == "/ask":
            question = body["question"].lower()
            if "trigger unavailable" in question:
                self._send_json(503, {"detail": "System is initializing"})
                return

            pipeline = (
                "LangGraph"
                if body.get("use_langgraph")
                else "LangChain"
                if body.get("use_langchain")
                else "Standard"
            )
            is_emergency = "chest pain" in question
            include_hallucination = "hallucination" in question
            self._send_json(
                200,
                {
                    "response_id": f"resp-{len([r for r in self.__class__.requests if r['url'] == '/ask'])}",
                    "question": body["question"],
                    "answer": (
                        "Call emergency services immediately for severe chest pain."
                        if is_emergency
                        else f"Mocked clinical answer for: {body['question']}"
                    ),
                    "sources": [
                        {
                            "source": "Mock Medical Reference",
                            "content": (
                                "Type 2 diabetes can cause increased thirst, frequent urination, and fatigue."
                            ),
                            "score": 0.91,
                            "url": "https://example.test/source",
                        }
                    ],
                    "confidence": {
                        "score": 0.86,
                        "level": "high",
                        "explanation": "Mocked high-confidence result from retrieved evidence.",
                    },
                    "attributions": [
                        {
                            "claim": "Symptoms can include increased thirst.",
                            "source": "Mock Medical Reference",
                            "evidence": "Mock evidence",
                            "similarity": 0.88,
                        }
                    ],
                    "disclaimer": "This information is for educational purposes only.",
                    "rationale": "The answer is based on the mocked retrieved source.",
                    "model_used": "TinyLlama 1.1B",
                    "pipeline_used": pipeline,
                    "session_id": "mcp-test-session",
                    "safety": (
                        {
                            "level": "emergency",
                            "flags": ["emergency_cardiac"],
                            "is_emergency": True,
                            "emergency_message": "Call emergency services immediately.",
                            "drug_warnings": [
                                "Avoid combining warfarin and aspirin without clinical advice."
                            ],
                        }
                        if is_emergency
                        else {"level": "safe", "flags": [], "is_emergency": False}
                    ),
                    "latency_ms": 42,
                    "confidence_breakdown": {
                        "retrieval_confidence": 0.87,
                        "generation_confidence": 0.82,
                        "consistency_score": 0.8,
                        "source_agreement": 0.78,
                        "medical_entity_coverage": 0.72,
                        "signal_weights": {
                            "retrieval": 0.3,
                            "generation": 0.25,
                            "consistency": 0.2,
                            "source_agreement": 0.15,
                            "entity_coverage": 0.1,
                        },
                    },
                    "hallucination": (
                        {
                            "score": 0.61,
                            "has_hallucination": True,
                            "type": "unsupported_claim",
                            "explanation": "Mock hallucination risk for UI coverage.",
                            "medical_accuracy_flags": ["unsupported dosage claim"],
                        }
                        if include_hallucination
                        else None
                    ),
                },
            )
            return

        if self.path == "/feedback":
            self._send_json(
                200,
                {
                    "status": "recorded",
                    "feedback_id": "feedback-1",
                    "response_id": body["response_id"],
                    "reward_signal": 1,
                    "trajectory_found": True,
                },
            )
            return

        if self.path == "/clear-cache":
            self._send_json(200, {"status": "cleared"})
            return

        self._send_json(404, {"detail": "Not found"})


class AppStack:
    def __init__(self):
        self.api_server = None
        self.api_thread = None
        self.streamlit = None
        self.api_port = None
        self.streamlit_port = None
        self.history_snapshot = None

    def start(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.history_snapshot = HISTORY_PATH.read_text() if HISTORY_PATH.exists() else None
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text("[]\n")

        MockApiHandler.requests = []
        self.api_port = get_free_port()
        self.api_server = ThreadingHTTPServer(("127.0.0.1", self.api_port), MockApiHandler)
        self.api_thread = threading.Thread(target=self.api_server.serve_forever, daemon=True)
        self.api_thread.start()

        self.streamlit_port = get_free_port()
        env = {
            **os.environ,
            "API_URL": f"http://127.0.0.1:{self.api_port}",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            "PYTHONUNBUFFERED": "1",
        }
        self.streamlit = subprocess.Popen(
            [
                str(PYTHON),
                "-m",
                "streamlit",
                "run",
                "frontend/streamlit_app.py",
                "--server.address=127.0.0.1",
                f"--server.port={self.streamlit_port}",
                "--server.headless=true",
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for_http(self.base_url)

    def stop(self):
        if self.streamlit:
            self.streamlit.terminate()
            try:
                self.streamlit.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.streamlit.kill()
        if self.api_server:
            self.api_server.shutdown()
            self.api_server.server_close()

        if self.history_snapshot is None:
            HISTORY_PATH.unlink(missing_ok=True)
        else:
            HISTORY_PATH.write_text(self.history_snapshot)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.streamlit_port}"

    @property
    def requests(self):
        return MockApiHandler.requests


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def log(message: str):
    print(message, flush=True)


def ask_requests(requests):
    return [r for r in requests if r["method"] == "POST" and r["url"] == "/ask"]


def text_blocks(result) -> str:
    parts = []
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


async def call_tool(session: ClientSession, name: str, arguments: dict | None = None):
    result = await session.call_tool(name, arguments or {})
    return result


@asynccontextmanager
async def mcp_session(server_name: str):
    config = SERVER_CONFIGS[server_name]
    params = StdioServerParameters(
        command=config["command"],
        args=config["args"],
        cwd=str(ROOT),
        env={**os.environ, "CI": "1"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def run_playwright_flow(base_url: str, requests: list[dict]):
    log("playwright: connect")
    async with mcp_session("playwright") as session:
        log("playwright: navigate")
        await call_tool(session, "browser_navigate", {"url": base_url})
        await call_tool(session, "browser_wait_for", {"text": "MediQuery AI"})
        await call_tool(session, "browser_wait_for", {"text": "Symptoms of Type 2 Diabetes"})

        log("playwright: suggested question")
        await call_tool(
            session,
            "browser_run_code",
            {
                "code": """
                async (page) => {
                  await page.getByRole('button', { name: 'Symptoms of Type 2 Diabetes' }).click();
                  return 'clicked question';
                }
                """
            },
        )
        await call_tool(
            session, "browser_wait_for", {"text": "Mocked clinical answer for: What are the symptoms of Type 2 Diabetes?"}
        )
        require(
            ask_requests(requests)[0]["body"]["question"] == "What are the symptoms of Type 2 Diabetes?",
            "Playwright MCP did not submit the suggested question",
        )

        log("playwright: langgraph")
        await call_tool(
            session,
            "browser_run_code",
            {
                "code": """
                async (page) => {
                  await page.getByText('LangGraph (Self-Correcting)', { exact: true }).click();
                  await page.getByPlaceholder('Ask a medical question…').fill('What causes acute migraine?');
                  await page.keyboard.press('Enter');
                  return 'submitted langgraph question';
                }
                """
            },
        )
        await call_tool(session, "browser_wait_for", {"text": "Mocked clinical answer for: What causes acute migraine?"})
        require(
            ask_requests(requests)[-1]["body"].get("use_langgraph") is True,
            "Playwright MCP did not propagate LangGraph selection",
        )

        log("playwright: 503 error path")
        await call_tool(
            session,
            "browser_run_code",
            {
                "code": """
                async (page) => {
                  await page.getByPlaceholder('Ask a medical question…').fill('trigger unavailable');
                  await page.keyboard.press('Enter');
                  return 'submitted unavailable trigger';
                }
                """
            },
        )
        await call_tool(session, "browser_wait_for", {"text": "System is initializing or busy"})

        log("playwright: safety path")
        await call_tool(
            session,
            "browser_run_code",
            {
                "code": """
                async (page) => {
                  await page.getByPlaceholder('Ask a medical question…').fill('I have severe chest pain with hallucination risk');
                  await page.keyboard.press('Enter');
                  await page.getByText('XAI Signal Breakdown', { exact: true }).click();
                  await page.getByText('Hallucination Analysis', { exact: true }).click();
                  return 'submitted emergency question';
                }
                """
            },
        )
        await call_tool(session, "browser_wait_for", {"text": "Call emergency services immediately"})

        log("playwright: clear cache")
        await call_tool(
            session,
            "browser_run_code",
            {
                "code": """
                async (page) => {
                  await page.getByRole('button', { name: 'Clear response cache' }).click();
                  return 'cleared cache';
                }
                """
            },
        )
        await call_tool(session, "browser_wait_for", {"text": "Cache cleared!"})
        require(any(r["url"] == "/clear-cache" for r in requests), "Playwright MCP did not call clear-cache")

        log("playwright: screenshot")
        await call_tool(
            session,
            "browser_take_screenshot",
            {"type": "png", "filename": str(ARTIFACTS / "playwright-mcp.png"), "fullPage": True},
        )
        await call_tool(session, "browser_close", {})


async def wait_for_body_text(session: ClientSession, text: str, timeout_s: float = 20.0):
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        result = await call_tool(
            session, "get_element_text", {"by": "tag", "value": "body", "timeout": 3000}
        )
        last = text_blocks(result)
        if text in last:
            return
        await anyio.sleep(0.5)
    raise AssertionError(f"Timed out waiting for text: {text}\nLast body text:\n{last[:2000]}")


async def run_selenium_flow(base_url: str, requests: list[dict]):
    log("selenium: connect")
    async with mcp_session("selenium") as session:
        log("selenium: start browser")
        await call_tool(
            session,
            "start_browser",
            {
                "browser": "chrome",
                "options": {
                    "headless": True,
                    "arguments": ["--no-sandbox", "--disable-dev-shm-usage", "--window-size=1440,1200"],
                },
            },
        )
        log("selenium: navigate")
        await call_tool(session, "navigate", {"url": base_url})
        await wait_for_body_text(session, "MediQuery AI")

        log("selenium: suggested question")
        await call_tool(
            session,
            "interact",
            {"action": "click", "by": "xpath", "value": "//button[contains(., 'Symptoms of Type 2 Diabetes')]"},
        )
        await wait_for_body_text(session, "Mocked clinical answer for: What are the symptoms of Type 2 Diabetes?")
        require(
            ask_requests(requests)[0]["body"]["question"] == "What are the symptoms of Type 2 Diabetes?",
            "Selenium MCP did not submit the suggested question",
        )

        log("selenium: langchain")
        await call_tool(
            session,
            "interact",
            {"action": "click", "by": "xpath", "value": "//label[contains(., 'LangChain (LCEL)')]"},
        )
        await call_tool(
            session,
            "send_keys",
            {
                "by": "css",
                "value": "textarea[placeholder*='Ask a medical question']",
                "text": "What is asthma?",
            },
        )
        await call_tool(session, "press_key", {"key": "Enter"})
        await wait_for_body_text(session, "Mocked clinical answer for: What is asthma?")
        require(
            ask_requests(requests)[-1]["body"].get("use_langchain") is True,
            "Selenium MCP did not propagate LangChain selection",
        )

        log("selenium: 503 error path")
        await call_tool(
            session,
            "send_keys",
            {
                "by": "css",
                "value": "textarea[placeholder*='Ask a medical question']",
                "text": "trigger unavailable",
            },
        )
        await call_tool(session, "press_key", {"key": "Enter"})
        await wait_for_body_text(session, "System is initializing or busy")

        log("selenium: clear cache")
        await call_tool(
            session,
            "interact",
            {"action": "click", "by": "xpath", "value": "//button[contains(., 'Clear response cache')]"},
        )
        await wait_for_body_text(session, "Cache cleared!")

        log("selenium: screenshot")
        await call_tool(
            session,
            "take_screenshot",
            {"outputPath": str(ARTIFACTS / "selenium-mcp.png")},
        )
        await call_tool(session, "close_session", {})


async def main():
    stack = AppStack()
    log("stack: start")
    stack.start()
    summary = {}
    try:
        stack.requests.clear()
        await run_playwright_flow(stack.base_url, stack.requests)
        summary["playwright_mcp"] = {
            "status": "passed",
            "ask_requests": len(ask_requests(stack.requests)),
            "cache_clear_calls": len([r for r in stack.requests if r["url"] == "/clear-cache"]),
        }

        stack.requests.clear()
        await run_selenium_flow(stack.base_url, stack.requests)
        summary["selenium_mcp"] = {
            "status": "passed",
            "ask_requests": len(ask_requests(stack.requests)),
            "cache_clear_calls": len([r for r in stack.requests if r["url"] == "/clear-cache"]),
        }
    finally:
        log("stack: stop")
        stack.stop()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    anyio.run(main)
