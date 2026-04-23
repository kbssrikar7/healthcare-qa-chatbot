const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { after, afterEach, before, beforeEach, describe, test } = require("node:test");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "../..");
const HISTORY_PATH = path.join(ROOT, "data/question_history.json");
const PYTHON = fs.existsSync(path.join(ROOT, "venv/bin/python"))
  ? path.join(ROOT, "venv/bin/python")
  : "python3";

let apiServer;
let apiPort;
let streamlitProcess;
let streamlitPort;
let browser;
let page;
let requests;
let historySnapshot;

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(error);
      }
    });
  });
}

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

async function startMockApi() {
  requests = [];
  apiPort = await getFreePort();

  apiServer = http.createServer(async (req, res) => {
    if (req.method === "OPTIONS") {
      sendJson(res, 200, {});
      return;
    }

    if (req.method === "GET" && req.url === "/models") {
      requests.push({ method: req.method, url: req.url });
      sendJson(res, 200, {
        tinyllama: {
          display_name: "TinyLlama 1.1B",
          description: "Fast local test model",
          parameters: "1.1B",
          requires_gpu: false,
          loaded: true,
        },
      });
      return;
    }

    if (req.method === "POST" && req.url === "/ask") {
      const body = await readJson(req);
      requests.push({ method: req.method, url: req.url, body });
      if (body.question.toLowerCase().includes("trigger unavailable")) {
        sendJson(res, 503, { detail: "System is initializing" });
        return;
      }
      const pipeline = body.use_langgraph
        ? "LangGraph"
        : body.use_langchain
          ? "LangChain"
          : "Standard";
      const isEmergency = body.question.toLowerCase().includes("chest pain");
      const includeHallucination = body.question.toLowerCase().includes("hallucination");
      sendJson(res, 200, {
        response_id: `resp-${requests.filter((r) => r.url === "/ask").length}`,
        question: body.question,
        answer: isEmergency
          ? "Call emergency services immediately for severe chest pain."
          : `Mocked clinical answer for: ${body.question}`,
        sources: [
          {
            source: "Mock Medical Reference",
            content: "Type 2 diabetes can cause increased thirst, frequent urination, and fatigue.",
            score: 0.91,
            url: "https://example.test/source",
          },
        ],
        confidence: {
          score: 0.86,
          level: "high",
          explanation: "Mocked high-confidence result from retrieved evidence.",
        },
        attributions: [
          {
            claim: "Symptoms can include increased thirst.",
            source: "Mock Medical Reference",
            evidence: "Mock evidence",
            similarity: 0.88,
          },
        ],
        disclaimer: "This information is for educational purposes only.",
        rationale: "The answer is based on the mocked retrieved source.",
        model_used: "TinyLlama 1.1B",
        pipeline_used: pipeline,
        session_id: "test-session",
        safety: isEmergency
          ? {
              level: "emergency",
              flags: ["emergency_cardiac"],
              is_emergency: true,
              emergency_message: "Call emergency services immediately.",
              drug_warnings: ["Avoid combining warfarin and aspirin without clinical advice."],
            }
          : { level: "safe", flags: [], is_emergency: false },
        latency_ms: 42,
        confidence_breakdown: {
          retrieval_confidence: 0.87,
          generation_confidence: 0.82,
          consistency_score: 0.8,
          source_agreement: 0.78,
          medical_entity_coverage: 0.72,
          signal_weights: {
            retrieval: 0.3,
            generation: 0.25,
            consistency: 0.2,
            source_agreement: 0.15,
            entity_coverage: 0.1,
          },
        },
        hallucination: includeHallucination
          ? {
              score: 0.61,
              has_hallucination: true,
              type: "unsupported_claim",
              explanation: "Mock hallucination risk for UI coverage.",
              medical_accuracy_flags: ["unsupported dosage claim"],
            }
          : null,
      });
      return;
    }

    if (req.method === "POST" && req.url === "/feedback") {
      const body = await readJson(req);
      requests.push({ method: req.method, url: req.url, body });
      sendJson(res, 200, {
        status: "recorded",
        feedback_id: "feedback-1",
        response_id: body.response_id,
        reward_signal: 1,
        trajectory_found: true,
      });
      return;
    }

    if (req.method === "POST" && req.url === "/clear-cache") {
      requests.push({ method: req.method, url: req.url });
      sendJson(res, 200, { status: "cleared" });
      return;
    }

    sendJson(res, 404, { detail: "Not found" });
  });

  await new Promise((resolve) => apiServer.listen(apiPort, "127.0.0.1", resolve));
}

async function waitForHttp(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;

  while (Date.now() < deadline) {
    try {
      const status = await new Promise((resolve, reject) => {
        const req = http.get(url, (res) => {
          res.resume();
          resolve(res.statusCode);
        });
        req.on("error", reject);
        req.setTimeout(1000, () => req.destroy(new Error("Request timed out")));
      });
      if (status && status < 500) return;
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  throw new Error(`Timed out waiting for ${url}: ${lastError?.message || "no response"}`);
}

async function startStreamlit() {
  streamlitPort = await getFreePort();
  const env = {
    ...process.env,
    API_URL: `http://127.0.0.1:${apiPort}`,
    STREAMLIT_BROWSER_GATHER_USAGE_STATS: "false",
    PYTHONUNBUFFERED: "1",
  };

  streamlitProcess = spawn(
    PYTHON,
    [
      "-m",
      "streamlit",
      "run",
      "frontend/streamlit_app.py",
      "--server.address=127.0.0.1",
      `--server.port=${streamlitPort}`,
      "--server.headless=true",
    ],
    { cwd: ROOT, env, stdio: ["ignore", "pipe", "pipe"] },
  );

  let output = "";
  streamlitProcess.stdout.on("data", (chunk) => {
    output += chunk.toString();
  });
  streamlitProcess.stderr.on("data", (chunk) => {
    output += chunk.toString();
  });

  streamlitProcess.on("exit", (code) => {
    if (code !== 0 && code !== null) {
      process.stderr.write(output);
    }
  });

  await waitForHttp(`http://127.0.0.1:${streamlitPort}`, 45000);
}

async function waitForText(text) {
  await page.getByText(text, { exact: false }).first().waitFor({ timeout: 30000 });
}

async function waitForInteractive() {
  // Streamlit disables all widgets while the server-side script is executing
  // (rerun). Wait until at least one button exists AND none are disabled.
  await page.waitForFunction(
    () => {
      const buttons = document.querySelectorAll('[data-testid^="stBaseButton"]');
      return buttons.length > 0 && [...buttons].every((b) => !b.disabled);
    },
    { timeout: 15000 },
  ).catch(() => {});
}

function askedRequests() {
  return requests.filter((request) => request.method === "POST" && request.url === "/ask");
}

describe("MediQuery Streamlit UI", () => {
  before(async () => {
    historySnapshot = fs.existsSync(HISTORY_PATH) ? fs.readFileSync(HISTORY_PATH, "utf8") : null;
    fs.mkdirSync(path.dirname(HISTORY_PATH), { recursive: true });
    fs.writeFileSync(HISTORY_PATH, "[]\n");

    await startMockApi();
    await startStreamlit();
    browser = await chromium.launch({ headless: true });
  });

  beforeEach(async () => {
    page = await browser.newPage();
    requests = requests.filter((request) => request.method === "GET" && request.url === "/models");
    await page.goto(`http://127.0.0.1:${streamlitPort}`, { waitUntil: "domcontentloaded" });
    await waitForText("MediQuery AI");
    await waitForInteractive();
  });

  afterEach(async () => {
    await page?.close().catch(() => {});
    page = undefined;
  });

  after(async () => {
    await browser?.close().catch(() => {});
    streamlitProcess?.kill("SIGTERM");
    await new Promise((resolve) => apiServer?.close(resolve));
    if (historySnapshot === null) {
      fs.rmSync(HISTORY_PATH, { force: true });
    } else {
      fs.writeFileSync(HISTORY_PATH, historySnapshot);
    }
  });

  test("renders the welcome state and sidebar controls", async () => {
    await waitForText("Symptoms of Type 2 Diabetes");
    await waitForText("Treatment for Hypertension");
    await waitForText("TinyLlama 1.1B");
    await waitForText("Pipeline");
    await waitForText("References");

    assert.ok(requests.some((request) => request.method === "GET" && request.url === "/models"));
  });

  test("renders the core question flow on a mobile viewport", async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.reload({ waitUntil: "domcontentloaded" });
    await waitForText("MediQuery AI");

    await page.getByPlaceholder(/Ask a medical question/).fill("What is bronchitis?");
    await page.keyboard.press("Enter");

    await waitForText("Mocked clinical answer for: What is bronchitis?");
    await waitForText("Medical Disclaimer");
  });

  test("submits a suggested medical question and renders evidence", async () => {
    await page
      .getByTestId("stMainBlockContainer")
      .getByRole("button", { name: "Symptoms of Type 2 Diabetes" })
      .click();

    await waitForText("Mocked clinical answer for: What are the symptoms of Type 2 Diabetes?");
    await waitForText("High confidence");
    await waitForText("Medical Disclaimer");
    await waitForText("TinyLlama 1.1B");

    await page.getByText("Sources & References", { exact: false }).click();
    await waitForText("Mock Medical Reference");

    const [ask] = askedRequests();
    assert.equal(ask.body.question, "What are the symptoms of Type 2 Diabetes?");
    assert.equal(ask.body.num_sources, 3);
    assert.equal(ask.body.include_explanation, true);
    assert.equal(ask.body.use_langchain, undefined);
    assert.equal(ask.body.use_langgraph, undefined);
  });

  test("passes the selected LangGraph pipeline flag to the backend", async () => {
    await page.getByText("LangGraph (Self-Correcting)", { exact: true }).click();
    await page.getByPlaceholder(/Ask a medical question/).fill("What causes acute migraine?");
    await page.keyboard.press("Enter");

    await waitForText("Mocked clinical answer for: What causes acute migraine?");
    await waitForText("LangGraph");

    const [ask] = askedRequests();
    assert.equal(ask.body.question, "What causes acute migraine?");
    assert.equal(ask.body.use_langgraph, true);
    assert.equal(ask.body.use_langchain, undefined);
  });

  test("passes the selected LangChain pipeline flag to the backend", async () => {
    await page.getByText("LangChain (LCEL)", { exact: true }).click();
    await page.getByPlaceholder(/Ask a medical question/).fill("What is asthma?");
    await page.keyboard.press("Enter");

    await waitForText("Mocked clinical answer for: What is asthma?");
    await waitForText("LangChain");

    const [ask] = askedRequests();
    assert.equal(ask.body.question, "What is asthma?");
    assert.equal(ask.body.use_langchain, true);
    assert.equal(ask.body.use_langgraph, undefined);
  });

  test("sends the selected source count to the backend", async () => {
    const slider = page.getByRole("slider").first();
    await slider.click();
    for (let i = 0; i < 4; i += 1) {
      await page.keyboard.press("ArrowRight");
    }
    await page.getByPlaceholder(/Ask a medical question/).fill("What are diabetes symptoms?");
    await page.keyboard.press("Enter");

    await waitForText("Mocked clinical answer for: What are diabetes symptoms?");

    const [ask] = askedRequests();
    assert.equal(ask.body.num_sources, 7);
  });

  test("shows initialization errors returned by the API", async () => {
    await page.getByPlaceholder(/Ask a medical question/).fill("trigger unavailable");
    await page.keyboard.press("Enter");

    await waitForText("System is initializing or busy");

    const [ask] = askedRequests();
    assert.equal(ask.body.question, "trigger unavailable");
  });

  test("renders emergency safety, drug warning, XAI, and hallucination panels", async () => {
    await page
      .getByPlaceholder(/Ask a medical question/)
      .fill("I have severe chest pain with hallucination risk");
    await page.keyboard.press("Enter");

    await waitForText("Call emergency services immediately");
    await waitForText("Avoid combining warfarin and aspirin");
    await waitForText("Safety: Emergency");

    await page.getByText("XAI Signal Breakdown", { exact: true }).click();
    await waitForText("Retrieval Quality");

    await page.getByText("Hallucination Analysis", { exact: true }).click();
    await waitForText("unsupported dosage claim");
  });

  test("clear cache button calls the backend and shows success", async () => {
    await page.getByRole("button", { name: "Clear response cache" }).click();
    await waitForText("Cache cleared!");

    assert.ok(requests.some((request) => request.method === "POST" && request.url === "/clear-cache"));
  });

  test("new conversation clears the visible chat state", async () => {
    await page.getByPlaceholder(/Ask a medical question/).fill("What is pneumonia?");
    await page.keyboard.press("Enter");
    await waitForText("Mocked clinical answer for: What is pneumonia?");

    await page.getByRole("button", { name: /New conversation/ }).click();

    await waitForText("MediQuery AI");
    assert.equal(
      await page.getByText("Mocked clinical answer for: What is pneumonia?").count(),
      0,
    );
  });

  test("submits positive answer feedback", async () => {
    await page
      .getByTestId("stMainBlockContainer")
      .getByRole("button", { name: "Treatment for Hypertension" })
      .click();
    await waitForText("Mocked clinical answer for: What is the treatment for Hypertension?");

    await page.getByRole("button", { name: /Accurate/ }).click();
    await waitForText("Feedback recorded: Accurate");

    const feedback = requests.find(
      (request) => request.method === "POST" && request.url === "/feedback",
    );
    assert.equal(feedback.body.response_id, "resp-1");
    assert.equal(feedback.body.rating, 5);
    assert.equal(feedback.body.was_accurate, true);
    assert.equal(feedback.body.was_safe, true);
  });
});
