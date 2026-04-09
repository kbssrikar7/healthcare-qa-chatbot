"""
Load test for Healthcare QA Chatbot API.

Tests realistic traffic patterns across all query types with the full
pipeline: retrieval → generation → XAI → safety.

Run:
    locust -f tests/locustfile.py --host http://localhost:8000 \
           --users 10 --spawn-rate 2 --run-time 60s --headless

Or open the Locust web UI:
    locust -f tests/locustfile.py --host http://localhost:8000
    # then open http://localhost:8089
"""
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# ---------------------------------------------------------------------------
# Realistic medical question bank — covers all 5 query types detected by
# the adaptive weight system (drug / definition / symptom / comparison / default)
# ---------------------------------------------------------------------------
DRUG_QUESTIONS = [
    "What is the dosage for metformin 500mg?",
    "What are the side effects of ibuprofen?",
    "Can I take aspirin with blood thinners?",
    "What is the maximum daily dose of paracetamol?",
    "What medications interact with warfarin?",
]

DEFINITION_QUESTIONS = [
    "What is type 2 diabetes?",
    "What is hypertension?",
    "What are the causes of anemia?",
    "Explain what COPD means",
    "What is atrial fibrillation?",
]

SYMPTOM_QUESTIONS = [
    "I have chest pain and shortness of breath",
    "What causes dizziness and nausea together?",
    "My joints are swollen and painful",
    "I feel fatigued all the time",
    "I have a persistent cough and fever",
]

COMPARISON_QUESTIONS = [
    "What is the difference between type 1 and type 2 diabetes?",
    "Compare ibuprofen vs paracetamol for pain relief",
    "What is better for hypertension, ACE inhibitors or beta blockers?",
]

DEFAULT_QUESTIONS = [
    "How long does recovery from a broken leg take?",
    "When should I see a doctor for a cold?",
    "How is blood pressure measured?",
    "What tests diagnose kidney disease?",
]

ALL_QUESTIONS = (
    DRUG_QUESTIONS + DEFINITION_QUESTIONS +
    SYMPTOM_QUESTIONS + COMPARISON_QUESTIONS + DEFAULT_QUESTIONS
)


class LightweightUser(HttpUser):
    """
    Stress-tests the fast, non-LLM endpoints:
    /health, /models, /feedback, /components
    Good for measuring API overhead without LLM latency.
    """
    wait_time = between(0.5, 2)
    weight = 3  # 3x more lightweight users than heavy users

    def on_start(self):
        with self.client.get("/health", catch_response=True, name="/health") as r:
            if r.status_code != 200:
                r.failure(f"Health check failed: {r.status_code}")

    @task(4)
    def health(self):
        self.client.get("/health", name="/health")

    @task(3)
    def models(self):
        self.client.get("/models", name="/models")

    @task(2)
    def feedback(self):
        self.client.post(
            "/feedback",
            json={
                "response_id": f"lt-{random.randint(0, 9999)}",
                "rating": 5,
                "was_helpful": True,
                "was_accurate": True,
                "was_safe": True,
            },
            name="/feedback",
        )

    @task(1)
    def docs(self):
        self.client.get("/docs", name="/docs")


class MedicalChatbotUser(HttpUser):
    """
    Simulates a realistic user session:
    - Health check on startup
    - Mix of question types weighted toward definition + symptom (most common)
    - Occasional session-based follow-ups
    - Feedback submission after some answers
    """
    wait_time = between(2, 6)  # think time between requests
    weight = 1  # fewer heavy LLM users
    session_id: str = None

    def on_start(self):
        """Called once per simulated user — ping health to confirm API is up."""
        with self.client.get("/health", catch_response=True, name="/health") as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    # ------------------------------------------------------------------
    # Weighted task mix: definition + symptom most common, drug moderate,
    # comparison + default less frequent (mirrors real usage)
    # ------------------------------------------------------------------

    @task(3)
    def ask_definition(self):
        self._ask(random.choice(DEFINITION_QUESTIONS))

    @task(3)
    def ask_symptom(self):
        self._ask(random.choice(SYMPTOM_QUESTIONS))

    @task(2)
    def ask_drug(self):
        self._ask(random.choice(DRUG_QUESTIONS))

    @task(1)
    def ask_comparison(self):
        self._ask(random.choice(COMPARISON_QUESTIONS))

    @task(1)
    def ask_default(self):
        self._ask(random.choice(DEFAULT_QUESTIONS))

    @task(1)
    def ask_with_followup(self):
        """Two-turn session: ask a question then a follow-up."""
        question = random.choice(DEFINITION_QUESTIONS + SYMPTOM_QUESTIONS)
        result = self._ask(question)
        if result and result.get("session_id"):
            self.session_id = result["session_id"]
            self._ask("Can you explain that in more detail?", use_session=True)
            self.session_id = None

    @task(1)
    def submit_feedback(self):
        """Simulate user submitting feedback (helpful/not helpful)."""
        with self.client.post(
            "/feedback",
            json={
                "response_id": f"load-test-{random.randint(1000, 9999)}",
                "rating": random.choice([4, 5]),
                "was_helpful": True,
                "was_accurate": True,
                "was_safe": True,
            },
            catch_response=True,
            name="/feedback",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Feedback failed: {resp.status_code}")

    @task(1)
    def list_models(self):
        with self.client.get("/models", name="/models", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Models endpoint failed: {resp.status_code}")

    # ------------------------------------------------------------------

    def _ask(self, question: str, use_session: bool = False) -> dict:
        payload = {
            "question": question,
            "model_choice": "tinyllama",
            "num_sources": 3,
            "include_explanation": True,
        }
        if use_session and self.session_id:
            payload["session_id"] = self.session_id

        with self.client.post(
            "/ask",
            json=payload,
            catch_response=True,
            name="/ask",
            timeout=180,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Empty answer in response")
                    return {}
                return data
            elif resp.status_code == 429:
                resp.success()  # rate limit is expected under load, not a failure
                return {}
            else:
                resp.failure(f"Ask failed: {resp.status_code} — {resp.text[:200]}")
                return {}


# ---------------------------------------------------------------------------
# Custom stats printed at end of run
# ---------------------------------------------------------------------------
@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    stats = environment.stats
    ask_stats = stats.get("/ask", "POST")
    if ask_stats:
        print("\n─── Load Test Summary ────────────────────────────────")
        print(f"  /ask  requests : {ask_stats.num_requests}")
        print(f"  /ask  failures : {ask_stats.num_failures}")
        print(f"  /ask  p50 (ms) : {ask_stats.get_response_time_percentile(0.50):.0f}")
        print(f"  /ask  p95 (ms) : {ask_stats.get_response_time_percentile(0.95):.0f}")
        print(f"  /ask  p99 (ms) : {ask_stats.get_response_time_percentile(0.99):.0f}")
        print(f"  /ask  RPS      : {ask_stats.current_rps:.2f}")
        print("──────────────────────────────────────────────────────\n")
