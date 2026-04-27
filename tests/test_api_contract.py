"""
FastAPI contract tests that avoid loading real model/RAG components.
"""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _fake_pipeline(answer_text="Mock API answer", source_score=0.91):
    class FakePipeline:
        def __init__(self):
            self.calls = []

        def prepare_for_stream(self, **kwargs):
            return self.answer(**kwargs)

        def answer(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                answer=answer_text,
                sources=[
                    {
                        "source": "Mock Source",
                        "content": "Mock evidence about a medical topic.",
                        "score": source_score,
                        "url": "https://example.test/source",
                    }
                ],
                confidence={
                    "score": 0.84,
                    "level": "high",
                    "explanation": "Mock confidence",
                },
                attributions=[
                    {
                        "claim": "Mock claim",
                        "source": "Mock Source",
                        "evidence": "Mock evidence",
                        "similarity": 0.82,
                    }
                ],
                disclaimer="Educational information only.",
                rationale="Mock rationale.",
                is_answerable=True,
                from_cache=False,
            )

    return FakePipeline()


class _FakeFeedback:
    feedback_id = "feedback-test"


class _FakeFeedbackCollector:
    def __init__(self):
        self.submissions = []

    def submit_feedback(self, **kwargs):
        self.submissions.append(kwargs)
        return _FakeFeedback()

    def get_statistics(self):
        return SimpleNamespace(
            total_feedback=3,
            avg_rating=4.3,
            helpful_rate=0.67,
            accuracy_rate=0.5,
            safety_rate=1.0,
            low_rated_count=1,
        )


class _FakeConversationSession:
    session_id = "session-test"

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "num_turns": 1,
            "turns": [{"question": "What is diabetes?", "answer": "Mock answer"}],
        }


class _FakeConversationManager:
    def __init__(self):
        self.session = _FakeConversationSession()

    def create_session(self):
        return self.session

    def get_session(self, session_id):
        if session_id == self.session.session_id:
            return self.session
        return None


def _patch_lightweight_api(monkeypatch, pipeline=None):
    import api.main as api_main

    pipeline = pipeline or _fake_pipeline()
    monkeypatch.setattr(api_main, "_init_guardrails", lambda: None)
    monkeypatch.setattr(api_main, "_init_conversation_manager", lambda: None)
    monkeypatch.setattr(api_main, "_log_response_trajectory", lambda payload: None)
    monkeypatch.setattr(api_main, "guardrails", None)
    monkeypatch.setattr(api_main, "drug_checker", None)
    monkeypatch.setattr(api_main, "conversation_manager", None)
    monkeypatch.setattr(api_main, "get_pipeline", lambda model_choice="tinyllama": pipeline)
    monkeypatch.setattr(api_main, "_get_langchain_pipeline", lambda model_choice="tinyllama": pipeline)
    monkeypatch.setattr(api_main, "_get_langgraph_pipeline", lambda model_choice="tinyllama": pipeline)
    return api_main, pipeline


def test_ask_standard_pipeline_contract(monkeypatch):
    api_main, pipeline = _patch_lightweight_api(monkeypatch)
    client = TestClient(api_main.app)

    response = client.post(
        "/ask",
        json={
            "question": "What are the symptoms of diabetes?",
            "model_choice": "tinyllama",
            "num_sources": 4,
            "include_explanation": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Mock API answer"
    assert payload["pipeline_used"] == "Standard"
    assert payload["model_used"] == "tinyllama"
    assert payload["sources"][0]["source"] == "Mock Source"
    assert payload["confidence"]["level"] == "high"
    assert payload.get("num_sources_requested") == 4
    assert payload.get("num_sources_effective") == 4
    assert payload.get("num_sources_returned") == 1
    assert pipeline.calls[0]["question"] == "What are the symptoms of diabetes?"
    assert pipeline.calls[0]["num_documents"] == 4
    assert pipeline.calls[0]["include_explanation"] is True


def test_ask_low_latency_profile_reduces_work(monkeypatch):
    import api.main as api_main

    api_main, pipeline = _patch_lightweight_api(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "_get_low_latency_settings",
        lambda: {
            "enabled": False,
            "num_sources": 2,
            "include_explanation": False,
            "max_new_tokens": 192,
        },
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/ask",
        json={
            "question": "What are first-line treatments for hypertension?",
            "num_sources": 6,
            "include_explanation": True,
            "low_latency": True,
        },
    )

    assert response.status_code == 200
    # low_latency no longer overrides num_sources — user's value (6) is respected
    assert pipeline.calls[0]["num_documents"] == 6
    assert pipeline.calls[0]["include_explanation"] is True
    assert pipeline.calls[0]["generation_max_tokens"] == 192


def test_ask_adaptive_token_budget_for_high_precision_query(monkeypatch):
    import api.main as api_main

    api_main, pipeline = _patch_lightweight_api(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "_get_low_latency_settings",
        lambda: {
            "enabled": False,
            "num_sources": 2,
            "include_explanation": False,
            "max_new_tokens": 192,
        },
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/ask",
        json={
            "question": "What are first-line therapies for COPD exacerbation?",
            "num_sources": 5,
            "include_explanation": True,
        },
    )

    assert response.status_code == 200
    assert pipeline.calls[0]["generation_max_tokens"] == 192
    assert pipeline.calls[0]["include_explanation"] is True


def test_ask_requires_api_key_when_configured(monkeypatch):
    api_main, _ = _patch_lightweight_api(monkeypatch)
    monkeypatch.setattr(api_main, "_API_KEY", "secret-test-key")
    monkeypatch.setattr(api_main, "_SCOPED_KEYS", {"secret-test-key": "read"})
    client = TestClient(api_main.app)

    missing = client.post("/ask", json={"question": "What is diabetes?"})
    valid = client.post(
        "/ask",
        json={"question": "What is diabetes?"},
        headers={"X-API-Key": "secret-test-key"},
    )

    assert missing.status_code == 401
    assert missing.json()["error_code"] == "HTTP_401"
    assert valid.status_code == 200


def test_ask_langchain_and_langgraph_flags_route_to_selected_pipeline(monkeypatch):
    api_main, pipeline = _patch_lightweight_api(monkeypatch)
    client = TestClient(api_main.app)

    langchain = client.post(
        "/ask",
        json={"question": "What is asthma?", "use_langchain": True},
    )
    langgraph = client.post(
        "/ask",
        json={"question": "What is migraine?", "use_langgraph": True},
    )

    assert langchain.status_code == 200
    assert langchain.json()["pipeline_used"] == "LangChain"
    assert langgraph.status_code == 200
    assert langgraph.json()["pipeline_used"] == "LangGraph"
    assert len(pipeline.calls) == 2


def test_ask_validation_rejects_bad_source_count(monkeypatch):
    api_main, _ = _patch_lightweight_api(monkeypatch)
    client = TestClient(api_main.app)

    response = client.post(
        "/ask",
        json={"question": "What are the symptoms of diabetes?", "num_sources": 99},
    )

    assert response.status_code == 422


def test_ask_emergency_redirect_does_not_load_pipeline(monkeypatch):
    import api.main as api_main

    class FakeGuardrails:
        def check_input(self, question):
            return SimpleNamespace(
                level=SimpleNamespace(value="emergency"),
                flags=["emergency_cardiac"],
                redirect_message="Call emergency services immediately.",
            )

    def fail_pipeline(*args, **kwargs):
        raise AssertionError("pipeline should not be loaded for emergency redirects")

    monkeypatch.setattr(api_main, "_init_guardrails", lambda: None)
    monkeypatch.setattr(api_main, "_log_response_trajectory", lambda payload: None)
    monkeypatch.setattr(api_main, "guardrails", FakeGuardrails())
    monkeypatch.setattr(api_main, "get_pipeline", fail_pipeline)
    client = TestClient(api_main.app)

    response = client.post(
        "/ask",
        json={"question": "I have severe chest pain and cannot breathe"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline_used"] == "EmergencyRedirect"
    assert payload["safety"]["is_emergency"] is True
    assert payload["sources"] == []
    assert "emergency" in payload["answer"].lower()


def test_ask_stream_ndjson_order_meta_before_tokens_then_done(monkeypatch):
    api_main, _ = _patch_lightweight_api(
        monkeypatch,
        pipeline=_fake_pipeline(answer_text="Alpha beta gamma delta."),
    )
    client = TestClient(api_main.app)

    with client.stream(
        "POST",
        "/ask/stream",
        json={"question": "What is diabetes?"},
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    types = [json.loads(line)["type"] for line in lines]
    assert types[0] == "meta"
    assert types[-1] == "done"
    assert types.count("token") >= 1
    first_token = next(i for i, t in enumerate(types) if t == "token")
    assert first_token > 0
    assert first_token < len(types) - 1


def test_ask_stream_returns_meta_tokens_and_done(monkeypatch):
    api_main, _ = _patch_lightweight_api(
        monkeypatch,
        pipeline=_fake_pipeline(answer_text="First streamed sentence. Second streamed sentence."),
    )
    client = TestClient(api_main.app)

    with client.stream(
        "POST",
        "/ask/stream",
        json={"question": "What is diabetes?"},
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert lines[0].startswith('{"type": "meta"')
    assert any('"type": "token"' in line for line in lines)
    assert lines[-1] == '{"type": "done", "data": ""}'


def test_ask_stream_returns_error_event_when_pipeline_is_unavailable(monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "_init_guardrails", lambda: None)
    monkeypatch.setattr(api_main, "guardrails", None)
    monkeypatch.setattr(api_main, "get_pipeline", lambda model_choice="tinyllama": None)
    client = TestClient(api_main.app)

    with client.stream(
        "POST",
        "/ask/stream",
        json={"question": "What is diabetes?"},
    ) as response:
        lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert len(lines) == 1
    assert lines[0].startswith('{"type": "error"')
    assert "Pipeline not initialized" in lines[0]


def test_models_endpoint_marks_loaded_models(monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "pipelines", {"tinyllama::base": object()})
    client = TestClient(api_main.app)

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert "tinyllama" in payload
    assert payload["tinyllama"]["loaded"] is True
    assert payload["tinyllama"]["requires_gpu"] is False


def test_health_endpoint_reports_loaded_pipeline(monkeypatch):
    import api.main as api_main

    fake_cache = SimpleNamespace(
        get_cache_stats=lambda: {
            "query_cache_size": 1,
            "embedding_cache_size": 0,
            "cache_dir": "test-cache",
            "ttl_seconds": 60,
        }
    )
    monkeypatch.setattr(
        api_main,
        "pipelines",
        {"tinyllama::base": SimpleNamespace(cache_manager=fake_cache)},
    )
    monkeypatch.setattr(api_main, "_shared_components", {})
    client = TestClient(api_main.app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["pipeline_ready"] is True
    assert payload["models_loaded"] == ["tinyllama::base"]
    assert payload["cache_stats"]["query_cache_size"] == 1


def test_component_health_reports_loaded_shared_components(monkeypatch):
    import api.main as api_main

    fake_collection = SimpleNamespace(count=lambda: 12)
    fake_retriever = SimpleNamespace(
        bm25=object(),
        corpus=["doc1", "doc2"],
        reranker=SimpleNamespace(model_name="mock-reranker"),
    )
    monkeypatch.setattr(
        api_main,
        "_shared_components",
        {
            "vector_store": SimpleNamespace(collection=fake_collection),
            "retriever": fake_retriever,
            "query_enhancer": object(),
            "corrective_rag": None,
            "factual_consistency_checker": None,
        },
    )
    monkeypatch.setattr(api_main, "pipelines", {})
    client = TestClient(api_main.app)

    response = client.get("/health/components")

    assert response.status_code == 200
    payload = response.json()
    assert payload["components"]["vector_store"] == {"status": "ok", "doc_count": 12}
    assert payload["components"]["bm25_index"] == {"status": "ok", "doc_count": 2}
    assert payload["components"]["query_enhancer"] == {"status": "ok"}
    assert payload["components"]["reranker"] == {"status": "ok", "model": "mock-reranker"}


def test_clear_cache_rejects_read_only_key_when_admin_key_configured(monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "_API_KEY", "read-only-key")
    monkeypatch.setattr(api_main, "_API_KEY_ADMIN", "admin-only-key")
    monkeypatch.setattr(
        api_main,
        "_SCOPED_KEYS",
        {"read-only-key": "read", "admin-only-key": "admin"},
    )
    monkeypatch.setattr(
        api_main,
        "pipelines",
        {"tinyllama::base": SimpleNamespace(cache_manager=SimpleNamespace(invalidate_cache=lambda: None))},
    )
    client = TestClient(api_main.app)

    bad = client.post("/clear-cache", headers={"X-API-Key": "read-only-key"})
    assert bad.status_code == 403  # read key → forbidden on admin endpoint

    ok = client.post("/clear-cache", headers={"X-API-Key": "admin-only-key"})
    assert ok.status_code == 200


def test_scoped_keys_read_can_ask_but_not_admin(monkeypatch):
    """KEY:scope format: read key passes /ask, is rejected on /clear-cache."""
    api_main, _ = _patch_lightweight_api(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "_SCOPED_KEYS",
        {"rk-abc": "read", "ak-xyz": "admin"},
    )
    monkeypatch.setattr(
        api_main,
        "pipelines",
        {"tinyllama::base": SimpleNamespace(cache_manager=SimpleNamespace(invalidate_cache=lambda: None))},
    )
    client = TestClient(api_main.app)

    # Read key can call /ask
    ask_ok = client.post("/ask", json={"question": "What is hypertension?"}, headers={"X-API-Key": "rk-abc"})
    assert ask_ok.status_code == 200

    # Read key cannot call admin endpoint
    cache_bad = client.post("/clear-cache", headers={"X-API-Key": "rk-abc"})
    assert cache_bad.status_code == 403

    # Admin key can call admin endpoint
    cache_ok = client.post("/clear-cache", headers={"X-API-Key": "ak-xyz"})
    assert cache_ok.status_code == 200


def test_scoped_keys_unknown_key_rejected(monkeypatch):
    """Unknown key is rejected with 401 on both read and admin endpoints."""
    api_main, _ = _patch_lightweight_api(monkeypatch)
    monkeypatch.setattr(api_main, "_SCOPED_KEYS", {"valid-key": "read"})
    client = TestClient(api_main.app)

    resp = client.post("/ask", json={"question": "test"}, headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_auth_disabled_when_no_keys_configured(monkeypatch):
    """When _SCOPED_KEYS is empty, all endpoints are open (dev mode)."""
    api_main, _ = _patch_lightweight_api(monkeypatch)
    monkeypatch.setattr(api_main, "_SCOPED_KEYS", {})
    client = TestClient(api_main.app)

    resp = client.post("/ask", json={"question": "What is anemia?"})
    assert resp.status_code == 200


def test_clear_cache_invalidates_pipeline_cache(monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(api_main, "_API_KEY", "")
    monkeypatch.setattr(api_main, "_API_KEY_ADMIN", "")

    class FakeCache:
        def __init__(self):
            self.invalidated = False

        def invalidate_cache(self):
            self.invalidated = True

    fake_cache = FakeCache()
    monkeypatch.setattr(
        api_main,
        "pipelines",
        {"tinyllama::base": SimpleNamespace(cache_manager=fake_cache)},
    )
    client = TestClient(api_main.app)

    response = client.post("/clear-cache")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "cleared": True}
    assert fake_cache.invalidated is True


def test_create_and_get_session_contract(monkeypatch):
    import api.main as api_main

    manager = _FakeConversationManager()
    monkeypatch.setattr(api_main, "_init_conversation_manager", lambda: None)
    monkeypatch.setattr(api_main, "conversation_manager", manager)
    client = TestClient(api_main.app)

    created = client.post("/sessions")
    fetched = client.get("/sessions/session-test")
    missing = client.get("/sessions/missing-session")

    assert created.status_code == 200
    assert created.json() == {"session_id": "session-test"}
    assert fetched.status_code == 200
    assert fetched.json()["num_turns"] == 1
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "HTTP_404"


def test_feedback_requires_rating_or_boolean_signal(monkeypatch):
    import api.main as api_main

    collector = _FakeFeedbackCollector()
    monkeypatch.setattr(api_main, "_init_feedback_system", lambda: None)
    monkeypatch.setattr(api_main, "feedback_collector", collector)
    monkeypatch.setattr(api_main, "trajectory_logger", None)
    client = TestClient(api_main.app)

    response = client.post("/feedback", json={"response_id": "resp-1"})

    assert response.status_code == 400
    assert "Provide either rating" in response.json()["detail"]


def test_feedback_derives_reward_from_boolean_signals(monkeypatch):
    import api.main as api_main

    collector = _FakeFeedbackCollector()
    monkeypatch.setattr(api_main, "_init_feedback_system", lambda: None)
    monkeypatch.setattr(api_main, "feedback_collector", collector)
    monkeypatch.setattr(api_main, "trajectory_logger", None)
    client = TestClient(api_main.app)

    response = client.post(
        "/feedback",
        json={
            "response_id": "resp-1",
            "was_helpful": True,
            "was_accurate": False,
            "was_safe": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["response_id"] == "resp-1"
    assert payload["trajectory_found"] is False
    assert payload["reward_signal"] == 0.7
    assert collector.submissions[0]["rating"] == 4
    assert collector.submissions[0]["was_accurate"] is False


def test_feedback_stats_contract(monkeypatch):
    import api.main as api_main

    collector = _FakeFeedbackCollector()
    monkeypatch.setattr(api_main, "_init_feedback_system", lambda: None)
    monkeypatch.setattr(api_main, "feedback_collector", collector)
    client = TestClient(api_main.app)

    response = client.get("/feedback/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_feedback": 3,
        "avg_rating": 4.3,
        "helpful_rate": 0.67,
        "accuracy_rate": 0.5,
        "safety_rate": 1.0,
        "low_rated_count": 1,
    }


def test_langsmith_status_reports_disabled_when_flag_off(monkeypatch):
    import api.main as api_main

    monkeypatch.setattr(
        api_main,
        "_get_langsmith_mcp_settings",
        lambda: SimpleNamespace(
            enabled=False,
            server_cmd="uvx",
            server_args="langsmith-mcp-server",
        ),
    )
    client = TestClient(api_main.app)

    response = client.get("/langsmith/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["connection_ok"] is False


def test_langsmith_tool_executes_when_enabled(monkeypatch):
    import api.main as api_main
    import src.mcp_client.langsmith_bridge as bridge

    monkeypatch.setattr(
        api_main,
        "_get_langsmith_mcp_settings",
        lambda: SimpleNamespace(
            enabled=True,
            server_cmd="uvx",
            server_args="langsmith-mcp-server",
            api_key="key",
            workspace_id=None,
            endpoint=None,
        ),
    )

    async def _fake_call(settings, tool_name, tool_args):
        return {"ok": True, "tool_name": tool_name, "data": {"result": [tool_args]}}

    monkeypatch.setattr(bridge, "call_langsmith_tool", _fake_call)
    client = TestClient(api_main.app)

    response = client.post(
        "/langsmith/tool",
        json={"tool_name": "fetch_runs", "tool_args": {"limit": 2, "page_number": 1}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["tool_name"] == "fetch_runs"


def test_langsmith_status_reports_tools_when_enabled(monkeypatch):
    import api.main as api_main
    import src.mcp_client.langsmith_bridge as bridge

    monkeypatch.setattr(
        api_main,
        "_get_langsmith_mcp_settings",
        lambda: SimpleNamespace(
            enabled=True,
            server_cmd="uvx",
            server_args="langsmith-mcp-server",
            api_key="key",
            workspace_id=None,
            endpoint=None,
        ),
    )

    async def _fake_list_tools(settings):
        return {
            "ok": True,
            "tools": [
                {"name": "fetch_runs", "description": "Fetch runs"},
                {"name": "list_projects", "description": "List projects"},
            ],
        }

    async def _fake_call(settings, tool_name, tool_args):
        return {"ok": True, "tool_name": tool_name, "data": [{"name": "demo-project"}]}

    monkeypatch.setattr(bridge, "list_langsmith_tools", _fake_list_tools)
    monkeypatch.setattr(bridge, "call_langsmith_tool", _fake_call)
    client = TestClient(api_main.app)

    response = client.get("/langsmith/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["connection_ok"] is True
    assert payload["available_tools"] == ["fetch_runs", "list_projects"]


def test_langsmith_improvement_report_aggregates_failures(monkeypatch):
    import api.main as api_main
    import src.mcp_client.langsmith_bridge as bridge

    monkeypatch.setattr(
        api_main,
        "_get_langsmith_mcp_settings",
        lambda: SimpleNamespace(
            enabled=True,
            server_cmd="uvx",
            server_args="langsmith-mcp-server",
            api_key="key",
            workspace_id=None,
            endpoint=None,
        ),
    )

    async def _fake_list_tools(settings):
        return {
            "ok": True,
            "tools": [{"name": "fetch_runs", "description": "Fetch runs"}],
        }

    async def _fake_call(settings, tool_name, tool_args):
        assert tool_name == "fetch_runs"
        assert tool_args["project_name"] == "healthcare-qa"
        assert tool_args["is_root"] == "true"
        return {
            "ok": True,
            "tool_name": "fetch_runs",
            "data": {
                "runs": [
                    {
                        "id": "r0",
                        "status": "pending",
                        "inputs": {"question": "Queued request"},
                        "outputs": {},
                    },
                    {
                        "id": "r1",
                        "status": "error",
                        "error": "tool crashed",
                        "latency_ms": 6200,
                        "inputs": {"question": "Drug A vs Drug B?"},
                        "outputs": {},
                    },
                    {
                        "id": "r2",
                        "status": "success",
                        "latency_ms": 5100,
                        "inputs": {"question": "Exam-style stem"},
                        "outputs": {"is_answerable": False},
                    },
                    {
                        "id": "r3",
                        "status": "success",
                        "latency_ms": 900,
                        "inputs": {"question": "Definition question"},
                        "outputs": {"answer": "A complete answer", "is_answerable": True},
                    },
                ]
            },
        }

    monkeypatch.setattr(bridge, "list_langsmith_tools", _fake_list_tools)
    monkeypatch.setattr(bridge, "call_langsmith_tool", _fake_call)
    client = TestClient(api_main.app)

    response = client.post(
        "/langsmith/improvement-report",
        json={"project_name": "healthcare-qa", "limit": 10, "page_number": 1, "slow_latency_ms": 4000},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fetched_runs"] == 4
    assert payload["analyzed_runs"] == 3
    assert payload["metrics"]["error_runs"] == 1
    assert payload["metrics"]["slow_runs"] == 2
    assert payload["metrics"]["unanswerable_runs"] == 1
    assert len(payload["recommendations"]) >= 1
