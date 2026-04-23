---
source_file: "/home/kbs/Documents/final_project/src/feedback/trajectory_logger.py"
type: "code"
community: "Community 2"
location: "L20"
tags:
  - graphify/code
  - graphify/INFERRED
  - community/Community_2
---

# TrajectoryLogger

## Connections
- [[.__init__()_50]] - `method` [EXTRACTED]
- [[._ensure_storage()]] - `method` [EXTRACTED]
- [[.get()_1]] - `method` [EXTRACTED]
- [[.log()]] - `method` [EXTRACTED]
- [[AnswerResponse_1]] - `uses` [INFERRED]
- [[Application lifespan startup warmup and shutdown cleanup.]] - `uses` [INFERRED]
- [[Ask a medical question and get an explainable answer.      - Set model_choice to]] - `uses` [INFERRED]
- [[Attach a unique request_id to every request for traceability.]] - `uses` [INFERRED]
- [[AttributionInfo]] - `uses` [INFERRED]
- [[Build simple retrieval score statistics.]] - `uses` [INFERRED]
- [[Catch unhandled exceptions and return structured JSON.]] - `uses` [INFERRED]
- [[Clear cached Q&A responses so fresh answers are generated.]] - `uses` [INFERRED]
- [[Compute a normalized reward signal (0-1) from feedback.      Weighted for RL exp]] - `uses` [INFERRED]
- [[ConfidenceInfo]] - `uses` [INFERRED]
- [[Create a new conversation session._1]] - `uses` [INFERRED]
- [[Detailed component health status for diagnostics.]] - `uses` [INFERRED]
- [[Enhanced health check with system metrics.]] - `uses` [INFERRED]
- [[ErrorResponse]] - `uses` [INFERRED]
- [[Execute an arbitrary LangSmith MCP tool.]] - `uses` [INFERRED]
- [[Extract numeric source scores from QA response sources.]] - `uses` [INFERRED]
- [[FastAPI application for Healthcare QA Chatbot.  Enhanced with - Single model T]] - `uses` [INFERRED]
- [[Feedback module for user feedback collection.]] - `uses` [INFERRED]
- [[FeedbackRequest]] - `uses` [INFERRED]
- [[FeedbackResponse]] - `uses` [INFERRED]
- [[Fetch recent runs and summarize primary qualitylatency failure patterns.]] - `uses` [INFERRED]
- [[Get conversation history for a session.]] - `uses` [INFERRED]
- [[Get or create a pipeline for the given model choice.]] - `uses` [INFERRED]
- [[HealthResponse_1]] - `uses` [INFERRED]
- [[Initialize conversation manager.]] - `uses` [INFERRED]
- [[Initialize feedback collector and trajectory logger.]] - `uses` [INFERRED]
- [[Initialize safety guardrails.]] - `uses` [INFERRED]
- [[LangSmithImprovementReportRequest]] - `uses` [INFERRED]
- [[LangSmithToolRequest]] - `uses` [INFERRED]
- [[Lazy load LangChain pipeline with specified model.      Args         model_choi]] - `uses` [INFERRED]
- [[Lazy load LangGraph pipeline with specified model.      Args         model_choi]] - `uses` [INFERRED]
- [[Load shared components (embedder, vector store, retriever) once.]] - `uses` [INFERRED]
- [[Parse API_KEYS env var into a {key scope} mapping.]] - `uses` [INFERRED]
- [[Persist RL-ready response trajectories.      Each line is one JSON object keyed]] - `rationale_for` [EXTRACTED]
- [[Persist a response trajectory for RL data collection.]] - `uses` [INFERRED]
- [[QuestionRequest_1]] - `uses` [INFERRED]
- [[Read LangSmith MCP settings from environment.]] - `uses` [INFERRED]
- [[Resolve adapter path with a sensible local default when present.]] - `uses` [INFERRED]
- [[Return LangSmith MCP connectivity and available tool names.]] - `uses` [INFERRED]
- [[Return a short non-reversible key identifier for audit logs.]] - `uses` [INFERRED]
- [[Return aggregate user feedback statistics.]] - `uses` [INFERRED]
- [[Return available models and their descriptions.]] - `uses` [INFERRED]
- [[Return low-latency knobs (env-overridable; monkeypatch-friendly for tests).]] - `uses` [INFERRED]
- [[Return structured JSON for HTTP exceptions.]] - `uses` [INFERRED]
- [[SafetyInfo]] - `uses` [INFERRED]
- [[Shared orchestration logic for standard and streaming requests.]] - `uses` [INFERRED]
- [[SourceInfo]] - `uses` [INFERRED]
- [[Stream a medical answer token by token (NDJSON).      Event types emitted]] - `uses` [INFERRED]
- [[Structured error response returned by all error handlers.]] - `uses` [INFERRED]
- [[Submit user feedback tied to a prior response_id.      Stores ratinglabels in F]] - `uses` [INFERRED]
- [[Tests for RL trajectory logging utilities.]] - `uses` [INFERRED]
- [[True when no keys are configured at all (open dev mode).]] - `uses` [INFERRED]
- [[Use tighter token budgets for low-latency and high-precision query types.]] - `uses` [INFERRED]
- [[Verify that the request carries a valid admin-scoped key.      Only keys with sc]] - `uses` [INFERRED]
- [[Verify that the request carries a valid read-or-admin key.      Accepts any key]] - `uses` [INFERRED]
- [[_init_feedback_system()]] - `calls` [INFERRED]
- [[test_get_missing_trajectory_returns_none()]] - `calls` [INFERRED]
- [[test_log_and_get_trajectory()]] - `calls` [INFERRED]
- [[test_log_requires_response_id()]] - `calls` [INFERRED]
- [[trajectory_logger.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/INFERRED #community/Community_2