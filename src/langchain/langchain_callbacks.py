"""
LangChain Callbacks for Monitoring and Observability.

Provides callback handlers for logging, tracing, and monitoring
LangChain-based medical QA interactions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseCallbackHandler = object
    LLMResult = object
    LANGCHAIN_AVAILABLE = False


class MedicalQACallbackHandler(BaseCallbackHandler):
    """
    Callback handler for logging and monitoring medical QA interactions.

    Logs:
    - Chain start/end with timing
    - Retrieval results
    - LLM generation details
    - Errors for debugging
    """

    def __init__(
        self,
        log_file: str = "logs/langchain_trace.jsonl",
        log_to_console: bool = True,
        include_inputs: bool = False,
        include_outputs: bool = False,
    ):
        """
        Initialize callback handler.

        Args:
            log_file: Path to JSONL log file
            log_to_console: Whether to also log to console
            include_inputs: Include full inputs in logs (privacy concern)
            include_outputs: Include full outputs in logs
        """
        super().__init__()
        self.log_file = Path(log_file)
        self.log_to_console = log_to_console
        self.include_inputs = include_inputs
        self.include_outputs = include_outputs

        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.run_id = None
        self.start_time = None
        self.chain_depth = 0

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Called when chain starts."""
        self.chain_depth += 1
        if self.chain_depth == 1:  # Top-level chain
            self.run_id = str(run_id) if run_id else "unknown"
            self.start_time = datetime.now()

            log_data = {
                "event": "chain_start",
                "run_id": self.run_id,
                "chain_name": (
                    serialized.get("name", serialized.get("id", ["unknown"])[-1])
                    if serialized
                    else "unknown"
                ),
            }

            if self.include_inputs:
                log_data["inputs"] = {k: str(v)[:200] for k, v in inputs.items()}
            else:
                # Only log question length for privacy
                if "question" in inputs:
                    log_data["question_length"] = len(str(inputs["question"]))

            self._log(log_data)

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: Optional[str] = None, **kwargs
    ) -> None:
        """Called when chain ends."""
        self.chain_depth -= 1
        if self.chain_depth == 0:  # Top-level chain
            duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

            log_data = {
                "event": "chain_end",
                "run_id": self.run_id,
                "duration_seconds": round(duration, 3),
                "has_answer": bool(outputs.get("answer")),
                "is_answerable": outputs.get("is_answerable", True),
            }

            # Include confidence if available
            if "confidence" in outputs:
                conf = outputs["confidence"]
                log_data["confidence_score"] = conf.get("score", 0)
                log_data["confidence_level"] = conf.get("level", "unknown")

            if self.include_outputs:
                log_data["answer_preview"] = str(outputs.get("answer", ""))[:100]

            self._log(log_data)

    def on_chain_error(self, error: Exception, *, run_id: Optional[str] = None, **kwargs) -> None:
        """Called when chain errors."""
        self._log(
            {
                "event": "chain_error",
                "run_id": self.run_id,
                "error_type": type(error).__name__,
                "error_message": str(error)[:200],
            }
        )
        logger.error(f"Chain error: {error}")

    def on_retriever_start(
        self, serialized: Dict[str, Any], query: str, *, run_id: Optional[str] = None, **kwargs
    ) -> None:
        """Called when retriever starts."""
        self._log({"event": "retrieval_start", "run_id": self.run_id, "query_length": len(query)})

    def on_retriever_end(self, documents: List, *, run_id: Optional[str] = None, **kwargs) -> None:
        """Called when retriever ends."""
        scores = []
        for doc in documents:
            if hasattr(doc, "metadata"):
                scores.append(doc.metadata.get("score", 0))

        self._log(
            {
                "event": "retrieval_complete",
                "run_id": self.run_id,
                "num_documents": len(documents),
                "top_score": max(scores) if scores else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
            }
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Called when LLM starts."""
        self._log(
            {
                "event": "llm_start",
                "run_id": self.run_id,
                "num_prompts": len(prompts),
                "prompt_length": sum(len(p) for p in prompts),
            }
        )

    def on_llm_end(self, response: LLMResult, *, run_id: Optional[str] = None, **kwargs) -> None:
        """Called when LLM ends."""
        output_length = 0
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    output_length += len(gen.text) if hasattr(gen, "text") else 0

        self._log({"event": "llm_complete", "run_id": self.run_id, "output_length": output_length})

    def on_llm_error(self, error: Exception, *, run_id: Optional[str] = None, **kwargs) -> None:
        """Called when LLM errors."""
        self._log(
            {
                "event": "llm_error",
                "run_id": self.run_id,
                "error_type": type(error).__name__,
                "error_message": str(error)[:200],
            }
        )

    def _log(self, data: Dict):
        """Write log entry."""
        data["timestamp"] = datetime.now().isoformat()

        # Write to file
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write log: {e}")

        # Console logging
        if self.log_to_console:
            event = data.get("event", "unknown")
            if event == "chain_start":
                logger.info(f"🔵 Chain started: {data.get('chain_name', 'unknown')}")
            elif event == "chain_end":
                duration = data.get("duration_seconds", 0)
                logger.info(f"🟢 Chain completed in {duration:.2f}s")
            elif event == "chain_error":
                logger.error(f"🔴 Chain error: {data.get('error_message', 'unknown')}")
            elif event == "retrieval_complete":
                logger.info(
                    f"📚 Retrieved {data.get('num_documents', 0)} documents (top score: {data.get('top_score', 0):.2f})"
                )


class MetricsAggregator:
    """
    Aggregate metrics from callback logs for monitoring dashboards.
    """

    def __init__(self, log_file: str = "logs/langchain_trace.jsonl"):
        self.log_file = Path(log_file)

    def get_summary(self, last_n_hours: int = 24) -> Dict:
        """Get summary metrics for the last N hours."""
        if not self.log_file.exists():
            return {"error": "No log file found"}

        cutoff = datetime.now().timestamp() - (last_n_hours * 3600)

        chains_completed = 0
        chains_failed = 0
        total_duration = 0
        confidence_scores = []
        retrieval_counts = []

        with open(self.log_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    ts = datetime.fromisoformat(data.get("timestamp", "")).timestamp()

                    if ts < cutoff:
                        continue

                    event = data.get("event")
                    if event == "chain_end":
                        chains_completed += 1
                        total_duration += data.get("duration_seconds", 0)
                        if "confidence_score" in data:
                            confidence_scores.append(data["confidence_score"])
                    elif event == "chain_error":
                        chains_failed += 1
                    elif event == "retrieval_complete":
                        retrieval_counts.append(data.get("num_documents", 0))

                except Exception as e:
                    logger.warning(f"Failed to parse log entry during analytics aggregation: {e}")
                    continue

        return {
            "period_hours": last_n_hours,
            "total_queries": chains_completed + chains_failed,
            "success_rate": (
                chains_completed / (chains_completed + chains_failed)
                if (chains_completed + chains_failed) > 0
                else 0
            ),
            "avg_duration_seconds": (
                total_duration / chains_completed if chains_completed > 0 else 0
            ),
            "avg_confidence": (
                sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            ),
            "avg_docs_retrieved": (
                sum(retrieval_counts) / len(retrieval_counts) if retrieval_counts else 0
            ),
        }
