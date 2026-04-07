"""
Focused regression tests for the local-safe rollout changes.
"""
from pathlib import Path
import time

import numpy as np

from evaluation.compute_calibration import build_calibration_pairs
from src.embeddings.embedding_models import MedicalEmbedder
from src.embeddings.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever


def test_query_embedding_cache_is_reused():
    """Repeated queries should hit the per-instance embedding cache."""
    embedder = MedicalEmbedder.__new__(MedicalEmbedder)
    embedder._embed_cache = {}
    embedder._embed_cache_order = []
    embedder._EMBED_CACHE_SIZE = 256

    def fake_embed(texts, **kwargs):
        time.sleep(0.02)
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    embedder.embed = fake_embed

    t0 = time.perf_counter()
    first = embedder.embed_query("diabetes symptoms")
    first_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    second = embedder.embed_query("diabetes symptoms")
    second_elapsed = time.perf_counter() - t1

    assert np.allclose(first, second)
    assert second_elapsed * 10 < first_elapsed


def test_vector_store_embedding_metadata_sidecar_and_legacy_fallback(tmp_path):
    """Vector store should verify metadata from both sidecar and legacy sentinel."""
    store = VectorStore(
        collection_name="test_meta",
        persist_directory=str(tmp_path / "kb"),
    )

    store.record_embedding_model("all-minilm", 384)
    assert store.verify_embedding_compatibility("all-minilm", 384) is True

    sidecar = Path(store._meta_path)
    assert sidecar.exists()

    sidecar.unlink()
    assert store.verify_embedding_compatibility("all-minilm", 384) is True
    assert store.verify_embedding_compatibility("all-minilm", 768) is False


def test_retriever_exposes_deterministic_stage_timings():
    """Retriever should publish the named timing keys used by the pipeline."""
    class DummyEmbedder:
        def embed_query(self, query):
            return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    class DummyVectorStore:
        def search(self, query_embedding, n_results=10):
            return {
                "documents": [["Document about diabetes"]],
                "distances": [[0.2]],
                "metadatas": [[{"source": "MedQuAD"}]],
                "ids": [["doc_1"]],
            }

    retriever = HybridRetriever(
        embedder=DummyEmbedder(),
        vector_store=DummyVectorStore(),
    )

    docs = retriever.retrieve("diabetes", k=1, use_hybrid=False, use_reranking=False)

    assert len(docs) == 1
    assert "embedding_ms" in retriever.last_timings
    assert "chroma_query_ms" in retriever.last_timings
    assert "safety_filter_ms" in retriever.last_timings
    assert "total_ms" in retriever.last_timings


def test_calibration_pairs_use_answer_text_when_available():
    """Calibration should use matched test cases and real answer text when present."""
    trajectories = [
        {
            "question": "Symptoms of Type 2 Diabetes",
            "answer": "Common symptoms include thirst and frequent urination.",
            "outcome": {
                "confidence_score": 0.8,
                "factual_consistency": None,
                "answer_length": 60,
                "num_sources_returned": 3,
                "is_answerable": True,
            },
            "retrieval": {"top_score": 0.015},
        }
    ]
    test_cases = {
        "Symptoms of Type 2 Diabetes": {
            "expected_keywords": ["thirst", "frequent urination", "fatigue"],
        }
    }

    confidences, labels, sources = build_calibration_pairs(trajectories, test_cases)

    assert confidences == [0.8]
    assert labels == [1]
    assert sources == ["keyword"]


def test_calibration_pairs_fall_back_to_matched_proxy_without_answer_text():
    """Calibration should still derive labels when trajectory rows omit answer text."""
    trajectories = [
        {
            "question": "Treatment for Hypertension",
            "outcome": {
                "confidence_score": 0.72,
                "factual_consistency": None,
                "answer_length": 120,
                "num_sources_returned": 3,
                "is_answerable": True,
            },
            "retrieval": {"top_score": 0.012},
        }
    ]
    test_cases = {
        "Treatment for Hypertension": {
            "expected_keywords": ["blood pressure", "lifestyle", "medication"],
        }
    }

    confidences, labels, sources = build_calibration_pairs(trajectories, test_cases)

    assert confidences == [0.72]
    assert labels == [1]
    assert sources == ["matched_proxy"]
