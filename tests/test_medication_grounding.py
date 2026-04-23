from dataclasses import dataclass

from src.pipeline.qa_pipeline import HealthcareQAPipeline


@dataclass
class _Doc:
    content: str
    source: str = "test"
    score: float = 0.02
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class _DummyRetriever:
    def retrieve_with_context(self, query: str, k: int = 5):
        docs = [_Doc(content="paracetamol can cause nausea and rash", source="med")]
        return docs[:k], "\n\n".join(d.content for d in docs[:k])


class _DummyPromptManager:
    def build_prompt(self, question: str, context: str, **kwargs):
        return f"Q: {question}\nC: {context}"

    def get_medical_disclaimer(self):
        return "medical disclaimer"


class _DummyLLM:
    model_name = "tinyllama"

    class _Gen:
        def __init__(self, response: str):
            self.response = response
            self.probabilities = None

    def generate(self, *args, **kwargs):
        return self._Gen("Paracetamol can cause nausea and rash.")


def _mk_pipeline() -> HealthcareQAPipeline:
    return HealthcareQAPipeline(
        retriever=_DummyRetriever(),
        llm=_DummyLLM(),
        prompt_manager=_DummyPromptManager(),
        enable_grounding_gate=False,
    )


def test_prepare_retrieval_queries_keeps_brand_and_generic_for_dolo():
    pipeline = _mk_pipeline()
    q = "What are the side effects of Dolo 650?"
    queries = pipeline._prepare_retrieval_queries(
        question=q,
        base_query=q,
        expanded_queries=[],
        conversation_context=None,
    )
    assert len(queries) == 2
    assert any("dolo 650" in x.lower() for x in queries)
    assert any("paracetamol" in x.lower() or "acetaminophen" in x.lower() for x in queries)


def test_medication_entity_verdict_rejects_wrong_drug_answer():
    pipeline = _mk_pipeline()
    docs = [_Doc(content="Paracetamol side effects include nausea and abdominal pain.")]
    answer = (
        "Based on retrieved evidence: older antihistamines cause dry mouth and blurred vision."
    )
    verdict = pipeline._medication_entity_verdict(
        "What are the side effects of Dolo 650?", answer, docs
    )
    assert verdict["supported"] is False
    assert verdict["reason"] == "answer_not_anchored_to_asked_medication"


def test_medication_entity_verdict_rejects_unrelated_retrieval_context():
    pipeline = _mk_pipeline()
    docs = [_Doc(content="Cetirizine side effects include drowsiness and dry mouth.")]
    answer = "Paracetamol side effects may include nausea."
    verdict = pipeline._medication_entity_verdict(
        "What are the side effects of Dolo 650?", answer, docs
    )
    assert verdict["supported"] is False
    assert verdict["reason"] == "retrieved_context_not_about_asked_medication"


def test_filter_docs_by_medication_terms_keeps_only_matching_docs():
    docs = [
        _Doc(content="Paracetamol may rarely cause rash."),
        _Doc(content="Cetirizine side effects include dry mouth."),
        _Doc(content="Dolophen side effects include dizziness."),
    ]
    kept = HealthcareQAPipeline._filter_docs_by_medication_terms(
        docs, ["paracetamol", "acetaminophen", "dolo"]
    )
    assert len(kept) == 1
    assert "paracetamol" in kept[0].content.lower()


def test_contains_medication_term_uses_word_boundaries():
    assert HealthcareQAPipeline._contains_medication_term("dolo 650 tablet", "dolo")
    assert not HealthcareQAPipeline._contains_medication_term("dolophen tablet", "dolo")
    assert HealthcareQAPipeline._contains_medication_term(
        "paracetamols may cause side effects", "paracetamol"
    )


def test_format_medication_extractive_answer_expands_effect_list():
    text = (
        "Paracetamol: Side effects from paracetamol are rare, but can include: "
        "an allergic reaction, rash and swelling, flushing."
    )
    formatted = HealthcareQAPipeline._format_medication_extractive_answer(
        text, ["paracetamol", "acetaminophen", "dolo"]
    )
    assert formatted is not None
    assert "paracetamol" in formatted.lower()
    assert "rare" in formatted.lower()
    assert "reported side effects include" in formatted.lower()


def test_is_medication_side_effect_query_detects_dolo_questions():
    assert HealthcareQAPipeline._is_medication_side_effect_query(
        "What are the side effects of Dolo 650?"
    )
    assert not HealthcareQAPipeline._is_medication_side_effect_query(
        "What is hypertension?"
    )
