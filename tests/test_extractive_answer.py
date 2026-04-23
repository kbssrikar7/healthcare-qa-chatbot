from src.pipeline.qa_pipeline import HealthcareQAPipeline


class _Doc:
    def __init__(self, content: str, score: float = 0.02):
        self.content = content
        self.source = "test"
        self.score = score
        self.metadata = {}


class _DummyRetriever:
    def retrieve_with_context(self, query: str, k: int = 5):
        docs = [_Doc("placeholder")]
        return docs[:k], docs[0].content


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
        return self._Gen("placeholder")


def _mk_pipeline() -> HealthcareQAPipeline:
    return HealthcareQAPipeline(
        retriever=_DummyRetriever(),
        llm=_DummyLLM(),
        prompt_manager=_DummyPromptManager(),
        enable_grounding_gate=False,
    )


def test_causation_extractive_prioritizes_cause_sentences():
    p = _mk_pipeline()
    docs = [
        _Doc("Fatigue and light sensitivity can happen before migraine."),
        _Doc("Acute migraine can be caused by vascular constriction and trigger-related neural changes."),
        _Doc("Common triggers include sleep deprivation and stress."),
    ]
    ans = p._build_extractive_answer("What causes acute migraine?", docs)
    assert ans is not None
    low = ans.lower()
    assert "caused" in low or "trigger" in low


def test_is_causation_question_detection():
    assert HealthcareQAPipeline._is_causation_question("What causes acute migraine?")
    assert HealthcareQAPipeline._is_causation_question("Why does migraine happen?")
    assert not HealthcareQAPipeline._is_causation_question("What is migraine?")
