"""
LangChain integration package for Healthcare QA Chatbot.

This package keeps imports lazy so the rest of the codebase can import
``src.langchain.<module>`` safely even when optional LangChain dependencies
are unavailable in the active environment.
"""

from importlib import import_module

__all__ = [
    "LangChainMedicalLLM",
    "LangChainMedicalLLMFromExisting",
    "LangChainHybridRetriever",
    "format_docs_as_context",
    "docs_to_retrieved_documents",
    "MEDICAL_QA_CHAT_TEMPLATE",
    "EXPLAINABLE_QA_CHAT_TEMPLATE",
    "SIMPLE_QA_TEMPLATE",
    "MEDICAL_DISCLAIMER",
    "get_prompt_template",
    "format_context_for_prompt",
    "LangChainHealthcareQAPipeline",
    "LangChainQAResult",
    "create_langchain_pipeline",
]

_EXPORT_MAP = {
    "LangChainMedicalLLM": ("src.langchain.langchain_llm", "LangChainMedicalLLM"),
    "LangChainMedicalLLMFromExisting": (
        "src.langchain.langchain_llm",
        "LangChainMedicalLLMFromExisting",
    ),
    "LangChainHybridRetriever": (
        "src.langchain.langchain_retriever",
        "LangChainHybridRetriever",
    ),
    "format_docs_as_context": (
        "src.langchain.langchain_retriever",
        "format_docs_as_context",
    ),
    "docs_to_retrieved_documents": (
        "src.langchain.langchain_retriever",
        "docs_to_retrieved_documents",
    ),
    "MEDICAL_QA_CHAT_TEMPLATE": (
        "src.langchain.langchain_prompts",
        "MEDICAL_QA_CHAT_TEMPLATE",
    ),
    "EXPLAINABLE_QA_CHAT_TEMPLATE": (
        "src.langchain.langchain_prompts",
        "EXPLAINABLE_QA_CHAT_TEMPLATE",
    ),
    "SIMPLE_QA_TEMPLATE": ("src.langchain.langchain_prompts", "SIMPLE_QA_TEMPLATE"),
    "MEDICAL_DISCLAIMER": ("src.langchain.langchain_prompts", "MEDICAL_DISCLAIMER"),
    "get_prompt_template": ("src.langchain.langchain_prompts", "get_prompt_template"),
    "format_context_for_prompt": (
        "src.langchain.langchain_prompts",
        "format_context_for_prompt",
    ),
    "LangChainHealthcareQAPipeline": (
        "src.langchain.langchain_pipeline",
        "LangChainHealthcareQAPipeline",
    ),
    "LangChainQAResult": ("src.langchain.langchain_pipeline", "LangChainQAResult"),
    "create_langchain_pipeline": (
        "src.langchain.langchain_pipeline",
        "create_langchain_pipeline",
    ),
}


def __getattr__(name):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
