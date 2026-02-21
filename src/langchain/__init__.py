"""
LangChain integration package for Healthcare QA Chatbot.

This package provides LangChain-compatible wrappers for the existing
components, enabling LCEL-based pipeline composition.

Usage:
    from src.langchain import (
        LangChainHealthcareQAPipeline,
        create_langchain_pipeline,
        LangChainMedicalLLM,
        LangChainHybridRetriever
    )
    
    # Create pipeline with existing components
    pipeline = create_langchain_pipeline(
        retriever=my_hybrid_retriever,
        llm=my_medical_llm,
        confidence_scorer=my_scorer
    )
    
    # Answer a question
    result = pipeline.answer("What is diabetes?")
"""

from src.langchain.langchain_llm import (
    LangChainMedicalLLM,
    LangChainMedicalLLMFromExisting
)

from src.langchain.langchain_retriever import (
    LangChainHybridRetriever,
    format_docs_as_context,
    docs_to_retrieved_documents
)

from src.langchain.langchain_prompts import (
    MEDICAL_QA_CHAT_TEMPLATE,
    EXPLAINABLE_QA_CHAT_TEMPLATE,
    SIMPLE_QA_TEMPLATE,
    MEDICAL_DISCLAIMER,
    get_prompt_template,
    format_context_for_prompt
)

from src.langchain.langchain_pipeline import (
    LangChainHealthcareQAPipeline,
    LangChainQAResult,
    create_langchain_pipeline
)


__all__ = [
    # LLM Wrappers
    "LangChainMedicalLLM",
    "LangChainMedicalLLMFromExisting",
    
    # Retriever Wrappers
    "LangChainHybridRetriever",
    "format_docs_as_context",
    "docs_to_retrieved_documents",
    
    # Prompt Templates
    "MEDICAL_QA_CHAT_TEMPLATE",
    "EXPLAINABLE_QA_CHAT_TEMPLATE", 
    "SIMPLE_QA_TEMPLATE",
    "MEDICAL_DISCLAIMER",
    "get_prompt_template",
    "format_context_for_prompt",
    
    # Pipeline
    "LangChainHealthcareQAPipeline",
    "LangChainQAResult",
    "create_langchain_pipeline",
]
