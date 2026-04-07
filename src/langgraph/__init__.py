"""
LangGraph integration package for Healthcare QA Chatbot.

Exports are resolved lazily so importing ``src.langgraph`` does not eagerly
require optional LangGraph dependencies unless the caller actually uses them.
"""

from importlib import import_module

__all__ = [
    "HealthcareRAGState",
    "create_initial_state",
    "HealthcareRAGNodes",
    "MEDICAL_DISCLAIMER",
    "UNANSWERABLE_RESPONSE",
    "route_after_grading",
    "route_after_verify",
    "route_after_xai",
    "should_continue_retrieval",
    "LangGraphHealthcareQAPipeline",
    "LangGraphQAResult",
    "create_langgraph_pipeline",
]

_EXPORT_MAP = {
    "HealthcareRAGState": ("src.langgraph.langgraph_state", "HealthcareRAGState"),
    "create_initial_state": ("src.langgraph.langgraph_state", "create_initial_state"),
    "HealthcareRAGNodes": ("src.langgraph.langgraph_nodes", "HealthcareRAGNodes"),
    "MEDICAL_DISCLAIMER": ("src.langgraph.langgraph_nodes", "MEDICAL_DISCLAIMER"),
    "UNANSWERABLE_RESPONSE": (
        "src.langgraph.langgraph_nodes",
        "UNANSWERABLE_RESPONSE",
    ),
    "route_after_grading": ("src.langgraph.langgraph_routing", "route_after_grading"),
    "route_after_verify": ("src.langgraph.langgraph_routing", "route_after_verify"),
    "route_after_xai": ("src.langgraph.langgraph_routing", "route_after_xai"),
    "should_continue_retrieval": (
        "src.langgraph.langgraph_routing",
        "should_continue_retrieval",
    ),
    "LangGraphHealthcareQAPipeline": (
        "src.langgraph.langgraph_pipeline",
        "LangGraphHealthcareQAPipeline",
    ),
    "LangGraphQAResult": ("src.langgraph.langgraph_pipeline", "LangGraphQAResult"),
    "create_langgraph_pipeline": (
        "src.langgraph.langgraph_pipeline",
        "create_langgraph_pipeline",
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
