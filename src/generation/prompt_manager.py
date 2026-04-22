"""
Prompt management for medical question answering.

Uses TinyLlama chat template format for proper instruction following.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PromptTemplate:
    """A prompt template with formatting."""

    template: str
    name: str
    description: str


def _inject_context_question(template: str, context: str, question: str) -> str:
    """
    Insert context and question without ``str.format`` (user/brace-safe).
    Expects each placeholder at most once, as in the bundled templates.
    """
    if "{context}" not in template and "{question}" not in template:
        return template

    # Order-agnostic replacement while avoiding accidental replacement inside
    # user-provided text payloads.
    out = template.replace("{context}", "__CTX_SENTINEL__").replace(
        "{question}", "__Q_SENTINEL__"
    )
    out = out.replace("__CTX_SENTINEL__", context)
    out = out.replace("__Q_SENTINEL__", question)
    return out


class MedicalPromptManager:
    """Manage prompts for medical QA."""

    MEDICAL_QA_PROMPT = (
        "<|system|>\n"
        "You are a medical fact extractor. Read the REFERENCE TEXT and answer the question.\n"
        "RULES:\n"
        "1. ONLY use facts from the REFERENCE TEXT below.\n"
        "2. Do NOT add any information from your own knowledge.\n"
        "3. Do NOT guess or make up information.\n"
        "4. Do NOT copy greetings, names, or sign-offs from the text.\n"
        "5. If the reference text does not answer the question, say: "
        "I do not have enough information in my references to answer this.\n"
        "6. Keep your answer short and factual — 2 to 4 sentences maximum.\n"
        "7. STOP after answering. Do NOT write another Question or Answer.\n"
        "8. Do NOT reproduce multiple-choice options or exam questions.\n"
        "9. Do NOT explain how drugs work or describe mechanisms unless the REFERENCE TEXT explicitly states them.\n"
        "</s>\n"
        "<|user|>\n"
        "REFERENCE TEXT:\n"
        "{context}\n\n"
        "QUESTION: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
    )

    EXPLAINABLE_QA_PROMPT = (
        "<|system|>\n"
        "You are a medical fact extractor. Read the REFERENCE TEXT and answer the question.\n"
        "Cite which part of the reference text supports your answer.\n"
        "RULES:\n"
        "1. ONLY use facts from the REFERENCE TEXT below.\n"
        "2. Do NOT add any information from your own knowledge.\n"
        "3. Do NOT guess or make up information.\n"
        "4. Do NOT copy greetings, names, or sign-offs from the text.\n"
        "5. If the reference text does not answer the question, say: "
        "I do not have enough information in my references to answer this.\n"
        "6. Keep your answer short and factual — 2 to 4 sentences maximum.\n"
        "7. STOP after answering. Do NOT write another Question or Answer.\n"
        "8. Do NOT reproduce multiple-choice options or exam questions.\n"
        "9. Do NOT explain how drugs work or describe mechanisms unless the REFERENCE TEXT explicitly states them.\n"
        "</s>\n"
        "<|user|>\n"
        "REFERENCE TEXT:\n"
        "{context}\n\n"
        "QUESTION: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
    )

    SIMPLE_QA_PROMPT = (
        "<|system|>\n"
        "Answer the question using ONLY the reference text. Do NOT add your own knowledge.\n"
        "</s>\n"
        "<|user|>\n"
        "REFERENCE TEXT: {context}\n\n"
        "QUESTION: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
    )

    CONCISE_FACTOID_PROMPT = (
        "<|system|>\n"
        "You are a medical fact extractor. Answer in 1-2 sentences maximum.\n"
        "RULES:\n"
        "1. ONLY use facts from the REFERENCE TEXT.\n"
        "2. For 'drug of choice' or 'mechanism' questions, state the answer directly then stop.\n"
        "3. Do NOT add caveats, disclaimers, or treatment discussion unless in the REFERENCE TEXT.\n"
        "4. If the reference text does not answer the question, say: "
        "I do not have enough information.\n"
        "5. Do NOT reproduce multiple-choice options.\n"
        "</s>\n"
        "<|user|>\n"
        "REFERENCE TEXT:\n"
        "{context}\n\n"
        "QUESTION: {question}\n"
        "</s>\n"
        "<|assistant|>\n"
    )

    BIOMISTRAL_QA_PROMPT = (
        "<s>[INST] You are a medical assistant. Read the REFERENCE TEXT and answer the question.\n"
        "RULES:\n"
        "1. Use relevant facts from the REFERENCE TEXT to answer.\n"
        "2. The reference text may contain Q&A pairs from similar cases — extract useful advice from them.\n"
        "3. Do NOT copy patient names, greetings, or sign-offs.\n"
        "4. Keep your answer practical and factual — 2 to 4 sentences.\n"
        "5. STOP after answering. Do NOT write another Question or Answer.\n"
        "6. Do NOT reproduce multiple-choice options or exam questions.\n\n"
        "REFERENCE TEXT:\n"
        "{context}\n\n"
        "QUESTION: {question} [/INST]"
    )

    def __init__(self, default_prompt: str = "medical_qa"):
        self.templates = {
            "medical_qa": PromptTemplate(
                template=self.MEDICAL_QA_PROMPT,
                name="Medical QA",
                description="Strict context-grounded medical QA prompt",
            ),
            "explainable": PromptTemplate(
                template=self.EXPLAINABLE_QA_PROMPT,
                name="Explainable QA",
                description="Context-grounded prompt with citation requests",
            ),
            "simple": PromptTemplate(
                template=self.SIMPLE_QA_PROMPT,
                name="Simple QA",
                description="Minimal strict-grounding prompt for fast responses",
            ),
            "concise_factoid": PromptTemplate(
                template=self.CONCISE_FACTOID_PROMPT,
                name="Concise Factoid QA",
                description="Ultra-short 1-2 sentence answers for drug/mechanism factoid questions",
            ),
        }
        self.default_prompt = default_prompt

    def build_prompt(
        self,
        question: str,
        context: str,
        template_name: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """Build a complete prompt."""
        template_name = template_name or self.default_prompt

        if model_name == "biomistral":
            prompt = _inject_context_question(self.BIOMISTRAL_QA_PROMPT, context, question)
        else:
            template = self.templates.get(template_name, self.templates["medical_qa"])
            prompt = _inject_context_question(template.template, context, question)

        if additional_instructions:
            prompt += f"\n\nAdditional instructions: {additional_instructions}"

        return prompt

    def build_context_from_documents(self, documents: List[Dict], max_length: int = 2000) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents):
            content = doc.get("content", str(doc))
            source = doc.get("source", f"Source {i + 1}")

            entry = f"[{source}]: {content}"

            if total_length + len(entry) > max_length:
                break

            context_parts.append(entry)
            total_length += len(entry)

        return "\n\n".join(context_parts)

    def get_medical_disclaimer(self) -> str:
        """Get the standard medical disclaimer."""
        return (
            "MEDICAL DISCLAIMER: This information is for educational purposes only "
            "and is NOT a substitute for professional medical advice, diagnosis, or "
            "treatment. Always seek the advice of your physician or other qualified "
            "health provider with any questions you may have regarding a medical "
            "condition. Never disregard professional medical advice or delay in "
            "seeking it because of something you have read here."
        )

    def add_template(self, name: str, template: str, description: str = ""):
        """Add a custom prompt template."""
        self.templates[name] = PromptTemplate(template=template, name=name, description=description)
