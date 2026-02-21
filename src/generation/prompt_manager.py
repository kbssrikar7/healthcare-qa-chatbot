"""
Prompt management for medical question answering.

Uses TinyLlama chat template format for proper instruction following.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """A prompt template with formatting."""
    template: str
    name: str
    description: str

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
        "6. Keep your answer short and factual.\n"
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

    def __init__(self, default_prompt: str = "medical_qa"):
        self.templates = {
            "medical_qa": PromptTemplate(
                template=self.MEDICAL_QA_PROMPT,
                name="Medical QA",
                description="Strict context-grounded medical QA prompt"
            ),
            "explainable": PromptTemplate(
                template=self.EXPLAINABLE_QA_PROMPT,
                name="Explainable QA",
                description="Context-grounded prompt with citation requests"
            ),
            "simple": PromptTemplate(
                template=self.SIMPLE_QA_PROMPT,
                name="Simple QA",
                description="Minimal strict-grounding prompt for fast responses"
            )
        }
        self.default_prompt = default_prompt
    
    def build_prompt(
        self,
        question: str,
        context: str,
        template_name: Optional[str] = None,
        additional_instructions: Optional[str] = None
    ) -> str:
        """Build a complete prompt."""
        template_name = template_name or self.default_prompt
        template = self.templates.get(template_name, self.templates["medical_qa"])
        
        prompt = template.template.format(
            question=question,
            context=context
        )
        
        if additional_instructions:
            prompt += f"\n\nAdditional instructions: {additional_instructions}"
        
        return prompt
    
    def build_context_from_documents(
        self,
        documents: List[Dict],
        max_length: int = 2000
    ) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.get("content", str(doc))
            source = doc.get("source", f"Source {i+1}")
            
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
        self.templates[name] = PromptTemplate(
            template=template,
            name=name,
            description=description
        )
