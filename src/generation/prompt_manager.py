"""
Prompt management for medical question answering.
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
    
    MEDICAL_QA_PROMPT = "\n".join([
        "You are a medical information assistant. Your job is to extract",
        "medical facts from the reference material and present them clearly.",
        "",
        "STRICT RULES you MUST follow:",
        "- DO NOT copy names, greetings, salutations, or letter formats from references",
        "- DO NOT address the user as Dear anyone or sign off as a doctor",
        "- DO NOT roleplay as a doctor giving a personal consultation",
        "- DO NOT mention medications or treatments NOT relevant to the question",
        "- ONLY extract medical facts that DIRECTLY answer the question asked",
        "- Present information as clear bullet points or short paragraphs",
        "- Use simple, patient-friendly language",
        "- If the references lack relevant information, say so honestly",
        "",
        "Reference Material:",
        "{context}",
        "",
        "Question: {question}",
        "",
        "Factual Answer:",
    ])

    EXPLAINABLE_QA_PROMPT = "\n".join([
        "You are a medical information assistant providing transparent, evidence-based answers.",
        "",
        "STRICT RULES:",
        "- DO NOT copy names, greetings, or conversational patterns from the references",
        "- DO NOT roleplay as a doctor or sign letters",
        "- Extract ONLY relevant medical facts from the reference material",
        "- Cite which reference supports each point",
        "- Note any limitations or uncertainties",
        "",
        "Reference Material:",
        "{context}",
        "",
        "Question: {question}",
        "",
        "Evidence-Based Answer:",
    ])

    SIMPLE_QA_PROMPT = "\n".join([
        "Extract medical facts from the references to answer the question.",
        "Do NOT copy names or greetings from references. Be concise and factual.",
        "",
        "References: {context}",
        "",
        "Question: {question}",
        "",
        "Answer:",
    ])

    def __init__(self, default_prompt: str = "medical_qa"):
        self.templates = {
            "medical_qa": PromptTemplate(
                template=self.MEDICAL_QA_PROMPT,
                name="Medical QA",
                description="Standard medical question answering prompt"
            ),
            "explainable": PromptTemplate(
                template=self.EXPLAINABLE_QA_PROMPT,
                name="Explainable QA",
                description="Prompt that requests confidence and citations"
            ),
            "simple": PromptTemplate(
                template=self.SIMPLE_QA_PROMPT,
                name="Simple QA",
                description="Minimal prompt for fast responses"
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
