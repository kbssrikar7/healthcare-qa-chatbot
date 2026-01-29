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
    
    # Default medical QA prompt
    MEDICAL_QA_PROMPT = """You are a knowledgeable medical assistant. Your role is to provide accurate, helpful health information based on the context provided. Always be clear about limitations and recommend consulting healthcare professionals for medical decisions.

### Context:
{context}

### Question:
{question}

### Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Use clear, patient-friendly language
4. Include relevant medical terms with explanations
5. NEVER provide diagnoses or prescriptions

### Answer:"""

    # Prompt with explanation request
    EXPLAINABLE_QA_PROMPT = """You are a knowledgeable medical assistant focused on providing transparent, explainable answers.

### Context:
{context}

### Question:
{question}

### Instructions:
1. Answer the question based on the provided context
2. Cite which parts of the context support your answer
3. Indicate your confidence level (High/Medium/Low)
4. Note any limitations or uncertainties
5. Use clear, patient-friendly language

### Answer:"""

    # Simple prompt for fast responses
    SIMPLE_QA_PROMPT = """Context: {context}

Question: {question}

Provide a helpful, accurate answer based on the context above. Be concise but thorough.

Answer:"""

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
        return """
⚠️ MEDICAL DISCLAIMER: This information is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read here.
"""
    
    def add_template(self, name: str, template: str, description: str = ""):
        """Add a custom prompt template."""
        self.templates[name] = PromptTemplate(
            template=template,
            name=name,
            description=description
        )
