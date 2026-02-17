"""
Rationale generator for explaining medical QA answers.
"""
from typing import List, Dict, Optional

class RationaleGenerator:
    """Generates natural language explanations (rationales) for answers."""
    
    RATIONALE_TEMPLATE = """You are a medical expert assistant. 
Your task is to explain WHY the answer provided below is correct based on the given context.
Focus on the specific medical facts in the text that support the answer.

Context:
{context}

Question: {question}
Answer: {answer}

Explanation:"""

    def __init__(self, llm):
        """
        Initialize with an LLM instance.
        
        Args:
            llm: MedicalLLM instance to use for generation
        """
        self.llm = llm
    
    def generate_rationale(
        self, 
        question: str, 
        answer: str, 
        context: str,
        max_tokens: int = 150
    ) -> str:
        """
        Generate an explanation for why the answer is correct.
        
        Args:
            question: The medical question
            answer: The answer provided by the system
            context: The retrieved context used to answer
            max_tokens: Maximum length of explanation
            
        Returns:
            Generated string explanation
        """
        prompt = self.RATIONALE_TEMPLATE.format(
            question=question,
            answer=answer,
            context=context,
        )
        
        # Use the LLM to generate just the explanation
        result = self.llm.generate(
            prompt, 
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7
        )
        
        return result.response.strip()
