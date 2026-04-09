"""
Rationale generator for explaining medical QA answers.

Provides two generation paths:
- LLM-based chain-of-thought rationale (when GPU/LLM is available)
- Template-based rationale (always available, CPU-safe fallback)
"""

from typing import Dict, List, Optional


class RationaleGenerator:
    """Generates natural language explanations (rationales) for answers."""

    # Chain-of-thought prompt: 3-step reasoning
    COT_TEMPLATE = """You are a medical expert assistant explaining an answer to a patient.
Follow these three steps to produce a clear rationale:

Step 1 — Identify the key medical claims in the answer.
Step 2 — Find supporting evidence in the context below.
Step 3 — Explain how the evidence supports the answer in plain language.

Context:
{context}

Question: {question}
Answer: {answer}

Now produce a concise rationale (2-4 sentences) following the three steps above:
Step 1 (Key claims):
Step 2 (Evidence):
Step 3 (Explanation):"""

    def __init__(self, llm=None):
        """
        Initialize with an optional LLM instance.

        Args:
            llm: MedicalLLM instance for GPU-based generation. If None,
                 template-based rationale is always used.
        """
        self.llm = llm

    def generate_rationale(
        self,
        question: str,
        answer: str,
        context: str,
        max_tokens: int = 200,
    ) -> str:
        """
        Generate a chain-of-thought explanation for the answer.

        Falls back to template-based rationale when LLM is unavailable.

        Args:
            question: The medical question
            answer: The answer provided by the system
            context: The retrieved context used to answer
            max_tokens: Maximum length of LLM explanation

        Returns:
            Generated string explanation
        """
        if self.llm is not None:
            try:
                prompt = self.COT_TEMPLATE.format(
                    question=question,
                    answer=answer,
                    context=context[:3000],  # guard against very long contexts
                )
                result = self.llm.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                )
                return result.response.strip()
            except Exception as e:
                from loguru import logger

                logger.warning(f"LLM rationale generation failed, falling back to template: {e}")

        return self.generate_template_rationale(question, answer, context=context)

    def generate_template_rationale(
        self,
        question: str,
        answer: str,
        confidence: Optional[Dict] = None,
        sources: Optional[List[Dict]] = None,
        attributions: Optional[List[Dict]] = None,
        context: str = "",
    ) -> str:
        """
        Build a rationale from confidence signals and sources without LLM inference.

        Safe to call on CPU; used as the default fallback.

        Args:
            question: The medical question
            answer: The system answer
            confidence: Optional confidence dict with 'score' and 'level' keys
            sources: Optional list of source dicts with 'source' key
            attributions: Optional list of attribution dicts
            context: Raw retrieved context (used when sources not provided)

        Returns:
            Template-based rationale string
        """
        parts = []

        # Confidence statement
        if confidence:
            score = confidence.get("score", 0)
            level = confidence.get("level", "moderate")
            parts.append(f"This answer was generated with {level} confidence ({score:.0%}).")
        else:
            parts.append("This answer is based on retrieved medical literature.")

        # Source statement
        if sources:
            source_names = list({s.get("source", "Unknown") for s in sources[:3]})
            parts.append(
                f"It draws on {len(sources)} retrieved passage(s) from: {', '.join(source_names)}."
            )
        elif context:
            word_count = len(context.split())
            parts.append(f"It is grounded in {word_count} words of retrieved medical context.")

        # Attribution statement
        if attributions:
            supported = [a for a in attributions if a.get("source") != "Unsupported"]
            parts.append(
                f"{len(supported)} of {len(attributions)} claim(s) in the answer "
                f"were directly verified against source passages."
            )

        parts.append("Always verify medical information with a qualified healthcare professional.")

        return " ".join(parts)
