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

    def __init__(self, llm=None, force_template_on_cpu: bool = True):
        """
        Initialize with an optional LLM instance.

        Args:
            llm: MedicalLLM instance for GPU-based generation. If None,
                 template-based rationale is always used.
            force_template_on_cpu: If True (default), skip LLM inference on
                 CPU and use the template rationale directly.  On CPU the CoT
                 call takes 2-4 minutes and typically produces lower quality
                 than the structured template.  Set to False only when you
                 explicitly want LLM rationales on CPU (e.g. for ablation).
        """
        self.llm = llm
        self._force_template_on_cpu = force_template_on_cpu

        # Detect CPU vs GPU once at init time (avoids per-call torch overhead)
        self._use_template_only = False
        if force_template_on_cpu and llm is not None:
            try:
                import torch
                self._use_template_only = not torch.cuda.is_available()
                if self._use_template_only:
                    from loguru import logger
                    logger.info(
                        "RationaleGenerator: CPU detected — using template rationale "
                        "(set force_template_on_cpu=False to enable LLM CoT on CPU)"
                    )
            except ImportError:
                self._use_template_only = True

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
        # Short-circuit to template on CPU to avoid 2-4 min inference
        if self._use_template_only:
            return self.generate_template_rationale(question, answer, context=context)

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


    def generate_rationale_with_attributions(
        self,
        question: str,
        answer: str,
        attributions: Optional[List[Dict]] = None,
        confidence: Optional[Dict] = None,
        sources: Optional[List[Dict]] = None,
        context: str = "",
    ) -> str:
        """Generate a rationale enriched with the top attribution result.

        Provides a more transparent explanation by including the specific
        source passage that best supports the answer.
        """
        # Build base template rationale
        base = self.generate_template_rationale(
            question, answer,
            confidence=confidence,
            sources=sources,
            attributions=attributions,
            context=context,
        )

        # Enrich with top attribution if available
        if attributions:
            top_attr = max(attributions, key=lambda a: a.get("similarity", 0.0), default=None)
            if top_attr:
                src = top_attr.get("source", "retrieved source")
                sim = top_attr.get("similarity", 0.0)
                evidence = top_attr.get("evidence", "")
                if evidence:
                    evidence_snippet = evidence[:120].rstrip() + ("..." if len(evidence) > 120 else "")
                    base += (
                        f" The highest-confidence supporting evidence (similarity: {sim:.0%}) "
                        f"from {src}: \"{evidence_snippet}\""
                    )
        return base
