"""
Attention and Token Importance Visualization for Explainable AI.

Provides visualization of:
- Token importance scores (simplified SHAP-like analysis)
- Attention patterns from transformer models
- Counterfactual explanations
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class TokenImportance:
    """Token with its importance score."""

    token: str
    importance: float
    position: int
    is_computed: bool = True  # False when gradient failed and scores are placeholder uniform 0.5


@dataclass
class AttentionVisualization:
    """Container for attention visualization data."""

    tokens: List[str]
    attention_weights: List[List[float]]  # Layer x Token x Token
    importance_scores: List[float]


class TokenImportanceAnalyzer:
    """
    Analyze token importance for model predictions.

    Uses gradient-based saliency as a lightweight alternative to SHAP.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize with a model and tokenizer.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def compute_token_importance(
        self, text: str, target_text: Optional[str] = None
    ) -> List[TokenImportance]:
        """
        Compute importance scores for each token in the input.

        Uses input gradient saliency: ||dL/d(embedding)||

        Args:
            text: Input text to analyze
            target_text: Optional target output for gradient computation

        Returns:
            List of TokenImportance objects
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings with gradient tracking
        self.model.eval()

        try:
            # Get input embeddings
            if hasattr(self.model, "get_input_embeddings"):
                embedding_layer = self.model.get_input_embeddings()
            else:
                embedding_layer = self.model.model.embed_tokens

            embeddings = embedding_layer(inputs["input_ids"])
            embeddings.retain_grad()

            # Forward pass
            outputs = self.model(
                inputs_embeds=embeddings, attention_mask=inputs.get("attention_mask")
            )

            # Use mean of logits as target for gradient
            if hasattr(outputs, "logits"):
                target = outputs.logits.mean()
            else:
                target = outputs[0].mean()

            # Backward pass
            target.backward()

            # Get gradient magnitudes
            is_computed = True
            if embeddings.grad is not None:
                gradients = embeddings.grad.detach().cpu().numpy()
                # L2 norm of gradients per token
                importance_scores = np.linalg.norm(gradients[0], axis=1)
                # Normalize to 0-1
                if importance_scores.max() > 0:
                    importance_scores = importance_scores / importance_scores.max()
            else:
                importance_scores = np.ones(len(tokens)) * 0.5
                is_computed = False

        except Exception as e:
            from loguru import logger

            logger.warning(f"Gradient computation failed: {e}")
            importance_scores = np.ones(len(tokens)) * 0.5
            is_computed = False

        # Build result
        result = []
        for i, (token, score) in enumerate(zip(tokens, importance_scores)):
            result.append(
                TokenImportance(
                    token=token,
                    importance=float(score),
                    position=i,
                    is_computed=is_computed,
                )
            )

        return result

    def get_top_important_tokens(self, text: str, top_k: int = 10) -> List[TokenImportance]:
        """Get top-k most important tokens."""
        all_tokens = self.compute_token_importance(text)
        sorted_tokens = sorted(all_tokens, key=lambda x: x.importance, reverse=True)
        return sorted_tokens[:top_k]


class AttentionExtractor:
    """
    Extract and visualize attention patterns from transformer models.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract_attention(
        self, text: str, layer: int = -1  # -1 for last layer
    ) -> AttentionVisualization:
        """
        Extract attention weights for visualization.

        Args:
            text: Input text
            layer: Which layer's attention to extract (-1 = last)

        Returns:
            AttentionVisualization object
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass with attention output
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Extract attention from specified layer
        attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)

        if attentions:
            layer_attention = attentions[layer][0]  # (heads, seq, seq)
            # Average across heads
            avg_attention = layer_attention.mean(dim=0).cpu().numpy()  # (seq, seq)

            # Compute importance as attention received by each token
            importance_scores = avg_attention.sum(axis=0).tolist()
            # Normalize
            max_imp = max(importance_scores) if importance_scores else 1
            importance_scores = [s / max_imp for s in importance_scores]
        else:
            avg_attention = np.eye(len(tokens))
            importance_scores = [0.5] * len(tokens)

        return AttentionVisualization(
            tokens=tokens,
            attention_weights=avg_attention.tolist(),
            importance_scores=importance_scores,
        )


class CounterfactualExplainer:
    """
    Generate counterfactual explanations: "Why X instead of Y?"
    """

    def __init__(self, llm):
        """
        Initialize with an LLM for generating explanations.

        Args:
            llm: MedicalLLM instance
        """
        self.llm = llm

    COUNTERFACTUAL_TEMPLATE = """You are a medical expert providing explanations.

The system answered the following medical question:

Question: {question}
Answer Given: {answer}
Alternative Answer: {alternative}

Explain why the given answer is more appropriate than the alternative based on the medical context provided.

Context: {context}

Explanation:"""

    def explain_why_not(self, question: str, answer: str, alternative: str, context: str) -> str:
        """
        Explain why the given answer was chosen over an alternative.

        Args:
            question: The original question
            answer: The answer that was given
            alternative: An alternative answer to compare against
            context: The context used for answering

        Returns:
            Explanation string
        """
        prompt = self.COUNTERFACTUAL_TEMPLATE.format(
            question=question,
            answer=answer,
            alternative=alternative,
            context=context[:1000],  # Limit context length
        )

        result = self.llm.generate(prompt, max_new_tokens=200, temperature=0.7)
        return result.response.strip()

    def generate_alternatives(
        self, question: str, answer: str, num_alternatives: int = 2
    ) -> List[str]:
        """
        Generate plausible alternative answers for comparison.

        Args:
            question: The original question
            answer: The given answer
            num_alternatives: Number of alternatives to generate

        Returns:
            List of alternative answer strings
        """
        prompt = f"""Given this medical question and answer, suggest {num_alternatives} plausible but incorrect alternative answers that a layperson might consider:

Question: {question}
Correct Answer: {answer}

Alternative 1:"""

        result = self.llm.generate(prompt, max_new_tokens=150, temperature=0.8)

        # Parse alternatives from response
        alternatives = []
        lines = result.response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Alternative"):
                alternatives.append(line)
            if len(alternatives) >= num_alternatives:
                break

        return alternatives


def format_importance_html(tokens: List[TokenImportance]) -> str:
    """
    Format token importance as HTML with color coding.

    Returns HTML string with tokens colored by importance.
    """
    html_parts = []
    for t in tokens:
        # Map importance to color intensity (red for important)
        intensity = int(255 * (1 - t.importance))
        color = f"rgb(255, {intensity}, {intensity})"
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px;">{t.token}</span>'
        )
    return " ".join(html_parts)
