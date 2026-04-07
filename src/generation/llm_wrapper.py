"""
LLM wrapper for medical question answering.
Uses TinyLlama as the sole model backend via Hugging Face transformers.
"""
import re
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
from loguru import logger


# Custom exceptions for granular error handling
class LLMError(Exception):
    """Base exception for LLM errors."""
    pass

class ModelNotFoundError(LLMError):
    """Raised when model cannot be found/downloaded."""
    pass

class GPUOutOfMemoryError(LLMError):
    """Raised when GPU runs out of memory."""
    pass

class GenerationError(LLMError):
    """Raised when generation fails."""
    pass


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    response: str
    input_tokens: int
    generated_tokens: int
    probabilities: Optional[List[float]] = None


class MedicalLLM:
    """
    Medical domain LLM wrapper.
    Supports TinyLlama (transformers) and BioMistral-7B (GGUF via llama-cpp-python).
    """

    SUPPORTED_MODELS = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # BioMistral uses the local GGUF path set in config/settings.py
        "biomistral": None,  # resolved via AVAILABLE_MODELS at runtime
    }

    # Patterns that indicate the model has leaked training data
    STOP_PATTERNS = [
        r'\nQuestion:', r'\nQ:', r'\nAnswer:',
        r'Best regards', r'Kind regards', r'Sincerely',
        r'Yours truly', r'Warm regards', r'With best wishes',
        r'\[Your Name\]', r"\[Doctor'?s? Name\]",
        r'Chat Doctor', r'ChatDoctor', r'HealthCareMagic',
        r'Thank you for choosing', r'Thank you for using',
        r'Thank you for reaching out', r'Thank you for contacting',
        r'If you have any further questions',
        r'please do not hesitate', r"don't hesitate to ask",
        r'I hope this (?:helps|information|answers)',
        r'Wishing you (?:good|the best)', r'Take care',
        r'\nHi,?\s', r'\nHello,?\s', r'\nDear ',
        r'\nHi doctor', r'\nHello doctor', r'\nHi,\s*\n',
        r'\[\d+\]\s*Source:',
        r'\n---', r'<\|', r'\[/INST\]', r'</s>',
        r'<\|im_end\|>', r'<\|endoftext\|>',
    ]

    def __init__(
        self,
        model_name: str = "tinyllama",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict] = None,
        adapter_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        **kwargs,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter_path = adapter_path
        self.hf_token = hf_token
        self.backend = "transformers"
        _ = max_memory  # kept for backward compat

        model_path = self._resolve_model_path(model_name)
        self.model_name = model_name
        logger.info("Loading LLM: {} on {}", model_path, self.device)

        # Route to the correct backend
        if self._is_gguf(model_name, model_path):
            self.backend = "gguf"
            self._init_gguf_model(model_path)
        else:
            self._init_transformers_model(
                model_path=model_path,
                load_in_4bit=load_in_4bit,
                adapter_path=adapter_path,
            )

    # ------------------------------------------------------------------
    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve a shorthand model key to a full model path."""
        if model_name == "biomistral":
            # Resolve via AVAILABLE_MODELS config
            try:
                from config.settings import AVAILABLE_MODELS
                return AVAILABLE_MODELS["biomistral"]["model_name"]
            except Exception as e:
                logger.warning(f"Could not resolve biomistral model path from config: {e}")
        if model_name in self.SUPPORTED_MODELS and self.SUPPORTED_MODELS[model_name]:
            return self.SUPPORTED_MODELS[model_name]
        return model_name

    @staticmethod
    def _is_gguf(model_name: str, model_path: str) -> bool:
        """Return True if this model should use the GGUF/llama-cpp backend."""
        return model_name == "biomistral" or (model_path and str(model_path).endswith(".gguf"))

    # ------------------------------------------------------------------
    def _init_gguf_model(self, model_path: str) -> None:
        """Initialize llama-cpp-python backend for GGUF models."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install llama-cpp-python"
            ) from e

        if not Path(model_path).exists():
            raise ModelNotFoundError(
                f"GGUF model not found at '{model_path}'. "
                "Run the BioMistral download script first."
            )

        logger.info("Loading GGUF model from {} (n_ctx=2048, n_threads=4)", model_path)
        import os
        n_threads = min(4, os.cpu_count() or 4)
        self._gguf_llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_threads=n_threads,
            n_gpu_layers=0,    # CPU-only
            verbose=False,
        )
        logger.info("GGUF model loaded successfully")

    # ------------------------------------------------------------------
    def _init_transformers_model(
        self,
        model_path: str,
        load_in_4bit: bool,
        adapter_path: Optional[str],
    ) -> None:
        """Initialize Hugging Face transformers backend."""
        if load_in_4bit and self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as e:
                quantization_config = None
                logger.warning(f"BitsAndBytes quantization config failed, loading without quantization: {e}")
        else:
            quantization_config = None

        # Tokenizer
        tokenizer_path = adapter_path if adapter_path else model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
        except OSError as e:
            if "not found" in str(e).lower() or "does not appear" in str(e).lower():
                raise ModelNotFoundError(
                    f"Model '{tokenizer_path}' not found. "
                    "Check the model name or internet connection."
                ) from e
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        try:
            if self.device == "cpu":
                import os
                threads = min(4, os.cpu_count() or 4)
                torch.set_num_threads(threads)

            if quantization_config and self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=(
                        torch.float16
                        if self.device == "cuda"
                        else getattr(torch, "bfloat16", torch.float32)
                    ),
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)

            # PEFT adapter
            if adapter_path:
                try:
                    from peft import PeftModel
                    logger.info("Loading PEFT adapter from {}", adapter_path)
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)
                    logger.info("Adapter loaded successfully")
                except ImportError:
                    logger.warning("PEFT not installed, using base model without adapter")
                except Exception as e:
                    logger.warning("Failed to load adapter: {}, using base model", e)

            self.model.eval()
            logger.info("Model loaded successfully")
        except torch.cuda.OutOfMemoryError as e:
            raise GPUOutOfMemoryError(
                f"GPU out of memory loading '{model_path}'. "
                "Try: 1) load_in_4bit=True, 2) Reducing batch size, 3) device='cpu'"
            ) from e
        except OSError as e:
            if "not found" in str(e).lower() or "does not appear" in str(e).lower():
                raise ModelNotFoundError(f"Model '{model_path}' not found: {e}") from e
            raise
        except Exception as e:
            logger.error("Failed to load model: {}", e)
            raise

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.85,
        do_sample: bool = True,
        return_probabilities: bool = False,
    ) -> GenerationResult:
        """Generate response from the LLM (routes to correct backend)."""
        if self.backend == "gguf":
            return self._generate_gguf(
                prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
        return self._generate_transformers(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            do_sample=do_sample, return_probabilities=return_probabilities,
        )

    # ------------------------------------------------------------------
    def _generate_gguf(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.85,
    ) -> GenerationResult:
        """Generate using llama-cpp-python (GGUF backend)."""
        # BioMistral uses native Mistral instruction formatting. If the caller
        # already supplied an [INST] prompt, preserve it instead of double-wrap.
        prompt = prompt.strip()
        formatted = prompt if "[INST]" in prompt else f"<s>[INST] {prompt} [/INST]"
        output = self._gguf_llm(
            formatted,
            max_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),  # llama-cpp requires > 0
            top_p=top_p,
            echo=False,
        )
        text = output["choices"][0]["text"]
        text = self._clean_response(text)
        n_prompt = output["usage"]["prompt_tokens"]
        n_gen = output["usage"]["completion_tokens"]
        return GenerationResult(
            response=text.strip(),
            input_tokens=n_prompt,
            generated_tokens=n_gen,
            probabilities=None,  # llama-cpp doesn't easily expose token probs
        )

    # ------------------------------------------------------------------
    def _generate_transformers(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.85,
        do_sample: bool = True,
        return_probabilities: bool = False,
    ) -> GenerationResult:
        """Generate using HuggingFace transformers backend."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=return_probabilities,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        probabilities = None
        if return_probabilities and hasattr(outputs, "scores") and outputs.scores:
            import math
            log_probs = []
            for i, score in enumerate(outputs.scores):
                probs = torch.softmax(score[0], dim=-1)
                token_id = generated_ids[i].item()
                log_probs.append(torch.log(probs[token_id] + 1e-10).item())
            if log_probs:
                mean_log_prob = sum(log_probs) / len(log_probs)
                probabilities = [math.exp(mean_log_prob)]

        response = self._clean_response(response)

        return GenerationResult(
            response=response.strip(),
            input_tokens=input_length,
            generated_tokens=len(generated_ids),
            probabilities=probabilities,
        )

    # ------------------------------------------------------------------
    def generate_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        """Generate response with context (for RAG) using TinyLlama chat format."""
        SYS = "<|system|>"
        USR = "<|user|>"
        AST = "<|assistant|>"
        END = "</s>"

        prompt = (
            f"{SYS}\n"
            "Answer the question using ONLY the reference text. "
            "Do NOT add your own knowledge.\n"
            f"{END}\n"
            f"{USR}\n"
            f"REFERENCE TEXT: {context}\n\n"
            f"QUESTION: {question}\n"
            f"{END}\n"
            f"{AST}\n"
        )

        return self.generate(prompt, max_new_tokens=max_new_tokens)

    # ------------------------------------------------------------------
    def _clean_response(self, response: str) -> str:
        """
        Clean the LLM response by removing leaked training data.

        Delegates to the shared text cleaning utility.
        """
        from src.utils.text_cleaning import clean_llm_response
        return clean_llm_response(response)

    # ------------------------------------------------------------------
    def cleanup(self):
        """Free memory for all backends."""
        if self.backend == "gguf":
            if hasattr(self, "_gguf_llm"):
                del self._gguf_llm
        else:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
