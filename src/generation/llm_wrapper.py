"""
LLM wrapper for medical question answering.
Supports TinyLlama (transformers), BioMistral (GGUF), and Extractive QA (no model needed).
"""

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList


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


class KeywordStoppingCriteria(StoppingCriteria):
    """
    Stop generation as soon as any of the configured keyword byte-sequences
    appears in the decoded output.

    Checking is done on the *decoded* prefix of newly generated tokens only,
    so there is no cost for the prompt tokens already in input_ids.

    Patterns checked (covers the most common TinyLlama leakage signatures):
      - Exam-format continuation markers: "Question:", "\nQ:"
      - ChatDoctor / HealthCareMagic identity strings
      - Sign-off phrases: "Thank you for", "I hope this"
      - Template token leaks: "<|", "</s>", "[/INST]"
    """

    # Keep this list short — each candidate is checked O(output_len) per step.
    _STOP_STRINGS = [
        "Question:",
        "\nQ:",
        "ChatDoctor",
        "HealthCareMagic",
        "Thank you for",
        "I hope this",
        "\n---",
        "<|",
        "</s>",
        "[/INST]",
    ]

    def __init__(self, tokenizer):
        super().__init__()
        self._tokenizer = tokenizer
        self._input_length: int = 0  # set before each generate() call

    def set_input_length(self, length: int) -> None:
        """Call before model.generate() with the prompt token count."""
        self._input_length = length

    def __call__(self, input_ids: "torch.LongTensor", scores: "torch.FloatTensor", **kwargs) -> bool:
        # Only decode newly generated tokens (exclude the prompt)
        new_ids = input_ids[0][self._input_length:]
        if len(new_ids) == 0:
            return False
        decoded = self._tokenizer.decode(new_ids, skip_special_tokens=False)
        return any(s in decoded for s in self._STOP_STRINGS)


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    response: str
    input_tokens: int
    generated_tokens: int
    probabilities: Optional[List[float]] = None
    # Which backend actually produced this answer (e.g. "ollama", "extractive",
    # "extractive_fallback" when Ollama was requested but unreachable).
    generation_backend_used: str = ""


class ExtractiveQA:
    """
    Answer extractor — no model weights needed.

    Strategy (in order):
    1. Parse "Answer: ..." spans from structured Q&A sources (MedQuAD, MedMCQA, etc.)
    2. Pick the sentence(s) from the top-scored passage that best overlap the query tokens
    3. Return the candidate with the highest token overlap as the answer
    """

    # Matches "Answer: <text>" or "A: <text>" at any position in a document chunk
    _ANSWER_RE = re.compile(
        r"(?:Answer|Answers?|A)\s*:\s*(?:nan\s*)?(.+?)(?=\s*(?:Question|Q)\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    # Sentence splitter (period/exclamation/question followed by space or end)
    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def generate(self, question: str, context: str) -> GenerationResult:
        """Extract the best answer from context.

        The pipeline's context_builder already extracts Answer: spans from Q&A chunks,
        so block 0 (highest-ranked source) is typically a clean medical fact.
        Strategy:
        1. Check all blocks for any residual Answer: pattern not yet stripped.
        2. If block 0 is short and clean (context_builder already extracted it), return it.
        3. Fall back to sentence-level overlap scoring with positional bonus.
        """
        # Split into per-source blocks, preserving retrieval order
        blocks = re.split(r"\[\d+\]\s*Source:[^\n]*\n|\[[^\]]+\]:\s*", context)
        blocks = [b.strip() for b in blocks if b.strip()]

        if not blocks:
            return GenerationResult(
                response="I do not have enough information in my references to answer this.",
                input_tokens=0, generated_tokens=0,
                generation_backend_used="extractive",
            )

        # Pass 1: look for any residual "Answer:" pattern not stripped by context_builder
        for position, block in enumerate(blocks):
            for m in self._ANSWER_RE.finditer(block):
                ans = m.group(1).strip()
                ans = re.sub(r"\s+", " ", ans)
                if len(ans) < 8 or ans.lower() in ("nan", "none", "n/a"):
                    continue
                ans = re.split(r"\n[A-E][.)]\s", ans)[0].strip()
                ans = " ".join(self._SENT_RE.split(ans)[:3]).strip()
                logger.debug("ExtractiveQA Answer: hit pos={}: {}", position, ans[:80])
                return GenerationResult(response=ans,
                                        input_tokens=len(context.split()),
                                        generated_tokens=len(ans.split()),
                                        generation_backend_used="extractive")

        # Pass 2: the top-ranked block is already the best source — return its first 5 sentences.
        # context_builder strips "Question:..." leaving only the answer text, so this is safe.
        # We skip blocks that still start with a Question: marker (no Answer: found in that doc).
        top_block = blocks[0]
        if not re.match(r"^\s*question", top_block, re.IGNORECASE):
            sentences = [s.strip() for s in self._SENT_RE.split(top_block) if len(s.strip()) >= 8]
            ans = " ".join(sentences[:5]).strip()
            if ans:
                logger.debug("ExtractiveQA top-block direct ({}s): {}", len(sentences[:5]), ans[:80])
                return GenerationResult(response=ans,
                                        input_tokens=len(context.split()),
                                        generated_tokens=len(ans.split()),
                                        generation_backend_used="extractive")

        # Pass 3: top block had no clean answer — fall back to best sentence from any block
        stopwords = {"what", "is", "are", "the", "a", "an", "of", "for", "in",
                     "and", "or", "to", "how", "does", "do", "was", "were"}
        query_tokens = set(w for w in re.findall(r"\w+", question.lower())
                           if w not in stopwords and not w.isdigit())
        sent_candidates: list[tuple[float, str]] = []
        for position, block in enumerate(blocks):
            pos_bonus = 1.0 / (1 + position)
            for sent in self._SENT_RE.split(block):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                text_tokens = set(re.findall(r"\w+", sent.lower()))
                overlap = len(query_tokens & text_tokens) / len(query_tokens) if query_tokens else 0
                if overlap > 0:
                    sent_candidates.append((overlap * pos_bonus, sent))

        if sent_candidates:
            sent_candidates.sort(key=lambda x: x[0], reverse=True)
            answer = " ".join(self._SENT_RE.split(sent_candidates[0][1])[:5]).strip()
            logger.debug("ExtractiveQA sentence score={:.3f}: {}", sent_candidates[0][0], answer[:80])
            return GenerationResult(response=answer,
                                    input_tokens=len(context.split()),
                                    generated_tokens=len(answer.split()),
                                    generation_backend_used="extractive")

        return GenerationResult(
            response="I do not have enough information in my references to answer this.",
            input_tokens=len(context.split()),
            generated_tokens=0,
            generation_backend_used="extractive",
        )

    def generate_stream(self, question: str, context: str):
        """Yield full answer as a single chunk (no token-by-token streaming needed)."""
        result = self.generate(question, context)
        yield result.response

    @staticmethod
    def _overlap(query_tokens: set[str], text: str) -> float:
        """Fraction of query tokens that appear in text (case-insensitive)."""
        if not query_tokens:
            return 0.0
        text_tokens = set(re.findall(r"\w+", text.lower()))
        # Ignore stopwords for overlap scoring
        stopwords = {"what", "is", "are", "the", "a", "an", "of", "for", "in", "and", "or", "to", "how", "does", "do"}
        q = query_tokens - stopwords
        if not q:
            q = query_tokens
        return len(q & text_tokens) / len(q)


class OllamaLLM:
    """
    Remote Ollama backend for serving larger models (e.g. Qwen2.5-7B) on a GPU server.

    Reads from environment:
      OLLAMA_BASE_URL  — default http://localhost:11434
      OLLAMA_MODEL     — default qwen2.5:7b
      OLLAMA_API_KEY   — optional bearer token for endpoint protection

    Falls back to ExtractiveQA on any connection error or timeout.
    """

    def __init__(self) -> None:
        import os

        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self._api_key = os.getenv("OLLAMA_API_KEY", "")
        self.backend = "ollama"
        self.model_name = self.model
        self._fallback = ExtractiveQA()
        logger.info("OllamaLLM: url={} model={}", self.base_url, self.model)

    def generate_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 512,
    ) -> GenerationResult:
        """Call Ollama /api/generate with faithfulness-enforcing system prompt."""
        import requests

        system_prompt = (
            "You are a medical information assistant. "
            "Answer the question using ONLY the reference text provided. "
            "If the reference text does not contain the answer, respond with: "
            "'I don't have enough information to answer this.' "
            "Do not use external knowledge. Be concise and accurate."
        )
        user_message = f"Reference text:\n{context}\n\nQuestion: {question}"

        headers: dict = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self.model,
            "prompt": user_message,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=headers,
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("response", "").strip()
            if not response_text:
                raise ValueError("Empty response from Ollama")
            return GenerationResult(
                response=response_text,
                input_tokens=data.get("prompt_eval_count", 0),
                generated_tokens=data.get("eval_count", 0),
                generation_backend_used="ollama",
            )
        except Exception as e:
            logger.warning("OllamaLLM unreachable or failed ({}), falling back to ExtractiveQA", e)
            result = self._fallback.generate(question=question, context=context)
            result.generation_backend_used = "extractive_fallback"
            return result

    def cleanup(self) -> None:
        pass  # no local resources to free


class MedicalLLM:
    """
    Medical domain LLM wrapper.
    Supports TinyLlama (transformers), BioMistral-7B (GGUF), and ExtractiveQA (no model).
    """

    SUPPORTED_MODELS = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "biomistral": None,  # resolved via AVAILABLE_MODELS at runtime
        "extractive": None,  # no model weights — uses ExtractiveQA
    }

    # Patterns that indicate the model has leaked training data
    STOP_PATTERNS = [
        r"\nQuestion:",
        r"\nQ:",
        r"\nAnswer:",
        r"Best regards",
        r"Kind regards",
        r"Sincerely",
        r"Yours truly",
        r"Warm regards",
        r"With best wishes",
        r"\[Your Name\]",
        r"\[Doctor'?s? Name\]",
        r"Chat Doctor",
        r"ChatDoctor",
        r"HealthCareMagic",
        r"Thank you for choosing",
        r"Thank you for using",
        r"Thank you for reaching out",
        r"Thank you for contacting",
        r"If you have any further questions",
        r"please do not hesitate",
        r"don't hesitate to ask",
        r"I hope this (?:helps|information|answers)",
        r"Wishing you (?:good|the best)",
        r"Take care",
        r"\nHi,?\s",
        r"\nHello,?\s",
        r"\nDear ",
        r"\nHi doctor",
        r"\nHello doctor",
        r"\nHi,\s*\n",
        r"\[\d+\]\s*Source:",
        r"\n---",
        r"<\|",
        r"\[/INST\]",
        r"</s>",
        r"<\|im_end\|>",
        r"<\|endoftext\|>",
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

        self.model_name = model_name

        if model_name == "extractive":
            self.backend = "extractive"
            self._extractive = ExtractiveQA()
            logger.info("ExtractiveQA backend loaded (no model weights)")
            return

        model_path = self._resolve_model_path(model_name)
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

        import os

        n_threads = min(4, os.cpu_count() or 4)
        logger.info(
            "Loading GGUF model from {} (n_ctx=4096, n_threads={})",
            model_path,
            n_threads,
        )
        n_gpu_layers = -1 if torch.cuda.is_available() else 0  # -1 = all layers on GPU
        self._gguf_llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,  # Mistral-7B supports 4096; 2048 caused prompt truncation with RAG context
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info("GGUF backend: n_gpu_layers={}", n_gpu_layers)
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
                logger.warning(
                    f"BitsAndBytes quantization config failed, loading without quantization: {e}"
                )
        else:
            quantization_config = None

        # Tokenizer
        tokenizer_path = adapter_path if adapter_path else model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        return_probabilities: bool = False,
    ) -> GenerationResult:
        """Generate response from the LLM (routes to correct backend)."""
        if self.backend == "extractive":
            # prompt is a raw string — extractive backend needs question + context
            # which are joined here; parse them back out
            return self._extractive.generate(question=prompt, context=prompt)
        if self.backend == "gguf":
            return self._generate_gguf(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else 0.1,
                top_p=top_p if top_p is not None else 0.9,
            )
        return self._generate_transformers(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature is not None else 0.3,
            top_p=top_p if top_p is not None else 0.85,
            do_sample=do_sample,
            return_probabilities=return_probabilities,
        )

    # ------------------------------------------------------------------
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """Yield decoded text tokens one at a time (routes to correct backend).

        Falls back to yielding the full response as a single chunk when the
        backend doesn't support native streaming.

        Yields:
            str: Decoded token text fragments as they are generated.
        """
        if self.backend == "extractive":
            yield from self._extractive.generate_stream(question=prompt, context=prompt)
        elif self.backend == "gguf":
            yield from self._generate_stream_gguf(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else 0.1,
                top_p=top_p if top_p is not None else 0.9,
            )
        else:
            yield from self._generate_stream_transformers(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else 0.3,
                top_p=top_p if top_p is not None else 0.85,
            )

    def _generate_stream_gguf(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ):
        """Stream tokens from the llama-cpp GGUF backend."""
        prompt = prompt.strip()
        formatted = prompt if "[INST]" in prompt else f"<s>[INST] {prompt} [/INST]"
        stream = self._gguf_llm(
            formatted,
            max_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            echo=False,
            stream=True,
        )
        for chunk in stream:
            token_text = chunk["choices"][0].get("text", "")
            if token_text:
                yield token_text

    def _generate_stream_transformers(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.85,
    ):
        """Stream tokens from the HuggingFace transformers backend using TextIteratorStreamer."""
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(
            self.model.device
        )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Attach keyword stopping criteria to prevent leakage tokens in stream mode.
        # Same logic as _generate_transformers; shared _keyword_stopper instance is safe
        # because set_input_length() is always called before Thread.start().
        if not hasattr(self, "_keyword_stopper"):
            self._keyword_stopper = KeywordStoppingCriteria(self.tokenizer)
        input_length = inputs.input_ids.shape[1]
        self._keyword_stopper.set_input_length(input_length)
        stopping_criteria = StoppingCriteriaList([self._keyword_stopper])

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 4,
            "stopping_criteria": stopping_criteria,
        }

        # Run generation in a background thread so we can yield from the streamer
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token_text in streamer:
            if token_text:
                yield token_text

        thread.join()

    # ------------------------------------------------------------------
    def _generate_gguf(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ) -> GenerationResult:
        """Generate using llama-cpp-python (GGUF backend).

        BioMistral Q4_K_M performs best with low temperature (0.1) and a mild
        repeat penalty (1.1) to prevent degenerate looping on CPU.
        """
        # BioMistral uses native Mistral instruction formatting. If the caller
        # already supplied an [INST] prompt, preserve it instead of double-wrap.
        prompt = prompt.strip()
        formatted = prompt if "[INST]" in prompt else f"<s>[INST] {prompt} [/INST]"
        output = self._gguf_llm(
            formatted,
            max_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),  # llama-cpp requires > 0
            top_p=top_p,
            repeat_penalty=repeat_penalty,
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
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(
            self.model.device
        )

        input_length = inputs.input_ids.shape[1]

        # Build keyword stopping criteria (prevents generating garbage tokens after
        # ChatDoctor sign-offs, exam-format re-starts, or template token leaks).
        if not hasattr(self, "_keyword_stopper"):
            self._keyword_stopper = KeywordStoppingCriteria(self.tokenizer)
        self._keyword_stopper.set_input_length(input_length)
        stopping_criteria = StoppingCriteriaList([self._keyword_stopper])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3,   # prevents ChatDoctor loop patterns
                no_repeat_ngram_size=4,   # no exact 4-gram repetition
                stopping_criteria=stopping_criteria,
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
        """Generate response with context (for RAG)."""
        if self.backend == "extractive":
            return self._extractive.generate(question=question, context=context)

        if self.backend == "gguf":
            # Mistral instruction format used by BioMistral
            prompt = (
                f"<s>[INST] You are a medical assistant. "
                f"Answer the question using ONLY the reference text. "
                f"Do NOT add your own knowledge.\n\n"
                f"REFERENCE TEXT: {context}\n\n"
                f"QUESTION: {question} [/INST]"
            )
        else:
            # TinyLlama chat format
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
        if self.backend == "extractive":
            return
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
