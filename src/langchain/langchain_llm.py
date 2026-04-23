"""
LangChain-compatible LLM wrapper for MedicalLLM.

Wraps the existing MedicalLLM to work with LangChain's LCEL pipelines.
"""

# Import the existing MedicalLLM
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import ConfigDict


from src.generation.llm_wrapper import GenerationResult, MedicalLLM


class LangChainMedicalLLM(LLM):
    """
    LangChain-compatible wrapper for MedicalLLM.

    This wrapper allows using the existing MedicalLLM with LangChain's
    LCEL (LangChain Expression Language) for declarative pipeline composition.

    Example:
        llm = LangChainMedicalLLM(model_name="tinyllama")
        response = llm.invoke("What are the symptoms of diabetes?")
    """

    # Declare fields for Pydantic
    model_name: str = "tinyllama"
    load_in_4bit: bool = True
    adapter_path: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512

    # Internal state (not serialized)
    _llm: Optional[MedicalLLM] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        model_name: str = "tinyllama",
        load_in_4bit: bool = True,
        adapter_path: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        llm: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize LangChain Medical LLM wrapper.

        Args:
            model_name: Name of the model (tinyllama, biomistral, mistral)
            load_in_4bit: Whether to use 4-bit quantization
            adapter_path: Optional path to PEFT adapter
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum tokens to generate
            llm: Optional pre-initialized LLM (used mainly for tests/injection)
        """
        resolved_model_name = model_name
        if llm is not None and (not model_name or model_name == "tinyllama"):
            # Keep a stable identifier when wrapping an existing external LLM.
            resolved_model_name = "medical-llm"

        super().__init__(
            model_name=resolved_model_name,
            load_in_4bit=load_in_4bit,
            adapter_path=adapter_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        self._llm = llm  # Lazy init unless injected

    def _get_llm(self) -> MedicalLLM:
        """Lazily initialize the underlying MedicalLLM."""
        if self._llm is None:
            self._llm = MedicalLLM(
                model_name=self.model_name,
                load_in_4bit=self.load_in_4bit,
                adapter_path=self.adapter_path,
            )
        return self._llm

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        if self.model_name == "medical-llm":
            return "medical-llm"
        return f"medical-llm-{self.model_name}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return parameters that identify this LLM."""
        return {
            "model_name": self.model_name,
            "load_in_4bit": self.load_in_4bit,
            "adapter_path": self.adapter_path,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from the prompt.

        Args:
            prompt: Input prompt
            stop: Optional stop sequences
            run_manager: Callback manager for streaming

        Returns:
            Generated text
        """
        llm = self._get_llm()

        # Get generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        # Generate response
        result: GenerationResult = llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

        response = result.response

        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]

        return response

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[GenerationChunk]:
        """
        Stream generation (falls back to non-streaming for now).

        Note: The underlying MedicalLLM doesn't support streaming natively.
        This provides a compatibility layer that yields the full response.
        """
        response = self._call(prompt, stop, run_manager, **kwargs)
        yield GenerationChunk(text=response)

    def generate_with_metadata(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate with full metadata including token counts and probabilities.

        Returns:
            Dict with response, input_tokens, generated_tokens, probabilities
        """
        llm = self._get_llm()

        result: GenerationResult = llm.generate(
            prompt=prompt,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            return_probabilities=True,
        )

        return {
            "response": result.response,
            "input_tokens": result.input_tokens,
            "generated_tokens": result.generated_tokens,
            "probabilities": result.probabilities,
        }

    def cleanup(self):
        """Free GPU memory."""
        if self._llm is not None:
            self._llm.cleanup()
            self._llm = None


class LangChainMedicalLLMFromExisting(LLM):
    """
    LangChain wrapper that uses an existing MedicalLLM instance.

    Use this when you already have a MedicalLLM instance loaded
    and want to use it with LangChain without reloading.

    Example:
        existing_llm = MedicalLLM(model_name="tinyllama")
        langchain_llm = LangChainMedicalLLMFromExisting(llm=existing_llm)
    """

    llm: Any  # MedicalLLM instance
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        model_name = getattr(self.llm, "model_name", "unknown")
        return f"medical-llm-{model_name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Generate text from the prompt."""
        result = self.llm.generate(
            prompt=prompt,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
        )

        response = result.response

        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]

        return response
