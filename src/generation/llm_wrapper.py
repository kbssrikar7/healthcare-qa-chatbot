"""
LLM wrapper for medical question answering.
"""
import re
import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc


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
    Supports: BioMistral, TinyLlama, Mistral, Llama
    """
    
    SUPPORTED_MODELS = {
        "biomistral": "BioMistral/BioMistral-7B",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    }
    
    # Patterns that indicate the model has leaked training data
    # and is generating beyond the intended answer
    STOP_PATTERNS = [
        # New Q&A pair starting (ChatDoctor / HealthCareMagic training data leak)
        r'\nQuestion:',
        r'\nQ:',
        r'\nAnswer:',
        # Sign-offs and letter-style endings
        r'Best regards',
        r'Kind regards',
        r'Sincerely',
        r'Yours truly',
        r'Warm regards',
        r'With best wishes',
        r'\[Your Name\]',
        r'\[Doctor\'?s? Name\]',
        # ChatDoctor / HealthCareMagic specific leaks
        r'Chat Doctor',
        r'ChatDoctor',
        r'HealthCareMagic',
        r'Thank you for choosing',
        r'Thank you for using',
        r'Thank you for reaching out',
        r'Thank you for contacting',
        r'If you have any further questions',
        r'please do not hesitate',
        r'don\'t hesitate to ask',
        r'I hope this (?:helps|information|answers)',
        r'Wishing you (?:good|the best)',
        r'Take care',
        # Greetings that indicate a new response is starting
        r'\nHi,?\s',
        r'\nHello,?\s',
        r'\nDear ',
        r'\nHi doctor',
        r'\nHello doctor',
        r'\nHi,\s*\n',
        # Source reference markers leaking from context
        r'\[\d+\]\s*Source:',
        # System-level markers
        r'\n---',
        r'<\|',
        r'\[/INST\]',
        r'</s>',
        r'<\|im_end\|>',
        r'<\|endoftext\|>',
    ]
    
    def __init__(
        self,
        model_name: str = "tinyllama",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        max_memory: Optional[Dict] = None,
        adapter_path: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter_path = adapter_path
        
        # Get model path
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name
        
        self.model_name = model_name
        print(f"🔄 Loading LLM: {model_path} on {self.device}")
        
        # Quantization config for 4-bit loading
        if load_in_4bit and self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except Exception:
                quantization_config = None
                print("⚠️ BitsAndBytes not available, loading without quantization")
        else:
            quantization_config = None
        
        # Load tokenizer - from adapter if available, otherwise base model
        tokenizer_path = adapter_path if adapter_path else model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
        except OSError as e:
            if "not found" in str(e).lower() or "does not appear" in str(e).lower():
                raise ModelNotFoundError(f"Model '{tokenizer_path}' not found. Check the model name or internet connection.")
            raise
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with granular error handling
        try:
            if quantization_config and self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            # Load PEFT adapter if specified
            if adapter_path:
                try:
                    from peft import PeftModel
                    print(f"🔄 Loading PEFT adapter from {adapter_path}")
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)
                    print(f"✅ Adapter loaded successfully")
                except ImportError:
                    print("⚠️ PEFT not installed, using base model without adapter")
                except Exception as e:
                    print(f"⚠️ Failed to load adapter: {e}, using base model")
            
            self.model.eval()
            print(f"✅ Model loaded successfully")
        except torch.cuda.OutOfMemoryError as e:
            raise GPUOutOfMemoryError(
                f"GPU out of memory loading '{model_path}'. "
                "Try: 1) Using load_in_4bit=True, 2) Using a smaller model like 'tinyllama', "
                "3) Reducing batch size, 4) Using device='cpu'"
            ) from e
        except OSError as e:
            if "not found" in str(e).lower() or "does not appear" in str(e).lower():
                raise ModelNotFoundError(f"Model '{model_path}' not found: {e}")
            raise
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_probabilities: bool = False
    ) -> GenerationResult:
        """Generate response from the LLM."""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        input_length = inputs.input_ids.shape[1]
        
        # Generate
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
                return_dict_in_generate=True
            )
        
        # Decode response
        generated_ids = outputs.sequences[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Get probabilities if requested
        probabilities = None
        if return_probabilities and hasattr(outputs, 'scores'):
            probs = []
            for score in outputs.scores:
                prob = torch.softmax(score[0], dim=-1)
                probs.append(prob.max().item())
            probabilities = probs
        
        # Post-process: truncate leaked training data
        response = self._clean_response(response)
        
        return GenerationResult(
            response=response.strip(),
            input_tokens=input_length,
            generated_tokens=len(generated_ids),
            probabilities=probabilities
        )
    
    def generate_with_context(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 512
    ) -> GenerationResult:
        """Generate response with context (for RAG)."""
        prompt = f"""You are a helpful medical assistant. Use the following context to answer the question. If you're unsure, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.generate(prompt, max_new_tokens=max_new_tokens)
    
    def _clean_response(self, response: str) -> str:
        """
        Clean the LLM response by removing leaked training data.
        
        The fine-tuned model (trained on ChatDoctor/HealthCareMagic data)
        sometimes generates beyond the answer boundary, producing sign-offs,
        new Q&A pairs, or other training artifacts. This method detects and
        truncates the response at the earliest such pattern.
        """
        if not response:
            return response
        
        # Find the earliest occurrence of any stop pattern
        earliest_pos = len(response)
        for pattern in self.STOP_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match and match.start() < earliest_pos:
                earliest_pos = match.start()
        
        # Truncate at the earliest stop pattern
        if earliest_pos < len(response):
            response = response[:earliest_pos]
        
        # Remove inline source reference markers like "[1]", "[2]", "[3]" etc
        # that leak from the RAG context into the answer
        response = re.sub(r'\[\d+\]', '', response)
        
        # Remove trailing incomplete sentences (if response was truncated mid-sentence)
        # Only do this if the response doesn't end with proper punctuation
        response = response.rstrip()
        if response and response[-1] not in '.!?:)"':
            # Find the last complete sentence
            last_period = max(
                response.rfind('.'),
                response.rfind('!'),
                response.rfind('?')
            )
            if last_period > len(response) * 0.3:  # Only trim if we keep at least 30%
                response = response[:last_period + 1]
        
        # Clean up extra whitespace left from removals
        response = re.sub(r'  +', ' ', response)
        response = re.sub(r'\n\n\n+', '\n\n', response)
        
        return response.strip()
    
    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
