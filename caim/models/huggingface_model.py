"""HuggingFace model integration for CAIM framework."""

import logging
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
import asyncio

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from accelerate import Accelerator
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base_model import BaseModel, ModelResponse
from ..core.config import CAIMConfig
from ..core.exceptions import ModelException


logger = logging.getLogger(__name__)


class HuggingFaceModel(BaseModel):
    """HuggingFace model implementation for Qwen, Gemma, and other models."""
    
    def __init__(
        self,
        config: CAIMConfig,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        use_accelerate: bool = True
    ):
        super().__init__(config, model_name)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required but not installed")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_accelerate = use_accelerate
        
        self.tokenizer = None
        self.model = None
        self.accelerator = None
        self.generation_config = None
        
        # Model configuration
        self.default_max_tokens = 512
        self.default_temperature = 0.7
        self.max_context_length = 4096
        
        logger.info(f"Initialized HuggingFace model: {model_name} on {self.device}")
    
    async def initialize(self) -> None:
        """Initialize the HuggingFace model and tokenizer."""
        try:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            
            # Initialize accelerator if requested
            if self.use_accelerate:
                self.accelerator = Accelerator()
            
            # Load tokenizer
            await self._load_tokenizer()
            
            # Load model
            await self._load_model()
            
            # Setup generation config
            self._setup_generation_config()
            
            self.is_initialized = True
            logger.info(f"HuggingFace model {self.model_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing HuggingFace model: {e}")
            raise ModelException(f"HuggingFace model initialization failed: {e}")
    
    async def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        try:
            loop = asyncio.get_event_loop()
            self.tokenizer = await loop.run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    token=self.config.huggingface_api_token
                )
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise ModelException(f"Tokenizer loading failed: {e}")
    
    async def _load_model(self) -> None:
        """Load the model."""
        try:
            loop = asyncio.get_event_loop()
            
            # Determine model loading parameters
            model_kwargs = {
                "trust_remote_code": True,
                "token": self.config.huggingface_api_token,
            }
            
            # Set appropriate torch_dtype based on device
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
            elif self.device == "mps":
                # MPS can be finicky with float16, try float32 first
                model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["torch_dtype"] = torch.float32
            
            # Add device map for multi-GPU setups
            if self.device == "cuda" and torch.cuda.device_count() > 1:
                model_kwargs["device_map"] = "auto"
            
            self.model = await loop.run_in_executor(
                None,
                lambda: AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                try:
                    self.model = self.model.to(self.device)
                except Exception as e:
                    if self.device == "mps":
                        logger.warning(f"Failed to move model to MPS ({e}), falling back to CPU")
                        self.device = "cpu"
                        self.model = self.model.to("cpu")
                    else:
                        raise
            
            # Use accelerator if available
            if self.accelerator:
                self.model = self.accelerator.prepare(self.model)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelException(f"Model loading failed: {e}")
    
    def _setup_generation_config(self) -> None:
        """Setup generation configuration."""
        try:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.default_max_tokens,
                temperature=self.default_temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            logger.info("Generation config setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up generation config: {e}")
            raise ModelException(f"Generation config setup failed: {e}")
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using the HuggingFace model."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_context_prompt(prompt, context, memory_context)
            
            # Format prompt for chat models
            formatted_prompt = self._format_chat_prompt(full_prompt)
            
            # Generate response
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                self._generate_text,
                formatted_prompt,
                max_tokens or self.default_max_tokens,
                temperature,
                kwargs
            )
            
            # Extract actual response (remove prompt)
            actual_response = self._extract_response_text(response_text, formatted_prompt)
            
            return ModelResponse(
                content=actual_response,
                model_name=self.model_name,
                timestamp=datetime.utcnow(),
                metadata={
                    "prompt_length": len(formatted_prompt),
                    "response_length": len(actual_response),
                    "device": str(self.device),
                    "model_type": "huggingface"
                },
                usage_stats=self._calculate_token_usage(formatted_prompt, actual_response),
                confidence_score=self._estimate_confidence(actual_response)
            )
            
        except Exception as e:
            logger.error(f"Error generating HuggingFace response: {e}")
            raise ModelException(f"HuggingFace response generation failed: {e}")
    
    async def generate_streaming_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response using the HuggingFace model."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_context_prompt(prompt, context, memory_context)
            formatted_prompt = self._format_chat_prompt(full_prompt)
            
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            
            # Setup generation parameters
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Move inputs to model device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate tokens one by one
            with torch.no_grad():
                current_ids = inputs['input_ids']
                
                for _ in range(max_tokens or self.default_max_tokens):
                    # Generate next token
                    outputs = self.model(current_ids)
                    logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature and sampling
                    if temperature > 0:
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)
                        next_token_id = torch.multinomial(probs, 1)
                    else:
                        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # Check for EOS token
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Decode and yield new token
                    new_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                    yield new_token
                    
                    # Update current_ids
                    current_ids = torch.cat([current_ids, next_token_id], dim=-1)
                    
                    # Yield control to allow other tasks
                    await asyncio.sleep(0)
                    
        except Exception as e:
            logger.error(f"Error generating HuggingFace streaming response: {e}")
            raise ModelException(f"HuggingFace streaming response failed: {e}")
    
    def _format_chat_prompt(self, prompt: str) -> str:
        """Format prompt for chat models."""
        try:
            # Check if model supports chat templates
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat templates
                system_prompt = self._get_system_prompt()
                return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
                
        except Exception as e:
            logger.error(f"Error formatting chat prompt: {e}")
            return f"User: {prompt}\nAssistant:"
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the model."""
        return """You are a helpful AI assistant with access to a long-term memory system. Use the provided context and memories to give personalized, relevant responses."""
    
    def _generate_text(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate text synchronously."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to model device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Debug device placement
            logger.debug(f"Model device: {self.device}")
            logger.debug(f"Input IDs device: {inputs['input_ids'].device}")
            if 'attention_mask' in inputs:
                logger.debug(f"Attention mask device: {inputs['attention_mask'].device}")
            
            # Prepare generation kwargs with proper device handling
            generation_kwargs = {
                'max_new_tokens': max_tokens,
                'do_sample': temperature > 0,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            if temperature > 0:
                generation_kwargs['temperature'] = temperature
            
            # Add attention mask if available
            if 'attention_mask' in inputs:
                generation_kwargs['attention_mask'] = inputs['attention_mask']
            
            # Generate with all tensors on correct device
            with torch.no_grad():
                # Ensure all inputs are on the correct device
                input_ids = inputs['input_ids']
                
                # For MPS, we need to be extra careful about device placement
                if self.device == "mps":
                    # Force all tensors to MPS explicitly
                    input_ids = input_ids.to("mps")
                    if 'attention_mask' in generation_kwargs:
                        generation_kwargs['attention_mask'] = generation_kwargs['attention_mask'].to("mps")
                
                try:
                    outputs = self.model.generate(
                        input_ids,
                        **generation_kwargs
                    )
                except RuntimeError as e:
                    if "mps" in str(e).lower() or "placeholder" in str(e).lower():
                        logger.warning(f"MPS generation failed ({e}), falling back to CPU")
                        # Fallback to CPU for this generation
                        cpu_model = self.model.to("cpu")
                        cpu_input_ids = input_ids.to("cpu")
                        cpu_kwargs = generation_kwargs.copy()
                        if 'attention_mask' in cpu_kwargs:
                            cpu_kwargs['attention_mask'] = cpu_kwargs['attention_mask'].to("cpu")
                        
                        outputs = cpu_model.generate(cpu_input_ids, **cpu_kwargs)
                        
                        # Move model back to MPS for next time
                        self.model = cpu_model.to(self.device)
                    else:
                        raise
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            raise ModelException(f"Text generation failed: {e}")
    
    def _extract_response_text(self, full_response: str, prompt: str) -> str:
        """Extract the actual response from the full generated text."""
        try:
            # Remove the prompt from the response
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                response = full_response.strip()
            
            # Clean up common artifacts
            response = response.replace("<|endoftext|>", "").strip()
            response = response.replace("</s>", "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error extracting response text: {e}")
            return full_response
    
    def _calculate_token_usage(self, prompt: str, response: str) -> Dict[str, Any]:
        """Calculate token usage statistics."""
        try:
            prompt_tokens = len(self.tokenizer.encode(prompt))
            response_tokens = len(self.tokenizer.encode(response))
            
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens
            }
            
        except Exception as e:
            logger.error(f"Error calculating token usage: {e}")
            return {}
    
    def _estimate_confidence(self, response: str) -> Optional[float]:
        """Estimate confidence based on response characteristics."""
        try:
            # Simple heuristics for confidence estimation
            if not response or len(response.strip()) < 10:
                return 0.3
            
            # Check for uncertainty indicators
            uncertainty_indicators = ["i'm not sure", "maybe", "possibly", "i think", "perhaps"]
            uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in response.lower())
            
            # Base confidence
            confidence = 0.8
            
            # Reduce confidence for uncertainty indicators
            confidence -= uncertainty_count * 0.1
            
            # Adjust based on length (very short or very long responses might be less confident)
            if len(response) < 50:
                confidence -= 0.1
            elif len(response) > 500:
                confidence -= 0.05
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error estimating confidence: {e}")
            return None
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the HuggingFace model."""
        try:
            model_info = {
                "model_name": self.model_name,
                "provider": "HuggingFace",
                "device": str(self.device),
                "supports_streaming": True,
                "max_context_length": self.max_context_length,
                "default_max_tokens": self.default_max_tokens
            }
            
            if self.model:
                model_info.update({
                    "num_parameters": sum(p.numel() for p in self.model.parameters()),
                    "model_dtype": str(next(self.model.parameters()).dtype),
                    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 1
                })
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    async def estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage of the model."""
        try:
            memory_info = {}
            
            if torch.cuda.is_available() and self.device == "cuda":
                memory_info.update({
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
                    "gpu_memory_available": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                })
            
            if self.model:
                model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**3  # GB
                memory_info["model_size_gb"] = model_size
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error estimating memory usage: {e}")
            return {}
    
    async def shutdown(self) -> None:
        """Shutdown the HuggingFace model and free resources."""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            await super().shutdown()
            
        except Exception as e:
            logger.error(f"Error shutting down HuggingFace model: {e}")