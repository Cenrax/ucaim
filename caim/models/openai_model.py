"""OpenAI model integration for CAIM framework."""

import logging
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
import asyncio

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_model import BaseModel, ModelResponse
from ..core.config import CAIMConfig
from ..core.exceptions import ModelException


logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """OpenAI model implementation for benchmarking and comparison."""
    
    def __init__(
        self,
        config: CAIMConfig,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        super().__init__(config, model_name)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required but not installed")
        
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise ModelException("OpenAI API key is required")
        
        self.client = None
        self.system_prompt = self._get_system_prompt()
        
        # Model configuration
        self.default_max_tokens = 1000
        self.default_temperature = 0.7
        
        logger.info(f"Initialized OpenAI model: {model_name}")
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            
            # Test the connection
            await self._test_connection()
            
            self.is_initialized = True
            logger.info(f"OpenAI model {self.model_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI model: {e}")
            raise ModelException(f"OpenAI initialization failed: {e}")
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """Generate a response using OpenAI's API."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_context_prompt(prompt, context, memory_context)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_prompt}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Extract metadata
            usage_stats = self._extract_usage_stats(response)
            confidence_score = self._calculate_confidence_score(response)
            
            return ModelResponse(
                content=content,
                model_name=self.model_name,
                timestamp=datetime.utcnow(),
                metadata={
                    "prompt_length": len(full_prompt),
                    "response_length": len(content),
                    "finish_reason": response.choices[0].finish_reason,
                    "model_version": response.model
                },
                usage_stats=usage_stats,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            raise ModelException(f"OpenAI response generation failed: {e}")
    
    async def generate_streaming_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI's API."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Prepare the full prompt with context
            full_prompt = self._prepare_context_prompt(prompt, context, memory_context)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_prompt}
            ]
            
            # Make streaming API call
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            # Yield response chunks
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating OpenAI streaming response: {e}")
            raise ModelException(f"OpenAI streaming response failed: {e}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the OpenAI model."""
        return """You are an AI assistant with access to a long-term memory system called CAIM (Cognitive AI Memory). 

You have access to:
1. Relevant memories from previous conversations
2. Current conversation context
3. User interaction history

Use this information to provide personalized, contextually aware responses. Reference past interactions when relevant, but don't overwhelm the user with too much historical information.

Be helpful, accurate, and maintain consistency with your past responses and the user's preferences."""
    
    async def _test_connection(self) -> None:
        """Test the OpenAI API connection."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            
            if not response.choices:
                raise ModelException("No response from OpenAI API")
                
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            raise ModelException(f"OpenAI connection failed: {e}")
    
    def _extract_usage_stats(self, response) -> Dict[str, Any]:
        """Extract usage statistics from OpenAI response."""
        try:
            if hasattr(response, 'usage') and response.usage:
                return {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            return {}
        except Exception as e:
            logger.error(f"Error extracting usage stats: {e}")
            return {}
    
    def _calculate_confidence_score(self, response) -> Optional[float]:
        """Calculate confidence score for OpenAI response."""
        try:
            # OpenAI doesn't provide confidence scores directly
            # We can use finish_reason as a proxy
            if hasattr(response, 'choices') and response.choices:
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "stop":
                    return 0.9  # High confidence for complete responses
                elif finish_reason == "length":
                    return 0.7  # Medium confidence for length-limited responses
                else:
                    return 0.5  # Lower confidence for other reasons
            return None
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return None
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model."""
        try:
            # OpenAI doesn't provide a direct model info endpoint
            # Return basic information
            return {
                "model_name": self.model_name,
                "provider": "OpenAI",
                "type": "chat_completion",
                "supports_streaming": True,
                "max_context_length": self._get_context_length(),
                "default_max_tokens": self.default_max_tokens
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def _get_context_length(self) -> int:
        """Get the context length for the model."""
        context_lengths = {
            "gpt-4o-mini": 128000,
            "gpt-4o-nano": 32000,
            "gpt-4o": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384
        }
        return context_lengths.get(self.model_name, 4096)
    
    async def estimate_cost(
        self,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Estimate the cost of a request."""
        try:
            # Rough token estimation (4 characters ≈ 1 token)
            prompt_tokens = len(prompt) // 4
            completion_tokens = max_tokens or self.default_max_tokens
            
            # Pricing per 1K tokens (as of 2024, subject to change)
            pricing = {
                "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
                "gpt-4o-nano": {"prompt": 0.00005, "completion": 0.0002},
                "gpt-4o": {"prompt": 0.005, "completion": 0.015},
                "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
                "gpt-4": {"prompt": 0.03, "completion": 0.06},
                "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002}
            }
            
            model_pricing = pricing.get(self.model_name, pricing["gpt-4o-mini"])
            
            prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
            completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
            total_cost = prompt_cost + completion_cost
            
            return {
                "estimated_prompt_tokens": prompt_tokens,
                "estimated_completion_tokens": completion_tokens,
                "estimated_total_tokens": prompt_tokens + completion_tokens,
                "estimated_prompt_cost": prompt_cost,
                "estimated_completion_cost": completion_cost,
                "estimated_total_cost": total_cost,
                "currency": "USD"
            }
            
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return {}
    
    async def shutdown(self) -> None:
        """Shutdown the OpenAI model."""
        try:
            self.client = None
            await super().shutdown()
        except Exception as e:
            logger.error(f"Error shutting down OpenAI model: {e}")