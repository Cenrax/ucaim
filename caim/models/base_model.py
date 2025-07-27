"""Base model interface for CAIM framework."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from ..core.config import CAIMConfig


logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from a model."""
    content: str
    model_name: str
    timestamp: datetime
    metadata: Dict[str, Any]
    usage_stats: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None


class BaseModel(ABC):
    """Abstract base class for all models in CAIM framework."""
    
    def __init__(self, config: CAIMConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self.is_initialized = False
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            context: Additional context
            memory_context: Relevant memories for context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse object
        """
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the model.
        
        Args:
            prompt: The input prompt
            context: Additional context
            memory_context: Relevant memories for context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Yields:
            Response chunks as they are generated
        """
        pass
    
    def _prepare_context_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        memory_context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Prepare the full prompt with context and memory.
        
        Args:
            prompt: Base prompt
            context: Additional context
            memory_context: Relevant memories
            
        Returns:
            Full prompt with context
        """
        try:
            prompt_parts = []
            
            # Add memory context if available
            if memory_context:
                prompt_parts.append("Relevant memories from previous interactions:")
                for i, memory in enumerate(memory_context[:5], 1):  # Limit to top 5 memories
                    content = memory.get("content", "")
                    importance = memory.get("importance", 0.5)
                    timestamp = memory.get("timestamp", "")
                    
                    prompt_parts.append(f"{i}. [{importance:.1f}] {content[:200]}...")
                    if timestamp:
                        prompt_parts.append(f"   (from: {timestamp})")
                
                prompt_parts.append("")
            
            # Add current context if available
            if context:
                prompt_parts.append(f"Current conversation context:\n{context}\n")
            
            # Add the main prompt
            prompt_parts.append(f"User: {prompt}")
            prompt_parts.append("Assistant:")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Error preparing context prompt: {e}")
            return prompt
    
    def _extract_usage_stats(self, response_data: Any) -> Dict[str, Any]:
        """Extract usage statistics from model response."""
        return {}
    
    def _calculate_confidence_score(self, response_data: Any) -> Optional[float]:
        """Calculate confidence score for the response."""
        return None
    
    @property
    def is_available(self) -> bool:
        """Check if the model is available and ready to use."""
        return self.is_initialized
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the model."""
        try:
            if not self.is_initialized:
                return {
                    "status": "not_initialized",
                    "model_name": self.model_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Try a simple generation
            test_response = await self.generate_response(
                "Test", max_tokens=10, temperature=0.1
            )
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "test_response_length": len(test_response.content),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.model_name}: {e}")
            return {
                "status": "unhealthy",
                "model_name": self.model_name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def shutdown(self) -> None:
        """Shutdown the model and cleanup resources."""
        try:
            self.is_initialized = False
            logger.info(f"Shutdown {self.__class__.__name__}")
        except Exception as e:
            logger.error(f"Error during model shutdown: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            import asyncio
            try:
                asyncio.create_task(self.shutdown())
            except:
                pass