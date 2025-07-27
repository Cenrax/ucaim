"""Model factory for creating different model instances."""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from .base_model import BaseModel
from .openai_model import OpenAIModel
from .huggingface_model import HuggingFaceModel
from ..core.config import CAIMConfig
from ..core.exceptions import ModelException


logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class SupportedModels:
    """Registry of supported models."""
    
    OPENAI_MODELS = [
        "gpt-4o-mini",
        "gpt-4o-nano",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4-32k"
    ]
    
    HUGGINGFACE_MODELS = [
        # Qwen models
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct", 
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        
        # Gemma models
        "google/gemma-2b-it",
        "google/gemma-7b-it",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        
        # Other popular models
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    
    @classmethod
    def get_all_models(cls) -> Dict[str, List[str]]:
        """Get all supported models by provider."""
        return {
            "openai": cls.OPENAI_MODELS,
            "huggingface": cls.HUGGINGFACE_MODELS
        }
    
    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """Check if a model is supported."""
        return (model_name in cls.OPENAI_MODELS or 
                model_name in cls.HUGGINGFACE_MODELS)
    
    @classmethod
    def get_provider(cls, model_name: str) -> Optional[str]:
        """Get the provider for a model."""
        if model_name in cls.OPENAI_MODELS:
            return "openai"
        elif model_name in cls.HUGGINGFACE_MODELS:
            return "huggingface"
        return None


def create_model(
    model_name: str,
    config: CAIMConfig,
    provider: Optional[str] = None,
    **kwargs
) -> BaseModel:
    """
    Create a model instance.
    
    Args:
        model_name: Name of the model to create
        config: CAIM configuration
        provider: Model provider (auto-detected if None)
        **kwargs: Additional model-specific parameters
        
    Returns:
        Configured model instance
        
    Raises:
        ModelException: If model creation fails
    """
    try:
        # Auto-detect provider if not specified
        if provider is None:
            provider = SupportedModels.get_provider(model_name)
            if provider is None:
                raise ModelException(f"Unknown model: {model_name}")
        
        # Validate provider
        if provider not in [p.value for p in ModelProvider]:
            raise ModelException(f"Unsupported provider: {provider}")
        
        # Create model instance
        if provider == ModelProvider.OPENAI.value:
            return OpenAIModel(config, model_name, **kwargs)
        elif provider == ModelProvider.HUGGINGFACE.value:
            return HuggingFaceModel(config, model_name, **kwargs)
        else:
            raise ModelException(f"No implementation for provider: {provider}")
            
    except Exception as e:
        logger.error(f"Error creating model {model_name}: {e}")
        raise ModelException(f"Model creation failed: {e}")


def create_benchmark_models(config: CAIMConfig) -> List[BaseModel]:
    """
    Create a set of models for benchmarking.
    
    Args:
        config: CAIM configuration
        
    Returns:
        List of model instances for benchmarking
    """
    try:
        models = []
        
        # OpenAI models (if API key available)
        if config.openai_api_key:
            try:
                models.append(create_model("gpt-4o-mini", config))
                logger.info("Added OpenAI GPT-4o-mini for benchmarking")
            except Exception as e:
                logger.warning(f"Failed to create OpenAI model: {e}")
            
            # Also add nano if available
            try:
                models.append(create_model("gpt-4o-nano", config))
                logger.info("Added OpenAI GPT-4o-nano for benchmarking")
            except Exception as e:
                logger.warning(f"Failed to create GPT-4o-nano: {e}")
        
        # HuggingFace models
        benchmark_hf_models = [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "google/gemma-2b-it"
        ]
        
        for model_name in benchmark_hf_models:
            try:
                models.append(create_model(model_name, config))
                logger.info(f"Added {model_name} for benchmarking")
            except Exception as e:
                logger.warning(f"Failed to create {model_name}: {e}")
        
        if not models:
            logger.warning("No models available for benchmarking")
        
        return models
        
    except Exception as e:
        logger.error(f"Error creating benchmark models: {e}")
        return []


def get_model_recommendations(
    use_case: str = "general",
    resource_constraints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Get model recommendations based on use case and constraints.
    
    Args:
        use_case: Use case ("general", "chat", "memory", "research")
        resource_constraints: Resource constraints (memory, compute, etc.)
        
    Returns:
        List of recommended models with metadata
    """
    try:
        recommendations = []
        
        # Default resource constraints
        if resource_constraints is None:
            resource_constraints = {}
        
        max_memory_gb = resource_constraints.get("max_memory_gb", 8)
        require_local = resource_constraints.get("require_local", False)
        max_cost_per_1k_tokens = resource_constraints.get("max_cost_per_1k_tokens", 0.01)
        
        # Model recommendations by use case
        if use_case == "general":
            recommendations.extend([
                {
                    "model_name": "gpt-4o-mini",
                    "provider": "openai",
                    "pros": ["Fast", "Very cost-effective", "Latest OpenAI tech", "Excellent performance"],
                    "cons": ["Requires API key", "Online only"],
                    "estimated_memory_gb": 0,  # API-based
                    "estimated_cost_per_1k": 0.00015,
                    "use_cases": ["general", "chat", "memory"]
                },
                {
                    "model_name": "gpt-4o-nano",
                    "provider": "openai",
                    "pros": ["Ultra-fast", "Extremely cost-effective", "Ultra-low latency"],
                    "cons": ["Requires API key", "Online only", "Limited capabilities"],
                    "estimated_memory_gb": 0,  # API-based
                    "estimated_cost_per_1k": 0.00005,
                    "use_cases": ["general", "chat", "quick_responses"]
                },
                {
                    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                    "provider": "huggingface",
                    "pros": ["Local deployment", "Good performance", "Multilingual"],
                    "cons": ["Requires GPU for best performance"],
                    "estimated_memory_gb": 3,
                    "estimated_cost_per_1k": 0,  # Local
                    "use_cases": ["general", "chat", "memory", "research"]
                }
            ])
        
        elif use_case == "research":
            recommendations.extend([
                {
                    "model_name": "Qwen/Qwen2.5-3B-Instruct",
                    "provider": "huggingface",
                    "pros": ["Better reasoning", "Local deployment", "Research-friendly"],
                    "cons": ["Higher memory requirements"],
                    "estimated_memory_gb": 6,
                    "estimated_cost_per_1k": 0,
                    "use_cases": ["research", "memory", "analysis"]
                },
                {
                    "model_name": "google/gemma-2b-it",
                    "provider": "huggingface",
                    "pros": ["Google-developed", "Good instruction following"],
                    "cons": ["Newer model, less tested"],
                    "estimated_memory_gb": 4,
                    "estimated_cost_per_1k": 0,
                    "use_cases": ["research", "chat", "memory"]
                }
            ])
        
        # Filter by constraints
        filtered_recommendations = []
        for rec in recommendations:
            # Memory constraint
            if rec["estimated_memory_gb"] > max_memory_gb:
                continue
            
            # Local requirement
            if require_local and rec["provider"] == "openai":
                continue
            
            # Cost constraint
            if rec["estimated_cost_per_1k"] > max_cost_per_1k_tokens:
                continue
            
            # Use case match
            if use_case not in rec["use_cases"]:
                continue
            
            filtered_recommendations.append(rec)
        
        # Sort by suitability score
        def suitability_score(rec):
            score = 0
            if use_case in rec["use_cases"]:
                score += 2
            if rec["estimated_memory_gb"] <= max_memory_gb / 2:
                score += 1
            if rec["estimated_cost_per_1k"] == 0:  # Local models
                score += 1
            return score
        
        filtered_recommendations.sort(key=suitability_score, reverse=True)
        
        return filtered_recommendations
        
    except Exception as e:
        logger.error(f"Error getting model recommendations: {e}")
        return []


async def benchmark_model_performance(
    models: List[BaseModel],
    test_prompts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Benchmark performance of multiple models.
    
    Args:
        models: List of models to benchmark
        test_prompts: List of test prompts
        
    Returns:
        Benchmark results
    """
    try:
        if test_prompts is None:
            test_prompts = [
                "What is artificial intelligence?",
                "Explain the concept of memory in AI systems.",
                "How can AI systems maintain context across conversations?"
            ]
        
        results = {}
        
        for model in models:
            model_name = model.model_name
            logger.info(f"Benchmarking model: {model_name}")
            
            model_results = {
                "model_name": model_name,
                "responses": [],
                "average_response_time": 0,
                "total_tokens": 0,
                "errors": 0
            }
            
            total_time = 0
            
            for i, prompt in enumerate(test_prompts):
                try:
                    import time
                    start_time = time.time()
                    
                    response = await model.generate_response(prompt, max_tokens=100)
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    total_time += response_time
                    
                    model_results["responses"].append({
                        "prompt": prompt,
                        "response": response.content,
                        "response_time": response_time,
                        "usage_stats": response.usage_stats
                    })
                    
                    if response.usage_stats:
                        model_results["total_tokens"] += response.usage_stats.get("total_tokens", 0)
                    
                except Exception as e:
                    logger.error(f"Error benchmarking {model_name} with prompt {i}: {e}")
                    model_results["errors"] += 1
            
            model_results["average_response_time"] = total_time / len(test_prompts)
            results[model_name] = model_results
        
        return results
        
    except Exception as e:
        logger.error(f"Error benchmarking models: {e}")
        return {}