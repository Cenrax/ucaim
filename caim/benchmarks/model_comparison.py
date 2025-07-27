"""Model comparison utilities for CAIM framework."""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models.base_model import BaseModel
from ..core.config import CAIMConfig
from .evaluation_metrics import EvaluationMetrics


logger = logging.getLogger(__name__)


class ModelComparison:
    """Utilities for comparing different model implementations."""
    
    def __init__(self, config: CAIMConfig):
        self.config = config
        self.evaluation_metrics = EvaluationMetrics()
        logger.info("Initialized ModelComparison")
    
    async def compare_models(
        self,
        models: List[BaseModel],
        test_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple models across various metrics."""
        try:
            if not models:
                return {"error": "No models provided for comparison"}
            
            if test_prompts is None:
                test_prompts = self._get_default_test_prompts()
            
            comparison_results = {
                "comparison_timestamp": datetime.utcnow().isoformat(),
                "models_tested": len(models),
                "test_prompts_count": len(test_prompts),
                "model_results": {},
                "comparison_summary": {}
            }
            
            # Test each model
            for model in models:
                model_results = await self._test_model_performance(model, test_prompts)
                comparison_results["model_results"][model.model_name] = model_results
            
            # Generate comparison summary
            comparison_results["comparison_summary"] = self._generate_comparison_summary(
                comparison_results["model_results"]
            )
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {"error": str(e)}
    
    async def _test_model_performance(
        self,
        model: BaseModel,
        test_prompts: List[str]
    ) -> Dict[str, Any]:
        """Test a single model's performance."""
        try:
            logger.info(f"Testing model: {model.model_name}")
            
            results = {
                "model_name": model.model_name,
                "response_times": [],
                "response_lengths": [],
                "successful_responses": 0,
                "failed_responses": 0,
                "average_response_time": 0.0,
                "average_response_length": 0.0,
                "success_rate": 0.0,
                "responses": []
            }
            
            for i, prompt in enumerate(test_prompts):
                try:
                    start_time = time.time()
                    
                    response = await model.generate_response(
                        prompt=prompt,
                        max_tokens=200,
                        temperature=0.7
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    results["response_times"].append(response_time)
                    results["response_lengths"].append(len(response.content))
                    results["successful_responses"] += 1
                    
                    results["responses"].append({
                        "prompt_index": i,
                        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                        "response_length": len(response.content),
                        "response_time": response_time,
                        "confidence_score": response.confidence_score
                    })
                    
                except Exception as e:
                    logger.error(f"Error testing prompt {i} with model {model.model_name}: {e}")
                    results["failed_responses"] += 1
                    results["responses"].append({
                        "prompt_index": i,
                        "error": str(e)
                    })
            
            # Calculate averages
            if results["response_times"]:
                results["average_response_time"] = sum(results["response_times"]) / len(results["response_times"])
            
            if results["response_lengths"]:
                results["average_response_length"] = sum(results["response_lengths"]) / len(results["response_lengths"])
            
            total_tests = results["successful_responses"] + results["failed_responses"]
            results["success_rate"] = results["successful_responses"] / total_tests if total_tests > 0 else 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing model {model.model_name}: {e}")
            return {
                "model_name": model.model_name,
                "error": str(e)
            }
    
    def _generate_comparison_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary comparison of model results."""
        try:
            summary = {
                "fastest_model": None,
                "most_reliable_model": None,
                "most_verbose_model": None,
                "performance_rankings": []
            }
            
            # Find fastest model (lowest average response time)
            fastest_time = float('inf')
            for model_name, results in model_results.items():
                if "error" not in results:
                    avg_time = results.get("average_response_time", float('inf'))
                    if avg_time < fastest_time:
                        fastest_time = avg_time
                        summary["fastest_model"] = model_name
            
            # Find most reliable model (highest success rate)
            highest_success_rate = 0.0
            for model_name, results in model_results.items():
                if "error" not in results:
                    success_rate = results.get("success_rate", 0.0)
                    if success_rate > highest_success_rate:
                        highest_success_rate = success_rate
                        summary["most_reliable_model"] = model_name
            
            # Find most verbose model (highest average response length)
            longest_response = 0.0
            for model_name, results in model_results.items():
                if "error" not in results:
                    avg_length = results.get("average_response_length", 0.0)
                    if avg_length > longest_response:
                        longest_response = avg_length
                        summary["most_verbose_model"] = model_name
            
            # Create performance rankings
            rankings = []
            for model_name, results in model_results.items():
                if "error" not in results:
                    # Composite score: success_rate * 0.5 + (1/response_time) * 0.3 + response_length/1000 * 0.2
                    success_rate = results.get("success_rate", 0.0)
                    response_time = results.get("average_response_time", float('inf'))
                    response_length = results.get("average_response_length", 0.0)
                    
                    time_score = 1.0 / response_time if response_time > 0 else 0.0
                    length_score = min(1.0, response_length / 1000.0)
                    
                    composite_score = (success_rate * 0.5 + 
                                     time_score * 0.3 + 
                                     length_score * 0.2)
                    
                    rankings.append({
                        "model_name": model_name,
                        "composite_score": composite_score,
                        "success_rate": success_rate,
                        "avg_response_time": response_time,
                        "avg_response_length": response_length
                    })
            
            # Sort by composite score
            rankings.sort(key=lambda x: x["composite_score"], reverse=True)
            summary["performance_rankings"] = rankings
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating comparison summary: {e}")
            return {"error": str(e)}
    
    def _get_default_test_prompts(self) -> List[str]:
        """Get default test prompts for model comparison."""
        return [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How do neural networks work?",
            "What are the applications of AI in healthcare?",
            "Describe the difference between supervised and unsupervised learning.",
            "What is natural language processing?",
            "How can AI help solve climate change?",
            "What are the ethical considerations in AI development?",
            "Explain computer vision and its uses.",
            "What does the future hold for artificial intelligence?"
        ]