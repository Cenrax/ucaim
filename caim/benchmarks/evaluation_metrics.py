"""Evaluation metrics for CAIM framework benchmarking."""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..memory.memory_types import Memory
from ..agents.base_agent import AgentResponse


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""
    metric_name: str
    score: float
    max_score: float
    percentage: float
    details: Dict[str, Any]
    timestamp: datetime


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for CAIM framework performance.
    
    Implements various metrics to evaluate:
    - Memory system performance
    - Agent response quality
    - Model comparison
    - System efficiency
    """
    
    def __init__(self):
        self.results_history: List[BenchmarkResult] = []
        logger.info("Initialized EvaluationMetrics")
    
    def evaluate_memory_retrieval_accuracy(
        self,
        retrieved_memories: List[Memory],
        relevant_memories: List[Memory],
        query: str
    ) -> BenchmarkResult:
        """
        Evaluate memory retrieval accuracy using precision, recall, and F1-score.
        
        Args:
            retrieved_memories: Memories retrieved by the system
            relevant_memories: Ground truth relevant memories
            query: Original query
            
        Returns:
            BenchmarkResult with retrieval accuracy metrics
        """
        try:
            retrieved_ids = {m.id for m in retrieved_memories}
            relevant_ids = {m.id for m in relevant_memories}
            
            if not relevant_ids:
                return BenchmarkResult(
                    metric_name="memory_retrieval_accuracy",
                    score=0.0,
                    max_score=1.0,
                    percentage=0.0,
                    details={"error": "No relevant memories provided"},
                    timestamp=datetime.utcnow()
                )
            
            # Calculate precision, recall, F1
            true_positives = len(retrieved_ids & relevant_ids)
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0
            recall = true_positives / len(relevant_ids)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate additional metrics
            false_positives = len(retrieved_ids - relevant_ids)
            false_negatives = len(relevant_ids - retrieved_ids)
            
            details = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "retrieved_count": len(retrieved_memories),
                "relevant_count": len(relevant_memories),
                "query": query[:100]
            }
            
            result = BenchmarkResult(
                metric_name="memory_retrieval_accuracy",
                score=f1_score,
                max_score=1.0,
                percentage=f1_score * 100,
                details=details,
                timestamp=datetime.utcnow()
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating memory retrieval accuracy: {e}")
            return BenchmarkResult(
                metric_name="memory_retrieval_accuracy",
                score=0.0,
                max_score=1.0,
                percentage=0.0,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    def evaluate_response_coherence(
        self,
        response: str,
        context: str,
        relevant_memories: List[Memory]
    ) -> BenchmarkResult:
        """
        Evaluate response coherence with context and retrieved memories.
        
        Args:
            response: Agent response
            context: Conversation context
            relevant_memories: Retrieved memories
            
        Returns:
            BenchmarkResult with coherence metrics
        """
        try:
            # Simple coherence evaluation based on word overlap and semantic consistency
            coherence_scores = []
            
            # Context coherence
            if context:
                context_coherence = self._calculate_text_overlap(response, context)
                coherence_scores.append(context_coherence)
            
            # Memory coherence
            if relevant_memories:
                memory_contents = [m.content for m in relevant_memories]
                for memory_content in memory_contents:
                    memory_coherence = self._calculate_text_overlap(response, memory_content)
                    coherence_scores.append(memory_coherence)
            
            # Overall coherence
            overall_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
            
            # Response quality metrics
            response_length = len(response)
            word_count = len(response.split())
            sentence_count = len([s for s in response.split('.') if s.strip()])
            avg_sentence_length = word_count / max(1, sentence_count)
            
            details = {
                "overall_coherence": overall_coherence,
                "context_coherence": coherence_scores[0] if coherence_scores and context else 0.0,
                "memory_coherence": sum(coherence_scores[1:]) / max(1, len(coherence_scores) - 1) if len(coherence_scores) > 1 else 0.0,
                "response_length": response_length,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length,
                "memories_used": len(relevant_memories)
            }
            
            result = BenchmarkResult(
                metric_name="response_coherence",
                score=overall_coherence,
                max_score=1.0,
                percentage=overall_coherence * 100,
                details=details,
                timestamp=datetime.utcnow()
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating response coherence: {e}")
            return BenchmarkResult(
                metric_name="response_coherence",
                score=0.0,
                max_score=1.0,
                percentage=0.0,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    def evaluate_memory_consolidation_quality(
        self,
        original_memories: List[Memory],
        consolidated_memory: Memory,
        consolidation_metadata: Dict[str, Any]
    ) -> BenchmarkResult:
        """
        Evaluate the quality of memory consolidation.
        
        Args:
            original_memories: Original memories that were consolidated
            consolidated_memory: Resulting consolidated memory
            consolidation_metadata: Metadata about the consolidation process
            
        Returns:
            BenchmarkResult with consolidation quality metrics
        """
        try:
            if not original_memories or not consolidated_memory:
                return BenchmarkResult(
                    metric_name="memory_consolidation_quality",
                    score=0.0,
                    max_score=1.0,
                    percentage=0.0,
                    details={"error": "Invalid input memories"},
                    timestamp=datetime.utcnow()
                )
            
            # Information preservation score
            original_content = " ".join(m.content for m in original_memories)
            consolidated_content = consolidated_memory.content
            
            information_preservation = self._calculate_information_preservation(
                original_content, consolidated_content
            )
            
            # Compression ratio
            original_length = sum(len(m.content) for m in original_memories)
            consolidated_length = len(consolidated_content)
            compression_ratio = consolidated_length / original_length if original_length > 0 else 0.0
            
            # Pattern extraction quality
            patterns_found = len(consolidation_metadata.get("consolidation_patterns", []))
            pattern_quality = min(1.0, patterns_found / 3.0)  # Normalize to 0-1
            
            # Importance preservation
            original_avg_importance = sum(m.importance.value for m in original_memories) / len(original_memories)
            consolidated_importance = consolidated_memory.importance.value
            importance_preservation = min(1.0, consolidated_importance / original_avg_importance)
            
            # Overall quality score
            quality_score = (
                information_preservation * 0.4 +
                pattern_quality * 0.3 +
                importance_preservation * 0.2 +
                (1.0 - compression_ratio) * 0.1  # Lower compression ratio is better
            )
            
            details = {
                "information_preservation": information_preservation,
                "compression_ratio": compression_ratio,
                "pattern_quality": pattern_quality,
                "importance_preservation": importance_preservation,
                "patterns_found": patterns_found,
                "original_memories_count": len(original_memories),
                "original_total_length": original_length,
                "consolidated_length": consolidated_length,
                "consolidation_confidence": consolidation_metadata.get("confidence", 0.0)
            }
            
            result = BenchmarkResult(
                metric_name="memory_consolidation_quality",
                score=quality_score,
                max_score=1.0,
                percentage=quality_score * 100,
                details=details,
                timestamp=datetime.utcnow()
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating memory consolidation quality: {e}")
            return BenchmarkResult(
                metric_name="memory_consolidation_quality",
                score=0.0,
                max_score=1.0,
                percentage=0.0,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    def evaluate_contextual_awareness(
        self,
        current_response: str,
        conversation_history: List[Dict[str, str]],
        relevant_memories: List[Memory]
    ) -> BenchmarkResult:
        """
        Evaluate how well the system maintains contextual awareness.
        
        Args:
            current_response: Current agent response
            conversation_history: Previous conversation turns
            relevant_memories: Retrieved relevant memories
            
        Returns:
            BenchmarkResult with contextual awareness metrics
        """
        try:
            awareness_scores = []
            
            # Conversation context awareness
            if conversation_history:
                recent_context = " ".join([
                    turn.get("user_message", "") + " " + turn.get("agent_response", "")
                    for turn in conversation_history[-3:]  # Last 3 turns
                ])
                
                context_awareness = self._calculate_text_overlap(current_response, recent_context)
                awareness_scores.append(context_awareness)
            
            # Memory context awareness
            if relevant_memories:
                memory_context = " ".join(m.content for m in relevant_memories)
                memory_awareness = self._calculate_text_overlap(current_response, memory_context)
                awareness_scores.append(memory_awareness)
            
            # Reference consistency (check for contradictions)
            consistency_score = self._evaluate_consistency(
                current_response, conversation_history, relevant_memories
            )
            awareness_scores.append(consistency_score)
            
            # Overall contextual awareness
            overall_awareness = sum(awareness_scores) / len(awareness_scores) if awareness_scores else 0.0
            
            details = {
                "overall_awareness": overall_awareness,
                "conversation_awareness": awareness_scores[0] if len(awareness_scores) > 0 else 0.0,
                "memory_awareness": awareness_scores[1] if len(awareness_scores) > 1 else 0.0,
                "consistency_score": consistency_score,
                "conversation_turns": len(conversation_history),
                "relevant_memories_count": len(relevant_memories)
            }
            
            result = BenchmarkResult(
                metric_name="contextual_awareness",
                score=overall_awareness,
                max_score=1.0,
                percentage=overall_awareness * 100,
                details=details,
                timestamp=datetime.utcnow()
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating contextual awareness: {e}")
            return BenchmarkResult(
                metric_name="contextual_awareness",
                score=0.0,
                max_score=1.0,
                percentage=0.0,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    def evaluate_personalization_quality(
        self,
        responses: List[str],
        user_preferences: Dict[str, Any],
        interaction_history: List[Dict[str, Any]]
    ) -> BenchmarkResult:
        """
        Evaluate the quality of personalization in responses.
        
        Args:
            responses: Agent responses to evaluate
            user_preferences: Known user preferences
            interaction_history: History of user interactions
            
        Returns:
            BenchmarkResult with personalization metrics
        """
        try:
            personalization_scores = []
            
            # Preference alignment
            if user_preferences:
                preference_alignment = self._calculate_preference_alignment(
                    responses, user_preferences
                )
                personalization_scores.append(preference_alignment)
            
            # Adaptation over time
            if len(responses) > 1:
                adaptation_score = self._calculate_adaptation_score(responses)
                personalization_scores.append(adaptation_score)
            
            # User-specific language patterns
            if interaction_history:
                language_adaptation = self._calculate_language_adaptation(
                    responses, interaction_history
                )
                personalization_scores.append(language_adaptation)
            
            # Overall personalization
            overall_personalization = sum(personalization_scores) / len(personalization_scores) if personalization_scores else 0.0
            
            details = {
                "overall_personalization": overall_personalization,
                "preference_alignment": personalization_scores[0] if len(personalization_scores) > 0 else 0.0,
                "adaptation_score": personalization_scores[1] if len(personalization_scores) > 1 else 0.0,
                "language_adaptation": personalization_scores[2] if len(personalization_scores) > 2 else 0.0,
                "responses_analyzed": len(responses),
                "preferences_count": len(user_preferences),
                "interaction_history_length": len(interaction_history)
            }
            
            result = BenchmarkResult(
                metric_name="personalization_quality",
                score=overall_personalization,
                max_score=1.0,
                percentage=overall_personalization * 100,
                details=details,
                timestamp=datetime.utcnow()
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating personalization quality: {e}")
            return BenchmarkResult(
                metric_name="personalization_quality",
                score=0.0,
                max_score=1.0,
                percentage=0.0,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    def evaluate_system_efficiency(
        self,
        response_times: List[float],
        memory_usage: Dict[str, Any],
        resource_usage: Dict[str, Any]
    ) -> BenchmarkResult:
        """
        Evaluate system efficiency metrics.
        
        Args:
            response_times: List of response times in seconds
            memory_usage: Memory usage statistics
            resource_usage: Resource usage statistics
            
        Returns:
            BenchmarkResult with efficiency metrics
        """
        try:
            efficiency_scores = []
            
            # Response time efficiency
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                response_efficiency = max(0.0, 1.0 - (avg_response_time / 10.0))  # Normalize to 10 seconds max
                efficiency_scores.append(response_efficiency)
            
            # Memory efficiency
            if memory_usage:
                total_memories = memory_usage.get("total_memories", 0)
                storage_size = memory_usage.get("storage_size_mb", 0)
                memory_efficiency = max(0.0, 1.0 - (storage_size / (total_memories * 0.1))) if total_memories > 0 else 1.0
                efficiency_scores.append(memory_efficiency)
            
            # Resource efficiency
            if resource_usage:
                cpu_usage = resource_usage.get("cpu_percent", 0)
                ram_usage = resource_usage.get("ram_percent", 0)
                resource_efficiency = max(0.0, 1.0 - ((cpu_usage + ram_usage) / 200.0))  # Normalize to 100% each
                efficiency_scores.append(resource_efficiency)
            
            # Overall efficiency
            overall_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0
            
            details = {
                "overall_efficiency": overall_efficiency,
                "response_efficiency": efficiency_scores[0] if len(efficiency_scores) > 0 else 0.0,
                "memory_efficiency": efficiency_scores[1] if len(efficiency_scores) > 1 else 0.0,
                "resource_efficiency": efficiency_scores[2] if len(efficiency_scores) > 2 else 0.0,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
                "response_time_std": self._calculate_std(response_times) if response_times else 0.0,
                "memory_usage": memory_usage,
                "resource_usage": resource_usage
            }
            
            result = BenchmarkResult(
                metric_name="system_efficiency",
                score=overall_efficiency,
                max_score=1.0,
                percentage=overall_efficiency * 100,
                details=details,
                timestamp=datetime.utcnow()
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating system efficiency: {e}")
            return BenchmarkResult(
                metric_name="system_efficiency",
                score=0.0,
                max_score=1.0,
                percentage=0.0,
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate text overlap using Jaccard similarity."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating text overlap: {e}")
            return 0.0
    
    def _calculate_information_preservation(self, original: str, consolidated: str) -> float:
        """Calculate how well information is preserved in consolidation."""
        try:
            original_words = set(original.lower().split())
            consolidated_words = set(consolidated.lower().split())
            
            if not original_words:
                return 1.0 if not consolidated_words else 0.0
            
            preserved_words = len(original_words & consolidated_words)
            preservation_ratio = preserved_words / len(original_words)
            
            # Bonus for extracting key concepts (longer words are often more important)
            important_words = {word for word in original_words if len(word) > 5}
            preserved_important = len(important_words & consolidated_words)
            importance_bonus = preserved_important / len(important_words) if important_words else 0.0
            
            return min(1.0, preservation_ratio + (importance_bonus * 0.2))
            
        except Exception as e:
            logger.error(f"Error calculating information preservation: {e}")
            return 0.0
    
    def _evaluate_consistency(
        self,
        response: str,
        conversation_history: List[Dict[str, str]],
        relevant_memories: List[Memory]
    ) -> float:
        """Evaluate consistency of response with history and memories."""
        try:
            # Simple consistency check - look for contradictory statements
            response_lower = response.lower()
            
            # Check for contradiction keywords
            contradiction_indicators = ["but", "however", "although", "despite", "contrary"]
            contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in response_lower)
            
            # Penalize excessive contradictions
            consistency_score = max(0.0, 1.0 - (contradiction_count * 0.2))
            
            return consistency_score
            
        except Exception as e:
            logger.error(f"Error evaluating consistency: {e}")
            return 0.5  # Neutral score on error
    
    def _calculate_preference_alignment(self, responses: List[str], preferences: Dict[str, Any]) -> float:
        """Calculate alignment with user preferences."""
        try:
            if not preferences:
                return 0.0
            
            alignment_scores = []
            
            for response in responses:
                response_lower = response.lower()
                
                # Check for preferred topics
                preferred_topics = preferences.get("topics", [])
                if preferred_topics:
                    topic_score = sum(1 for topic in preferred_topics if topic.lower() in response_lower)
                    topic_score = min(1.0, topic_score / len(preferred_topics))
                    alignment_scores.append(topic_score)
                
                # Check for preferred communication style
                preferred_style = preferences.get("communication_style", "")
                if preferred_style:
                    style_indicators = {
                        "formal": ["please", "thank you", "sincerely"],
                        "casual": ["hey", "cool", "awesome"],
                        "technical": ["specifically", "precisely", "algorithm"]
                    }
                    
                    indicators = style_indicators.get(preferred_style.lower(), [])
                    if indicators:
                        style_score = sum(1 for indicator in indicators if indicator in response_lower)
                        style_score = min(1.0, style_score / len(indicators))
                        alignment_scores.append(style_score)
            
            return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating preference alignment: {e}")
            return 0.0
    
    def _calculate_adaptation_score(self, responses: List[str]) -> float:
        """Calculate adaptation score based on response evolution."""
        try:
            if len(responses) < 2:
                return 0.0
            
            # Simple adaptation metric: vocabulary diversity over time
            vocab_sizes = []
            for i, response in enumerate(responses):
                # Consider cumulative vocabulary up to this point
                cumulative_text = " ".join(responses[:i+1])
                vocab_size = len(set(cumulative_text.lower().split()))
                vocab_sizes.append(vocab_size)
            
            # Calculate growth rate
            if len(vocab_sizes) > 1:
                growth_rate = (vocab_sizes[-1] - vocab_sizes[0]) / vocab_sizes[0]
                adaptation_score = min(1.0, growth_rate)
            else:
                adaptation_score = 0.0
            
            return max(0.0, adaptation_score)
            
        except Exception as e:
            logger.error(f"Error calculating adaptation score: {e}")
            return 0.0
    
    def _calculate_language_adaptation(self, responses: List[str], interaction_history: List[Dict[str, Any]]) -> float:
        """Calculate language adaptation based on user interaction patterns."""
        try:
            if not interaction_history:
                return 0.0
            
            # Extract user language patterns
            user_messages = [interaction.get("user_message", "") for interaction in interaction_history]
            user_text = " ".join(user_messages).lower()
            user_words = set(user_text.split())
            
            # Calculate how much agent responses adapt to user language
            adaptation_scores = []
            for response in responses:
                response_words = set(response.lower().split())
                overlap = len(user_words & response_words) / len(user_words) if user_words else 0.0
                adaptation_scores.append(overlap)
            
            return sum(adaptation_scores) / len(adaptation_scores) if adaptation_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating language adaptation: {e}")
            return 0.0
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        try:
            if len(values) < 2:
                return 0.0
            
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return math.sqrt(variance)
            
        except Exception as e:
            logger.error(f"Error calculating standard deviation: {e}")
            return 0.0
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        try:
            if not self.results_history:
                return {"error": "No evaluation results available"}
            
            # Group results by metric
            metric_groups = {}
            for result in self.results_history:
                metric_name = result.metric_name
                if metric_name not in metric_groups:
                    metric_groups[metric_name] = []
                metric_groups[metric_name].append(result)
            
            # Calculate summary statistics for each metric
            metric_summaries = {}
            for metric_name, results in metric_groups.items():
                scores = [r.score for r in results]
                
                metric_summaries[metric_name] = {
                    "count": len(scores),
                    "average_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "std_dev": self._calculate_std(scores),
                    "latest_score": scores[-1] if scores else 0.0
                }
            
            # Overall system score
            all_scores = [r.score for r in self.results_history]
            overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            return {
                "overall_score": overall_score,
                "overall_percentage": overall_score * 100,
                "total_evaluations": len(self.results_history),
                "metrics_evaluated": len(metric_groups),
                "metric_summaries": metric_summaries,
                "evaluation_period": {
                    "start": min(r.timestamp for r in self.results_history).isoformat(),
                    "end": max(r.timestamp for r in self.results_history).isoformat()
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {"error": str(e)}
    
    def clear_history(self) -> None:
        """Clear evaluation results history."""
        self.results_history.clear()
        logger.info("Cleared evaluation results history")