"""Memory consolidation and forgetting mechanisms for CAIM framework."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .memory_types import Memory, MemoryType, MemoryImportance, InductiveThought
from ..core.config import CAIMConfig
from ..core.exceptions import MemoryException
from ..storage.memory_store import MemoryStore
from ..retrieval.memory_retriever import MemoryRetriever


logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Advanced memory consolidation and forgetting mechanisms.
    
    This class implements:
    - Intelligent memory consolidation based on patterns and importance
    - Memory decay and forgetting algorithms
    - Interference-based memory management
    - Memory optimization and cleanup
    """
    
    def __init__(
        self,
        config: CAIMConfig,
        memory_store: MemoryStore,
        memory_retriever: MemoryRetriever
    ):
        self.config = config
        self.memory_store = memory_store
        self.memory_retriever = memory_retriever
        
        # Consolidation parameters
        self.consolidation_threshold = config.consolidation_threshold
        self.min_memories_for_consolidation = 3
        self.max_consolidation_group_size = 10
        
        # Forgetting parameters
        self.base_decay_rate = 0.01  # Daily decay rate
        self.interference_threshold = 0.8  # Similarity threshold for interference
        self.importance_protection_factor = 2.0  # Protection for important memories
        
        # Pattern recognition
        self.pattern_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.consolidation_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized MemoryConsolidator")
    
    async def consolidate_memories(
        self,
        session_id: Optional[str] = None,
        force_consolidation: bool = False,
        max_consolidations: int = 10
    ) -> Dict[str, Any]:
        """
        Perform intelligent memory consolidation.
        
        Args:
            session_id: Optional session to consolidate (None for all)
            force_consolidation: Force consolidation even if thresholds not met
            max_consolidations: Maximum number of consolidations to perform
            
        Returns:
            Consolidation results
        """
        try:
            logger.info(f"Starting memory consolidation for session: {session_id or 'all'}")
            
            # Get memories to consolidate
            if session_id:
                memories = await self.memory_store.get_memories_by_session(session_id)
            else:
                memories = await self.memory_store.get_all_memories()
            
            if len(memories) < self.min_memories_for_consolidation:
                return {
                    "status": "insufficient_memories",
                    "memory_count": len(memories),
                    "min_required": self.min_memories_for_consolidation
                }
            
            # Find consolidation candidates
            consolidation_groups = await self._find_consolidation_groups(
                memories, force_consolidation
            )
            
            if not consolidation_groups:
                return {
                    "status": "no_consolidation_needed",
                    "memory_count": len(memories),
                    "threshold": self.consolidation_threshold
                }
            
            # Perform consolidations
            consolidation_results = []
            consolidations_performed = 0
            
            for group in consolidation_groups[:max_consolidations]:
                try:
                    result = await self._consolidate_memory_group(group, session_id)
                    if result:
                        consolidation_results.append(result)
                        consolidations_performed += 1
                        
                except Exception as e:
                    logger.error(f"Error consolidating memory group: {e}")
            
            # Update consolidation history
            self._update_consolidation_history(session_id, consolidations_performed)
            
            return {
                "status": "success",
                "session_id": session_id,
                "total_memories": len(memories),
                "consolidation_groups_found": len(consolidation_groups),
                "consolidations_performed": consolidations_performed,
                "consolidation_results": consolidation_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
            raise MemoryException(f"Memory consolidation failed: {e}")
    
    async def apply_memory_decay(
        self,
        max_age_days: int = 30,
        decay_function: str = "exponential"
    ) -> Dict[str, Any]:
        """
        Apply memory decay algorithms to reduce memory importance over time.
        
        Args:
            max_age_days: Maximum age before aggressive decay
            decay_function: Type of decay function ("exponential", "linear", "power")
            
        Returns:
            Decay application results
        """
        try:
            logger.info(f"Applying memory decay with function: {decay_function}")
            
            all_memories = await self.memory_store.get_all_memories()
            current_time = datetime.utcnow()
            
            decay_stats = {
                "total_memories": len(all_memories),
                "memories_decayed": 0,
                "memories_forgotten": 0,
                "average_decay_applied": 0.0
            }
            
            total_decay_applied = 0.0
            
            for memory in all_memories:
                # Skip critical memories and recent inductive thoughts
                if (memory.importance == MemoryImportance.CRITICAL or
                    (memory.memory_type == MemoryType.INDUCTIVE_THOUGHT and 
                     (current_time - memory.timestamp).days < 7)):
                    continue
                
                # Calculate memory age
                age_days = (current_time - memory.timestamp).total_seconds() / 86400
                
                # Apply decay function
                decay_amount = self._calculate_decay_amount(
                    age_days, memory.importance, decay_function, max_age_days
                )
                
                if decay_amount > 0:
                    # Apply decay to memory
                    original_decay = memory.decay_factor
                    new_decay_factor = max(0.1, memory.decay_factor - decay_amount)
                    memory.decay_factor = new_decay_factor
                    
                    # Update memory in store
                    await self.memory_store.update(memory)
                    
                    decay_stats["memories_decayed"] += 1
                    total_decay_applied += (original_decay - new_decay_factor)
                    
                    # Check if memory should be forgotten
                    if memory.decay_factor < 0.2 and memory.importance != MemoryImportance.CRITICAL:
                        await self._forget_memory(memory)
                        decay_stats["memories_forgotten"] += 1
            
            if decay_stats["memories_decayed"] > 0:
                decay_stats["average_decay_applied"] = total_decay_applied / decay_stats["memories_decayed"]
            
            logger.info(f"Memory decay complete: {decay_stats}")
            return decay_stats
            
        except Exception as e:
            logger.error(f"Error applying memory decay: {e}")
            return {"error": str(e)}
    
    async def handle_memory_interference(
        self,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Handle memory interference by identifying and resolving conflicting memories.
        
        Args:
            similarity_threshold: Threshold for considering memories as interfering
            
        Returns:
            Interference handling results
        """
        try:
            threshold = similarity_threshold or self.interference_threshold
            logger.info(f"Handling memory interference with threshold: {threshold}")
            
            all_memories = await self.memory_store.get_all_memories()
            
            # Group memories by session for interference analysis
            session_memories = defaultdict(list)
            for memory in all_memories:
                session_memories[memory.session_id].append(memory)
            
            interference_results = {
                "sessions_analyzed": len(session_memories),
                "interference_groups_found": 0,
                "resolutions_applied": 0,
                "memories_merged": 0,
                "memories_removed": 0
            }
            
            for session_id, memories in session_memories.items():
                if len(memories) < 2:
                    continue
                
                # Find interfering memory groups
                interfering_groups = await self._find_interfering_memories(
                    memories, threshold
                )
                
                interference_results["interference_groups_found"] += len(interfering_groups)
                
                # Resolve interference for each group
                for group in interfering_groups:
                    resolution = await self._resolve_memory_interference(group)
                    
                    if resolution["action"] == "merge":
                        interference_results["memories_merged"] += len(group) - 1
                    elif resolution["action"] == "remove":
                        interference_results["memories_removed"] += resolution["removed_count"]
                    
                    interference_results["resolutions_applied"] += 1
            
            logger.info(f"Memory interference handling complete: {interference_results}")
            return interference_results
            
        except Exception as e:
            logger.error(f"Error handling memory interference: {e}")
            return {"error": str(e)}
    
    async def optimize_memory_storage(self) -> Dict[str, Any]:
        """
        Optimize memory storage by removing redundant and low-value memories.
        
        Returns:
            Optimization results
        """
        try:
            logger.info("Starting memory storage optimization")
            
            all_memories = await self.memory_store.get_all_memories()
            original_count = len(all_memories)
            
            optimization_results = {
                "original_memory_count": original_count,
                "redundant_memories_removed": 0,
                "low_value_memories_removed": 0,
                "empty_memories_removed": 0,
                "final_memory_count": 0
            }
            
            # Remove empty or invalid memories
            for memory in all_memories:
                if not memory.content or len(memory.content.strip()) < 5:
                    await self.memory_store.delete(memory.id)
                    optimization_results["empty_memories_removed"] += 1
            
            # Remove low-value memories
            current_time = datetime.utcnow()
            for memory in all_memories:
                relevance_score = memory.calculate_relevance_score(current_time)
                
                if (relevance_score < 0.1 and 
                    memory.importance != MemoryImportance.CRITICAL and
                    memory.memory_type != MemoryType.INDUCTIVE_THOUGHT):
                    
                    await self.memory_store.delete(memory.id)
                    optimization_results["low_value_memories_removed"] += 1
            
            # Find and remove near-duplicate memories
            remaining_memories = await self.memory_store.get_all_memories()
            duplicates_removed = await self._remove_duplicate_memories(remaining_memories)
            optimization_results["redundant_memories_removed"] = duplicates_removed
            
            # Final count
            final_memories = await self.memory_store.get_all_memories()
            optimization_results["final_memory_count"] = len(final_memories)
            
            logger.info(f"Memory optimization complete: {optimization_results}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing memory storage: {e}")
            return {"error": str(e)}
    
    async def _find_consolidation_groups(
        self,
        memories: List[Memory],
        force_consolidation: bool = False
    ) -> List[List[Memory]]:
        """Find groups of memories that should be consolidated."""
        try:
            consolidation_groups = []
            processed_ids = set()
            
            # Group memories by type and session for better consolidation
            grouped_memories = defaultdict(list)
            for memory in memories:
                if memory.memory_type != MemoryType.INDUCTIVE_THOUGHT:  # Don't consolidate inductive thoughts
                    key = (memory.session_id, memory.memory_type)
                    grouped_memories[key].append(memory)
            
            for (session_id, memory_type), group_memories in grouped_memories.items():
                if len(group_memories) < self.min_memories_for_consolidation:
                    continue
                
                # Find similar memories within the group
                for i, memory in enumerate(group_memories):
                    if memory.id in processed_ids:
                        continue
                    
                    similar_group = [memory]
                    processed_ids.add(memory.id)
                    
                    for j, other_memory in enumerate(group_memories[i+1:], i+1):
                        if other_memory.id in processed_ids:
                            continue
                        
                        similarity = await self.memory_retriever.calculate_similarity(
                            memory.content, other_memory.content
                        )
                        
                        if similarity > self.consolidation_threshold or force_consolidation:
                            similar_group.append(other_memory)
                            processed_ids.add(other_memory.id)
                            
                            if len(similar_group) >= self.max_consolidation_group_size:
                                break
                    
                    if len(similar_group) >= self.min_memories_for_consolidation:
                        consolidation_groups.append(similar_group)
            
            return consolidation_groups
            
        except Exception as e:
            logger.error(f"Error finding consolidation groups: {e}")
            return []
    
    async def _consolidate_memory_group(
        self,
        memory_group: List[Memory],
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Consolidate a group of similar memories into an inductive thought."""
        try:
            if len(memory_group) < 2:
                return None
            
            # Extract patterns and themes from the memory group
            patterns = await self._extract_memory_patterns(memory_group)
            
            # Generate consolidated content
            consolidated_content = await self._generate_consolidated_content(
                memory_group, patterns
            )
            
            if not consolidated_content:
                return None
            
            # Calculate consolidated memory properties
            avg_importance = sum(m.importance.value for m in memory_group) / len(memory_group)
            consolidated_importance = min(MemoryImportance.HIGH, 
                                        MemoryImportance(avg_importance * 1.2))
            
            # Create inductive thought
            inductive_thought = InductiveThought(
                content=consolidated_content,
                session_id=session_id or memory_group[0].session_id,
                source_memories=[m.id for m in memory_group],
                confidence=min(0.9, 0.5 + (len(memory_group) * 0.1)),
                generalization_level=len(set(m.session_id for m in memory_group)),
                importance=consolidated_importance,
                metadata={
                    "consolidation_timestamp": datetime.utcnow().isoformat(),
                    "source_memory_count": len(memory_group),
                    "consolidation_patterns": patterns,
                    "consolidation_method": "pattern_based"
                }
            )
            
            # Store the inductive thought
            stored_thought = await self.memory_store.store(inductive_thought)
            
            # Remove original memories (optional, based on configuration)
            removed_memories = []
            for memory in memory_group:
                if memory.importance != MemoryImportance.CRITICAL:
                    await self.memory_store.delete(memory.id)
                    removed_memories.append(memory.id)
            
            return {
                "inductive_thought_id": stored_thought.id,
                "consolidated_content": consolidated_content,
                "source_memories": len(memory_group),
                "removed_memories": len(removed_memories),
                "patterns_found": patterns,
                "confidence": inductive_thought.confidence
            }
            
        except Exception as e:
            logger.error(f"Error consolidating memory group: {e}")
            return None
    
    async def _extract_memory_patterns(self, memories: List[Memory]) -> List[str]:
        """Extract patterns and themes from a group of memories."""
        try:
            patterns = []
            
            # Simple pattern extraction based on common words and phrases
            all_content = " ".join(m.content for m in memories).lower()
            words = all_content.split()
            
            # Find frequent words (excluding common stopwords)
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            word_counts = {}
            for word in words:
                if word not in stopwords and len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Extract patterns based on frequency
            frequent_words = [word for word, count in word_counts.items() if count >= 2]
            if frequent_words:
                patterns.append(f"frequent_terms: {', '.join(frequent_words[:5])}")
            
            # Extract temporal patterns
            timestamps = [m.timestamp for m in memories]
            if len(timestamps) > 1:
                time_span = max(timestamps) - min(timestamps)
                patterns.append(f"time_span: {time_span.days} days")
            
            # Extract importance patterns
            importance_levels = [m.importance.value for m in memories]
            avg_importance = sum(importance_levels) / len(importance_levels)
            patterns.append(f"avg_importance: {avg_importance:.2f}")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting memory patterns: {e}")
            return []
    
    async def _generate_consolidated_content(
        self,
        memories: List[Memory],
        patterns: List[str]
    ) -> Optional[str]:
        """Generate consolidated content from memory group."""
        try:
            # Simple consolidation - can be enhanced with NLP models
            contents = [m.content for m in memories]
            
            # Create a summary of the consolidated memories
            consolidated_parts = [
                f"Consolidated insight from {len(memories)} related memories:",
                f"Patterns identified: {'; '.join(patterns)}",
                f"Key themes: {self._extract_key_themes(contents)}",
                f"Summary: {self._create_simple_summary(contents)}"
            ]
            
            return "\n".join(consolidated_parts)
            
        except Exception as e:
            logger.error(f"Error generating consolidated content: {e}")
            return None
    
    def _extract_key_themes(self, contents: List[str]) -> str:
        """Extract key themes from content list."""
        try:
            # Simple theme extraction
            all_text = " ".join(contents).lower()
            
            # Look for repeated phrases
            words = all_text.split()
            themes = []
            
            # Find 2-word phrases that appear multiple times
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 6 and all_text.count(phrase) >= 2:
                    themes.append(phrase)
            
            return ", ".join(list(set(themes))[:3]) if themes else "general discussion"
            
        except Exception as e:
            logger.error(f"Error extracting themes: {e}")
            return "general discussion"
    
    def _create_simple_summary(self, contents: List[str]) -> str:
        """Create a simple summary of contents."""
        try:
            # Take the first sentence of the longest content as summary
            longest_content = max(contents, key=len)
            sentences = longest_content.split('. ')
            return sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return "Summary of consolidated memories"
    
    def _calculate_decay_amount(
        self,
        age_days: float,
        importance: MemoryImportance,
        decay_function: str,
        max_age_days: int
    ) -> float:
        """Calculate decay amount based on age and importance."""
        try:
            # Base decay rate adjusted by importance
            protection_factor = importance.value * self.importance_protection_factor
            adjusted_decay_rate = self.base_decay_rate / protection_factor
            
            if decay_function == "exponential":
                decay_amount = 1 - (1 - adjusted_decay_rate) ** age_days
            elif decay_function == "linear":
                decay_amount = adjusted_decay_rate * age_days
            elif decay_function == "power":
                decay_amount = adjusted_decay_rate * (age_days ** 0.5)
            else:
                decay_amount = adjusted_decay_rate * age_days
            
            # Apply accelerated decay for very old memories
            if age_days > max_age_days:
                decay_multiplier = 1 + ((age_days - max_age_days) / max_age_days)
                decay_amount *= decay_multiplier
            
            return min(0.9, decay_amount)  # Cap at 90% decay
            
        except Exception as e:
            logger.error(f"Error calculating decay amount: {e}")
            return 0.0
    
    async def _forget_memory(self, memory: Memory) -> bool:
        """Forget (delete) a memory that has decayed sufficiently."""
        try:
            # Final check before forgetting
            if memory.importance == MemoryImportance.CRITICAL:
                return False
            
            # Log forgetting event
            logger.info(f"Forgetting memory {memory.id} due to decay (factor: {memory.decay_factor})")
            
            # Delete memory
            success = await self.memory_store.delete(memory.id)
            return success
            
        except Exception as e:
            logger.error(f"Error forgetting memory: {e}")
            return False
    
    async def _find_interfering_memories(
        self,
        memories: List[Memory],
        threshold: float
    ) -> List[List[Memory]]:
        """Find groups of interfering memories."""
        try:
            interfering_groups = []
            processed = set()
            
            for i, memory in enumerate(memories):
                if memory.id in processed:
                    continue
                
                interfering_group = [memory]
                processed.add(memory.id)
                
                for j, other_memory in enumerate(memories[i+1:], i+1):
                    if other_memory.id in processed:
                        continue
                    
                    # Check for conflicting or highly similar content
                    similarity = await self.memory_retriever.calculate_similarity(
                        memory.content, other_memory.content
                    )
                    
                    if similarity > threshold:
                        interfering_group.append(other_memory)
                        processed.add(other_memory.id)
                
                if len(interfering_group) > 1:
                    interfering_groups.append(interfering_group)
            
            return interfering_groups
            
        except Exception as e:
            logger.error(f"Error finding interfering memories: {e}")
            return []
    
    async def _resolve_memory_interference(
        self,
        interfering_memories: List[Memory]
    ) -> Dict[str, Any]:
        """Resolve interference between similar memories."""
        try:
            if len(interfering_memories) < 2:
                return {"action": "none", "reason": "insufficient memories"}
            
            # Sort by importance and recency
            sorted_memories = sorted(
                interfering_memories,
                key=lambda m: (m.importance.value, m.timestamp),
                reverse=True
            )
            
            # Keep the most important and recent memory
            memory_to_keep = sorted_memories[0]
            memories_to_remove = sorted_memories[1:]
            
            # Check if memories are very similar (merge) or conflicting (remove)
            similarities = []
            for memory in memories_to_remove:
                similarity = await self.memory_retriever.calculate_similarity(
                    memory_to_keep.content, memory.content
                )
                similarities.append(similarity)
            
            avg_similarity = sum(similarities) / len(similarities)
            
            if avg_similarity > 0.9:  # Very similar - merge
                # Enhance the kept memory with information from others
                enhanced_content = self._merge_memory_contents(
                    memory_to_keep, memories_to_remove
                )
                memory_to_keep.content = enhanced_content
                await self.memory_store.update(memory_to_keep)
                
                # Remove the redundant memories
                removed_count = 0
                for memory in memories_to_remove:
                    if await self.memory_store.delete(memory.id):
                        removed_count += 1
                
                return {
                    "action": "merge",
                    "kept_memory_id": memory_to_keep.id,
                    "removed_count": removed_count,
                    "avg_similarity": avg_similarity
                }
            
            else:  # Conflicting - remove less important ones
                removed_count = 0
                for memory in memories_to_remove:
                    if memory.importance != MemoryImportance.CRITICAL:
                        if await self.memory_store.delete(memory.id):
                            removed_count += 1
                
                return {
                    "action": "remove",
                    "kept_memory_id": memory_to_keep.id,
                    "removed_count": removed_count,
                    "avg_similarity": avg_similarity
                }
                
        except Exception as e:
            logger.error(f"Error resolving memory interference: {e}")
            return {"action": "error", "error": str(e)}
    
    def _merge_memory_contents(
        self,
        primary_memory: Memory,
        secondary_memories: List[Memory]
    ) -> str:
        """Merge contents of similar memories."""
        try:
            all_contents = [primary_memory.content]
            all_contents.extend(m.content for m in secondary_memories)
            
            # Simple merge - take the longest content and add unique information
            primary_content = primary_memory.content
            
            # Extract unique information from secondary memories
            unique_info = []
            for memory in secondary_memories:
                # Simple uniqueness check
                if memory.content not in primary_content:
                    # Add if significantly different
                    similarity = len(set(memory.content.split()) & set(primary_content.split()))
                    total_words = len(set(memory.content.split()) | set(primary_content.split()))
                    
                    if similarity / total_words < 0.8:  # Less than 80% overlap
                        unique_info.append(memory.content[:100])  # Add first 100 chars
            
            if unique_info:
                merged_content = f"{primary_content}\n\nAdditional context: {'; '.join(unique_info)}"
            else:
                merged_content = primary_content
            
            return merged_content
            
        except Exception as e:
            logger.error(f"Error merging memory contents: {e}")
            return primary_memory.content
    
    async def _remove_duplicate_memories(self, memories: List[Memory]) -> int:
        """Remove near-duplicate memories."""
        try:
            removed_count = 0
            
            for i in range(len(memories)):
                for j in range(i + 1, len(memories)):
                    similarity = await self.memory_retriever.calculate_similarity(
                        memories[i].content, memories[j].content
                    )
                    
                    if similarity > 0.95:  # Very high similarity
                        # Remove the less important or older memory
                        if memories[i].importance.value >= memories[j].importance.value:
                            if memories[j].importance != MemoryImportance.CRITICAL:
                                await self.memory_store.delete(memories[j].id)
                                removed_count += 1
                        else:
                            if memories[i].importance != MemoryImportance.CRITICAL:
                                await self.memory_store.delete(memories[i].id)
                                removed_count += 1
                                break  # Stop checking this memory
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error removing duplicate memories: {e}")
            return 0
    
    def _update_consolidation_history(self, session_id: Optional[str], count: int) -> None:
        """Update consolidation history for analysis."""
        try:
            self.consolidation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "consolidations_performed": count
            })
            
            # Keep only last 100 entries
            if len(self.consolidation_history) > 100:
                self.consolidation_history = self.consolidation_history[-100:]
                
        except Exception as e:
            logger.error(f"Error updating consolidation history: {e}")
    
    async def get_consolidation_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory consolidation."""
        try:
            all_memories = await self.memory_store.get_all_memories()
            inductive_thoughts = [m for m in all_memories if m.memory_type == MemoryType.INDUCTIVE_THOUGHT]
            
            stats = {
                "total_memories": len(all_memories),
                "inductive_thoughts": len(inductive_thoughts),
                "consolidation_ratio": len(inductive_thoughts) / max(1, len(all_memories)),
                "recent_consolidations": len([
                    h for h in self.consolidation_history
                    if (datetime.utcnow() - datetime.fromisoformat(h["timestamp"])).days < 7
                ]),
                "consolidation_threshold": self.consolidation_threshold,
                "decay_rate": self.base_decay_rate,
                "interference_threshold": self.interference_threshold
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting consolidation statistics: {e}")
            return {}