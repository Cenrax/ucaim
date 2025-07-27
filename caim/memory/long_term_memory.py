"""Long-Term Memory implementation for CAIM framework."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

from .memory_types import Memory, MemoryType, MemoryImportance, InductiveThought, MemoryCluster
from ..core.config import CAIMConfig
from ..core.exceptions import MemoryException
from ..storage.memory_store import MemoryStore
from ..retrieval.memory_retriever import MemoryRetriever


logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Long-Term Memory (LTM) stores historical data as 'inductive thoughts' - 
    concise summaries of key events and patterns across conversations.
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
        
        self.consolidation_threshold = config.consolidation_threshold
        self.decay_factor = config.memory_decay_factor
        
        self.memory_clusters: Dict[str, MemoryCluster] = {}
        self.inductive_thoughts: List[InductiveThought] = []
        
        logger.info("Initialized Long-Term Memory")
    
    async def store_memory(
        self,
        memory: Memory,
        should_consolidate: bool = False
    ) -> Memory:
        """
        Store a memory in long-term memory.
        
        Args:
            memory: Memory to store
            should_consolidate: Whether to trigger consolidation
            
        Returns:
            Stored memory object
        """
        try:
            # Apply decay factor for long-term storage
            memory.decay_factor = self.decay_factor
            
            # Store in persistent storage
            stored_memory = await self.memory_store.store(memory)
            
            # Update clusters
            await self._update_memory_clusters(stored_memory)
            
            # Consolidate if needed
            if should_consolidate:
                await self.consolidate_memories(memory.session_id)
            
            logger.debug(f"Stored memory in LTM: {memory.content[:50]}...")
            return stored_memory
            
        except Exception as e:
            logger.error(f"Error storing memory in LTM: {e}")
            raise MemoryException(f"Failed to store memory in LTM: {e}")
    
    async def retrieve_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        importance_threshold: Optional[MemoryImportance] = None,
        limit: int = 10,
        include_clusters: bool = False
    ) -> List[Memory]:
        """
        Retrieve memories from long-term memory.
        
        Args:
            query: Search query
            session_id: Optional session filter
            memory_types: Optional memory type filters
            importance_threshold: Minimum importance level
            limit: Maximum number of memories
            include_clusters: Whether to include cluster information
            
        Returns:
            List of relevant memories
        """
        try:
            memories = await self.memory_retriever.retrieve(
                query=query,
                session_id=session_id,
                memory_types=memory_types,
                importance_threshold=importance_threshold,
                limit=limit
            )
            
            # Filter by importance if specified
            if importance_threshold:
                memories = [
                    m for m in memories 
                    if m.importance.value >= importance_threshold.value
                ]
            
            # Apply temporal decay to relevance scores
            current_time = datetime.utcnow()
            for memory in memories:
                memory.update_access()
                relevance_score = memory.calculate_relevance_score(current_time)
                memory.metadata["current_relevance"] = relevance_score
            
            # Sort by relevance
            memories.sort(
                key=lambda m: m.metadata.get("current_relevance", 0),
                reverse=True
            )
            
            logger.debug(f"Retrieved {len(memories)} memories from LTM")
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving memories from LTM: {e}")
            return []
    
    async def create_inductive_thought(
        self,
        source_memories: List[Memory],
        thought_content: str,
        confidence: float = 0.5,
        session_id: Optional[str] = None
    ) -> InductiveThought:
        """
        Create an inductive thought from source memories.
        
        Args:
            source_memories: Memories that inspired this thought
            thought_content: Content of the inductive thought
            confidence: Confidence level in the thought
            session_id: Session identifier
            
        Returns:
            Created inductive thought
        """
        try:
            # Determine session_id if not provided
            if not session_id and source_memories:
                session_id = source_memories[0].session_id
            
            # Calculate generalization level based on source diversity
            unique_sessions = set(m.session_id for m in source_memories)
            generalization_level = len(unique_sessions)
            
            # Create metadata
            metadata = {
                "creation_method": "consolidation",
                "source_count": len(source_memories),
                "unique_sessions": len(unique_sessions),
                "avg_importance": sum(m.importance.value for m in source_memories) / len(source_memories)
            }
            
            thought = InductiveThought(
                content=thought_content,
                session_id=session_id or "global",
                source_memories=[m.id for m in source_memories],
                confidence=confidence,
                generalization_level=generalization_level,
                importance=MemoryImportance.HIGH,
                metadata=metadata,
                timestamp=datetime.utcnow()
            )
            
            # Store the thought
            stored_thought = await self.store_memory(thought)
            self.inductive_thoughts.append(stored_thought)
            
            logger.info(f"Created inductive thought: {thought_content[:50]}...")
            return stored_thought
            
        except Exception as e:
            logger.error(f"Error creating inductive thought: {e}")
            raise MemoryException(f"Failed to create inductive thought: {e}")
    
    async def consolidate_memories(
        self,
        session_id: str,
        max_consolidations: int = 5
    ) -> List[InductiveThought]:
        """
        Consolidate memories for a session into inductive thoughts.
        
        Args:
            session_id: Session to consolidate
            max_consolidations: Maximum number of consolidations to perform
            
        Returns:
            List of created inductive thoughts
        """
        try:
            # Get memories for consolidation
            memories = await self.memory_store.get_memories_by_session(session_id)
            
            if len(memories) < 3:  # Need at least 3 memories to consolidate
                return []
            
            # Group similar memories
            memory_groups = await self._group_memories_for_consolidation(memories)
            
            thoughts = []
            for i, group in enumerate(memory_groups[:max_consolidations]):
                if len(group) >= 2:  # Only consolidate groups with 2+ memories
                    thought_content = await self._generate_inductive_thought(group)
                    if thought_content:
                        thought = await self.create_inductive_thought(
                            source_memories=group,
                            thought_content=thought_content,
                            session_id=session_id,
                            confidence=min(0.9, 0.5 + (len(group) * 0.1))
                        )
                        thoughts.append(thought)
            
            logger.info(f"Consolidated {len(thoughts)} inductive thoughts for session {session_id}")
            return thoughts
            
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return []
    
    async def forget_irrelevant_memories(
        self,
        relevance_threshold: float = 0.1,
        max_age_days: int = 365
    ) -> int:
        """
        Remove memories that are no longer relevant.
        
        Args:
            relevance_threshold: Minimum relevance score to keep
            max_age_days: Maximum age in days
            
        Returns:
            Number of memories forgotten
        """
        try:
            current_time = datetime.utcnow()
            cutoff_date = current_time - timedelta(days=max_age_days)
            
            all_memories = await self.memory_store.get_all_memories()
            forgotten_count = 0
            
            for memory in all_memories:
                # Skip inductive thoughts and high importance memories
                if (memory.memory_type == MemoryType.INDUCTIVE_THOUGHT or 
                    memory.importance == MemoryImportance.CRITICAL):
                    continue
                
                # Check relevance and age
                relevance_score = memory.calculate_relevance_score(current_time)
                is_too_old = memory.timestamp < cutoff_date
                is_irrelevant = relevance_score < relevance_threshold
                
                if is_irrelevant or is_too_old:
                    await self.memory_store.delete(memory.id)
                    forgotten_count += 1
            
            logger.info(f"Forgot {forgotten_count} irrelevant memories")
            return forgotten_count
            
        except Exception as e:
            logger.error(f"Error forgetting memories: {e}")
            return 0
    
    async def get_memory_timeline(
        self,
        session_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Memory]:
        """
        Get a chronological timeline of memories for a session.
        
        Args:
            session_id: Session identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Chronologically sorted list of memories
        """
        try:
            memories = await self.memory_store.get_memories_by_session(session_id)
            
            # Apply date filters
            if start_date:
                memories = [m for m in memories if m.timestamp >= start_date]
            if end_date:
                memories = [m for m in memories if m.timestamp <= end_date]
            
            # Sort chronologically
            memories.sort(key=lambda m: m.timestamp)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting memory timeline: {e}")
            return []
    
    async def get_ltm_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about long-term memory.
        
        Returns:
            Dictionary containing LTM statistics
        """
        try:
            all_memories = await self.memory_store.get_all_memories()
            
            # Basic counts
            total_memories = len(all_memories)
            inductive_thoughts_count = len([m for m in all_memories if m.memory_type == MemoryType.INDUCTIVE_THOUGHT])
            
            # Memory type distribution
            type_distribution = {}
            importance_distribution = {}
            session_distribution = {}
            
            for memory in all_memories:
                # Type distribution
                mem_type = memory.memory_type.value
                type_distribution[mem_type] = type_distribution.get(mem_type, 0) + 1
                
                # Importance distribution
                importance = memory.importance.value
                importance_distribution[str(importance)] = importance_distribution.get(str(importance), 0) + 1
                
                # Session distribution
                session = memory.session_id
                session_distribution[session] = session_distribution.get(session, 0) + 1
            
            # Age analysis
            current_time = datetime.utcnow()
            age_buckets = {"0-1d": 0, "1-7d": 0, "7-30d": 0, "30d+": 0}
            
            for memory in all_memories:
                age_days = (current_time - memory.timestamp).days
                if age_days <= 1:
                    age_buckets["0-1d"] += 1
                elif age_days <= 7:
                    age_buckets["1-7d"] += 1
                elif age_days <= 30:
                    age_buckets["7-30d"] += 1
                else:
                    age_buckets["30d+"] += 1
            
            # Cluster information
            cluster_stats = {
                "total_clusters": len(self.memory_clusters),
                "avg_cluster_size": sum(c.size() for c in self.memory_clusters.values()) / max(1, len(self.memory_clusters))
            }
            
            return {
                "total_memories": total_memories,
                "inductive_thoughts": inductive_thoughts_count,
                "type_distribution": type_distribution,
                "importance_distribution": importance_distribution,
                "session_distribution": session_distribution,
                "age_distribution": age_buckets,
                "cluster_statistics": cluster_stats,
                "consolidation_threshold": self.consolidation_threshold,
                "decay_factor": self.decay_factor
            }
            
        except Exception as e:
            logger.error(f"Error getting LTM statistics: {e}")
            return {}
    
    async def _update_memory_clusters(self, memory: Memory) -> None:
        """Update memory clusters with new memory."""
        try:
            # Find best matching cluster
            best_cluster = None
            best_similarity = 0.0
            
            for cluster in self.memory_clusters.values():
                if cluster.cluster_embedding and memory.embedding:
                    similarity = await self.memory_retriever.calculate_similarity_vectors(
                        memory.embedding,
                        cluster.cluster_embedding
                    )
                    if similarity > best_similarity and similarity > self.consolidation_threshold:
                        best_similarity = similarity
                        best_cluster = cluster
            
            if best_cluster:
                best_cluster.add_memory(memory.id)
            else:
                # Create new cluster
                cluster = MemoryCluster(
                    name=f"Cluster_{len(self.memory_clusters) + 1}",
                    description=f"Auto-generated cluster for memory type: {memory.memory_type.value}",
                    memory_ids=[memory.id],
                    cluster_embedding=memory.embedding,
                    tags=[memory.memory_type.value]
                )
                self.memory_clusters[cluster.id] = cluster
            
        except Exception as e:
            logger.error(f"Error updating memory clusters: {e}")
    
    async def _group_memories_for_consolidation(
        self,
        memories: List[Memory]
    ) -> List[List[Memory]]:
        """Group similar memories for consolidation."""
        try:
            groups = []
            processed = set()
            
            for memory in memories:
                if memory.id in processed or memory.memory_type == MemoryType.INDUCTIVE_THOUGHT:
                    continue
                
                group = [memory]
                processed.add(memory.id)
                
                for other_memory in memories:
                    if other_memory.id in processed or other_memory.memory_type == MemoryType.INDUCTIVE_THOUGHT:
                        continue
                    
                    similarity = await self.memory_retriever.calculate_similarity(
                        memory.content,
                        other_memory.content
                    )
                    
                    if similarity > self.consolidation_threshold:
                        group.append(other_memory)
                        processed.add(other_memory.id)
                
                if len(group) >= 2:
                    groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error grouping memories: {e}")
            return []
    
    async def _generate_inductive_thought(self, memories: List[Memory]) -> Optional[str]:
        """Generate an inductive thought from a group of memories."""
        try:
            contents = [memory.content for memory in memories]
            
            # Simple pattern extraction (can be enhanced with NLP)
            common_themes = set()
            for content in contents:
                words = content.lower().split()
                common_themes.update(words)
            
            # Find words that appear in multiple memories
            word_counts = {}
            for content in contents:
                words = set(content.lower().split())
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            common_words = [
                word for word, count in word_counts.items()
                if count >= 2 and len(word) > 3
            ]
            
            if common_words:
                # Create a simple inductive thought
                thought = f"Pattern identified involving: {', '.join(common_words[:5])}. "
                thought += f"Observed across {len(memories)} related memories."
                return thought
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating inductive thought: {e}")
            return None