"""Memory Controller: Central decision-making unit for memory access."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

from ..memory.memory_types import Memory, MemoryType, MemoryImportance
from ..retrieval.memory_retriever import MemoryRetriever
from ..storage.memory_store import MemoryStore
from .config import CAIMConfig
from .exceptions import MemoryException, RetrievalException


logger = logging.getLogger(__name__)


class MemoryController:
    """
    Central decision-making unit that determines when to access agent's memory.
    Acts as a gatekeeper, analyzing user input to decide if historical information is required.
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
        self.access_patterns: Dict[str, int] = {}
        
    async def should_access_memory(
        self,
        user_input: str,
        current_context: str,
        session_id: str
    ) -> bool:
        """
        Determine if memory access is necessary for the current input.
        
        Args:
            user_input: Current user input
            current_context: Current conversation context
            session_id: Current session identifier
            
        Returns:
            Boolean indicating if memory access is needed
        """
        try:
            memory_indicators = [
                "remember",
                "last time",
                "before",
                "previously",
                "earlier",
                "what did I",
                "recall",
                "history"
            ]
            
            temporal_indicators = [
                "yesterday",
                "last week",
                "ago",
                "when we talked",
                "before"
            ]
            
            personal_indicators = [
                "my",
                "I told you",
                "we discussed",
                "you know that I"
            ]
            
            user_lower = user_input.lower()
            
            has_memory_indicator = any(indicator in user_lower for indicator in memory_indicators)
            has_temporal_indicator = any(indicator in user_lower for indicator in temporal_indicators)
            has_personal_indicator = any(indicator in user_lower for indicator in personal_indicators)
            
            context_length = len(current_context.split())
            needs_context = context_length < 10
            
            session_frequency = self.access_patterns.get(session_id, 0)
            frequent_user = session_frequency > 5
            
            should_access = (
                has_memory_indicator or
                has_temporal_indicator or
                has_personal_indicator or
                (needs_context and frequent_user)
            )
            
            logger.debug(f"Memory access decision: {should_access} for input: {user_input[:50]}...")
            return should_access
            
        except Exception as e:
            logger.error(f"Error in memory access decision: {e}")
            return False
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        session_id: str,
        context: Optional[str] = None,
        limit: int = None
    ) -> List[Memory]:
        """
        Retrieve relevant memories based on query and context.
        
        Args:
            query: Search query
            session_id: Current session ID
            context: Additional context
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        try:
            if limit is None:
                limit = self.config.retrieval_top_k
                
            memories = await self.memory_retriever.retrieve(
                query=query,
                session_id=session_id,
                context=context,
                limit=limit
            )
            
            self.access_patterns[session_id] = self.access_patterns.get(session_id, 0) + 1
            
            logger.info(f"Retrieved {len(memories)} memories for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            raise RetrievalException(f"Failed to retrieve memories: {e}")
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        session_id: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Store a new memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            session_id: Session identifier
            importance: Memory importance level
            metadata: Additional metadata
            
        Returns:
            Created memory object
        """
        try:
            memory = Memory(
                content=content,
                memory_type=memory_type,
                session_id=session_id,
                importance=importance,
                metadata=metadata or {},
                timestamp=datetime.utcnow()
            )
            
            stored_memory = await self.memory_store.store(memory)
            logger.info(f"Stored memory: {content[:50]}...")
            return stored_memory
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise MemoryException(f"Failed to store memory: {e}")
    
    async def update_memory_importance(
        self,
        memory_id: str,
        new_importance: MemoryImportance
    ) -> bool:
        """
        Update the importance level of a memory.
        
        Args:
            memory_id: Memory identifier
            new_importance: New importance level
            
        Returns:
            Success status
        """
        try:
            success = await self.memory_store.update_importance(memory_id, new_importance)
            if success:
                logger.info(f"Updated importance for memory {memory_id} to {new_importance}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")
            return False
    
    async def consolidate_memories(self, session_id: str) -> int:
        """
        Consolidate memories for a given session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of memories consolidated
        """
        try:
            memories = await self.memory_store.get_memories_by_session(session_id)
            
            consolidated_count = 0
            
            similar_groups = await self._group_similar_memories(memories)
            
            for group in similar_groups:
                if len(group) > 1:
                    consolidated_memory = await self._create_consolidated_memory(group)
                    if consolidated_memory:
                        await self.memory_store.store(consolidated_memory)
                        
                        for memory in group:
                            await self.memory_store.delete(memory.id)
                        
                        consolidated_count += len(group) - 1
            
            logger.info(f"Consolidated {consolidated_count} memories for session {session_id}")
            return consolidated_count
            
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return 0
    
    async def _group_similar_memories(self, memories: List[Memory]) -> List[List[Memory]]:
        """Group similar memories together for consolidation."""
        groups = []
        processed = set()
        
        for i, memory in enumerate(memories):
            if memory.id in processed:
                continue
                
            group = [memory]
            processed.add(memory.id)
            
            for j, other_memory in enumerate(memories[i+1:], i+1):
                if other_memory.id in processed:
                    continue
                    
                similarity = await self.memory_retriever.calculate_similarity(
                    memory.content,
                    other_memory.content
                )
                
                if similarity > self.config.consolidation_threshold:
                    group.append(other_memory)
                    processed.add(other_memory.id)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    async def _create_consolidated_memory(self, memories: List[Memory]) -> Optional[Memory]:
        """Create a consolidated memory from a group of similar memories."""
        try:
            contents = [memory.content for memory in memories]
            combined_content = " | ".join(contents)
            
            avg_importance = sum(
                memory.importance.value for memory in memories
            ) / len(memories)
            
            importance = MemoryImportance.HIGH if avg_importance > 0.7 else MemoryImportance.MEDIUM
            
            consolidated_metadata = {
                "consolidated_from": [memory.id for memory in memories],
                "original_count": len(memories),
                "consolidation_timestamp": datetime.utcnow().isoformat()
            }
            
            for memory in memories:
                for key, value in memory.metadata.items():
                    if key not in consolidated_metadata:
                        consolidated_metadata[key] = value
            
            return Memory(
                content=f"Consolidated memory: {combined_content}",
                memory_type=memories[0].memory_type,
                session_id=memories[0].session_id,
                importance=importance,
                metadata=consolidated_metadata,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error creating consolidated memory: {e}")
            return None
    
    async def get_memory_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics for monitoring and debugging.
        
        Args:
            session_id: Optional session ID to filter statistics
            
        Returns:
            Dictionary containing memory statistics
        """
        try:
            if session_id:
                memories = await self.memory_store.get_memories_by_session(session_id)
            else:
                memories = await self.memory_store.get_all_memories()
            
            total_memories = len(memories)
            
            type_counts = {}
            importance_counts = {}
            
            for memory in memories:
                memory_type = memory.memory_type.value
                importance = memory.importance.value
                
                type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
                importance_counts[importance] = importance_counts.get(importance, 0) + 1
            
            access_count = self.access_patterns.get(session_id, 0) if session_id else sum(self.access_patterns.values())
            
            return {
                "total_memories": total_memories,
                "memory_types": type_counts,
                "importance_distribution": importance_counts,
                "access_count": access_count,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {}