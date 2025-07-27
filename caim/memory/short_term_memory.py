"""Short-Term Memory implementation for CAIM framework."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque

from .memory_types import Memory, MemoryType, MemoryImportance, ConversationMemory
from ..core.config import CAIMConfig
from ..core.exceptions import MemoryException


logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Short-Term Memory (STM) holds the context of the current conversation.
    Manages recent interactions and provides immediate context for responses.
    """
    
    def __init__(self, config: CAIMConfig, max_size: int = 20, retention_time: int = 3600):
        self.config = config
        self.max_size = max_size
        self.retention_time = retention_time  # seconds
        
        self.memories: deque = deque(maxlen=max_size)
        self.session_memories: Dict[str, deque] = {}
        self.conversation_context: Dict[str, str] = {}
        
        logger.info(f"Initialized STM with max_size={max_size}, retention_time={retention_time}s")
    
    def add_memory(
        self,
        content: str,
        session_id: str,
        memory_type: MemoryType = MemoryType.CONVERSATIONAL,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        speaker: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Add a new memory to short-term memory.
        
        Args:
            content: Memory content
            session_id: Session identifier
            memory_type: Type of memory
            importance: Importance level
            speaker: Who created this memory
            metadata: Additional metadata
            
        Returns:
            Created memory object
        """
        try:
            if session_id not in self.session_memories:
                self.session_memories[session_id] = deque(maxlen=self.max_size)
            
            if memory_type == MemoryType.CONVERSATIONAL:
                turn_number = len(self.session_memories[session_id]) + 1
                memory = ConversationMemory(
                    content=content,
                    memory_type=MemoryType.CONVERSATIONAL,
                    session_id=session_id,
                    importance=importance,
                    speaker=speaker,
                    turn_number=turn_number,
                    conversation_id=session_id,
                    metadata=metadata or {},
                    timestamp=datetime.utcnow()
                )
            else:
                memory = Memory(
                    content=content,
                    memory_type=memory_type,
                    session_id=session_id,
                    importance=importance,
                    metadata=metadata or {},
                    timestamp=datetime.utcnow()
                )
            
            self.memories.append(memory)
            self.session_memories[session_id].append(memory)
            
            self._update_conversation_context(session_id)
            
            logger.debug(f"Added STM memory: {content[:50]}...")
            return memory
            
        except Exception as e:
            logger.error(f"Error adding memory to STM: {e}")
            raise MemoryException(f"Failed to add memory to STM: {e}")
    
    def get_recent_memories(
        self,
        session_id: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None
    ) -> List[Memory]:
        """
        Get recent memories from short-term memory.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of memories to return
            memory_type: Filter by memory type
            
        Returns:
            List of recent memories
        """
        try:
            if session_id not in self.session_memories:
                return []
            
            memories = list(self.session_memories[session_id])
            
            if memory_type:
                memories = [m for m in memories if m.memory_type == memory_type]
            
            # Sort by timestamp (most recent first)
            memories.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Remove expired memories
            current_time = datetime.utcnow()
            valid_memories = []
            
            for memory in memories[:limit]:
                if self._is_memory_valid(memory, current_time):
                    valid_memories.append(memory)
            
            return valid_memories
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return []
    
    def get_conversation_context(self, session_id: str) -> str:
        """
        Get the current conversation context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation context string
        """
        return self.conversation_context.get(session_id, "")
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all memories for a specific session.
        
        Args:
            session_id: Session identifier
        """
        try:
            if session_id in self.session_memories:
                self.session_memories[session_id].clear()
                
            if session_id in self.conversation_context:
                del self.conversation_context[session_id]
                
            # Remove from main memory deque
            self.memories = deque(
                [m for m in self.memories if m.session_id != session_id],
                maxlen=self.max_size
            )
            
            logger.info(f"Cleared STM for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
    
    def cleanup_expired_memories(self) -> int:
        """
        Remove expired memories from short-term memory.
        
        Returns:
            Number of memories removed
        """
        try:
            current_time = datetime.utcnow()
            removed_count = 0
            
            # Clean main memories
            valid_memories = []
            for memory in self.memories:
                if self._is_memory_valid(memory, current_time):
                    valid_memories.append(memory)
                else:
                    removed_count += 1
            
            self.memories = deque(valid_memories, maxlen=self.max_size)
            
            # Clean session memories
            for session_id in list(self.session_memories.keys()):
                session_memories = self.session_memories[session_id]
                valid_session_memories = deque(maxlen=self.max_size)
                
                for memory in session_memories:
                    if self._is_memory_valid(memory, current_time):
                        valid_session_memories.append(memory)
                    else:
                        removed_count += 1
                
                if valid_session_memories:
                    self.session_memories[session_id] = valid_session_memories
                    self._update_conversation_context(session_id)
                else:
                    del self.session_memories[session_id]
                    if session_id in self.conversation_context:
                        del self.conversation_context[session_id]
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired memories from STM")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up memories: {e}")
            return 0
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about short-term memory usage.
        
        Returns:
            Dictionary containing STM statistics
        """
        try:
            total_memories = len(self.memories)
            active_sessions = len(self.session_memories)
            
            memory_types = {}
            importance_levels = {}
            
            for memory in self.memories:
                mem_type = memory.memory_type.value
                importance = memory.importance.value
                
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                importance_levels[str(importance)] = importance_levels.get(str(importance), 0) + 1
            
            session_sizes = {
                session_id: len(memories)
                for session_id, memories in self.session_memories.items()
            }
            
            return {
                "total_memories": total_memories,
                "active_sessions": active_sessions,
                "memory_types": memory_types,
                "importance_levels": importance_levels,
                "session_sizes": session_sizes,
                "max_size": self.max_size,
                "retention_time": self.retention_time
            }
            
        except Exception as e:
            logger.error(f"Error getting STM statistics: {e}")
            return {}
    
    def _is_memory_valid(self, memory: Memory, current_time: datetime) -> bool:
        """Check if a memory is still valid based on retention time."""
        time_diff = (current_time - memory.timestamp).total_seconds()
        return time_diff <= self.retention_time
    
    def _update_conversation_context(self, session_id: str) -> None:
        """Update the conversation context for a session."""
        try:
            if session_id not in self.session_memories:
                return
            
            recent_memories = list(self.session_memories[session_id])[-5:]  # Last 5 memories
            
            context_parts = []
            for memory in recent_memories:
                if isinstance(memory, ConversationMemory):
                    context_parts.append(f"{memory.speaker}: {memory.content}")
                else:
                    context_parts.append(memory.content)
            
            self.conversation_context[session_id] = "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error updating conversation context: {e}")
    
    def promote_to_ltm(
        self,
        memory_ids: List[str],
        session_id: str
    ) -> List[Memory]:
        """
        Get memories that should be promoted to long-term memory.
        
        Args:
            memory_ids: List of memory IDs to promote
            session_id: Session identifier
            
        Returns:
            List of memories to be promoted
        """
        try:
            memories_to_promote = []
            
            if session_id in self.session_memories:
                for memory in self.session_memories[session_id]:
                    if memory.id in memory_ids:
                        memories_to_promote.append(memory)
            
            logger.info(f"Promoting {len(memories_to_promote)} memories to LTM")
            return memories_to_promote
            
        except Exception as e:
            logger.error(f"Error promoting memories to LTM: {e}")
            return []
    
    def search_memories(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Memory]:
        """
        Search memories in short-term memory.
        
        Args:
            query: Search query
            session_id: Optional session filter
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        try:
            query_lower = query.lower()
            matching_memories = []
            
            memories_to_search = []
            if session_id and session_id in self.session_memories:
                memories_to_search = list(self.session_memories[session_id])
            else:
                memories_to_search = list(self.memories)
            
            for memory in memories_to_search:
                if query_lower in memory.content.lower():
                    matching_memories.append(memory)
            
            # Sort by relevance and timestamp
            matching_memories.sort(
                key=lambda x: (x.importance.value, x.timestamp),
                reverse=True
            )
            
            return matching_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error searching STM memories: {e}")
            return []