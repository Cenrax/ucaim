"""Memory storage interface and implementations."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..memory.memory_types import Memory, MemoryType, MemoryImportance
from ..core.config import CAIMConfig
from ..core.exceptions import StorageException


logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """Abstract base class for memory storage implementations."""
    
    def __init__(self, config: CAIMConfig):
        self.config = config
    
    @abstractmethod
    async def store(self, memory: Memory) -> Memory:
        """Store a memory and return the stored memory with any updates."""
        pass
    
    @abstractmethod
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        pass
    
    @abstractmethod
    async def update(self, memory: Memory) -> Memory:
        """Update an existing memory."""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        pass
    
    @abstractmethod
    async def get_memories_by_session(self, session_id: str) -> List[Memory]:
        """Get all memories for a specific session."""
        pass
    
    @abstractmethod
    async def get_all_memories(self, limit: Optional[int] = None) -> List[Memory]:
        """Get all memories with optional limit."""
        pass
    
    @abstractmethod
    async def search_memories(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Search memories by content and filters."""
        pass
    
    @abstractmethod
    async def update_importance(
        self,
        memory_id: str,
        importance: MemoryImportance
    ) -> bool:
        """Update memory importance level."""
        pass
    
    @abstractmethod
    async def cleanup_expired_memories(self, max_age_days: int = 365) -> int:
        """Remove expired memories and return count removed."""
        pass


class InMemoryStore(MemoryStore):
    """In-memory implementation of MemoryStore for testing and development."""
    
    def __init__(self, config: CAIMConfig):
        super().__init__(config)
        self.memories: Dict[str, Memory] = {}
        self.session_index: Dict[str, List[str]] = {}
        logger.info("Initialized InMemoryStore")
    
    async def store(self, memory: Memory) -> Memory:
        """Store a memory in memory."""
        try:
            self.memories[memory.id] = memory
            
            # Update session index
            if memory.session_id not in self.session_index:
                self.session_index[memory.session_id] = []
            
            if memory.id not in self.session_index[memory.session_id]:
                self.session_index[memory.session_id].append(memory.id)
            
            logger.debug(f"Stored memory {memory.id} in InMemoryStore")
            return memory
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise StorageException(f"Failed to store memory: {e}")
    
    async def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        try:
            memory = self.memories.get(memory_id)
            if memory:
                memory.update_access()
            return memory
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    async def update(self, memory: Memory) -> Memory:
        """Update an existing memory."""
        try:
            if memory.id not in self.memories:
                raise StorageException(f"Memory {memory.id} not found")
            
            self.memories[memory.id] = memory
            logger.debug(f"Updated memory {memory.id}")
            return memory
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            raise StorageException(f"Failed to update memory: {e}")
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            if memory_id not in self.memories:
                return False
            
            memory = self.memories[memory_id]
            
            # Remove from session index
            if memory.session_id in self.session_index:
                if memory_id in self.session_index[memory.session_id]:
                    self.session_index[memory.session_id].remove(memory_id)
                
                # Clean up empty session entries
                if not self.session_index[memory.session_id]:
                    del self.session_index[memory.session_id]
            
            # Remove memory
            del self.memories[memory_id]
            
            logger.debug(f"Deleted memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    async def get_memories_by_session(self, session_id: str) -> List[Memory]:
        """Get all memories for a specific session."""
        try:
            if session_id not in self.session_index:
                return []
            
            memories = []
            for memory_id in self.session_index[session_id]:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append(memory)
            
            # Sort by timestamp
            memories.sort(key=lambda m: m.timestamp)
            return memories
            
        except Exception as e:
            logger.error(f"Error getting memories for session {session_id}: {e}")
            return []
    
    async def get_all_memories(self, limit: Optional[int] = None) -> List[Memory]:
        """Get all memories with optional limit."""
        try:
            memories = list(self.memories.values())
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            
            if limit:
                memories = memories[:limit]
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting all memories: {e}")
            return []
    
    async def search_memories(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Search memories by content and filters."""
        try:
            query_lower = query.lower()
            matching_memories = []
            
            for memory in self.memories.values():
                # Text search
                if query_lower in memory.content.lower():
                    match = True
                    
                    # Apply filters
                    if filters:
                        if "session_id" in filters and memory.session_id != filters["session_id"]:
                            match = False
                        if "memory_type" in filters and memory.memory_type != filters["memory_type"]:
                            match = False
                        if "importance" in filters and memory.importance != filters["importance"]:
                            match = False
                        if "start_date" in filters and memory.timestamp < filters["start_date"]:
                            match = False
                        if "end_date" in filters and memory.timestamp > filters["end_date"]:
                            match = False
                    
                    if match:
                        matching_memories.append(memory)
            
            # Sort by relevance (importance + recency)
            matching_memories.sort(
                key=lambda m: (m.importance.value, m.timestamp),
                reverse=True
            )
            
            return matching_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def update_importance(
        self,
        memory_id: str,
        importance: MemoryImportance
    ) -> bool:
        """Update memory importance level."""
        try:
            if memory_id not in self.memories:
                return False
            
            self.memories[memory_id].importance = importance
            logger.debug(f"Updated importance for memory {memory_id} to {importance}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")
            return False
    
    async def cleanup_expired_memories(self, max_age_days: int = 365) -> int:
        """Remove expired memories and return count removed."""
        try:
            current_time = datetime.utcnow()
            expired_ids = []
            
            for memory_id, memory in self.memories.items():
                age_days = (current_time - memory.timestamp).days
                
                # Don't remove critical memories or inductive thoughts
                if (memory.importance == MemoryImportance.CRITICAL or 
                    memory.memory_type == MemoryType.INDUCTIVE_THOUGHT):
                    continue
                
                if age_days > max_age_days:
                    expired_ids.append(memory_id)
            
            # Delete expired memories
            for memory_id in expired_ids:
                await self.delete(memory_id)
            
            logger.info(f"Cleaned up {len(expired_ids)} expired memories")
            return len(expired_ids)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired memories: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_memories": len(self.memories),
            "active_sessions": len(self.session_index),
            "memory_types": {
                mem_type.value: len([
                    m for m in self.memories.values()
                    if m.memory_type == mem_type
                ])
                for mem_type in MemoryType
            }
        }