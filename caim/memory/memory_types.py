"""Memory types and data structures for CAIM framework."""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class MemoryType(Enum):
    """Types of memories in the CAIM framework."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EMOTIONAL = "emotional"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    INDUCTIVE_THOUGHT = "inductive_thought"


class MemoryImportance(Enum):
    """Importance levels for memories."""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    MINIMAL = 0.2


class Memory(BaseModel):
    """Base memory class representing a single memory unit."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The actual content of the memory")
    memory_type: MemoryType = Field(..., description="Type of memory")
    importance: MemoryImportance = Field(default=MemoryImportance.MEDIUM, description="Importance level")
    
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the memory was created")
    last_accessed: Optional[datetime] = Field(default=None, description="When the memory was last accessed")
    access_count: int = Field(default=0, description="Number of times memory was accessed")
    
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the memory")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    decay_factor: float = Field(default=1.0, description="Factor for memory decay over time")
    consolidation_level: int = Field(default=0, description="Level of memory consolidation")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def calculate_relevance_score(
        self,
        current_time: Optional[datetime] = None,
        time_weight: float = 0.3,
        importance_weight: float = 0.4,
        access_weight: float = 0.3
    ) -> float:
        """
        Calculate relevance score based on various factors.
        
        Args:
            current_time: Current timestamp
            time_weight: Weight for time factor
            importance_weight: Weight for importance factor
            access_weight: Weight for access frequency factor
            
        Returns:
            Relevance score between 0 and 1
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        time_diff = (current_time - self.timestamp).total_seconds() / 86400
        time_score = max(0, 1 - (time_diff * 0.01))
        
        importance_score = self.importance.value
        
        access_score = min(1.0, self.access_count / 10.0)
        
        total_score = (
            time_score * time_weight +
            importance_score * importance_weight +
            access_score * access_weight
        ) * self.decay_factor
        
        return min(1.0, max(0.0, total_score))
    
    def should_forget(self, threshold: float = 0.1) -> bool:
        """Determine if this memory should be forgotten."""
        return self.calculate_relevance_score() < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return self.model_dump()


class ConversationMemory(Memory):
    """Memory specifically for conversation history."""
    
    speaker: str = Field(..., description="Who said this")
    turn_number: int = Field(..., description="Turn number in conversation")
    conversation_id: str = Field(..., description="Conversation identifier")
    
    def __init__(self, **data):
        super().__init__(**data)
        self.memory_type = MemoryType.CONVERSATIONAL


class InductiveThought(Memory):
    """Memory representing an inductive thought or insight."""
    
    source_memories: List[str] = Field(default_factory=list, description="IDs of source memories")
    confidence: float = Field(default=0.5, description="Confidence in the thought")
    generalization_level: int = Field(default=1, description="Level of generalization")
    
    def __init__(self, **data):
        super().__init__(**data)
        self.memory_type = MemoryType.INDUCTIVE_THOUGHT


class EmotionalMemory(Memory):
    """Memory with emotional context."""
    
    emotion: str = Field(..., description="Primary emotion")
    emotion_intensity: float = Field(default=0.5, description="Intensity of emotion (0-1)")
    valence: float = Field(default=0.0, description="Emotional valence (-1 to 1)")
    
    def __init__(self, **data):
        super().__init__(**data)
        self.memory_type = MemoryType.EMOTIONAL


class FactualMemory(Memory):
    """Memory for factual information."""
    
    fact_type: str = Field(..., description="Type of fact")
    source: Optional[str] = Field(default=None, description="Source of the fact")
    verification_status: str = Field(default="unverified", description="Verification status")
    
    def __init__(self, **data):
        super().__init__(**data)
        self.memory_type = MemoryType.FACTUAL


class MemoryCluster(BaseModel):
    """A cluster of related memories."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Name of the cluster")
    description: Optional[str] = Field(default=None, description="Description of the cluster")
    
    memory_ids: List[str] = Field(default_factory=list, description="IDs of memories in this cluster")
    cluster_embedding: Optional[List[float]] = Field(default=None, description="Cluster centroid embedding")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    tags: List[str] = Field(default_factory=list, description="Cluster tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def add_memory(self, memory_id: str) -> None:
        """Add a memory to this cluster."""
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)
            self.updated_at = datetime.utcnow()
    
    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from this cluster."""
        if memory_id in self.memory_ids:
            self.memory_ids.remove(memory_id)
            self.updated_at = datetime.utcnow()
    
    def size(self) -> int:
        """Get the number of memories in this cluster."""
        return len(self.memory_ids)