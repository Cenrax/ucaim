"""Main CAIM Framework implementation."""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from .config import CAIMConfig
from .memory_controller import MemoryController
from .exceptions import CAIMException, ConfigurationException
from ..memory.short_term_memory import ShortTermMemory
from ..memory.long_term_memory import LongTermMemory
from ..memory.memory_types import Memory, MemoryType, MemoryImportance
from ..storage.memory_store import InMemoryStore
from ..storage.vector_store import InMemoryVectorStore
from ..retrieval.memory_retriever import MemoryRetriever
from ..retrieval.embedding_generator import create_embedding_generator


logger = logging.getLogger(__name__)


class CAIMFramework:
    """
    Main CAIM Framework class that orchestrates all components.
    
    The CAIM framework implements a human-like memory system with:
    - Short-Term Memory (STM) for current conversation context
    - Long-Term Memory (LTM) for persistent storage and inductive thoughts
    - Memory Controller for intelligent memory access decisions
    - Cognitive processing pipeline for memory consolidation
    """
    
    def __init__(self, config: Optional[CAIMConfig] = None):
        """
        Initialize CAIM Framework.
        
        Args:
            config: Configuration object, uses default if None
        """
        self.config = config or CAIMConfig.from_env()
        
        # Initialize storage components
        self.memory_store = None
        self.vector_store = None
        self.embedding_generator = None
        self.memory_retriever = None
        
        # Initialize memory components
        self.stm = None
        self.ltm = None
        self.memory_controller = None
        
        # Framework state
        self.is_initialized = False
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("CAIM Framework initialized")
    
    async def initialize(self) -> None:
        """Initialize all framework components asynchronously."""
        try:
            if self.is_initialized:
                logger.warning("Framework already initialized")
                return
            
            logger.info("Initializing CAIM Framework components...")
            
            # Initialize storage layer
            await self._initialize_storage()
            
            # Initialize memory components
            await self._initialize_memory_components()
            
            # Initialize processing pipeline
            await self._initialize_processing_pipeline()
            
            self.is_initialized = True
            logger.info("CAIM Framework initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing CAIM Framework: {e}")
            raise CAIMException(f"Framework initialization failed: {e}")
    
    async def _initialize_storage(self) -> None:
        """Initialize storage components."""
        try:
            # Initialize memory store
            self.memory_store = InMemoryStore(self.config)
            
            # Initialize vector store
            self.vector_store = InMemoryVectorStore(self.config)
            
            # Initialize embedding generator
            try:
                self.embedding_generator = create_embedding_generator(
                    self.config, provider="sentence_transformers"
                )
            except ImportError:
                logger.warning("SentenceTransformers not available, using simple embeddings")
                from ..retrieval.embedding_generator import SimpleEmbedding
                self.embedding_generator = SimpleEmbedding(self.config)
            
            # Initialize memory retriever
            self.memory_retriever = MemoryRetriever(
                config=self.config,
                memory_store=self.memory_store,
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator
            )
            
            logger.info("Storage components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            raise ConfigurationException(f"Storage initialization failed: {e}")
    
    async def _initialize_memory_components(self) -> None:
        """Initialize memory components."""
        try:
            # Initialize Short-Term Memory
            self.stm = ShortTermMemory(
                config=self.config,
                max_size=20,
                retention_time=3600
            )
            
            # Initialize Long-Term Memory
            self.ltm = LongTermMemory(
                config=self.config,
                memory_store=self.memory_store,
                memory_retriever=self.memory_retriever
            )
            
            # Initialize Memory Controller
            self.memory_controller = MemoryController(
                config=self.config,
                memory_store=self.memory_store,
                memory_retriever=self.memory_retriever
            )
            
            logger.info("Memory components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing memory components: {e}")
            raise ConfigurationException(f"Memory initialization failed: {e}")
    
    async def _initialize_processing_pipeline(self) -> None:
        """Initialize cognitive processing pipeline."""
        try:
            # Start background tasks
            maintenance_task = asyncio.create_task(self._memory_maintenance_loop())
            self.background_tasks.append(maintenance_task)
            
            logger.info("Processing pipeline initialized")
            
        except Exception as e:
            logger.error(f"Error initializing processing pipeline: {e}")
            raise ConfigurationException(f"Processing pipeline initialization failed: {e}")
    
    async def process_input(
        self,
        user_input: str,
        session_id: str,
        context: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input through the CAIM framework.
        
        Args:
            user_input: User's input text
            session_id: Session identifier
            context: Additional context
            user_metadata: User metadata
            
        Returns:
            Processing result with relevant memories and context
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Initialize session if new
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "created_at": datetime.utcnow(),
                    "last_activity": datetime.utcnow(),
                    "interaction_count": 0
                }
            
            # Update session activity
            self.active_sessions[session_id]["last_activity"] = datetime.utcnow()
            self.active_sessions[session_id]["interaction_count"] += 1
            
            # Store user input in STM
            user_memory = self.stm.add_memory(
                content=user_input,
                session_id=session_id,
                memory_type=MemoryType.CONVERSATIONAL,
                speaker="user",
                metadata=user_metadata or {}
            )
            
            # Get current conversation context
            current_context = self.stm.get_conversation_context(session_id)
            
            # Determine if we need to access LTM
            should_access_ltm = await self.memory_controller.should_access_memory(
                user_input=user_input,
                current_context=current_context,
                session_id=session_id
            )
            
            relevant_memories = []
            if should_access_ltm:
                # Retrieve relevant memories from LTM
                relevant_memories = await self.memory_controller.retrieve_relevant_memories(
                    query=user_input,
                    session_id=session_id,
                    context=context or current_context,
                    limit=self.config.retrieval_top_k
                )
            
            # Get recent STM memories for context
            recent_memories = self.stm.get_recent_memories(session_id, limit=5)
            
            # Prepare response data
            response_data = {
                "session_id": session_id,
                "user_input": user_input,
                "current_context": current_context,
                "relevant_ltm_memories": [m.to_dict() for m in relevant_memories],
                "recent_stm_memories": [m.to_dict() for m in recent_memories],
                "should_access_ltm": should_access_ltm,
                "memory_statistics": await self._get_session_statistics(session_id),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Processed input for session {session_id}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise CAIMException(f"Input processing failed: {e}")
    
    async def add_response_memory(
        self,
        response: str,
        session_id: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Add agent response to memory.
        
        Args:
            response: Agent's response
            session_id: Session identifier
            importance: Response importance
            metadata: Additional metadata
            
        Returns:
            Created memory object
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Add to STM
            response_memory = self.stm.add_memory(
                content=response,
                session_id=session_id,
                memory_type=MemoryType.CONVERSATIONAL,
                importance=importance,
                speaker="agent",
                metadata=metadata or {}
            )
            
            # Promote important responses to LTM
            if importance in [MemoryImportance.HIGH, MemoryImportance.CRITICAL]:
                # Generate embedding
                response_memory.embedding = await self.embedding_generator.generate_embedding(response)
                
                # Store in LTM
                await self.ltm.store_memory(response_memory)
                
                # Store embedding in vector store
                await self.vector_store.store_embedding(
                    response_memory.id,
                    response_memory.embedding,
                    metadata={
                        "session_id": session_id,
                        "memory_type": response_memory.memory_type.value,
                        "importance": response_memory.importance.value,
                        "speaker": "agent"
                    }
                )
            
            return response_memory
            
        except Exception as e:
            logger.error(f"Error adding response memory: {e}")
            raise CAIMException(f"Failed to add response memory: {e}")
    
    async def consolidate_session_memories(self, session_id: str) -> Dict[str, Any]:
        """
        Consolidate memories for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Consolidation results
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get STM memories for promotion
            stm_memories = self.stm.get_recent_memories(session_id)
            
            # Promote important STM memories to LTM
            promoted_count = 0
            for memory in stm_memories:
                if memory.importance.value >= 0.6:  # Medium or higher importance
                    # Generate embedding if not present
                    if not memory.embedding:
                        memory.embedding = await self.embedding_generator.generate_embedding(memory.content)
                    
                    # Store in LTM
                    await self.ltm.store_memory(memory)
                    
                    # Store embedding
                    await self.vector_store.store_embedding(
                        memory.id,
                        memory.embedding,
                        metadata={
                            "session_id": session_id,
                            "memory_type": memory.memory_type.value,
                            "importance": memory.importance.value
                        }
                    )
                    
                    promoted_count += 1
            
            # Consolidate LTM memories
            inductive_thoughts = await self.ltm.consolidate_memories(session_id)
            
            # Store inductive thought embeddings
            for thought in inductive_thoughts:
                if not thought.embedding:
                    thought.embedding = await self.embedding_generator.generate_embedding(thought.content)
                    
                await self.vector_store.store_embedding(
                    thought.id,
                    thought.embedding,
                    metadata={
                        "session_id": session_id,
                        "memory_type": thought.memory_type.value,
                        "importance": thought.importance.value,
                        "is_inductive_thought": True
                    }
                )
            
            results = {
                "session_id": session_id,
                "promoted_memories": promoted_count,
                "inductive_thoughts": len(inductive_thoughts),
                "consolidation_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Consolidated session {session_id}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error consolidating session memories: {e}")
            raise CAIMException(f"Memory consolidation failed: {e}")
    
    async def get_memory_insights(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get insights about memory usage and patterns.
        
        Args:
            session_id: Optional session filter
            
        Returns:
            Memory insights and statistics
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get STM statistics
            stm_stats = self.stm.get_memory_statistics()
            
            # Get LTM statistics
            ltm_stats = await self.ltm.get_ltm_statistics()
            
            # Get memory controller statistics
            controller_stats = await self.memory_controller.get_memory_statistics(session_id)
            
            # Session-specific insights
            session_insights = {}
            if session_id:
                session_insights = await self._get_session_insights(session_id)
            
            insights = {
                "short_term_memory": stm_stats,
                "long_term_memory": ltm_stats,
                "memory_controller": controller_stats,
                "session_insights": session_insights,
                "active_sessions": len(self.active_sessions),
                "framework_uptime": self._get_uptime(),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting memory insights: {e}")
            return {}
    
    async def cleanup_expired_data(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Clean up expired data from all components.
        
        Args:
            max_age_days: Maximum age for data retention
            
        Returns:
            Cleanup statistics
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Cleanup STM
            stm_cleaned = self.stm.cleanup_expired_memories()
            
            # Cleanup LTM
            ltm_cleaned = await self.ltm.forget_irrelevant_memories(max_age_days=max_age_days)
            
            # Cleanup memory store
            store_cleaned = await self.memory_store.cleanup_expired_memories(max_age_days)
            
            # Clear retrieval cache
            self.memory_retriever.clear_cache()
            
            cleanup_stats = {
                "stm_memories_cleaned": stm_cleaned,
                "ltm_memories_cleaned": ltm_cleaned,
                "store_memories_cleaned": store_cleaned,
                "cleanup_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {}
    
    async def _memory_maintenance_loop(self) -> None:
        """Background task for memory maintenance."""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup expired STM memories
                self.stm.cleanup_expired_memories()
                
                # Periodic LTM consolidation for active sessions
                for session_id in list(self.active_sessions.keys()):
                    try:
                        await self.consolidate_session_memories(session_id)
                    except Exception as e:
                        logger.error(f"Error in maintenance consolidation for {session_id}: {e}")
                
                logger.debug("Memory maintenance cycle completed")
                
        except asyncio.CancelledError:
            logger.info("Memory maintenance loop cancelled")
        except Exception as e:
            logger.error(f"Error in memory maintenance loop: {e}")
    
    async def _get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        try:
            session_data = self.active_sessions.get(session_id, {})
            
            stm_memories = self.stm.get_recent_memories(session_id)
            ltm_memories = await self.memory_store.get_memories_by_session(session_id)
            
            return {
                "session_id": session_id,
                "interaction_count": session_data.get("interaction_count", 0),
                "stm_memory_count": len(stm_memories),
                "ltm_memory_count": len(ltm_memories),
                "session_age_hours": self._calculate_session_age(session_id),
                "last_activity": session_data.get("last_activity", datetime.utcnow()).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {}
    
    async def _get_session_insights(self, session_id: str) -> Dict[str, Any]:
        """Get detailed insights for a session."""
        try:
            memories = await self.memory_store.get_memories_by_session(session_id)
            
            if not memories:
                return {}
            
            # Analyze memory types
            type_counts = {}
            importance_distribution = {}
            
            for memory in memories:
                mem_type = memory.memory_type.value
                importance = memory.importance.value
                
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                importance_distribution[str(importance)] = importance_distribution.get(str(importance), 0) + 1
            
            # Calculate engagement metrics
            total_interactions = len([m for m in memories if m.memory_type == MemoryType.CONVERSATIONAL])
            avg_importance = sum(m.importance.value for m in memories) / len(memories)
            
            return {
                "total_memories": len(memories),
                "memory_types": type_counts,
                "importance_distribution": importance_distribution,
                "total_interactions": total_interactions,
                "average_importance": avg_importance,
                "first_interaction": min(m.timestamp for m in memories).isoformat(),
                "last_interaction": max(m.timestamp for m in memories).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting session insights: {e}")
            return {}
    
    def _calculate_session_age(self, session_id: str) -> float:
        """Calculate session age in hours."""
        if session_id not in self.active_sessions:
            return 0.0
        
        created_at = self.active_sessions[session_id].get("created_at", datetime.utcnow())
        age = datetime.utcnow() - created_at
        return age.total_seconds() / 3600
    
    def _get_uptime(self) -> str:
        """Get framework uptime."""
        # This would need to track initialization time
        return "N/A"
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the framework."""
        try:
            logger.info("Shutting down CAIM Framework...")
            
            # Cancel our background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error cancelling background task: {e}")
            
            self.background_tasks.clear()
            
            # Final cleanup
            await self.cleanup_expired_data()
            
            self.is_initialized = False
            logger.info("CAIM Framework shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if self.is_initialized:
            asyncio.create_task(self.shutdown())