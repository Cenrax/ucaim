"""CAIM Agent implementation with memory-enhanced interactions."""

import logging
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime

from .base_agent import BaseAgent, AgentResponse
from ..core.config import CAIMConfig
from ..core.caim_framework import CAIMFramework
from ..models.base_model import BaseModel
from ..memory.memory_types import MemoryImportance
from ..core.exceptions import CAIMException


logger = logging.getLogger(__name__)


class CAIMAgent(BaseAgent):
    """
    CAIM Agent that leverages the CAIM framework for memory-enhanced interactions.
    
    This agent integrates with the CAIM memory system to provide:
    - Context-aware responses based on conversation history
    - Long-term memory consolidation
    - Personalized interactions based on user patterns
    - Intelligent memory retrieval and utilization
    """
    
    def __init__(
        self,
        name: str,
        config: CAIMConfig,
        model: BaseModel,
        caim_framework: Optional[CAIMFramework] = None
    ):
        super().__init__(name, config, model, caim_framework)
        
        # Agent-specific settings
        self.memory_importance_threshold = 0.6
        self.max_context_memories = 5
        self.auto_consolidate = True
        self.consolidation_interval = 10  # Consolidate every N interactions
        
        # Interaction tracking
        self.interaction_counts: Dict[str, int] = {}
        self.last_consolidation: Dict[str, int] = {}
        
        logger.info(f"Initialized CAIM Agent: {name}")
    
    async def initialize(self) -> None:
        """Initialize the CAIM agent."""
        try:
            # Initialize model
            if not self.model.is_initialized:
                await self.model.initialize()
            
            # Initialize CAIM framework if provided
            if self.caim_framework and not self.caim_framework.is_initialized:
                await self.caim_framework.initialize()
            
            self.is_initialized = True
            logger.info(f"CAIM Agent {self.name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing CAIM agent: {e}")
            raise CAIMException(f"CAIM agent initialization failed: {e}")
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a user message with CAIM memory integration.
        
        Args:
            message: User message
            session_id: Session identifier
            context: Additional context
            
        Returns:
            Memory-enhanced agent response
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Track interaction
            self.interaction_counts[session_id] = self.interaction_counts.get(session_id, 0) + 1
            
            # Prepare context
            processed_context = self._prepare_context(message, session_id, context)
            
            # Process through CAIM framework if available
            caim_data = None
            relevant_memories = []
            
            if self.caim_framework:
                caim_data = await self.caim_framework.process_input(
                    user_input=message,
                    session_id=session_id,
                    context=processed_context.get("conversation_context"),
                    user_metadata=context
                )
                relevant_memories = caim_data.get("relevant_ltm_memories", [])
            
            # Prepare memory context for model
            memory_context = self._prepare_memory_context(relevant_memories)
            
            # Get conversation context
            conversation_context = self._build_conversation_context(session_id)
            
            # Generate response using model
            model_response = await self.model.generate_response(
                prompt=message,
                context=conversation_context,
                memory_context=memory_context,
                max_tokens=self.config.max_memory_size // 10,  # Reasonable default
                temperature=0.7
            )
            
            # Store agent response in memory
            if self.caim_framework:
                await self.caim_framework.add_response_memory(
                    response=model_response.content,
                    session_id=session_id,
                    importance=self._determine_response_importance(model_response),
                    metadata={
                        "model_name": self.model.model_name,
                        "agent_name": self.name,
                        "relevant_memories_count": len(relevant_memories)
                    }
                )
            
            # Update conversation history
            self._update_conversation_history(session_id, message, model_response.content)
            
            # Auto-consolidate if needed
            if self.auto_consolidate and self._should_consolidate(session_id):
                await self._consolidate_session_memories(session_id)
            
            # Create agent response
            agent_response = AgentResponse(
                content=model_response.content,
                agent_name=self.name,
                session_id=session_id,
                timestamp=datetime.utcnow(),
                confidence_score=model_response.confidence_score,
                relevant_memories=relevant_memories,
                model_response=model_response,
                metadata={
                    "interaction_count": self.interaction_counts[session_id],
                    "memory_enhanced": len(relevant_memories) > 0,
                    "caim_data": caim_data
                }
            )
            
            logger.info(f"Processed message for session {session_id} with {len(relevant_memories)} memories")
            return agent_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise CAIMException(f"Message processing failed: {e}")
    
    async def process_streaming_message(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Process a user message and generate a streaming response.
        
        Args:
            message: User message
            session_id: Session identifier
            context: Additional context
            
        Yields:
            Response chunks
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Track interaction
            self.interaction_counts[session_id] = self.interaction_counts.get(session_id, 0) + 1
            
            # Prepare context
            processed_context = self._prepare_context(message, session_id, context)
            
            # Process through CAIM framework if available
            relevant_memories = []
            if self.caim_framework:
                caim_data = await self.caim_framework.process_input(
                    user_input=message,
                    session_id=session_id,
                    context=processed_context.get("conversation_context"),
                    user_metadata=context
                )
                relevant_memories = caim_data.get("relevant_ltm_memories", [])
            
            # Prepare memory context for model
            memory_context = self._prepare_memory_context(relevant_memories)
            
            # Get conversation context
            conversation_context = self._build_conversation_context(session_id)
            
            # Generate streaming response
            response_chunks = []
            async for chunk in self.model.generate_streaming_response(
                prompt=message,
                context=conversation_context,
                memory_context=memory_context,
                max_tokens=self.config.max_memory_size // 10,
                temperature=0.7
            ):
                response_chunks.append(chunk)
                yield chunk
            
            # Reconstruct full response for memory storage
            full_response = "".join(response_chunks)
            
            # Store agent response in memory
            if self.caim_framework and full_response:
                await self.caim_framework.add_response_memory(
                    response=full_response,
                    session_id=session_id,
                    importance=MemoryImportance.MEDIUM,
                    metadata={
                        "model_name": self.model.model_name,
                        "agent_name": self.name,
                        "streaming": True
                    }
                )
            
            # Update conversation history
            if full_response:
                self._update_conversation_history(session_id, message, full_response)
            
            # Auto-consolidate if needed
            if self.auto_consolidate and self._should_consolidate(session_id):
                await self._consolidate_session_memories(session_id)
                
        except Exception as e:
            logger.error(f"Error processing streaming message: {e}")
            yield f"Error: {e}"
    
    def _prepare_memory_context(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare memory context for the model."""
        try:
            if not memories:
                return []
            
            # Limit to max context memories
            limited_memories = memories[:self.max_context_memories]
            
            # Format memories for model context
            formatted_memories = []
            for memory in limited_memories:
                formatted_memory = {
                    "content": memory.get("content", ""),
                    "importance": memory.get("importance", 0.5),
                    "timestamp": memory.get("timestamp", ""),
                    "memory_type": memory.get("memory_type", "unknown")
                }
                formatted_memories.append(formatted_memory)
            
            return formatted_memories
            
        except Exception as e:
            logger.error(f"Error preparing memory context: {e}")
            return []
    
    def _build_conversation_context(self, session_id: str) -> str:
        """Build conversation context string."""
        try:
            if session_id not in self.conversation_history:
                return ""
            
            history = self.conversation_history[session_id]
            if not history:
                return ""
            
            # Use last 3 interactions for context
            recent_history = history[-3:]
            context_parts = []
            
            for entry in recent_history:
                user_msg = entry.get("user_message", "")
                agent_resp = entry.get("agent_response", "")
                
                if user_msg:
                    context_parts.append(f"User: {user_msg}")
                if agent_resp:
                    context_parts.append(f"Assistant: {agent_resp}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return ""
    
    def _determine_response_importance(self, model_response) -> MemoryImportance:
        """Determine the importance of a response."""
        try:
            # Use confidence score if available
            if model_response.confidence_score:
                if model_response.confidence_score > 0.8:
                    return MemoryImportance.HIGH
                elif model_response.confidence_score > 0.6:
                    return MemoryImportance.MEDIUM
                else:
                    return MemoryImportance.LOW
            
            # Use response length as a proxy
            response_length = len(model_response.content)
            if response_length > 200:
                return MemoryImportance.MEDIUM
            elif response_length > 50:
                return MemoryImportance.LOW
            else:
                return MemoryImportance.MINIMAL
                
        except Exception as e:
            logger.error(f"Error determining response importance: {e}")
            return MemoryImportance.MEDIUM
    
    def _should_consolidate(self, session_id: str) -> bool:
        """Determine if memory consolidation should be triggered."""
        try:
            current_interactions = self.interaction_counts.get(session_id, 0)
            last_consolidation = self.last_consolidation.get(session_id, 0)
            
            return (current_interactions - last_consolidation) >= self.consolidation_interval
            
        except Exception as e:
            logger.error(f"Error checking consolidation condition: {e}")
            return False
    
    async def _consolidate_session_memories(self, session_id: str) -> None:
        """Consolidate memories for a session."""
        try:
            if not self.caim_framework:
                return
            
            results = await self.caim_framework.consolidate_session_memories(session_id)
            self.last_consolidation[session_id] = self.interaction_counts.get(session_id, 0)
            
            logger.info(f"Consolidated memories for session {session_id}: {results}")
            
        except Exception as e:
            logger.error(f"Error consolidating session memories: {e}")
    
    async def get_memory_insights(self, session_id: str) -> Dict[str, Any]:
        """Get memory insights for a session."""
        try:
            if not self.caim_framework:
                return {"error": "CAIM framework not available"}
            
            insights = await self.caim_framework.get_memory_insights(session_id)
            
            # Add agent-specific insights
            agent_insights = {
                "interaction_count": self.interaction_counts.get(session_id, 0),
                "last_consolidation": self.last_consolidation.get(session_id, 0),
                "next_consolidation_at": self.last_consolidation.get(session_id, 0) + self.consolidation_interval,
                "conversation_summary": await self.get_conversation_summary(session_id)
            }
            
            insights["agent_insights"] = agent_insights
            return insights
            
        except Exception as e:
            logger.error(f"Error getting memory insights: {e}")
            return {"error": str(e)}
    
    async def force_consolidation(self, session_id: str) -> Dict[str, Any]:
        """Force memory consolidation for a session."""
        try:
            if not self.caim_framework:
                return {"error": "CAIM framework not available"}
            
            results = await self.caim_framework.consolidate_session_memories(session_id)
            self.last_consolidation[session_id] = self.interaction_counts.get(session_id, 0)
            
            return {
                "consolidation_results": results,
                "forced": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error forcing consolidation: {e}")
            return {"error": str(e)}
    
    async def get_relevant_memories(
        self,
        query: str,
        session_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant memories for a query."""
        try:
            if not self.caim_framework:
                return []
            
            memories = await self.caim_framework.memory_controller.retrieve_relevant_memories(
                query=query,
                session_id=session_id,
                limit=limit
            )
            
            return [memory.to_dict() for memory in memories]
            
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        try:
            base_stats = await super().get_agent_statistics()
            
            # Add CAIM-specific statistics
            caim_stats = {
                "total_interactions": sum(self.interaction_counts.values()),
                "sessions_with_consolidation": len(self.last_consolidation),
                "avg_interactions_per_session": (
                    sum(self.interaction_counts.values()) / max(1, len(self.interaction_counts))
                ),
                "memory_importance_threshold": self.memory_importance_threshold,
                "auto_consolidate_enabled": self.auto_consolidate,
                "consolidation_interval": self.consolidation_interval
            }
            
            base_stats.update(caim_stats)
            
            # Add CAIM framework statistics if available
            if self.caim_framework:
                framework_insights = await self.caim_framework.get_memory_insights()
                base_stats["caim_framework_stats"] = framework_insights
            
            return base_stats
            
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {}
    
    async def configure_agent(self, **kwargs) -> Dict[str, Any]:
        """Configure agent parameters."""
        try:
            updated_params = {}
            
            if "memory_importance_threshold" in kwargs:
                self.memory_importance_threshold = kwargs["memory_importance_threshold"]
                updated_params["memory_importance_threshold"] = self.memory_importance_threshold
            
            if "max_context_memories" in kwargs:
                self.max_context_memories = kwargs["max_context_memories"]
                updated_params["max_context_memories"] = self.max_context_memories
            
            if "auto_consolidate" in kwargs:
                self.auto_consolidate = kwargs["auto_consolidate"]
                updated_params["auto_consolidate"] = self.auto_consolidate
            
            if "consolidation_interval" in kwargs:
                self.consolidation_interval = kwargs["consolidation_interval"]
                updated_params["consolidation_interval"] = self.consolidation_interval
            
            logger.info(f"Updated agent configuration: {updated_params}")
            
            return {
                "status": "success",
                "updated_parameters": updated_params,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error configuring agent: {e}")
            return {"status": "error", "error": str(e)}
    
    async def reset_session(self, session_id: str) -> Dict[str, Any]:
        """Reset all data for a session."""
        try:
            # Clear conversation history
            await self.clear_conversation_history(session_id)
            
            # Reset interaction tracking
            if session_id in self.interaction_counts:
                del self.interaction_counts[session_id]
            
            if session_id in self.last_consolidation:
                del self.last_consolidation[session_id]
            
            # Note: CAIM framework memories are not cleared automatically
            # This requires explicit user action for data persistence
            
            return {
                "status": "success",
                "session_id": session_id,
                "reset_timestamp": datetime.utcnow().isoformat(),
                "note": "CAIM framework memories preserved (clear separately if needed)"
            }
            
        except Exception as e:
            logger.error(f"Error resetting session: {e}")
            return {"status": "error", "error": str(e)}
    
    def __str__(self) -> str:
        """String representation of the CAIM agent."""
        caim_status = "with CAIM" if self.caim_framework else "no CAIM"
        return f"CAIMAgent(name='{self.name}', model='{self.model.model_name if self.model else 'None'}', {caim_status})"