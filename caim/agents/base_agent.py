"""Base agent interface for CAIM framework."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
from dataclasses import dataclass

from ..core.config import CAIMConfig
from ..models.base_model import BaseModel, ModelResponse


logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    agent_name: str
    session_id: str
    timestamp: datetime
    confidence_score: Optional[float] = None
    relevant_memories: Optional[List[Dict[str, Any]]] = None
    model_response: Optional[ModelResponse] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Abstract base class for all agents in CAIM framework."""
    
    def __init__(
        self,
        name: str,
        config: CAIMConfig,
        model: BaseModel,
        caim_framework = None
    ):
        self.name = name
        self.config = config
        self.model = model
        self.caim_framework = caim_framework
        
        self.is_initialized = False
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"Initialized agent: {name}")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent."""
        pass
    
    @abstractmethod
    async def process_message(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a user message and generate a response.
        
        Args:
            message: User message
            session_id: Session identifier
            context: Additional context
            
        Returns:
            Agent response
        """
        pass
    
    @abstractmethod
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
        pass
    
    def _prepare_context(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare context for processing."""
        try:
            processed_context = {
                "message": message,
                "session_id": session_id,
                "timestamp": datetime.utcnow(),
                "agent_name": self.name
            }
            
            if context:
                processed_context.update(context)
            
            # Add conversation history
            if session_id in self.conversation_history:
                processed_context["conversation_history"] = self.conversation_history[session_id][-5:]  # Last 5 messages
            
            return processed_context
            
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            return {"message": message, "session_id": session_id}
    
    def _update_conversation_history(
        self,
        session_id: str,
        user_message: str,
        agent_response: str
    ) -> None:
        """Update conversation history."""
        try:
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            self.conversation_history[session_id].append({
                "timestamp": datetime.utcnow().isoformat(),
                "user_message": user_message,
                "agent_response": agent_response
            })
            
            # Keep only last 20 interactions
            if len(self.conversation_history[session_id]) > 20:
                self.conversation_history[session_id] = self.conversation_history[session_id][-20:]
                
        except Exception as e:
            logger.error(f"Error updating conversation history: {e}")
    
    async def get_conversation_summary(self, session_id: str) -> str:
        """Get a summary of the conversation."""
        try:
            if session_id not in self.conversation_history:
                return "No conversation history found."
            
            history = self.conversation_history[session_id]
            if not history:
                return "No conversation history found."
            
            # Simple summary - could be enhanced with summarization model
            total_messages = len(history)
            recent_topics = []
            
            for entry in history[-3:]:  # Last 3 interactions
                user_msg = entry.get("user_message", "")
                if len(user_msg) > 50:
                    recent_topics.append(user_msg[:50] + "...")
                else:
                    recent_topics.append(user_msg)
            
            summary = f"Conversation with {total_messages} interactions. "
            if recent_topics:
                summary += f"Recent topics: {', '.join(recent_topics)}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return "Error generating conversation summary."
    
    async def clear_conversation_history(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        try:
            if session_id in self.conversation_history:
                del self.conversation_history[session_id]
                logger.info(f"Cleared conversation history for session: {session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            return False
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        try:
            total_sessions = len(self.conversation_history)
            total_interactions = sum(len(history) for history in self.conversation_history.values())
            
            active_sessions = [
                session_id for session_id, history in self.conversation_history.items()
                if history and (datetime.utcnow() - datetime.fromisoformat(history[-1]["timestamp"])).total_seconds() < 3600
            ]
            
            return {
                "agent_name": self.name,
                "total_sessions": total_sessions,
                "active_sessions": len(active_sessions),
                "total_interactions": total_interactions,
                "model_name": self.model.model_name if self.model else "None",
                "is_initialized": self.is_initialized,
                "uptime": self._get_uptime()
            }
            
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {}
    
    def _get_uptime(self) -> str:
        """Get agent uptime (placeholder)."""
        return "N/A"
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform agent health check."""
        try:
            health_status = {
                "agent_name": self.name,
                "status": "healthy" if self.is_initialized else "not_initialized",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check model health
            if self.model:
                model_health = await self.model.health_check()
                health_status["model_health"] = model_health
                
                if model_health.get("status") != "healthy":
                    health_status["status"] = "unhealthy"
            
            # Check CAIM framework health
            if self.caim_framework:
                health_status["caim_framework_initialized"] = self.caim_framework.is_initialized
                if not self.caim_framework.is_initialized:
                    health_status["status"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            return {
                "agent_name": self.name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def shutdown(self) -> None:
        """Shutdown the agent."""
        try:
            # Clear conversation history
            self.conversation_history.clear()
            
            # Shutdown model if owned by agent
            if self.model:
                await self.model.shutdown()
            
            self.is_initialized = False
            logger.info(f"Agent {self.name} shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down agent: {e}")
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model.model_name if self.model else 'None'}')"
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"model='{self.model.model_name if self.model else 'None'}', "
                f"initialized={self.is_initialized})")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            import asyncio
            try:
                asyncio.create_task(self.shutdown())
            except:
                pass