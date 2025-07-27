"""Agent factory for creating different agent instances."""

import logging
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from .caim_agent import CAIMAgent
from ..core.config import CAIMConfig
from ..core.caim_framework import CAIMFramework
from ..models.base_model import BaseModel
from ..models.model_factory import create_model
from ..core.exceptions import CAIMException


logger = logging.getLogger(__name__)


def create_agent(
    agent_type: str = "caim",
    name: str = "CAIMAgent",
    config: Optional[CAIMConfig] = None,
    model: Optional[BaseModel] = None,
    model_name: Optional[str] = None,
    caim_framework: Optional[CAIMFramework] = None,
    **kwargs
) -> BaseAgent:
    """
    Create an agent instance.
    
    Args:
        agent_type: Type of agent to create ("caim", "base")
        name: Agent name
        config: CAIM configuration
        model: Pre-initialized model (takes precedence over model_name)
        model_name: Name of model to create if model not provided
        caim_framework: CAIM framework instance
        **kwargs: Additional agent-specific parameters
        
    Returns:
        Configured agent instance
        
    Raises:
        CAIMException: If agent creation fails
    """
    try:
        # Use default config if not provided
        if config is None:
            config = CAIMConfig.from_env()
        
        # Create model if not provided
        if model is None:
            if model_name is None:
                model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Default model
            
            model = create_model(model_name, config)
        
        # Create CAIM framework if not provided and agent type is caim
        if agent_type == "caim" and caim_framework is None:
            caim_framework = CAIMFramework(config)
        
        # Create agent based on type
        if agent_type == "caim":
            agent = CAIMAgent(
                name=name,
                config=config,
                model=model,
                caim_framework=caim_framework
            )
        elif agent_type == "base":
            agent = BaseAgent(
                name=name,
                config=config,
                model=model,
                caim_framework=caim_framework
            )
        else:
            raise CAIMException(f"Unknown agent type: {agent_type}")
        
        logger.info(f"Created {agent_type} agent: {name}")
        return agent
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise CAIMException(f"Agent creation failed: {e}")


def create_multi_model_agents(
    config: Optional[CAIMConfig] = None,
    agent_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    shared_caim_framework: bool = True
) -> List[BaseAgent]:
    """
    Create multiple agents with different models for comparison.
    
    Args:
        config: CAIM configuration
        agent_names: List of agent names
        model_names: List of model names
        shared_caim_framework: Whether agents share the same CAIM framework
        
    Returns:
        List of agent instances
    """
    try:
        if config is None:
            config = CAIMConfig.from_env()
        
        # Default model names for comparison
        if model_names is None:
            model_names = [
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "google/gemma-2b-it"
            ]
            
            # Add OpenAI model if API key available
            if config.openai_api_key:
                model_names.append("gpt-4o-mini")
        
        # Default agent names
        if agent_names is None:
            agent_names = [f"Agent_{i+1}" for i in range(len(model_names))]
        
        # Ensure same length
        if len(agent_names) != len(model_names):
            raise CAIMException("Number of agent names must match number of models")
        
        # Create shared CAIM framework if requested
        caim_framework = None
        if shared_caim_framework:
            caim_framework = CAIMFramework(config)
        
        agents = []
        
        for agent_name, model_name in zip(agent_names, model_names):
            try:
                # Create individual CAIM framework if not shared
                agent_caim_framework = caim_framework
                if not shared_caim_framework:
                    agent_caim_framework = CAIMFramework(config)
                
                agent = create_agent(
                    agent_type="caim",
                    name=agent_name,
                    config=config,
                    model_name=model_name,
                    caim_framework=agent_caim_framework
                )
                
                agents.append(agent)
                logger.info(f"Created agent {agent_name} with model {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to create agent {agent_name} with model {model_name}: {e}")
        
        if not agents:
            raise CAIMException("No agents could be created")
        
        logger.info(f"Created {len(agents)} agents for comparison")
        return agents
        
    except Exception as e:
        logger.error(f"Error creating multi-model agents: {e}")
        return []


def create_benchmark_agents(
    config: Optional[CAIMConfig] = None,
    include_baselines: bool = True
) -> Dict[str, BaseAgent]:
    """
    Create agents for benchmarking different models and configurations.
    
    Args:
        config: CAIM configuration
        include_baselines: Whether to include baseline (non-CAIM) agents
        
    Returns:
        Dictionary of agent_name -> agent mappings
    """
    try:
        if config is None:
            config = CAIMConfig.from_env()
        
        agents = {}
        
        # CAIM-enhanced agents
        caim_models = [
            ("Qwen_1.5B_CAIM", "Qwen/Qwen2.5-1.5B-Instruct"),
            ("Qwen_3B_CAIM", "Qwen/Qwen2.5-3B-Instruct"),
            ("Gemma_2B_CAIM", "google/gemma-2b-it")
        ]
        
        # Add OpenAI if available
        if config.openai_api_key:
            caim_models.append(("GPT4O_Mini_CAIM", "gpt-4o-mini"))
        
        # Create CAIM framework
        caim_framework = CAIMFramework(config)
        
        # Create CAIM agents
        for agent_name, model_name in caim_models:
            try:
                agent = create_agent(
                    agent_type="caim",
                    name=agent_name,
                    config=config,
                    model_name=model_name,
                    caim_framework=caim_framework
                )
                agents[agent_name] = agent
                
            except Exception as e:
                logger.warning(f"Failed to create CAIM agent {agent_name}: {e}")
        
        # Create baseline agents (without CAIM)
        if include_baselines:
            baseline_models = [
                ("Qwen_1.5B_Baseline", "Qwen/Qwen2.5-1.5B-Instruct"),
                ("Qwen_3B_Baseline", "Qwen/Qwen2.5-3B-Instruct"),
                ("Gemma_2B_Baseline", "google/gemma-2b-it")
            ]
            
            if config.openai_api_key:
                baseline_models.append(("GPT4O_Mini_Baseline", "gpt-4o-mini"))
            
            for agent_name, model_name in baseline_models:
                try:
                    agent = create_agent(
                        agent_type="base",
                        name=agent_name,
                        config=config,
                        model_name=model_name,
                        caim_framework=None  # No CAIM for baselines
                    )
                    agents[agent_name] = agent
                    
                except Exception as e:
                    logger.warning(f"Failed to create baseline agent {agent_name}: {e}")
        
        logger.info(f"Created {len(agents)} benchmark agents")
        return agents
        
    except Exception as e:
        logger.error(f"Error creating benchmark agents: {e}")
        return {}


async def initialize_agents(agents: List[BaseAgent]) -> Dict[str, Any]:
    """
    Initialize multiple agents.
    
    Args:
        agents: List of agents to initialize
        
    Returns:
        Initialization results
    """
    try:
        results = {
            "total_agents": len(agents),
            "successful_initializations": 0,
            "failed_initializations": 0,
            "errors": []
        }
        
        for agent in agents:
            try:
                await agent.initialize()
                results["successful_initializations"] += 1
                logger.info(f"Successfully initialized agent: {agent.name}")
                
            except Exception as e:
                results["failed_initializations"] += 1
                error_msg = f"Failed to initialize agent {agent.name}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return results
        
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        return {"error": str(e)}


async def shutdown_agents(agents: List[BaseAgent]) -> Dict[str, Any]:
    """
    Shutdown multiple agents.
    
    Args:
        agents: List of agents to shutdown
        
    Returns:
        Shutdown results
    """
    try:
        results = {
            "total_agents": len(agents),
            "successful_shutdowns": 0,
            "failed_shutdowns": 0,
            "errors": []
        }
        
        for agent in agents:
            try:
                await agent.shutdown()
                results["successful_shutdowns"] += 1
                logger.info(f"Successfully shutdown agent: {agent.name}")
                
            except Exception as e:
                results["failed_shutdowns"] += 1
                error_msg = f"Failed to shutdown agent {agent.name}: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        return results
        
    except Exception as e:
        logger.error(f"Error shutting down agents: {e}")
        return {"error": str(e)}


def get_agent_capabilities(agent_type: str = "caim") -> Dict[str, Any]:
    """
    Get capabilities information for an agent type.
    
    Args:
        agent_type: Type of agent
        
    Returns:
        Capabilities information
    """
    capabilities = {
        "base": {
            "memory_integration": False,
            "conversation_history": True,
            "model_integration": True,
            "streaming_support": True,
            "health_monitoring": True,
            "statistics": True
        },
        "caim": {
            "memory_integration": True,
            "conversation_history": True,
            "model_integration": True,
            "streaming_support": True,
            "health_monitoring": True,
            "statistics": True,
            "long_term_memory": True,
            "memory_consolidation": True,
            "context_awareness": True,
            "personalization": True,
            "memory_insights": True
        }
    }
    
    return capabilities.get(agent_type, {})