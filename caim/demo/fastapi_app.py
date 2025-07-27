"""FastAPI demo application for CAIM framework."""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..core.caim_framework import CAIMFramework
from ..core.config import CAIMConfig
from ..agents.caim_agent import CAIMAgent
from ..models.model_factory import create_model
from ..benchmarks.benchmark_runner import BenchmarkRunner


logger = logging.getLogger(__name__)


# Pydantic models for API
if FASTAPI_AVAILABLE:
    class ChatRequest(BaseModel):
        message: str
        session_id: str
        context: Optional[Dict[str, Any]] = None
    
    class ChatResponse(BaseModel):
        response: str
        session_id: str
        relevant_memories_count: int
        confidence_score: Optional[float] = None
        timestamp: str
    
    class MemoryInsightRequest(BaseModel):
        session_id: Optional[str] = None
    
    class BenchmarkRequest(BaseModel):
        benchmark_type: str = "quick"
        models: Optional[List[str]] = None
    
    class SystemStatusResponse(BaseModel):
        caim_framework_status: str
        agent_status: str
        model_name: str
        uptime: str
        memory_stats: Dict[str, Any]


class FastAPIDemo:
    """FastAPI web application demo for CAIM framework."""
    
    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for web demo")
        
        self.caim_framework: Optional[CAIMFramework] = None
        self.agent: Optional[CAIMAgent] = None
        self.config: Optional[CAIMConfig] = None
        self.benchmark_runner: Optional[BenchmarkRunner] = None
        self.startup_time = datetime.utcnow()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="CAIM Framework API",
            description="Cognitive AI Memory Framework for Enhanced Long-Term Agent Interactions",
            version="0.1.0",
            lifespan=self.lifespan
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan manager."""
        # Startup
        await self._initialize_system()
        yield
        # Shutdown
        await self._cleanup_system()
    
    async def _initialize_system(self):
        """Initialize CAIM system."""
        try:
            logger.info("Initializing CAIM system for FastAPI demo")
            
            # Create configuration
            self.config = CAIMConfig.from_env()
            
            # Initialize CAIM framework
            self.caim_framework = CAIMFramework(self.config)
            await self.caim_framework.initialize()
            
            # Create model and agent
            try:
                model = create_model("Qwen/Qwen2.5-1.5B-Instruct", self.config)
            except Exception as e:
                logger.warning(f"Could not create advanced model: {e}")
                # Use mock model
                from ..models.base_model import BaseModel, ModelResponse
                
                class MockModel(BaseModel):
                    def __init__(self):
                        super().__init__(self.config, "MockModel")
                    
                    async def initialize(self):
                        self.is_initialized = True
                    
                    async def generate_response(self, prompt, **kwargs):
                        return ModelResponse(
                            content=f"FastAPI mock response to: {prompt[:50]}...",
                            model_name=self.model_name,
                            timestamp=datetime.utcnow(),
                            metadata={"mock": True, "api": "fastapi"}
                        )
                    
                    async def generate_streaming_response(self, prompt, **kwargs):
                        response = await self.generate_response(prompt, **kwargs)
                        for word in response.content.split():
                            yield word + " "
                
                model = MockModel()
            
            self.agent = CAIMAgent(
                name="FastAPI-Agent",
                config=self.config,
                model=model,
                caim_framework=self.caim_framework
            )
            await self.agent.initialize()
            
            # Initialize benchmark runner
            self.benchmark_runner = BenchmarkRunner(self.config)
            
            logger.info("FastAPI CAIM system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FastAPI system: {e}")
            raise e
    
    async def _cleanup_system(self):
        """Cleanup CAIM system."""
        try:
            if self.agent:
                await self.agent.shutdown()
            if self.caim_framework:
                await self.caim_framework.shutdown()
            
            logger.info("FastAPI CAIM system cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "message": "CAIM Framework API",
                "version": "0.1.0",
                "status": "active",
                "endpoints": {
                    "chat": "/chat",
                    "memory_insights": "/memory/insights",
                    "system_status": "/system/status",
                    "benchmark": "/benchmark",
                    "docs": "/docs"
                }
            }
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Chat with CAIM agent."""
            try:
                if not self.agent:
                    raise HTTPException(status_code=500, detail="Agent not initialized")
                
                response = await self.agent.process_message(
                    request.message,
                    request.session_id,
                    request.context
                )
                
                return ChatResponse(
                    response=response.content,
                    session_id=request.session_id,
                    relevant_memories_count=len(response.relevant_memories or []),
                    confidence_score=response.confidence_score,
                    timestamp=datetime.utcnow().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error in chat endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/memory/insights")
        async def get_memory_insights(request: MemoryInsightRequest):
            """Get memory system insights."""
            try:
                if not self.caim_framework:
                    raise HTTPException(status_code=500, detail="CAIM framework not initialized")
                
                insights = await self.caim_framework.get_memory_insights(request.session_id)
                return insights
                
            except Exception as e:
                logger.error(f"Error getting memory insights: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get system status."""
            try:
                # Calculate uptime
                uptime_delta = datetime.utcnow() - self.startup_time
                uptime_str = str(uptime_delta).split('.')[0]  # Remove microseconds
                
                # Get memory stats
                memory_stats = {}
                if self.caim_framework:
                    insights = await self.caim_framework.get_memory_insights()
                    memory_stats = {
                        "stm_memories": insights.get("short_term_memory", {}).get("total_memories", 0),
                        "ltm_memories": insights.get("long_term_memory", {}).get("total_memories", 0),
                        "active_sessions": insights.get("active_sessions", 0)
                    }
                
                return SystemStatusResponse(
                    caim_framework_status="active" if self.caim_framework and self.caim_framework.is_initialized else "inactive",
                    agent_status="ready" if self.agent and self.agent.is_initialized else "not_ready",
                    model_name=self.agent.model.model_name if self.agent and self.agent.model else "unknown",
                    uptime=uptime_str,
                    memory_stats=memory_stats
                )
                
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/benchmark")
        async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
            """Run benchmarks."""
            try:
                if not self.benchmark_runner or not self.agent:
                    raise HTTPException(status_code=500, detail="Benchmark system not ready")
                
                if request.benchmark_type == "quick":
                    # Run quick benchmark
                    test_prompts = [
                        "Hello, how are you?",
                        "What is AI?",
                        "Explain machine learning."
                    ]
                    
                    results = {
                        "benchmark_type": "quick",
                        "timestamp": datetime.utcnow().isoformat(),
                        "response_times": [],
                        "total_time": 0
                    }
                    
                    import time
                    start_time = time.time()
                    
                    for prompt in test_prompts:
                        prompt_start = time.time()
                        response = await self.agent.process_message(
                            prompt,
                            f"benchmark_{int(prompt_start)}"
                        )
                        prompt_end = time.time()
                        
                        results["response_times"].append(prompt_end - prompt_start)
                    
                    results["total_time"] = time.time() - start_time
                    results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
                    
                    return results
                
                elif request.benchmark_type == "memory":
                    # Run memory benchmark
                    results = await self.benchmark_runner.run_memory_benchmark_suite(
                        self.caim_framework
                    )
                    return results
                
                else:
                    raise HTTPException(status_code=400, detail="Unknown benchmark type")
                
            except Exception as e:
                logger.error(f"Error running benchmark: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/memory/consolidate")
        async def consolidate_memories(session_id: str):
            """Force memory consolidation for a session."""
            try:
                if not self.caim_framework:
                    raise HTTPException(status_code=500, detail="CAIM framework not initialized")
                
                result = await self.caim_framework.consolidate_session_memories(session_id)
                return result
                
            except Exception as e:
                logger.error(f"Error consolidating memories: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/memory/cleanup")
        async def cleanup_expired_memories(max_age_days: int = 30):
            """Clean up expired memories."""
            try:
                if not self.caim_framework:
                    raise HTTPException(status_code=500, detail="CAIM framework not initialized")
                
                result = await self.caim_framework.cleanup_expired_data(max_age_days)
                return result
                
            except Exception as e:
                logger.error(f"Error cleaning up memories: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                health_status = {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "components": {}
                }
                
                # Check CAIM framework
                if self.caim_framework:
                    health_status["components"]["caim_framework"] = "healthy" if self.caim_framework.is_initialized else "unhealthy"
                else:
                    health_status["components"]["caim_framework"] = "not_initialized"
                
                # Check agent
                if self.agent:
                    agent_health = await self.agent.health_check()
                    health_status["components"]["agent"] = agent_health.get("status", "unknown")
                else:
                    health_status["components"]["agent"] = "not_initialized"
                
                # Overall status
                if all(status == "healthy" for status in health_status["components"].values()):
                    health_status["status"] = "healthy"
                else:
                    health_status["status"] = "degraded"
                
                return health_status
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    demo = FastAPIDemo()
    return demo.app


# For running with uvicorn
app = create_app()


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(
            "caim.demo.fastapi_app:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except ImportError:
        print("uvicorn is required to run the FastAPI demo")
        print("Install with: pip install uvicorn")