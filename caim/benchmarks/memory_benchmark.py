"""Memory-specific benchmarking for CAIM framework."""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..core.caim_framework import CAIMFramework
from ..core.config import CAIMConfig
from ..memory.memory_types import Memory, MemoryType, MemoryImportance
from .evaluation_metrics import EvaluationMetrics


logger = logging.getLogger(__name__)


class MemoryBenchmark:
    """Specialized benchmarking for memory system performance."""
    
    def __init__(self, config: CAIMConfig):
        self.config = config
        self.evaluation_metrics = EvaluationMetrics()
        logger.info("Initialized MemoryBenchmark")
    
    async def run_comprehensive_memory_tests(
        self,
        caim_framework: CAIMFramework,
        test_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run comprehensive memory system tests."""
        try:
            results = {
                "test_type": "comprehensive_memory_tests",
                "timestamp": datetime.utcnow().isoformat(),
                "total_scenarios": len(test_scenarios),
                "scenario_results": [],
                "overall_score": 0.0
            }
            
            total_score = 0.0
            
            for i, scenario in enumerate(test_scenarios):
                scenario_result = await self._run_memory_scenario(caim_framework, scenario, i)
                results["scenario_results"].append(scenario_result)
                total_score += scenario_result.get("score", 0.0)
            
            results["overall_score"] = total_score / len(test_scenarios) if test_scenarios else 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Error running comprehensive memory tests: {e}")
            return {"error": str(e)}
    
    async def _run_memory_scenario(
        self,
        caim_framework: CAIMFramework,
        scenario: Dict[str, Any],
        scenario_index: int
    ) -> Dict[str, Any]:
        """Run a single memory test scenario."""
        try:
            scenario_name = scenario.get("name", f"scenario_{scenario_index}")
            logger.info(f"Running memory scenario: {scenario_name}")
            
            result = {
                "scenario_name": scenario_name,
                "description": scenario.get("description", ""),
                "score": 0.0,
                "details": {}
            }
            
            # Test storage and retrieval
            storage_score = await self._test_memory_storage(caim_framework, scenario)
            retrieval_score = await self._test_memory_retrieval(caim_framework, scenario)
            consolidation_score = await self._test_memory_consolidation(caim_framework, scenario)
            
            # Calculate overall score
            result["score"] = (storage_score + retrieval_score + consolidation_score) / 3.0
            result["details"] = {
                "storage_score": storage_score,
                "retrieval_score": retrieval_score,
                "consolidation_score": consolidation_score
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error running memory scenario: {e}")
            return {
                "scenario_name": scenario.get("name", f"scenario_{scenario_index}"),
                "score": 0.0,
                "error": str(e)
            }
    
    async def _test_memory_storage(
        self,
        caim_framework: CAIMFramework,
        scenario: Dict[str, Any]
    ) -> float:
        """Test memory storage functionality."""
        try:
            memories_data = scenario.get("memories", [])
            if not memories_data:
                return 0.0
            
            session_id = f"storage_test_{int(time.time())}"
            stored_count = 0
            
            for memory_data in memories_data:
                try:
                    await caim_framework.process_input(
                        user_input=memory_data["content"],
                        session_id=session_id,
                        user_metadata={"importance": memory_data.get("importance", "medium")}
                    )
                    stored_count += 1
                except Exception as e:
                    logger.error(f"Error storing memory: {e}")
            
            storage_score = stored_count / len(memories_data) if memories_data else 0.0
            return storage_score
            
        except Exception as e:
            logger.error(f"Error testing memory storage: {e}")
            return 0.0
    
    async def _test_memory_retrieval(
        self,
        caim_framework: CAIMFramework,
        scenario: Dict[str, Any]
    ) -> float:
        """Test memory retrieval accuracy."""
        try:
            queries = scenario.get("queries", [])
            if not queries:
                return 0.0
            
            session_id = f"retrieval_test_{int(time.time())}"
            retrieval_scores = []
            
            for query in queries:
                try:
                    result = await caim_framework.process_input(
                        user_input=query,
                        session_id=session_id
                    )
                    
                    relevant_memories = result.get("relevant_ltm_memories", [])
                    
                    # Simple scoring based on whether memories were retrieved
                    if relevant_memories:
                        retrieval_scores.append(1.0)
                    else:
                        retrieval_scores.append(0.0)
                        
                except Exception as e:
                    logger.error(f"Error testing retrieval for query '{query}': {e}")
                    retrieval_scores.append(0.0)
            
            avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
            return avg_retrieval_score
            
        except Exception as e:
            logger.error(f"Error testing memory retrieval: {e}")
            return 0.0
    
    async def _test_memory_consolidation(
        self,
        caim_framework: CAIMFramework,
        scenario: Dict[str, Any]
    ) -> float:
        """Test memory consolidation functionality."""
        try:
            session_id = f"consolidation_test_{int(time.time())}"
            
            # First store some memories
            memories_data = scenario.get("memories", [])
            for memory_data in memories_data:
                await caim_framework.process_input(
                    user_input=memory_data["content"],
                    session_id=session_id
                )
            
            # Trigger consolidation
            consolidation_result = await caim_framework.consolidate_session_memories(session_id)
            
            # Score based on successful consolidation
            promoted_memories = consolidation_result.get("promoted_memories", 0)
            inductive_thoughts = consolidation_result.get("inductive_thoughts", 0)
            
            if promoted_memories > 0 or inductive_thoughts > 0:
                return 1.0
            else:
                return 0.5  # Partial credit if no consolidation needed
                
        except Exception as e:
            logger.error(f"Error testing memory consolidation: {e}")
            return 0.0