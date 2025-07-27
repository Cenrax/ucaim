"""Benchmark runner for comprehensive CAIM framework evaluation."""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .evaluation_metrics import EvaluationMetrics, BenchmarkResult
from .memory_benchmark import MemoryBenchmark
from .model_comparison import ModelComparison
from ..agents.base_agent import BaseAgent
from ..core.caim_framework import CAIMFramework
from ..core.config import CAIMConfig


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSuite:
    """Definition of a benchmark test suite."""
    name: str
    description: str
    test_cases: List[Dict[str, Any]]
    expected_results: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 300


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for CAIM framework evaluation.
    
    Runs various benchmark suites to evaluate:
    - Memory system performance
    - Agent capabilities
    - Model comparisons
    - System efficiency
    - Overall framework performance
    """
    
    def __init__(self, config: Optional[CAIMConfig] = None):
        self.config = config or CAIMConfig.from_env()
        self.evaluation_metrics = EvaluationMetrics()
        self.memory_benchmark = MemoryBenchmark(self.config)
        self.model_comparison = ModelComparison(self.config)
        
        self.benchmark_results: Dict[str, List[BenchmarkResult]] = {}
        self.benchmark_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized BenchmarkRunner")
    
    async def run_full_benchmark_suite(
        self,
        agents: List[BaseAgent],
        caim_framework: Optional[CAIMFramework] = None,
        custom_suites: Optional[List[BenchmarkSuite]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.
        
        Args:
            agents: List of agents to benchmark
            caim_framework: CAIM framework instance
            custom_suites: Additional custom benchmark suites
            
        Returns:
            Comprehensive benchmark results
        """
        try:
            logger.info("Starting full benchmark suite")
            start_time = time.time()
            
            # Initialize agents
            initialization_results = await self._initialize_agents(agents)
            
            # Run core benchmarks
            core_results = await self._run_core_benchmarks(agents, caim_framework)
            
            # Run memory benchmarks
            memory_results = await self._run_memory_benchmarks(caim_framework)
            
            # Run model comparison benchmarks
            model_results = await self._run_model_comparison_benchmarks(agents)
            
            # Run custom benchmarks if provided
            custom_results = {}
            if custom_suites:
                custom_results = await self._run_custom_benchmarks(custom_suites, agents)
            
            # Run performance benchmarks
            performance_results = await self._run_performance_benchmarks(agents, caim_framework)
            
            # Generate comprehensive report
            end_time = time.time()
            execution_time = end_time - start_time
            
            benchmark_summary = {
                "benchmark_id": f"benchmark_{int(time.time())}",
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time_seconds": execution_time,
                "agents_tested": len(agents),
                "initialization_results": initialization_results,
                "core_benchmark_results": core_results,
                "memory_benchmark_results": memory_results,
                "model_comparison_results": model_results,
                "custom_benchmark_results": custom_results,
                "performance_benchmark_results": performance_results,
                "overall_metrics": self.evaluation_metrics.get_comprehensive_report()
            }
            
            # Store in history
            self.benchmark_history.append(benchmark_summary)
            
            logger.info(f"Full benchmark suite completed in {execution_time:.2f} seconds")
            return benchmark_summary
            
        except Exception as e:
            logger.error(f"Error running full benchmark suite: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def run_memory_benchmark_suite(
        self,
        caim_framework: CAIMFramework,
        test_scenarios: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run focused memory system benchmarks.
        
        Args:
            caim_framework: CAIM framework instance
            test_scenarios: Custom test scenarios
            
        Returns:
            Memory benchmark results
        """
        try:
            logger.info("Starting memory benchmark suite")
            
            # Use default scenarios if none provided
            if test_scenarios is None:
                test_scenarios = self._get_default_memory_scenarios()
            
            results = await self.memory_benchmark.run_comprehensive_memory_tests(
                caim_framework, test_scenarios
            )
            
            # Store results
            if "memory_benchmarks" not in self.benchmark_results:
                self.benchmark_results["memory_benchmarks"] = []
            
            benchmark_result = BenchmarkResult(
                metric_name="memory_benchmark_suite",
                score=results.get("overall_score", 0.0),
                max_score=1.0,
                percentage=results.get("overall_score", 0.0) * 100,
                details=results,
                timestamp=datetime.utcnow()
            )
            
            self.benchmark_results["memory_benchmarks"].append(benchmark_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running memory benchmark suite: {e}")
            return {"error": str(e)}
    
    async def run_agent_comparison_benchmark(
        self,
        agents: List[BaseAgent],
        test_prompts: Optional[List[str]] = None,
        evaluation_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comparative benchmarks between different agents.
        
        Args:
            agents: List of agents to compare
            test_prompts: Test prompts for evaluation
            evaluation_criteria: Specific criteria to evaluate
            
        Returns:
            Agent comparison results
        """
        try:
            logger.info(f"Starting agent comparison benchmark with {len(agents)} agents")
            
            # Use default prompts if none provided
            if test_prompts is None:
                test_prompts = self._get_default_test_prompts()
            
            # Use default criteria if none provided
            if evaluation_criteria is None:
                evaluation_criteria = ["response_quality", "coherence", "efficiency", "memory_usage"]
            
            comparison_results = {}
            
            for i, agent in enumerate(agents):
                logger.info(f"Testing agent {i+1}/{len(agents)}: {agent.name}")
                
                agent_results = {
                    "agent_name": agent.name,
                    "model_name": agent.model.model_name if agent.model else "Unknown",
                    "test_results": [],
                    "performance_metrics": {},
                    "errors": []
                }
                
                # Test each prompt
                for j, prompt in enumerate(test_prompts):
                    try:
                        test_start_time = time.time()
                        
                        # Generate response
                        session_id = f"benchmark_session_{i}_{j}"
                        response = await agent.process_message(prompt, session_id)
                        
                        test_end_time = time.time()
                        response_time = test_end_time - test_start_time
                        
                        # Evaluate response
                        evaluation_results = await self._evaluate_agent_response(
                            response, prompt, evaluation_criteria
                        )
                        
                        test_result = {
                            "prompt_index": j,
                            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                            "response_time": response_time,
                            "response_length": len(response.content),
                            "evaluation_results": evaluation_results
                        }
                        
                        agent_results["test_results"].append(test_result)
                        
                    except Exception as e:
                        error_msg = f"Error testing prompt {j}: {e}"
                        agent_results["errors"].append(error_msg)
                        logger.error(error_msg)
                
                # Calculate performance metrics
                if agent_results["test_results"]:
                    response_times = [r["response_time"] for r in agent_results["test_results"]]
                    agent_results["performance_metrics"] = {
                        "avg_response_time": sum(response_times) / len(response_times),
                        "min_response_time": min(response_times),
                        "max_response_time": max(response_times),
                        "total_tests": len(agent_results["test_results"]),
                        "success_rate": len(agent_results["test_results"]) / len(test_prompts)
                    }
                
                comparison_results[agent.name] = agent_results
            
            # Generate comparison summary
            summary = self._generate_agent_comparison_summary(comparison_results)
            
            return {
                "comparison_results": comparison_results,
                "summary": summary,
                "test_prompts_count": len(test_prompts),
                "agents_tested": len(agents),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error running agent comparison benchmark: {e}")
            return {"error": str(e)}
    
    async def run_stress_test(
        self,
        agent: BaseAgent,
        concurrent_requests: int = 10,
        total_requests: int = 100,
        request_delay: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run stress tests on an agent.
        
        Args:
            agent: Agent to stress test
            concurrent_requests: Number of concurrent requests
            total_requests: Total number of requests
            request_delay: Delay between requests
            
        Returns:
            Stress test results
        """
        try:
            logger.info(f"Starting stress test: {concurrent_requests} concurrent, {total_requests} total")
            
            test_prompts = [
                "What is the current time?",
                "Explain artificial intelligence briefly.",
                "How are you doing today?",
                "What can you help me with?",
                "Tell me a short joke."
            ]
            
            results = {
                "test_parameters": {
                    "concurrent_requests": concurrent_requests,
                    "total_requests": total_requests,
                    "request_delay": request_delay
                },
                "response_times": [],
                "successful_requests": 0,
                "failed_requests": 0,
                "errors": [],
                "start_time": time.time()
            }
            
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def make_request(request_id: int) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        prompt = test_prompts[request_id % len(test_prompts)]
                        session_id = f"stress_test_{request_id}"
                        
                        start_time = time.time()
                        response = await agent.process_message(prompt, session_id)
                        end_time = time.time()
                        
                        return {
                            "success": True,
                            "response_time": end_time - start_time,
                            "response_length": len(response.content),
                            "request_id": request_id
                        }
                        
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "request_id": request_id
                        }
                    
                    await asyncio.sleep(request_delay)
            
            # Execute requests
            tasks = [make_request(i) for i in range(total_requests)]
            request_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in request_results:
                if isinstance(result, Exception):
                    results["failed_requests"] += 1
                    results["errors"].append(str(result))
                elif result.get("success"):
                    results["successful_requests"] += 1
                    results["response_times"].append(result["response_time"])
                else:
                    results["failed_requests"] += 1
                    results["errors"].append(result.get("error", "Unknown error"))
            
            # Calculate statistics
            results["end_time"] = time.time()
            results["total_duration"] = results["end_time"] - results["start_time"]
            results["success_rate"] = results["successful_requests"] / total_requests
            
            if results["response_times"]:
                results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
                results["min_response_time"] = min(results["response_times"])
                results["max_response_time"] = max(results["response_times"])
                results["requests_per_second"] = results["successful_requests"] / results["total_duration"]
            
            return results
            
        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            return {"error": str(e)}
    
    async def _initialize_agents(self, agents: List[BaseAgent]) -> Dict[str, Any]:
        """Initialize all agents for benchmarking."""
        try:
            initialization_results = {
                "total_agents": len(agents),
                "successful_initializations": 0,
                "failed_initializations": 0,
                "initialization_times": [],
                "errors": []
            }
            
            for agent in agents:
                try:
                    start_time = time.time()
                    await agent.initialize()
                    end_time = time.time()
                    
                    initialization_results["successful_initializations"] += 1
                    initialization_results["initialization_times"].append(end_time - start_time)
                    
                except Exception as e:
                    initialization_results["failed_initializations"] += 1
                    error_msg = f"Failed to initialize {agent.name}: {e}"
                    initialization_results["errors"].append(error_msg)
                    logger.error(error_msg)
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            return {"error": str(e)}
    
    async def _run_core_benchmarks(
        self,
        agents: List[BaseAgent],
        caim_framework: Optional[CAIMFramework]
    ) -> Dict[str, Any]:
        """Run core functionality benchmarks."""
        try:
            core_results = {}
            
            # Test basic agent functionality
            for agent in agents:
                try:
                    agent_tests = await self._run_basic_agent_tests(agent)
                    core_results[f"{agent.name}_basic_tests"] = agent_tests
                    
                except Exception as e:
                    logger.error(f"Error in basic tests for {agent.name}: {e}")
                    core_results[f"{agent.name}_basic_tests"] = {"error": str(e)}
            
            # Test CAIM framework functionality if available
            if caim_framework:
                try:
                    caim_tests = await self._run_caim_framework_tests(caim_framework)
                    core_results["caim_framework_tests"] = caim_tests
                    
                except Exception as e:
                    logger.error(f"Error in CAIM framework tests: {e}")
                    core_results["caim_framework_tests"] = {"error": str(e)}
            
            return core_results
            
        except Exception as e:
            logger.error(f"Error running core benchmarks: {e}")
            return {"error": str(e)}
    
    async def _run_memory_benchmarks(
        self,
        caim_framework: Optional[CAIMFramework]
    ) -> Dict[str, Any]:
        """Run memory system benchmarks."""
        try:
            if not caim_framework:
                return {"skipped": "No CAIM framework provided"}
            
            return await self.memory_benchmark.run_comprehensive_memory_tests(
                caim_framework, self._get_default_memory_scenarios()
            )
            
        except Exception as e:
            logger.error(f"Error running memory benchmarks: {e}")
            return {"error": str(e)}
    
    async def _run_model_comparison_benchmarks(self, agents: List[BaseAgent]) -> Dict[str, Any]:
        """Run model comparison benchmarks."""
        try:
            return await self.model_comparison.compare_models(
                [agent.model for agent in agents if agent.model]
            )
            
        except Exception as e:
            logger.error(f"Error running model comparison benchmarks: {e}")
            return {"error": str(e)}
    
    async def _run_custom_benchmarks(
        self,
        benchmark_suites: List[BenchmarkSuite],
        agents: List[BaseAgent]
    ) -> Dict[str, Any]:
        """Run custom benchmark suites."""
        try:
            custom_results = {}
            
            for suite in benchmark_suites:
                try:
                    suite_results = await self._run_benchmark_suite(suite, agents)
                    custom_results[suite.name] = suite_results
                    
                except Exception as e:
                    logger.error(f"Error running custom benchmark suite {suite.name}: {e}")
                    custom_results[suite.name] = {"error": str(e)}
            
            return custom_results
            
        except Exception as e:
            logger.error(f"Error running custom benchmarks: {e}")
            return {"error": str(e)}
    
    async def _run_performance_benchmarks(
        self,
        agents: List[BaseAgent],
        caim_framework: Optional[CAIMFramework]
    ) -> Dict[str, Any]:
        """Run performance and efficiency benchmarks."""
        try:
            performance_results = {}
            
            # Test response times for each agent
            for agent in agents:
                try:
                    response_times = []
                    prompts = self._get_default_test_prompts()[:5]  # Use subset for performance test
                    
                    for prompt in prompts:
                        start_time = time.time()
                        await agent.process_message(prompt, f"perf_test_{int(time.time())}")
                        end_time = time.time()
                        response_times.append(end_time - start_time)
                    
                    performance_results[f"{agent.name}_response_times"] = {
                        "times": response_times,
                        "average": sum(response_times) / len(response_times),
                        "min": min(response_times),
                        "max": max(response_times)
                    }
                    
                except Exception as e:
                    logger.error(f"Error in performance test for {agent.name}: {e}")
                    performance_results[f"{agent.name}_response_times"] = {"error": str(e)}
            
            return performance_results
            
        except Exception as e:
            logger.error(f"Error running performance benchmarks: {e}")
            return {"error": str(e)}
    
    async def _run_basic_agent_tests(self, agent: BaseAgent) -> Dict[str, Any]:
        """Run basic functionality tests for an agent."""
        try:
            test_results = {
                "health_check": await agent.health_check(),
                "statistics": await agent.get_agent_statistics(),
                "basic_response_test": None
            }
            
            # Test basic response
            try:
                response = await agent.process_message(
                    "Hello, this is a test message.",
                    f"basic_test_{int(time.time())}"
                )
                test_results["basic_response_test"] = {
                    "success": True,
                    "response_length": len(response.content),
                    "has_content": bool(response.content.strip())
                }
                
            except Exception as e:
                test_results["basic_response_test"] = {
                    "success": False,
                    "error": str(e)
                }
            
            return test_results
            
        except Exception as e:
            logger.error(f"Error running basic agent tests: {e}")
            return {"error": str(e)}
    
    async def _run_caim_framework_tests(self, caim_framework: CAIMFramework) -> Dict[str, Any]:
        """Run CAIM framework specific tests."""
        try:
            if not caim_framework.is_initialized:
                await caim_framework.initialize()
            
            test_session_id = f"caim_test_{int(time.time())}"
            
            # Test input processing
            process_result = await caim_framework.process_input(
                user_input="Test input for CAIM framework",
                session_id=test_session_id
            )
            
            # Test memory insights
            insights = await caim_framework.get_memory_insights(test_session_id)
            
            return {
                "framework_initialized": caim_framework.is_initialized,
                "process_input_test": {
                    "success": bool(process_result),
                    "has_memories": len(process_result.get("relevant_ltm_memories", [])) > 0
                },
                "memory_insights_test": {
                    "success": bool(insights),
                    "has_statistics": "short_term_memory" in insights
                }
            }
            
        except Exception as e:
            logger.error(f"Error running CAIM framework tests: {e}")
            return {"error": str(e)}
    
    async def _run_benchmark_suite(self, suite: BenchmarkSuite, agents: List[BaseAgent]) -> Dict[str, Any]:
        """Run a custom benchmark suite."""
        try:
            suite_results = {
                "suite_name": suite.name,
                "description": suite.description,
                "test_cases_count": len(suite.test_cases),
                "results": []
            }
            
            for i, test_case in enumerate(suite.test_cases):
                test_result = {
                    "test_case_index": i,
                    "test_case": test_case,
                    "agent_results": {}
                }
                
                # Run test case on each agent
                for agent in agents:
                    try:
                        # Extract test parameters
                        prompt = test_case.get("prompt", "")
                        expected_keywords = test_case.get("expected_keywords", [])
                        
                        # Run test
                        start_time = time.time()
                        response = await agent.process_message(
                            prompt,
                            f"suite_{suite.name}_{i}_{agent.name}"
                        )
                        end_time = time.time()
                        
                        # Evaluate response
                        keyword_matches = sum(
                            1 for keyword in expected_keywords
                            if keyword.lower() in response.content.lower()
                        )
                        
                        test_result["agent_results"][agent.name] = {
                            "response_time": end_time - start_time,
                            "response_length": len(response.content),
                            "keyword_matches": keyword_matches,
                            "expected_keywords": len(expected_keywords),
                            "keyword_score": keyword_matches / len(expected_keywords) if expected_keywords else 0.0
                        }
                        
                    except Exception as e:
                        test_result["agent_results"][agent.name] = {
                            "error": str(e)
                        }
                
                suite_results["results"].append(test_result)
            
            return suite_results
            
        except Exception as e:
            logger.error(f"Error running benchmark suite {suite.name}: {e}")
            return {"error": str(e)}
    
    async def _evaluate_agent_response(
        self,
        response,
        prompt: str,
        criteria: List[str]
    ) -> Dict[str, Any]:
        """Evaluate an agent response against criteria."""
        try:
            evaluation = {}
            
            for criterion in criteria:
                if criterion == "response_quality":
                    # Simple quality metric based on length and completeness
                    quality_score = min(1.0, len(response.content) / 100.0)  # Normalize to 100 chars
                    evaluation[criterion] = quality_score
                    
                elif criterion == "coherence":
                    # Simple coherence check
                    coherence_result = self.evaluation_metrics.evaluate_response_coherence(
                        response.content, "", []
                    )
                    evaluation[criterion] = coherence_result.score
                    
                elif criterion == "efficiency":
                    # Use response metadata if available
                    if hasattr(response, 'model_response') and response.model_response:
                        usage_stats = response.model_response.usage_stats or {}
                        total_tokens = usage_stats.get("total_tokens", 0)
                        efficiency_score = max(0.0, 1.0 - (total_tokens / 1000.0))  # Normalize to 1000 tokens
                        evaluation[criterion] = efficiency_score
                    else:
                        evaluation[criterion] = 0.5  # Neutral score
                        
                elif criterion == "memory_usage":
                    # Check if memories were used
                    relevant_memories = getattr(response, 'relevant_memories', [])
                    memory_score = 1.0 if relevant_memories else 0.0
                    evaluation[criterion] = memory_score
                    
                else:
                    evaluation[criterion] = 0.0  # Unknown criterion
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating agent response: {e}")
            return {criterion: 0.0 for criterion in criteria}
    
    def _generate_agent_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of agent comparison results."""
        try:
            summary = {
                "agents_compared": len(comparison_results),
                "performance_rankings": {},
                "best_performers": {},
                "overall_winner": None
            }
            
            # Calculate average scores for each agent
            agent_scores = {}
            for agent_name, results in comparison_results.items():
                if results.get("test_results"):
                    avg_response_time = results["performance_metrics"].get("avg_response_time", float('inf'))
                    success_rate = results["performance_metrics"].get("success_rate", 0.0)
                    
                    # Simple scoring: fast response time + high success rate
                    score = success_rate - (avg_response_time * 0.1)  # Penalize slow responses
                    agent_scores[agent_name] = score
            
            # Rank agents
            if agent_scores:
                ranked_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
                summary["performance_rankings"] = {
                    f"rank_{i+1}": {"agent": agent, "score": score}
                    for i, (agent, score) in enumerate(ranked_agents)
                }
                summary["overall_winner"] = ranked_agents[0][0]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating agent comparison summary: {e}")
            return {"error": str(e)}
    
    def _get_default_test_prompts(self) -> List[str]:
        """Get default test prompts for benchmarking."""
        return [
            "What is artificial intelligence?",
            "Explain the concept of machine learning in simple terms.",
            "How do neural networks work?",
            "What are the benefits and risks of AI?",
            "Describe the difference between supervised and unsupervised learning.",
            "What is natural language processing?",
            "How can AI help in healthcare?",
            "What are the ethical considerations in AI development?",
            "Explain computer vision and its applications.",
            "What is the future of artificial intelligence?"
        ]
    
    def _get_default_memory_scenarios(self) -> List[Dict[str, Any]]:
        """Get default memory test scenarios."""
        return [
            {
                "name": "basic_memory_storage",
                "description": "Test basic memory storage and retrieval",
                "memories": [
                    {"content": "User likes pizza", "importance": "medium"},
                    {"content": "User is a software developer", "importance": "high"},
                    {"content": "User prefers morning meetings", "importance": "low"}
                ],
                "queries": [
                    "What food does the user like?",
                    "What is the user's profession?",
                    "When does the user prefer meetings?"
                ]
            },
            {
                "name": "memory_consolidation",
                "description": "Test memory consolidation capabilities",
                "memories": [
                    {"content": "User worked on Python project", "importance": "medium"},
                    {"content": "User debugged Python code", "importance": "medium"},
                    {"content": "User learned Python functions", "importance": "medium"}
                ],
                "queries": [
                    "What programming activities has the user done?"
                ]
            }
        ]
    
    def get_benchmark_history(self) -> List[Dict[str, Any]]:
        """Get the history of benchmark runs."""
        return self.benchmark_history
    
    def clear_benchmark_history(self) -> None:
        """Clear benchmark history."""
        self.benchmark_history.clear()
        self.benchmark_results.clear()
        self.evaluation_metrics.clear_history()
        logger.info("Cleared benchmark history")