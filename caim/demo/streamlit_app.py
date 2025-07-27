"""Streamlit demo application for CAIM framework."""

import logging
import asyncio
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

# Import CAIM components
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from caim.core.caim_framework import CAIMFramework
from caim.core.config import CAIMConfig
from caim.agents.caim_agent import CAIMAgent
from caim.models.model_factory import create_model
from caim.benchmarks.benchmark_runner import BenchmarkRunner


class StreamlitDemo:
    """Streamlit web application demo for CAIM framework."""
    
    def __init__(self):
        self.setup_logging()
        self.initialize_session_state()
    
    def setup_logging(self):
        """Setup logging for the demo."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'caim_framework' not in st.session_state:
            st.session_state.caim_framework = None
        if 'agent' not in st.session_state:
            st.session_state.agent = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"streamlit_session_{int(time.time())}"
        if 'benchmark_results' not in st.session_state:
            st.session_state.benchmark_results = None
        if 'cleanup_registered' not in st.session_state:
            st.session_state.cleanup_registered = False
            # Register cleanup function
            import atexit
            atexit.register(self.cleanup_on_exit)
    
    def cleanup_on_exit(self):
        """Cleanup function called on exit."""
        try:
            if hasattr(st.session_state, 'caim_framework') and st.session_state.caim_framework:
                # Run cleanup in a new event loop since we're in exit handler
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule cleanup for later if loop is running
                        asyncio.create_task(st.session_state.caim_framework.shutdown())
                    else:
                        # Run cleanup directly if no running loop
                        loop.run_until_complete(st.session_state.caim_framework.shutdown())
                except RuntimeError:
                    # No event loop, create new one
                    asyncio.run(st.session_state.caim_framework.shutdown())
            
            if hasattr(st.session_state, 'agent') and st.session_state.agent:
                try:
                    import asyncio
                    asyncio.run(st.session_state.agent.shutdown())
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def run(self):
        """Run the Streamlit demo application."""
        st.set_page_config(
            page_title="CAIM Framework Demo",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 CAIM: Cognitive AI Memory Framework")
        st.markdown("### Enhanced Long-Term Agent Interactions")
        
        # Sidebar for configuration
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "🧠 Memory Insights", "📊 Benchmarks", "⚙️ System Status"])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_memory_insights()
        
        with tab3:
            self.render_benchmarks()
        
        with tab4:
            self.render_system_status()
    
    def render_sidebar(self):
        """Render the sidebar configuration panel."""
        st.sidebar.header("🔧 Configuration")
        
        # Model selection
        model_options = [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct", 
            "google/gemma-2b-it",
            "gpt-4o-mini"
        ]
        
        selected_model = st.sidebar.selectbox(
            "Select Model",
            model_options,
            index=0,
            help="Choose the AI model for the agent"
        )
        
        # Agent configuration
        st.sidebar.subheader("Agent Settings")
        auto_consolidate = st.sidebar.checkbox(
            "Auto Memory Consolidation",
            value=True,
            help="Automatically consolidate memories during conversations"
        )
        
        consolidation_interval = st.sidebar.slider(
            "Consolidation Interval",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of interactions before consolidation"
        )
        
        max_context_memories = st.sidebar.slider(
            "Max Context Memories",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum memories to use in context"
        )
        
        # Initialize/Reinitialize button
        if st.sidebar.button("🚀 Initialize CAIM System", type="primary"):
            with st.sidebar:
                with st.spinner("Initializing CAIM system..."):
                    success = self.initialize_caim_system(
                        selected_model,
                        auto_consolidate,
                        consolidation_interval,
                        max_context_memories
                    )
                    
                    if success:
                        st.success("✅ CAIM system initialized!")
                    else:
                        st.error("❌ Failed to initialize CAIM system")
        
        # System info
        st.sidebar.subheader("📊 System Info")
        if st.session_state.caim_framework:
            st.sidebar.info(f"Framework: {'✅ Active' if st.session_state.caim_framework.is_initialized else '❌ Not Active'}")
        if st.session_state.agent:
            st.sidebar.info(f"Agent: {'✅ Ready' if st.session_state.agent.is_initialized else '❌ Not Ready'}")
            st.sidebar.info(f"Model: {selected_model}")
        
        # Clear conversation button
        if st.sidebar.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.session_id = f"streamlit_session_{int(time.time())}"
            st.rerun()
    
    def initialize_caim_system(
        self,
        model_name: str,
        auto_consolidate: bool,
        consolidation_interval: int,
        max_context_memories: int
    ) -> bool:
        """Initialize the CAIM system with given configuration."""
        try:
            # Create configuration
            config = CAIMConfig.from_env()
            
            # Initialize CAIM framework
            caim_framework = CAIMFramework(config)
            
            # Create model
            model = create_model(model_name, config)
            
            # Create agent
            agent = CAIMAgent(
                name="StreamlitAgent",
                config=config,
                model=model,
                caim_framework=caim_framework
            )
            
            # Configure agent
            agent.auto_consolidate = auto_consolidate
            agent.consolidation_interval = consolidation_interval
            agent.max_context_memories = max_context_memories
            
            # Initialize asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(caim_framework.initialize())
                loop.run_until_complete(agent.initialize())
                
                # Store in session state
                st.session_state.caim_framework = caim_framework
                st.session_state.agent = agent
                
                return True
                
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error initializing CAIM system: {e}")
            st.error(f"Initialization error: {e}")
            return False
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        st.header("💬 Chat with CAIM Agent")
        
        if not st.session_state.agent:
            st.warning("Please initialize the CAIM system using the sidebar configuration.")
            return
        
        # Display conversation history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.conversation_history):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        
                        # Show memory info if available
                        if message.get("relevant_memories"):
                            with st.expander(f"🧠 Memories Used ({len(message['relevant_memories'])})"):
                                for j, memory in enumerate(message["relevant_memories"]):
                                    st.write(f"**Memory {j+1}:** {memory.get('content', '')[:100]}...")
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                response = self.generate_agent_response(user_input)
            
            if response:
                # Add assistant response to history
                assistant_message = {
                    "role": "assistant",
                    "content": response.content,
                    "timestamp": datetime.now().isoformat(),
                    "relevant_memories": response.relevant_memories or [],
                    "confidence_score": response.confidence_score
                }
                
                st.session_state.conversation_history.append(assistant_message)
            
            st.rerun()
    
    def generate_agent_response(self, user_input: str):
        """Generate agent response asynchronously."""
        try:
            if not st.session_state.agent:
                return None
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    st.session_state.agent.process_message(
                        user_input,
                        st.session_state.session_id
                    )
                )
                return response
                
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            st.error(f"Response generation error: {e}")
            return None
    
    def render_memory_insights(self):
        """Render memory insights and analytics."""
        st.header("🧠 Memory System Insights")
        
        if not st.session_state.caim_framework:
            st.warning("Please initialize the CAIM system to view memory insights.")
            return
        
        # Get memory insights
        insights = self.get_memory_insights()
        
        if not insights:
            st.info("No memory insights available yet. Start a conversation to generate insights.")
            return
        
        # Memory overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stm_count = insights.get("short_term_memory", {}).get("total_memories", 0)
            st.metric("Short-Term Memories", stm_count)
        
        with col2:
            ltm_count = insights.get("long_term_memory", {}).get("total_memories", 0)
            st.metric("Long-Term Memories", ltm_count)
        
        with col3:
            inductive_thoughts = insights.get("long_term_memory", {}).get("inductive_thoughts", 0)
            st.metric("Inductive Thoughts", inductive_thoughts)
        
        with col4:
            active_sessions = insights.get("active_sessions", 0)
            st.metric("Active Sessions", active_sessions)
        
        # Memory type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Memory Type Distribution")
            ltm_types = insights.get("long_term_memory", {}).get("type_distribution", {})
            if ltm_types:
                fig = px.pie(
                    values=list(ltm_types.values()),
                    names=list(ltm_types.keys()),
                    title="Long-Term Memory Types"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Memory Importance Distribution")
            importance_dist = insights.get("long_term_memory", {}).get("importance_distribution", {})
            if importance_dist:
                fig = px.bar(
                    x=list(importance_dist.keys()),
                    y=list(importance_dist.values()),
                    title="Memory Importance Levels"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Session-specific insights
        if st.session_state.session_id:
            st.subheader(f"Current Session Insights")
            session_insights = insights.get("session_insights", {})
            
            if session_insights:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_memories = session_insights.get("total_memories", 0)
                    st.metric("Session Memories", total_memories)
                
                with col2:
                    interactions = session_insights.get("total_interactions", 0)
                    st.metric("Interactions", interactions)
                
                with col3:
                    avg_importance = session_insights.get("average_importance", 0)
                    st.metric("Avg Importance", f"{avg_importance:.2f}")
        
        # Memory timeline
        self.render_memory_timeline(insights)
    
    def render_memory_timeline(self, insights: Dict[str, Any]):
        """Render memory timeline visualization."""
        st.subheader("Memory Timeline")
        
        # This would need actual memory data with timestamps
        # For demo purposes, create sample data
        sample_data = {
            "Date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
            "Memories Created": [5, 8, 12, 6, 9, 15, 11, 7, 13, 10],
            "Memories Consolidated": [1, 2, 3, 1, 2, 4, 3, 2, 3, 2]
        }
        
        df = pd.DataFrame(sample_data)
        
        fig = px.line(
            df,
            x="Date",
            y=["Memories Created", "Memories Consolidated"],
            title="Memory Activity Over Time"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def get_memory_insights(self):
        """Get memory insights from the CAIM framework."""
        try:
            if not st.session_state.caim_framework:
                return None
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                insights = loop.run_until_complete(
                    st.session_state.caim_framework.get_memory_insights(
                        st.session_state.session_id
                    )
                )
                return insights
                
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error getting memory insights: {e}")
            return None
    
    def render_benchmarks(self):
        """Render benchmarking interface and results."""
        st.header("📊 Benchmarks & Performance")
        
        if not st.session_state.agent:
            st.warning("Please initialize the CAIM system to run benchmarks.")
            return
        
        # Benchmark controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏃 Run Quick Benchmark", type="primary"):
                with st.spinner("Running benchmark..."):
                    results = self.run_quick_benchmark()
                    st.session_state.benchmark_results = results
        
        with col2:
            if st.button("🔍 Memory Benchmark"):
                with st.spinner("Running memory benchmark..."):
                    results = self.run_memory_benchmark()
                    st.session_state.benchmark_results = results
        
        with col3:
            if st.button("💨 Stress Test"):
                with st.spinner("Running stress test..."):
                    results = self.run_stress_test()
                    st.session_state.benchmark_results = results
        
        # Display benchmark results
        if st.session_state.benchmark_results:
            self.display_benchmark_results(st.session_state.benchmark_results)
    
    def run_quick_benchmark(self):
        """Run a quick benchmark test."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                runner = BenchmarkRunner()
                
                # Simple response time test
                test_prompts = [
                    "Hello, how are you?",
                    "What is AI?",
                    "Explain machine learning."
                ]
                
                results = {
                    "test_type": "quick_benchmark",
                    "timestamp": datetime.now().isoformat(),
                    "response_times": [],
                    "total_time": 0
                }
                
                start_time = time.time()
                
                for prompt in test_prompts:
                    prompt_start = time.time()
                    response = loop.run_until_complete(
                        st.session_state.agent.process_message(
                            prompt,
                            f"benchmark_{int(prompt_start)}"
                        )
                    )
                    prompt_end = time.time()
                    
                    results["response_times"].append(prompt_end - prompt_start)
                
                results["total_time"] = time.time() - start_time
                results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
                
                return results
                
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error running quick benchmark: {e}")
            return {"error": str(e)}
    
    def run_memory_benchmark(self):
        """Run memory-specific benchmark."""
        try:
            if not st.session_state.caim_framework:
                return {"error": "CAIM framework not initialized"}
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                runner = BenchmarkRunner()
                
                results = loop.run_until_complete(
                    runner.run_memory_benchmark_suite(st.session_state.caim_framework)
                )
                
                results["test_type"] = "memory_benchmark"
                results["timestamp"] = datetime.now().isoformat()
                
                return results
                
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error running memory benchmark: {e}")
            return {"error": str(e)}
    
    def run_stress_test(self):
        """Run stress test on the agent."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                runner = BenchmarkRunner()
                
                results = loop.run_until_complete(
                    runner.run_stress_test(
                        st.session_state.agent,
                        concurrent_requests=5,
                        total_requests=20
                    )
                )
                
                results["test_type"] = "stress_test"
                results["timestamp"] = datetime.now().isoformat()
                
                return results
                
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
            return {"error": str(e)}
    
    def display_benchmark_results(self, results: Dict[str, Any]):
        """Display benchmark results with visualizations."""
        st.subheader(f"📊 {results.get('test_type', 'Benchmark').replace('_', ' ').title()} Results")
        
        if "error" in results:
            st.error(f"Benchmark failed: {results['error']}")
            return
        
        # Display metrics based on test type
        if results.get("test_type") == "quick_benchmark":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Response Time", f"{results.get('avg_response_time', 0):.2f}s")
            
            with col2:
                st.metric("Total Test Time", f"{results.get('total_time', 0):.2f}s")
            
            with col3:
                st.metric("Tests Completed", len(results.get('response_times', [])))
            
            # Response time chart
            if results.get("response_times"):
                fig = px.bar(
                    x=[f"Test {i+1}" for i in range(len(results["response_times"]))],
                    y=results["response_times"],
                    title="Response Times by Test"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif results.get("test_type") == "stress_test":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Success Rate", f"{results.get('success_rate', 0)*100:.1f}%")
            
            with col2:
                st.metric("Successful Requests", results.get('successful_requests', 0))
            
            with col3:
                st.metric("Failed Requests", results.get('failed_requests', 0))
            
            with col4:
                st.metric("Requests/Second", f"{results.get('requests_per_second', 0):.2f}")
            
            # Performance metrics
            if results.get("response_times"):
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        x=results["response_times"],
                        title="Response Time Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(
                        x=list(range(len(results["response_times"]))),
                        y=results["response_times"],
                        title="Response Times Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show raw results in expandable section
        with st.expander("📄 Raw Results"):
            st.json(results)
    
    def render_system_status(self):
        """Render system status and health information."""
        st.header("⚙️ System Status")
        
        # Health checks
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Health Checks")
            
            if st.session_state.caim_framework:
                framework_status = "✅ Active" if st.session_state.caim_framework.is_initialized else "❌ Inactive"
                st.info(f"CAIM Framework: {framework_status}")
            else:
                st.warning("CAIM Framework: ❌ Not Initialized")
            
            if st.session_state.agent:
                agent_status = "✅ Ready" if st.session_state.agent.is_initialized else "❌ Not Ready"
                st.info(f"Agent: {agent_status}")
                
                # Agent statistics
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        stats = loop.run_until_complete(st.session_state.agent.get_agent_statistics())
                        
                        st.metric("Total Sessions", stats.get("total_sessions", 0))
                        st.metric("Active Sessions", stats.get("active_sessions", 0))
                        st.metric("Total Interactions", stats.get("total_interactions", 0))
                        
                    finally:
                        loop.close()
                        
                except Exception as e:
                    st.error(f"Error getting agent statistics: {e}")
            else:
                st.warning("Agent: ❌ Not Initialized")
        
        with col2:
            st.subheader("📊 Performance Metrics")
            
            # Current session info
            st.info(f"Session ID: {st.session_state.session_id}")
            st.info(f"Conversation Length: {len(st.session_state.conversation_history)} messages")
            
            # Memory usage info
            if st.session_state.caim_framework:
                try:
                    insights = self.get_memory_insights()
                    if insights:
                        st.metric("Framework Uptime", insights.get("framework_uptime", "N/A"))
                        
                        # System statistics
                        stm_stats = insights.get("short_term_memory", {})
                        ltm_stats = insights.get("long_term_memory", {})
                        
                        st.write("**Memory Statistics:**")
                        st.write(f"- STM Memories: {stm_stats.get('total_memories', 0)}")
                        st.write(f"- LTM Memories: {ltm_stats.get('total_memories', 0)}")
                        st.write(f"- Consolidation Ratio: {ltm_stats.get('consolidation_ratio', 0):.3f}")
                        
                except Exception as e:
                    st.error(f"Error getting performance metrics: {e}")
        
        # Advanced controls
        st.subheader("🛠️ Advanced Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Force Memory Consolidation"):
                if st.session_state.agent and hasattr(st.session_state.agent, 'force_consolidation'):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            result = loop.run_until_complete(
                                st.session_state.agent.force_consolidation(st.session_state.session_id)
                            )
                            st.success(f"Consolidation completed: {result}")
                            
                        finally:
                            loop.close()
                            
                    except Exception as e:
                        st.error(f"Consolidation failed: {e}")
                else:
                    st.warning("Agent not available for consolidation")
        
        with col2:
            if st.button("🧹 Cleanup Expired Data"):
                if st.session_state.caim_framework:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            result = loop.run_until_complete(
                                st.session_state.caim_framework.cleanup_expired_data()
                            )
                            st.success(f"Cleanup completed: {result}")
                            
                        finally:
                            loop.close()
                            
                    except Exception as e:
                        st.error(f"Cleanup failed: {e}")
                else:
                    st.warning("CAIM framework not available")
        
        with col3:
            if st.button("📊 Export Session Data"):
                if st.session_state.conversation_history:
                    # Create export data
                    export_data = {
                        "session_id": st.session_state.session_id,
                        "export_timestamp": datetime.now().isoformat(),
                        "conversation_history": st.session_state.conversation_history,
                        "message_count": len(st.session_state.conversation_history)
                    }
                    
                    st.download_button(
                        label="⬇️ Download Session Data",
                        data=str(export_data),
                        file_name=f"caim_session_{st.session_state.session_id}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No conversation data to export")


def main():
    """Main function to run the Streamlit demo."""
    demo = StreamlitDemo()
    demo.run()


if __name__ == "__main__":
    main()