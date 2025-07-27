"""Basic usage example for CAIM framework."""

import asyncio
import logging
from pathlib import Path
import sys

# Add the parent directory to the path so we can import caim
sys.path.insert(0, str(Path(__file__).parent.parent))

from caim.core.caim_framework import CAIMFramework
from caim.core.config import CAIMConfig
from caim.agents.caim_agent import CAIMAgent
from caim.models.model_factory import create_model
from caim.memory.memory_types import MemoryImportance


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_usage_example():
    """Demonstrate basic CAIM framework usage."""
    
    print("🧠 CAIM Framework - Basic Usage Example")
    print("=" * 50)
    
    try:
        # 1. Create configuration
        print("📋 Step 1: Creating configuration...")
        config = CAIMConfig.from_env()
        
        # You can also configure manually:
        # config = CAIMConfig(
        #     max_memory_size=5000,
        #     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        #     retrieval_top_k=3
        # )
        
        print(f"   Max memory size: {config.max_memory_size}")
        print(f"   Embedding model: {config.embedding_model}")
        print(f"   Retrieval top-k: {config.retrieval_top_k}")
        
        # 2. Initialize CAIM framework
        print("\n🚀 Step 2: Initializing CAIM framework...")
        caim_framework = CAIMFramework(config)
        await caim_framework.initialize()
        print("   ✅ CAIM framework initialized")
        
        # 3. Create a model (using simple embedding for demo)
        print("\n🤖 Step 3: Creating AI model...")
        try:
            # Try to use a simple model that doesn't require external dependencies
            from caim.retrieval.embedding_generator import SimpleEmbedding
            
            # Create a mock model for demonstration
            class MockModel:
                def __init__(self):
                    self.model_name = "MockModel"
                    self.is_initialized = False
                
                async def initialize(self):
                    self.is_initialized = True
                
                async def generate_response(self, prompt, **kwargs):
                    from caim.models.base_model import ModelResponse
                    from datetime import datetime
                    
                    # Simple mock response
                    response_text = f"This is a mock response to: {prompt[:50]}..."
                    
                    return ModelResponse(
                        content=response_text,
                        model_name=self.model_name,
                        timestamp=datetime.utcnow(),
                        metadata={"mock": True}
                    )
                
                async def generate_streaming_response(self, prompt, **kwargs):
                    response = await self.generate_response(prompt, **kwargs)
                    for word in response.content.split():
                        yield word + " "
                
                async def health_check(self):
                    return {"status": "healthy", "model_name": self.model_name}
                
                async def shutdown(self):
                    self.is_initialized = False
            
            model = MockModel()
            await model.initialize()
            print("   ✅ Mock model created and initialized")
            
        except Exception as e:
            print(f"   ⚠️  Could not create advanced model: {e}")
            print("   Using mock model for demonstration")
        
        # 4. Create CAIM agent
        print("\n🤝 Step 4: Creating CAIM agent...")
        agent = CAIMAgent(
            name="ExampleAgent",
            config=config,
            model=model,
            caim_framework=caim_framework
        )
        await agent.initialize()
        print("   ✅ CAIM agent created and initialized")
        
        # 5. Simulate a conversation
        print("\n💬 Step 5: Simulating conversation...")
        session_id = "example_session_001"
        
        # First interaction
        user_message_1 = "Hello! My name is Alice and I love programming in Python."
        print(f"\n👤 User: {user_message_1}")
        
        response_1 = await agent.process_message(user_message_1, session_id)
        print(f"🤖 Agent: {response_1.content}")
        print(f"   Memories used: {len(response_1.relevant_memories or [])}")
        
        # Second interaction
        user_message_2 = "What programming languages do I enjoy?"
        print(f"\n👤 User: {user_message_2}")
        
        response_2 = await agent.process_message(user_message_2, session_id)
        print(f"🤖 Agent: {response_2.content}")
        print(f"   Memories used: {len(response_2.relevant_memories or [])}")
        
        # Third interaction
        user_message_3 = "Can you remind me what my name is?"
        print(f"\n👤 User: {user_message_3}")
        
        response_3 = await agent.process_message(user_message_3, session_id)
        print(f"🤖 Agent: {response_3.content}")
        print(f"   Memories used: {len(response_3.relevant_memories or [])}")
        
        # 6. Check memory insights
        print("\n🧠 Step 6: Checking memory insights...")
        insights = await caim_framework.get_memory_insights(session_id)
        
        stm_stats = insights.get("short_term_memory", {})
        ltm_stats = insights.get("long_term_memory", {})
        
        print(f"   Short-term memories: {stm_stats.get('total_memories', 0)}")
        print(f"   Long-term memories: {ltm_stats.get('total_memories', 0)}")
        print(f"   Active sessions: {insights.get('active_sessions', 0)}")
        
        # 7. Force memory consolidation
        print("\n🔄 Step 7: Forcing memory consolidation...")
        consolidation_result = await caim_framework.consolidate_session_memories(session_id)
        print(f"   Promoted memories: {consolidation_result.get('promoted_memories', 0)}")
        print(f"   Inductive thoughts: {consolidation_result.get('inductive_thoughts', 0)}")
        
        # 8. Check updated memory insights
        print("\n📊 Step 8: Updated memory insights...")
        updated_insights = await caim_framework.get_memory_insights(session_id)
        
        updated_stm = updated_insights.get("short_term_memory", {})
        updated_ltm = updated_insights.get("long_term_memory", {})
        
        print(f"   Short-term memories: {updated_stm.get('total_memories', 0)}")
        print(f"   Long-term memories: {updated_ltm.get('total_memories', 0)}")
        print(f"   Inductive thoughts: {updated_ltm.get('inductive_thoughts', 0)}")
        
        # 9. Test memory retrieval after consolidation
        print("\n🔍 Step 9: Testing memory retrieval after consolidation...")
        user_message_4 = "What do you remember about me?"
        print(f"\n👤 User: {user_message_4}")
        
        response_4 = await agent.process_message(user_message_4, session_id)
        print(f"🤖 Agent: {response_4.content}")
        print(f"   Memories used: {len(response_4.relevant_memories or [])}")
        
        # 10. Cleanup
        print("\n🧹 Step 10: Cleaning up...")
        await agent.shutdown()
        await caim_framework.shutdown()
        print("   ✅ Cleanup completed")
        
        print("\n🎉 Basic usage example completed successfully!")
        print("\nKey features demonstrated:")
        print("✅ CAIM framework initialization")
        print("✅ Agent creation and configuration")
        print("✅ Conversation with memory")
        print("✅ Memory consolidation")
        print("✅ Memory retrieval and insights")
        
    except Exception as e:
        logger.error(f"Error in basic usage example: {e}")
        print(f"\n❌ Error: {e}")
        raise


async def advanced_usage_example():
    """Demonstrate advanced CAIM framework features."""
    
    print("\n" + "=" * 50)
    print("🚀 CAIM Framework - Advanced Usage Example")
    print("=" * 50)
    
    try:
        # Create configuration with custom settings
        config = CAIMConfig(
            max_memory_size=1000,
            consolidation_threshold=0.7,
            retrieval_top_k=3,
            memory_decay_factor=0.9
        )
        
        # Initialize framework
        caim_framework = CAIMFramework(config)
        await caim_framework.initialize()
        
        # Create multiple agents for comparison
        print("👥 Creating multiple agents...")
        
        # Mock model for demo
        class AdvancedMockModel:
            def __init__(self, name):
                self.model_name = name
                self.is_initialized = False
            
            async def initialize(self):
                self.is_initialized = True
            
            async def generate_response(self, prompt, **kwargs):
                from caim.models.base_model import ModelResponse
                from datetime import datetime
                
                # More sophisticated mock responses
                if "programming" in prompt.lower():
                    response = f"I understand you're interested in programming. {self.model_name} suggests exploring different coding paradigms."
                elif "name" in prompt.lower():
                    response = f"Based on our conversation history, {self.model_name} recalls personal details you've shared."
                else:
                    response = f"{self.model_name} processing: {prompt[:30]}..."
                
                return ModelResponse(
                    content=response,
                    model_name=self.model_name,
                    timestamp=datetime.utcnow(),
                    metadata={"advanced_mock": True},
                    confidence_score=0.85
                )
            
            async def generate_streaming_response(self, prompt, **kwargs):
                response = await self.generate_response(prompt, **kwargs)
                words = response.content.split()
                for word in words:
                    yield word + " "
            
            async def health_check(self):
                return {"status": "healthy", "model_name": self.model_name}
            
            async def shutdown(self):
                self.is_initialized = False
        
        # Create agents with different configurations
        agent1 = CAIMAgent(
            name="MemoryExpert",
            config=config,
            model=AdvancedMockModel("MemoryExpert-Model"),
            caim_framework=caim_framework
        )
        agent1.auto_consolidate = True
        agent1.consolidation_interval = 3
        
        agent2 = CAIMAgent(
            name="QuickResponder", 
            config=config,
            model=AdvancedMockModel("QuickResponder-Model"),
            caim_framework=caim_framework
        )
        agent2.auto_consolidate = False
        agent2.max_context_memories = 2
        
        await agent1.initialize()
        await agent2.initialize()
        
        print("✅ Multiple agents created")
        
        # Advanced conversation simulation
        print("\n💬 Simulating advanced conversations...")
        
        session1 = "advanced_session_001"
        session2 = "advanced_session_002"
        
        # Conversation topics
        topics = [
            ("programming", "I'm learning machine learning and deep learning algorithms."),
            ("preferences", "I prefer working in the morning and drinking coffee."),
            ("projects", "I'm building a chatbot using transformer models."),
            ("goals", "My goal is to become an AI researcher and publish papers."),
            ("hobbies", "In my free time, I enjoy reading research papers and coding.")
        ]
        
        # Simulate conversations with both agents
        for i, (topic, message) in enumerate(topics):
            print(f"\n🔄 Round {i+1}: {topic}")
            
            # Agent 1 response
            response1 = await agent1.process_message(message, session1)
            print(f"🧠 MemoryExpert: {response1.content}")
            
            # Agent 2 response
            response2 = await agent2.process_message(message, session2)
            print(f"⚡ QuickResponder: {response2.content}")
        
        # Test cross-session memory retrieval
        print("\n🔍 Testing cross-session queries...")
        
        query = "What do you know about my interests and goals?"
        
        response1 = await agent1.process_message(query, session1)
        print(f"🧠 MemoryExpert (Session 1): {response1.content}")
        
        response2 = await agent2.process_message(query, session2) 
        print(f"⚡ QuickResponder (Session 2): {response2.content}")
        
        # Compare memory insights
        print("\n📊 Comparing memory insights...")
        
        insights1 = await agent1.get_memory_insights(session1)
        insights2 = await agent2.get_memory_insights(session2)
        
        print("MemoryExpert insights:")
        print(f"  Interactions: {insights1.get('agent_insights', {}).get('interaction_count', 0)}")
        print(f"  Auto-consolidate: {insights1.get('agent_insights', {}).get('auto_consolidate_enabled', False)}")
        
        print("QuickResponder insights:")
        print(f"  Interactions: {insights2.get('agent_insights', {}).get('interaction_count', 0)}")
        print(f"  Auto-consolidate: {insights2.get('agent_insights', {}).get('auto_consolidate_enabled', False)}")
        
        # Manual consolidation for QuickResponder
        print("\n🔄 Manual consolidation for QuickResponder...")
        consolidation_result = await agent2.force_consolidation(session2)
        print(f"Consolidation forced: {consolidation_result.get('forced', False)}")
        
        # Final cleanup
        print("\n🧹 Cleanup...")
        await agent1.shutdown()
        await agent2.shutdown()
        await caim_framework.shutdown()
        
        print("\n🎉 Advanced usage example completed!")
        print("\nAdvanced features demonstrated:")
        print("✅ Multiple agent configurations")
        print("✅ Cross-session memory management")
        print("✅ Manual vs automatic consolidation")
        print("✅ Agent-specific memory insights")
        print("✅ Comparative memory behavior")
        
    except Exception as e:
        logger.error(f"Error in advanced usage example: {e}")
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    print("🧠 CAIM Framework Examples")
    print("Choose an example to run:")
    print("1. Basic Usage")
    print("2. Advanced Usage")
    print("3. Both")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        asyncio.run(basic_usage_example())
    elif choice == "2":
        asyncio.run(advanced_usage_example())
    elif choice == "3":
        asyncio.run(basic_usage_example())
        asyncio.run(advanced_usage_example())
    else:
        print("Invalid choice. Running basic usage example...")
        asyncio.run(basic_usage_example())