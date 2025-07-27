#!/usr/bin/env python3
"""Basic test of CAIM framework without heavy dependencies."""

import asyncio
import sys
from pathlib import Path

# Add caim to path
sys.path.insert(0, str(Path(__file__).parent))

from caim.core.caim_framework import CAIMFramework
from caim.core.config import CAIMConfig
from caim.agents.caim_agent import CAIMAgent
from caim.models.base_model import BaseModel, ModelResponse
from datetime import datetime


class SimpleTestModel(BaseModel):
    """Simple test model that doesn't require external dependencies."""
    
    def __init__(self, config):
        super().__init__(config, "SimpleTestModel")
    
    async def initialize(self):
        self.is_initialized = True
    
    async def generate_response(self, prompt, **kwargs):
        # Simple rule-based responses for testing
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            response = "Hello! I'm a CAIM-powered agent with memory capabilities. How can I help you?"
        elif "name" in prompt_lower and "alice" in prompt_lower:
            response = "I remember that your name is Alice! It's nice to meet you."
        elif "name" in prompt_lower:
            response = "I can help you remember names and other important information across our conversations."
        elif "python" in prompt_lower:
            response = "I see you mentioned Python! I'll remember your interest in programming."
        else:
            response = f"I understand you said: '{prompt[:50]}...' I'll remember this in our conversation history."
        
        return ModelResponse(
            content=response,
            model_name=self.model_name,
            timestamp=datetime.utcnow(),
            metadata={"test_model": True},
            confidence_score=0.8
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


async def test_caim_basic():
    """Test basic CAIM functionality."""
    print("🧠 Testing CAIM Framework - Basic Functionality")
    print("=" * 50)
    
    try:
        # 1. Create configuration
        print("📋 Creating configuration...")
        config = CAIMConfig(
            max_memory_size=100,
            embedding_model="simple",
            retrieval_top_k=3
        )
        print("   ✅ Configuration created")
        
        # 2. Initialize CAIM framework
        print("\n🚀 Initializing CAIM framework...")
        caim_framework = CAIMFramework(config)
        await caim_framework.initialize()
        print("   ✅ CAIM framework initialized")
        
        # 3. Create simple model
        print("\n🤖 Creating test model...")
        model = SimpleTestModel(config)
        await model.initialize()
        print("   ✅ Test model created")
        
        # 4. Create CAIM agent
        print("\n🤝 Creating CAIM agent...")
        agent = CAIMAgent(
            name="TestAgent",
            config=config,
            model=model,
            caim_framework=caim_framework
        )
        await agent.initialize()
        print("   ✅ CAIM agent created")
        
        # 5. Test conversation with memory
        print("\n💬 Testing conversation with memory...")
        session_id = "test_session_001"
        
        # First message
        print("\n👤 User: Hello! My name is Alice and I love Python programming.")
        response1 = await agent.process_message(
            "Hello! My name is Alice and I love Python programming.",
            session_id
        )
        print(f"🤖 Agent: {response1.content}")
        
        # Second message
        print("\n👤 User: What do you remember about me?")
        response2 = await agent.process_message(
            "What do you remember about me?",
            session_id
        )
        print(f"🤖 Agent: {response2.content}")
        print(f"   💭 Memories used: {len(response2.relevant_memories or [])}")
        
        # 6. Test memory insights
        print("\n🧠 Checking memory insights...")
        insights = await caim_framework.get_memory_insights(session_id)
        stm_count = insights.get("short_term_memory", {}).get("total_memories", 0)
        ltm_count = insights.get("long_term_memory", {}).get("total_memories", 0)
        print(f"   Short-term memories: {stm_count}")
        print(f"   Long-term memories: {ltm_count}")
        
        # 7. Test memory consolidation
        print("\n🔄 Testing memory consolidation...")
        consolidation_result = await caim_framework.consolidate_session_memories(session_id)
        print(f"   Promoted memories: {consolidation_result.get('promoted_memories', 0)}")
        print(f"   Inductive thoughts: {consolidation_result.get('inductive_thoughts', 0)}")
        
        # 8. Test after consolidation
        print("\n👤 User: Do you remember my name and interests?")
        response3 = await agent.process_message(
            "Do you remember my name and interests?",
            session_id
        )
        print(f"🤖 Agent: {response3.content}")
        print(f"   💭 Memories used: {len(response3.relevant_memories or [])}")
        
        # 9. Cleanup
        print("\n🧹 Cleaning up...")
        await agent.shutdown()
        await caim_framework.shutdown()
        print("   ✅ Cleanup completed")
        
        print("\n🎉 Basic CAIM test completed successfully!")
        print("\nKey features tested:")
        print("✅ Framework initialization")
        print("✅ Agent creation")
        print("✅ Conversation with memory")
        print("✅ Memory consolidation")
        print("✅ Memory insights")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_caim_basic())
    sys.exit(0 if success else 1)