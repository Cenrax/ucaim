#!/usr/bin/env python3
"""Simple test to verify OpenAI integration is working."""

import asyncio
import sys
from pathlib import Path

# Add caim to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_openai_chat():
    """Test OpenAI chat functionality."""
    try:
        from caim.core.config import CAIMConfig
        from caim.models.openai_model import OpenAIModel
        
        print("🤖 Testing OpenAI Integration")
        print("=" * 40)
        
        # Create config
        config = CAIMConfig.from_env()
        
        if not config.openai_api_key:
            print("❌ No OpenAI API key found in config")
            return False
        
        print(f"✅ API key found: {config.openai_api_key[:8]}...{config.openai_api_key[-4:]}")
        
        # Create model
        model = OpenAIModel(config, "gpt-4o-mini")
        await model.initialize()
        print("✅ Model initialized successfully")
        
        # Test simple chat
        print("\n💬 Testing chat...")
        response = await model.generate_response("Hello! Can you respond with just 'Hi there!'?")
        print(f"🤖 Response: {response.content}")
        
        if response.content:
            print("✅ OpenAI integration working perfectly!")
            return True
        else:
            print("❌ Got empty response")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_openai_chat())
    if success:
        print("\n🎉 OpenAI is ready to use with CAIM!")
    else:
        print("\n🔧 Please check your API key setup")
    sys.exit(0 if success else 1)