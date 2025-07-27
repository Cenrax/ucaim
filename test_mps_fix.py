#!/usr/bin/env python3
"""Test MPS device fix for HuggingFace model."""

import asyncio
import sys
from pathlib import Path
import logging

# Add caim to path
sys.path.insert(0, str(Path(__file__).parent))

from caim.core.config import CAIMConfig
from caim.models.huggingface_model import HuggingFaceModel

# Setup logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

async def test_mps_fix():
    """Test MPS device handling."""
    print("🔧 Testing MPS device fix...")
    
    try:
        # Create config
        config = CAIMConfig()
        
        # Create model with a smaller model for testing
        model = HuggingFaceModel(
            config=config,
            model_name="microsoft/DialoGPT-small"  # Smaller model for testing
        )
        
        print(f"📱 Detected device: {model.device}")
        
        # Initialize model
        print("🚀 Initializing model...")
        await model.initialize()
        print("✅ Model initialized successfully")
        
        # Test generation
        print("💬 Testing generation...")
        response = await model.generate_response("Hello, how are you?")
        print(f"🤖 Response: {response.content}")
        print("✅ Generation successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mps_fix())
    sys.exit(0 if success else 1)