#!/usr/bin/env python3
"""Test script to diagnose OpenAI API key issue."""

import os
import sys
from pathlib import Path

# Add caim to path
sys.path.insert(0, str(Path(__file__).parent))

def test_environment_variables():
    """Test environment variables."""
    print("🔍 Environment Variable Diagnostics")
    print("=" * 50)
    
    # Check if OPENAI_API_KEY is set
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OPENAI_API_KEY found: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '[short]'}")
    else:
        print("❌ OPENAI_API_KEY not found in environment")
    
    # Check all environment variables that start with OPENAI
    openai_vars = {k: v for k, v in os.environ.items() if k.startswith("OPENAI")}
    if openai_vars:
        print(f"\n🔑 Found {len(openai_vars)} OPENAI environment variables:")
        for key, value in openai_vars.items():
            masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "[short]"
            print(f"   {key}: {masked_value}")
    else:
        print("\n❌ No OPENAI environment variables found")
    
    return api_key

def test_caim_config():
    """Test CAIM configuration."""
    print("\n🔧 CAIM Configuration Test")
    print("=" * 50)
    
    try:
        from caim.core.config import CAIMConfig
        
        # Test default config
        config = CAIMConfig()
        print(f"✅ Default config created")
        print(f"   openai_api_key: {'Set' if config.openai_api_key else 'Not set'}")
        
        # Test from_env config  
        config_env = CAIMConfig.from_env()
        print(f"✅ Environment config created")
        print(f"   openai_api_key: {'Set' if config_env.openai_api_key else 'Not set'}")
        
        if config_env.openai_api_key:
            key = config_env.openai_api_key
            print(f"   API key: {key[:8]}...{key[-4:] if len(key) > 12 else '[short]'}")
        
        return config_env
        
    except Exception as e:
        print(f"❌ Error creating CAIM config: {e}")
        return None

def test_openai_model():
    """Test OpenAI model creation."""
    print("\n🤖 OpenAI Model Test")
    print("=" * 50)
    
    try:
        from caim.core.config import CAIMConfig
        from caim.models.openai_model import OpenAIModel
        
        config = CAIMConfig.from_env()
        
        if not config.openai_api_key:
            print("❌ Cannot test OpenAI model: No API key in config")
            return False
        
        # Try to create model
        model = OpenAIModel(config, "gpt-4o-mini")
        print("✅ OpenAI model created successfully")
        print(f"   Model: {model.model_name}")
        print(f"   API key configured: Yes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating OpenAI model: {e}")
        return False

def suggest_fixes():
    """Suggest potential fixes."""
    print("\n💡 Suggested Fixes")
    print("=" * 50)
    
    print("1. **Check your shell/terminal:**")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print("   echo $OPENAI_API_KEY")
    
    print("\n2. **Create a .env file in the project root:**")
    print("   echo 'OPENAI_API_KEY=your-api-key-here' > .env")
    
    print("\n3. **Set in current Python session:**")
    print("   import os")
    print("   os.environ['OPENAI_API_KEY'] = 'your-api-key-here'")
    
    print("\n4. **Check if you're using the right terminal:**")
    print("   - Make sure you set the variable in the same terminal you're running the script")
    print("   - Try opening a new terminal and setting the variable again")
    
    print("\n5. **Verify API key format:**")
    print("   - Should start with 'sk-' or 'sk-proj-'")
    print("   - Should be around 51-56 characters long")

if __name__ == "__main__":
    print("🧪 CAIM OpenAI API Key Diagnostic Tool")
    print("=" * 60)
    
    # Test environment
    api_key = test_environment_variables()
    
    # Test config
    config = test_caim_config()
    
    # Test model creation
    model_success = test_openai_model()
    
    # Summary
    print("\n📋 Summary")
    print("=" * 50)
    print(f"Environment variable: {'✅ Found' if api_key else '❌ Missing'}")
    print(f"CAIM config: {'✅ Working' if config and config.openai_api_key else '❌ No API key'}")
    print(f"Model creation: {'✅ Success' if model_success else '❌ Failed'}")
    
    if not api_key or not config or not model_success:
        suggest_fixes()
    else:
        print("\n🎉 Everything looks good! Your OpenAI API key should work.")