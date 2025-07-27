"""Configuration utilities for CAIM framework."""

import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_env_file(env_file: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    env_path = Path(env_file)
    
    if not env_path.exists():
        return env_vars
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip().strip('"').strip("'")
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ.setdefault(key, value)
        
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
    
    return env_vars


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate CAIM configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "max_memory_size",
        "embedding_model", 
        "retrieval_top_k"
    ]
    
    for field in required_fields:
        if field not in config:
            print(f"Missing required config field: {field}")
            return False
    
    # Validate data types and ranges
    if not isinstance(config["max_memory_size"], int) or config["max_memory_size"] <= 0:
        print("max_memory_size must be a positive integer")
        return False
    
    if not isinstance(config["retrieval_top_k"], int) or config["retrieval_top_k"] <= 0:
        print("retrieval_top_k must be a positive integer")
        return False
    
    return True