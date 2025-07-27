"""
CAIM: A Cognitive AI Memory Framework for Enhanced Long-Term Agent Interactions

This package provides a comprehensive framework for implementing human-like memory
systems in AI agents, enabling long-term contextual understanding and personalized
interactions.
"""

__version__ = "0.1.0"
__author__ = "CAIM Team"
__email__ = "caim@example.com"

from .core.caim_framework import CAIMFramework
from .core.memory_controller import MemoryController
from .memory.short_term_memory import ShortTermMemory
from .memory.long_term_memory import LongTermMemory
from .agents.caim_agent import CAIMAgent

__all__ = [
    "CAIMFramework",
    "MemoryController",
    "ShortTermMemory",
    "LongTermMemory",
    "CAIMAgent",
]