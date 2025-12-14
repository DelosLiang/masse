"""
MASSE - Multi-Agent Structural System Engineering

A modern, multi-agent structural engineering framework built on Microsoft AutoGen.
"""

__version__ = "0.1.0"
__author__ = "MASSE Development Team"
__email__ = "team@masse-engineering.com"

from .default_config import get_default_config
from .workflows import StructuralAnalysisWorkflow
from .agents import (
    MasseAgentFactory, 
    FunctionRegistry, 
    StructuralMemoryManager
)

__all__ = [
    "get_default_config",
    "StructuralAnalysisWorkflow", 
    "MasseAgentFactory",
    "FunctionRegistry",
    "StructuralMemoryManager"
] 