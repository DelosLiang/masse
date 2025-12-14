# -*- coding: utf-8 -*-
"""
MASSE Default Configuration
"""

import os
from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for MASSE"""
    return {
        # Core settings
        "llm_model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 2000,  # Increased to prevent empty responses
        "max_round": 8,
        "human_input_mode": "NEVER",
        "max_consecutive_auto_reply": 5,  # Increased to prevent premature termination
        "speaker_selection_method": "auto",
        "allow_repeat_speaker": False,
        
        # LLM Providers configuration
        "llm_providers": {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "models": ["gpt-4o", "o4-mini", "gpt-5", "gpt-3.5-turbo"]
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "models": ["claude-3-5-sonnet-latest"]
            }
        },
        
        # Analysis parameters
        "top_k": 6,
        "chunk_overlap": 100,
        
        # Code execution
        "code_execution_config": {
            "use_docker": False,
            "last_n_messages": 3,
            "work_dir": "workspace"
        }
    }


def get_provider_for_model(model: str) -> str:
    """Get provider name for a given model"""
    if model.startswith("gpt-") or model.startswith("o4-"):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    else:
        return "openai"  # Default to OpenAI 