from typing import Dict, Any
from autogen import ConversableAgent, UserProxyAgent
from masseagents.default_config import get_provider_for_model


class MasseAgentFactory:
    """Unified AutoGen agent creation factory"""
    
    @staticmethod
    def _get_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM configuration - supports multiple providers"""
        model = config.get("llm_model", "gpt-4o")
        provider = get_provider_for_model(model)
        
        # Get provider-specific configuration
        providers_config = config.get("llm_providers", {})
        provider_config = providers_config.get(provider, {})
        
        # Handle model-specific temperature settings
        base_temperature = config.get("temperature", 0)
        if model == "o4-mini" or model == "gpt-5":
            # o4-mini and gpt-5 only support temperature=1 (default), not 0
            temperature = 1
        else:
            temperature = base_temperature
        
        if provider == "anthropic":
            # Anthropic Claude configuration
            api_key = provider_config.get("api_key") or config.get("api_key") or "placeholder"
            
            return {
                "config_list": [{
                    "model": model,
                    "api_key": api_key,
                    "base_url": provider_config.get("base_url", "https://api.anthropic.com"),
                    "api_type": "anthropic"
                }],
                "temperature": temperature,
                "max_tokens": config.get("max_tokens", 4000),
            }
        else:
            # OpenAI configuration (default)
            api_key = provider_config.get("api_key") or config.get("api_key") or "placeholder"
            
            return {
                "config_list": [{
                    "model": model,
                    "api_key": api_key,
                    "base_url": provider_config.get("base_url", "https://api.openai.com/v1")
                }],
                "temperature": temperature,
                # Note: AutoGen handles cost tracking internally via OpenAI client
            }
    
    @staticmethod
    def create_loading_analyst(config: Dict[str, Any]) -> ConversableAgent:
        """Create loading analyst agent"""
        system_message = """You are a LoadingAnalyst. Extract building info from LA_input using extract_building_info()."""
        
        return ConversableAgent(
            name="LoadingAnalyst",
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_model_engineer(config: Dict[str, Any]) -> ConversableAgent:
        """Create model engineer agent"""
        system_message = """You are a ModelEngineer. Use run_complete_opensees_analysis() with structural model from memory. One function call completes entire analysis."""
        
        return ConversableAgent(
            name="ModelEngineer",
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_dynamic_analyst(config: Dict[str, Any]) -> ConversableAgent:
        """Create dynamic analyst agent"""
        system_message = """You are a DynamicAnalyst. Get data from memory, then call calculate_seismic_loads(floor_elevations_ft, loads_lbs, seismic_parameters)."""
        
        return ConversableAgent(
            name="DynamicAnalyst",
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_seismic_analyst(config: Dict[str, Any]) -> ConversableAgent:
        """Create seismic analyst agent"""
        system_message = """You are a SeismicAnalyst. Call get_seismic_parameters(location) for the given location."""
        
        return ConversableAgent(
            name="SeismicAnalyst", 
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_design_engineer(config: Dict[str, Any]) -> ConversableAgent:
        """Create design engineer agent"""
        system_message = """You are a DesignEngineer. Extract section info from SDA_input, then calculate capacities. Use extract_section_info() → calculate_section_capacities()."""
        
        return ConversableAgent(
            name="DesignEngineer",
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_structural_analyst(config: Dict[str, Any]) -> ConversableAgent:
        """Create structural analyst agent"""
        system_message = """You are a StructuralAnalyst. Generate structural model from SAA_input_update using generate_structural_model()."""
        
        return ConversableAgent(
            name="StructuralAnalyst",
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_verification_engineer(config: Dict[str, Any]) -> ConversableAgent:
        """Create verification engineer agent"""
        system_message = """You are a VerificationEngineer. Use get_analysis_context() to get all data, then verify_structural_safety(capacities, demands)."""
        
        return ConversableAgent(
            name="VerificationEngineer",
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_project_manager(config: Dict[str, Any]) -> ConversableAgent:
        """Create project manager agent"""
        system_message = """You are a ProjectManager. Coordinate workflow: split_problem_description() → update_saa_input() → adjust_pallet_weights() → save_analysis_results()."""
        
        return ConversableAgent(
            name="ProjectManager",
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_safety_manager(config: Dict[str, Any]) -> ConversableAgent:
        """Create safety manager agent"""
        system_message = """You are a SafetyManager. Use get_analysis_context() to get all data. Provide final assessment: "FINAL RESULT: STRUCTURALLY ADEQUATE" or "FINAL RESULT: STRUCTURALLY INADEQUATE"."""
        
        return ConversableAgent(
            name="SafetyManager",
            system_message=system_message,
            llm_config=MasseAgentFactory._get_llm_config(config),
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        )
    
    @staticmethod
    def create_user_proxy(config: Dict[str, Any]) -> UserProxyAgent:
        """Create user proxy agent"""
        def is_final_conclusion(x):
            """Check if message is final conclusion"""
            content = x.get("content", "")
            # Ensure content is not None
            if content is None:
                return False
            # Only terminate when message starts with "FINAL RESULT:" or "CONCLUSION:" and contains conclusion
            return (("FINAL RESULT:" in content or "CONCLUSION:" in content) and 
                    ("STRUCTURALLY ADEQUATE" in content or "STRUCTURALLY INADEQUATE" in content))
        
        return UserProxyAgent(
            name="User",
            human_input_mode=config.get("human_input_mode", "NEVER"),
            max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 5),
            is_termination_msg=is_final_conclusion,
            code_execution_config=config.get("code_execution_config", {"use_docker": False})
        ) 