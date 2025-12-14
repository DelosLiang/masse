import json
import time
from typing import Dict, Any, List, Optional
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager

from masseagents.agents.memory import StructuralMemoryManager
from masseagents.agents.function_registry import FunctionRegistry
from masseagents.agents.agent_factory import MasseAgentFactory
from masseagents.utils.logger import AnalysisLogger


class StructuralAnalysisWorkflow:
    """MASSE Structural Analysis Workflow - Multi-agent coordination based on AutoGen GroupChat"""
    
    def __init__(self, config: Dict[str, Any], log_dir: Optional[str] = None):
        self.config = config
        
        # Initialize logging system first to get session directory
        self.logger = AnalysisLogger(log_dir="logs", session_dir=log_dir)
        self.session_dir = self.logger.get_session_dir()
        
        # Pass session directory to other components
        self.memory_manager = StructuralMemoryManager(config, session_dir=self.session_dir)
        self.function_registry = FunctionRegistry(config, self.memory_manager, session_dir=self.session_dir)
        
        # Create all agents
        self.agents = self._create_agents()
        
        # Create user agent
        self.user_proxy = MasseAgentFactory.create_user_proxy(config)
        
        # Register all functions
        self.function_registry.register_all_functions(self.agents, self.user_proxy)
        
        # Create group chat manager
        self.chat_manager = self._create_chat_manager()
    
    def _create_agents(self) -> Dict[str, ConversableAgent]:
        """Create all specialized agents"""
        agents = {}
        
        # Analyst team - four analysts
        agents["LoadingAnalyst"] = MasseAgentFactory.create_loading_analyst(self.config)
        agents["SeismicAnalyst"] = MasseAgentFactory.create_seismic_analyst(self.config)
        agents["DynamicAnalyst"] = MasseAgentFactory.create_dynamic_analyst(self.config)
        agents["StructuralAnalyst"] = MasseAgentFactory.create_structural_analyst(self.config)
        
        # Engineer team - three engineers
        agents["DesignEngineer"] = MasseAgentFactory.create_design_engineer(self.config)
        agents["ModelEngineer"] = MasseAgentFactory.create_model_engineer(self.config)
        agents["VerificationEngineer"] = MasseAgentFactory.create_verification_engineer(self.config)
        
        # Management team
        agents["ProjectManager"] = MasseAgentFactory.create_project_manager(self.config)
        agents["SafetyManager"] = MasseAgentFactory.create_safety_manager(self.config)
        
        return agents
    
    def _create_chat_manager(self) -> GroupChatManager:
        """Create GroupChat manager"""
        participants = list(self.agents.values()) + [self.user_proxy]
        
        group_chat = GroupChat(
            agents=participants,
            messages=[],
            max_round=self.config.get("max_round", 8),  # Reduce max_round for efficiency
            speaker_selection_method=self.config.get("speaker_selection_method", "auto"),
            allow_repeat_speaker=self.config.get("allow_repeat_speaker", False)
        )
        
        return GroupChatManager(
            groupchat=group_chat,
            llm_config=self.agents["ProjectManager"].llm_config,
            human_input_mode=self.config.get("human_input_mode", "NEVER")
        )
    
    def run_sequential_analysis(self, problem_description: str, location: str = "") -> Dict[str, Any]:
        """Run optimized sequential analysis workflow"""
        start_time = time.time()
        self.logger.log_analysis_start(problem_description, location)
        
        # Initialize AutoGen runtime logging for token tracking
        self.runtime_logging_session = None
        try:
            import autogen
            import tempfile
            import os
            
            print(f"🔍 AutoGen version: {autogen.__version__}")
            
            # Create a temporary database for this session
            temp_db = os.path.join(tempfile.gettempdir(), f"autogen_logs_{int(time.time())}.db")
            self.runtime_logging_session = autogen.runtime_logging.start(
                config={"dbname": temp_db}
            )
            self.runtime_logging_db_path = temp_db
            print(f"📊 Started AutoGen runtime logging: {self.runtime_logging_session}")
            print(f"📊 Database path: {temp_db}")
            
        except Exception as e:
            print(f"⚠️ Could not start AutoGen runtime logging: {e}")
            self.runtime_logging_session = None
        
        # Initialize token tracking
        self.total_token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "steps": {},
            "runtime_db": getattr(self, 'runtime_logging_session', None)
        }
        
        print(f"🔍 LLM Config: {self.config.get('llm_model', 'Unknown')}")
        print(f"🔍 Temperature: {self.config.get('temperature', 'Unknown')}")
        print(f"🔍 Max tokens: {self.config.get('max_tokens', 'Unknown')}")
        
        # System component verification
        try:
            self._verify_system_components()
        except Exception as e:
            self.logger.error(f"❌ System verification failed: {str(e)}")
            return {
                "status": "error",
                "error_message": f"System verification failed: {str(e)}",
                "memory_summary": self.memory_manager.get_summary()
            }
        
        # Initialize execution log
        self.execution_log = []
        self.logger.log_step("System Verification", "COMPLETED")
        
        try:
            # Step 1: ProjectManager - problem decomposition (merge and adjust tray weight)
            self.logger.info("📋 Step 1: Problem decomposition and setup...")
            result1 = self.user_proxy.initiate_chat(
                self.agents["ProjectManager"],
                message=f"MANDATORY: 1) Call split_problem_description() with: {problem_description}. 2) After split, call adjust_pallet_weights() using number_of_bays and number_of_pallets from memory to adjust LA_input. Both steps are required.",
                max_turns=5
            )
            self._extract_token_usage(result1, "Step1_ProjectManager")
            self.logger.log_agent_chat("ProjectManager", "Split problem and adjust pallet weights", 5)
            
            # Step 2: DesignEngineer - section design
            self.logger.info("🔧 Step 2: Section design...")
            result2 = self.user_proxy.initiate_chat(
                self.agents["DesignEngineer"],
                message="Extract section info from SDA_input and calculate capacities.",
                max_turns=5
            )
            self._extract_token_usage(result2, "Step2_DesignEngineer")
            self.logger.log_agent_chat("DesignEngineer", "Extract section info and calculate capacities", 5)
            
            # Step 3: LoadingAnalyst - building information extraction
            self.logger.info("🏢 Step 3: Building information...")
            result3 = self.user_proxy.initiate_chat(
                self.agents["LoadingAnalyst"],
                message="Extract building info from LA_input.",
                max_turns=5
            )
            self._extract_token_usage(result3, "Step3_LoadingAnalyst")
            self.logger.log_agent_chat("LoadingAnalyst", "Extract building info from LA_input", 5)
            
            # Step 4: SeismicAnalyst - seismic parameters
            self.logger.info("🌍 Step 4: Seismic parameters...")
            
            # Extract location from building_info memory (dynamic approach)
            building_info = self.memory_manager.get_memory('building_info')
            location_to_use = ""
            
            if building_info and isinstance(building_info, dict) and 'location' in building_info:
                location_to_use = building_info['location']
                self.logger.info(f"📍 Using location from building_info: {location_to_use}")
            elif "Vancouver" in problem_description:
                location_to_use = "Vancouver, BC"
                self.logger.info(f"📍 Extracted Vancouver from problem description")
            elif "Nanaimo" in problem_description:
                location_to_use = "Nanaimo, BC" 
                self.logger.info(f"📍 Extracted Nanaimo from problem description")
            elif location:
                location_to_use = location
                self.logger.info(f"📍 Using provided location parameter: {location_to_use}")
            else:
                location_to_use = "Vancouver, BC"  # Default fallback
                self.logger.warning(f"📍 No location found, using default: {location_to_use}")
            
            result4 = self.user_proxy.initiate_chat(
                self.agents["SeismicAnalyst"],
                message=f"Get seismic parameters for: {location_to_use}",
                max_turns=3
            )
            self._extract_token_usage(result4, "Step4_SeismicAnalyst")
            self.logger.log_agent_chat("SeismicAnalyst", f"Get seismic parameters for: {location_to_use}", 3)
            
            # Step 5: DynamicAnalyst - load calculation
            self.logger.info("⚖️ Step 5: Load calculations...")
            result5 = self.user_proxy.initiate_chat(
                self.agents["DynamicAnalyst"],
                message="Calculate seismic loads using data from memory.",
                max_turns=5
            )
            self._extract_token_usage(result5, "Step5_DynamicAnalyst")
            self.logger.log_agent_chat("DynamicAnalyst", "Calculate seismic loads", 5)
            
            # Step 6: Update SAA input
            print("🔄 Step 6: Update SAA input...")
            chat_result_6 = self.user_proxy.initiate_chat(
                self.agents["ProjectManager"],
                message="Call update_saa_input() to merge SAA, section, and load data from memory.",
                max_turns=6
            )
            self._extract_token_usage(chat_result_6, "Step6_ProjectManager_Update")
            
            # Step 7: Generate structural model
            print("🏗️ Step 7: Generate structural model...")
            try:
                chat_result_7 = self.user_proxy.initiate_chat(
                    self.agents["StructuralAnalyst"],
                    message="MANDATORY: Call generate_structural_model() function NOW using SAA_input_update data. This step cannot be skipped.",
                    max_turns=6
                )
                self._extract_token_usage(chat_result_7, "Step7_StructuralAnalyst")
                print(f"✅ Step 7 completed successfully")
                print(f"📋 Step 7 chat result summary: {type(chat_result_7)}")
                
                # Verify structural model generation
                structural_model = self.memory_manager.get_memory('structural_model')
                if structural_model:
                    print(f"✅ Structural model confirmed in memory: {len(structural_model.get('nodes', []))} nodes")
                else:
                    error_msg = "Step 7 failed: No structural model found in memory after generation attempt"
                    print(f"❌ SYSTEM FAILURE: {error_msg}")
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                error_msg = f"Step 7 (Structural model generation) failed: {str(e)}"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(error_msg)
            
            # Step 8: Complete structural analysis (merge original 3 sub-steps)
            print("🔬 Step 8: Complete structural analysis...")
            print("📍 About to execute Step 8 - this is the critical OpenSees step")
            
            # Verify ModelEngineer exists
            if "ModelEngineer" not in self.agents:
                error_msg = "Step 8 failed: ModelEngineer not found in agents dictionary"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            print("✅ ModelEngineer agent confirmed")
            
            # Verify structural model data
            structural_model = self.memory_manager.get_memory('structural_model')
            if not structural_model:
                error_msg = "Step 8 failed: No structural model available for OpenSees analysis"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            
            print(f"✅ Structural model data confirmed: {len(structural_model.get('nodes', []))} nodes, {len(structural_model.get('elements', []))} elements")
            
            try:
                chat_result_8 = self.user_proxy.initiate_chat(
                    self.agents["ModelEngineer"],
                    message="MANDATORY: Call run_complete_opensees_analysis() NOW. This step cannot be skipped. Execute OpenSees analysis immediately.",
                    max_turns=5
                )
                self._extract_token_usage(chat_result_8, "Step8_ModelEngineer")
                print(f"✅ Step 8 chat completed")
                
                # Verify if OpenSees analysis actually executed
                processed_forces = self.memory_manager.get_memory('processed_forces')
                if not processed_forces:
                    error_msg = "Step 8 failed: No processed forces found in memory after OpenSees analysis"
                    print(f"❌ SYSTEM FAILURE: {error_msg}")
                    raise RuntimeError(error_msg)
                
                print("✅ Step 8 SUCCESS: OpenSees analysis completed, processed forces found in memory")
                self._log_execution_step("Step 8", "SUCCESS", "OpenSees analysis completed")
                        
            except Exception as e:
                error_msg = f"Step 8 (OpenSees analysis) failed: {str(e)}"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(error_msg)
            
            # Step 9: Structural verification
            print("✅ Step 9: Structural verification...")
            try:
                chat_result_9 = self.user_proxy.initiate_chat(
                    self.agents["VerificationEngineer"],
                    message="Verify structural safety using all analysis data.",
                    max_turns=5
                )
                self._extract_token_usage(chat_result_9, "Step9_VerificationEngineer")
            except Exception as e:
                error_msg = f"Step 9 (Structural verification) failed: {str(e)}"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Step 10: Final safety assessment
            print("🛡️ Step 10: Final safety assessment...")
            try:
                chat_result_10 = self.user_proxy.initiate_chat(
                    self.agents["SafetyManager"],
                    message="Use get_analysis_context() to get all data. Provide final assessment: 'FINAL RESULT: STRUCTURALLY ADEQUATE' or 'FINAL RESULT: STRUCTURALLY INADEQUATE'.",
                    max_turns=5
                )
                self._extract_token_usage(chat_result_10, "Step10_SafetyManager")
            except Exception as e:
                error_msg = f"Step 10 (Final safety assessment) failed: {str(e)}"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            
            self.logger.info("✅ Optimized sequential analysis completed!")
            
            # Stop AutoGen runtime logging and collect actual token usage
            try:
                if self.runtime_logging_session:
                    import autogen
                    autogen.runtime_logging.stop()
                    print("📊 Stopped AutoGen runtime logging")
                    
                    # Extract actual token usage from the database
                    self._extract_runtime_logging_usage()
                    
                else:
                    print("⚠️ No runtime logging session available")
                    
            except Exception as e:
                error_msg = f"Token usage collection failed: {str(e)}"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            
            results = self._compile_results()
            
            # Record analysis end
            duration = time.time() - start_time
            final_result = results.get("final_result", "UNKNOWN")
            self.logger.log_analysis_end(final_result, duration)
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"❌ Error during sequential analysis: {str(e)}")
            self.logger.log_analysis_end("ERROR", duration)
            
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error_message": str(e),
                "memory_summary": self.memory_manager.get_summary()
            }
    
    def run_full_analysis(self, problem_description: str, location: str = "", problem_id: str = "") -> Dict[str, Any]:
        """Run complete structural analysis workflow - unified entry method"""
        
        # Store problem_id in memory for later use
        if problem_id:
            self.memory_manager.update_memory("problem_id", problem_id)
        
        # Initialize token tracking (in case this method is called directly)
        if not hasattr(self, 'total_token_usage'):
            self.total_token_usage = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "steps": {}
            }
        
        # First try sequential execution mode
        return self.run_sequential_analysis(problem_description, location)
        
        # Update memory initial input
        self.memory_manager.update_memory("original_description", problem_description)
        self.memory_manager.update_memory("location", location)
        
        # Prepare initial message
        initial_message = f"""
Please perform a complete structural analysis for this building:

{problem_description}

Steps needed:
1. ProjectManager: Split the problem description into sub-tasks for different specialists
2. DesignEngineer: Extract section information and calculate capacities using SDA_input
3. LoadingAnalyst: Extract building information using LA_input
4. SeismicAnalyst: Get seismic parameters for the location
5. DynamicAnalyst: Calculate seismic loads using the building info and seismic parameters
6. ProjectManager: Update SAA_input with section and load data
7. StructuralAnalyst: Generate structural model from SAA_input_update
8. ModelEngineer: Perform OpenSees analysis and extract maximum forces
9. VerificationEngineer: Verify structural adequacy
10. SafetyManager: Provide final safety assessment

Please coordinate and complete this analysis step by step.
"""
        
        # Start group chat analysis
        try:
            print("🔄 Initiating group chat analysis...")
            result = self.user_proxy.initiate_chat(
                self.chat_manager,
                message=initial_message
            )
            print(f"📋 Chat result: {result}")
            
            # Compile analysis results
            return self._compile_results()
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error_message": str(e),
                "memory_summary": self.memory_manager.get_summary()
            }
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final analysis results"""
        try:
            # Get all data in memory - use get_all_memory instead of get_summary
            memory_data = self.memory_manager.get_all_memory()
            memory_summary_text = self.memory_manager.get_summary()
            
            # Extract key results
            building_info = memory_data.get("building_info", {})
            seismic_parameters = memory_data.get("seismic_parameters", {})
            loads = memory_data.get("loads", {})
            structural_model = memory_data.get("structural_model", {})
            section_capacities = memory_data.get("section_properties", {})  # Note this is section_properties
            internal_forces = memory_data.get("processed_forces", {})  # Note this is processed_forces
            safety_evaluation = memory_data.get("safety_evaluation", {})
            
            # Extract beam capacity information
            beam_capacity = None
            if section_capacities and isinstance(section_capacities, dict):
                # Look for beam information in section_capacities
                beam_info = section_capacities.get("beam", {})
                if beam_info and "capacities" in beam_info:
                    beam_capacity = beam_info["capacities"]
                # Also check for beam_capacities directly
                elif "beam_capacities" in section_capacities:
                    beam_capacity = section_capacities["beam_capacities"]
                # Check for allowable_load in the main structure
                elif "allowable_load" in section_capacities:
                    beam_capacity = {"allowable_load": section_capacities["allowable_load"]}
            
            # Determine final result
            final_result = "ANALYSIS INCOMPLETE"
            safety_status = "UNKNOWN"
            
            if safety_evaluation:
                final_result = safety_evaluation.get("result", "UNKNOWN")
                safety_status = safety_evaluation.get("safety_status", "UNKNOWN")
            
            # Compile enhanced results
            results = {
                "status": "completed",
                "final_result": final_result,
                "safety_status": safety_status,
                "analysis_summary": {
                    "building_info": building_info,
                    "seismic_parameters": seismic_parameters,
                    "loads": loads,
                    "structural_model": structural_model,
                    "section_capacities": section_capacities,
                    "beam_capacity": beam_capacity,  # Add beam capacity specifically
                    "internal_forces": internal_forces,
                    "safety_evaluation": safety_evaluation
                },
                "memory_summary": memory_summary_text,
                "workflow_type": "sequential_analysis"
            }
            
            # Add problem_id if available
            problem_id = memory_data.get("problem_id")
            if problem_id:
                results["problem_id"] = problem_id
            
            # Add simplified token usage information
            if hasattr(self, 'total_token_usage') and self.total_token_usage:
                results["token_usage"] = {
                    "total_tokens": self.total_token_usage["total_tokens"],
                    "prompt_tokens": self.total_token_usage["prompt_tokens"], 
                    "completion_tokens": self.total_token_usage["completion_tokens"],
                    "note": "Token usage collected from AutoGen runtime logging"
                }
                
                # Add cost information if available
                if "total_cost" in self.total_token_usage:
                    results["token_usage"]["total_cost"] = self.total_token_usage["total_cost"]
                    
            else:
                results["token_usage"] = {
                    "total_tokens": 0,
                    "prompt_tokens": 0, 
                    "completion_tokens": 0,
                    "note": "Token tracking not available"
                }
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Failed to compile results: {str(e)}",
                "memory_summary": self.memory_manager.get_summary()
            }
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get analysis system status"""
        return {
            "available_agents": list(self.agents.keys()),
            "memory_items": len(self.memory_manager.get_summary()),
            "configuration": {
                "model": self.config.get("llm_model", "gpt-4o"),
                "temperature": self.config.get("temperature", 0),
                "max_rounds": self.config.get("max_round", 15)
            }
        }
    
    def _verify_system_components(self) -> None:
        """Verify system components are properly initialized"""
        print("🔍 Verifying system components...")
        
        # Verify required agents
        required_agents = [
            "ProjectManager", "DesignEngineer", "LoadingAnalyst", 
            "SeismicAnalyst", "DynamicAnalyst", "ModelEngineer",
            "StructuralAnalyst", "VerificationEngineer", "SafetyManager"
        ]
        
        missing_agents = []
        for agent_name in required_agents:
            if agent_name not in self.agents:
                missing_agents.append(agent_name)
            else:
                print(f"✅ {agent_name} initialized")
        
        if missing_agents:
            raise ValueError(f"Missing required agents: {missing_agents}")
        
        # Verify key functions are registered
        try:
            # Test if OpenSees function is available
            test_result = self.function_registry.run_complete_opensees_analysis("")
            if "Error: No structural model found in memory" in test_result:
                print("✅ OpenSees function available (expected error for empty input)")
            else:
                print(f"✅ OpenSees function available: {test_result}")
        except Exception as e:
            print(f"❌ OpenSees function error: {e}")
            raise
        
        print("✅ All system components verified")
    
    def _log_execution_step(self, step_name: str, status: str, details: str = "") -> None:
        """Log execution steps"""
        if not hasattr(self, 'execution_log'):
            self.execution_log = []
        
        log_entry = f"{step_name}: {status}"
        if details:
            log_entry += f" - {details}"
        
        self.execution_log.append(log_entry)
        print(f"📝 {log_entry}")
    
    def _extract_runtime_logging_usage(self) -> None:
        """Extract token usage from AutoGen runtime logging database"""
        try:
            print("\n📊 === EXTRACTING TOKEN USAGE FROM AUTOGEN LOGS ===")
            
            if not hasattr(self, 'total_token_usage'):
                self.total_token_usage = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "steps": {}
                }
            
            # Get the database path
            if not hasattr(self, 'runtime_logging_db_path'):
                print("⚠️ No runtime logging database path available")
                return
            
            temp_db = self.runtime_logging_db_path
            print(f"📊 Using database: {temp_db}")
            
            import sqlite3
            import json
            import os
            
            try:
                # Connect to the SQLite database
                con = sqlite3.connect(temp_db)
                
                # Get all completion records (since we're using a dedicated temp db)
                cursor = con.execute("SELECT * FROM chat_completions")
                rows = cursor.fetchall()
                print(f"📊 Found {len(rows)} completion records in database")
                
                column_names = [description[0] for description in cursor.description]
                
                total_tokens = 0
                total_prompt_tokens = 0
                total_completion_tokens = 0
                total_cost = 0
                
                for row in rows:
                    row_dict = dict(zip(column_names, row))
                    
                    # Parse the response JSON to get usage
                    if 'response' in row_dict and row_dict['response']:
                        try:
                            response_data = json.loads(row_dict['response'])
                            usage = response_data.get('usage', {})
                            
                            if usage:
                                tokens = usage.get('total_tokens', 0)
                                prompt_tokens = usage.get('prompt_tokens', 0)
                                completion_tokens = usage.get('completion_tokens', 0)
                                
                                total_tokens += tokens
                                total_prompt_tokens += prompt_tokens
                                total_completion_tokens += completion_tokens
                                
                                print(f"✅ Found usage: {tokens} tokens ({prompt_tokens} prompt + {completion_tokens} completion)")
                        except (json.JSONDecodeError, KeyError) as e:
                            continue
                    
                    # Also get cost information
                    if 'cost' in row_dict and row_dict['cost']:
                        try:
                            cost = float(row_dict['cost'])
                            total_cost += cost
                        except (ValueError, TypeError):
                            continue
                
                con.close()
                
                # Update total usage with real data
                self.total_token_usage.update({
                    "total_tokens": total_tokens,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_cost": total_cost,
                    "extraction_method": "autogen_runtime_logging"
                })
                
                print(f"✅ EXTRACTED REAL TOKEN USAGE:")
                print(f"   📊 Total tokens: {total_tokens:,}")
                print(f"   📊 Prompt tokens: {total_prompt_tokens:,}")
                print(f"   📊 Completion tokens: {total_completion_tokens:,}")
                print(f"   💰 Total cost: ${total_cost:.6f}")
                
                # Clean up temp database
                try:
                    os.remove(temp_db)
                    print(f"🗑️ Cleaned up temp database: {temp_db}")
                except:
                    pass
                    
            except Exception as db_error:
                error_msg = f"Database extraction error: {db_error}"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"Runtime logging extraction failed: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    

    
    def _extract_token_usage(self, chat_result, step_name: str) -> None:
        """Simplified token usage extraction"""
        try:
            if not hasattr(self, 'total_token_usage'):
                self.total_token_usage = {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "steps": {}
                }
            
            # Basic token extraction - try simple cases first
            step_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
            
            # Simple extraction attempts
            try:
                if hasattr(chat_result, 'cost') and chat_result.cost:
                    if isinstance(chat_result.cost, dict) and 'usage' in chat_result.cost:
                        usage = chat_result.cost['usage']
                        step_usage["total_tokens"] = usage.get('total_tokens', 0)
                        step_usage["prompt_tokens"] = usage.get('prompt_tokens', 0)
                        step_usage["completion_tokens"] = usage.get('completion_tokens', 0)
                        print(f"✅ {step_name} tokens: {step_usage['total_tokens']}")
                elif hasattr(chat_result, 'usage') and chat_result.usage:
                    usage = chat_result.usage
                    if isinstance(usage, dict):
                        step_usage["total_tokens"] = usage.get('total_tokens', 0)
                        step_usage["prompt_tokens"] = usage.get('prompt_tokens', 0)
                        step_usage["completion_tokens"] = usage.get('completion_tokens', 0)
                        print(f"✅ {step_name} tokens: {step_usage['total_tokens']}")
            except Exception:
                pass  # Ignore extraction errors
            
            # Store step data
            self.total_token_usage["steps"][step_name] = step_usage
            
            # Accumulate totals
            self.total_token_usage["total_tokens"] += step_usage["total_tokens"]
            self.total_token_usage["prompt_tokens"] += step_usage["prompt_tokens"] 
            self.total_token_usage["completion_tokens"] += step_usage["completion_tokens"]
            
            if step_usage["total_tokens"] == 0:
                print(f"📝 {step_name} - Token extraction not available")
            
        except Exception as e:
            print(f"⚠️ Token extraction error for {step_name}: {e}")
            # Store placeholder data for failed extractions
            self.total_token_usage["steps"][step_name] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e)
            } 