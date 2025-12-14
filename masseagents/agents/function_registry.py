import json
import os
from typing import Dict, List, Any
from autogen import ConversableAgent, UserProxyAgent
from masseagents.dataflows.interface import StructuralDataInterface
from masseagents.default_config import get_default_config


class FunctionRegistry:
    """Unified management of all AutoGen agent tool functions - streamlined approach"""
    
    def __init__(self, config: Dict[str, Any], memory_manager, session_dir: str = None):
        self.config = config
        self.memory_manager = memory_manager
        self.session_dir = session_dir
        self.data_interface = StructuralDataInterface(config, memory_manager, session_dir=session_dir)
    
    def run_complete_opensees_analysis(self, model_data: str = "") -> str:
        """Complete OpenSees analysis workflow in one step - accessible as class method"""
        try:
            # Auto-fetch model data if not provided
            if not model_data:
                model_data = self.memory_manager.get_memory("structural_model")
                if not model_data:
                    return "No structural model available"
            
            if isinstance(model_data, str):
                model_data = json.loads(model_data)
            
            # Step 1: Generate OpenSees script
            script_path = self.data_interface.generate_opensees_script(model_data)
            print(f"✅ OpenSees script generated: {script_path}")
            
            # Step 2: Run OpenSees analysis
            self.data_interface.run_opensees_analysis(script_path)
            print(f"✅ OpenSees analysis executed successfully")
            
            # Step 3: Process internal forces
            processed_forces = self.data_interface.process_internal_forces()
            
            # Step 4: Apply live load envelope analysis
            load_data = self.memory_manager.get_memory("load_data")
            if load_data and isinstance(load_data, dict):
                live_loads = load_data.get('load_cases', {}).get('live', {})
                if live_loads:
                    # Calculate live load combination: 1.5 * (F1 + F2 + F3 + ...)
                    total_live_load = sum(live_loads.get(f'F{i}', 0) for i in range(1, 10))  # F1, F2, F3...
                    live_load_combination = 1.5 * total_live_load
                    
                    # Get seismic max compression
                    seismic_max_compression = processed_forces.get('beams', {}).get('max_compression', {}).get('value', 0)
                    
                    # Apply envelope: max(seismic, live load combination)
                    envelope_max_compression = max(seismic_max_compression, live_load_combination)
                    
                    # Update processed forces with envelope result
                    if 'beams' in processed_forces:
                        processed_forces['beams']['max_compression']['value'] = envelope_max_compression
                        print(f"🔄 Applied live load envelope: seismic={seismic_max_compression:.3f}, live_combo={live_load_combination:.3f}, envelope={envelope_max_compression:.3f}")
            
            # Step 5: Save results to memory
            self.memory_manager.update_memory("processed_forces", processed_forces)
            
            # Step 6: Format output summary
            beams = processed_forces.get("beams", {})
            trusses = processed_forces.get("trusses", {})
            
            summary = (
                f"OpenSees Analysis Complete:\n"
                f"BEAMS: Tension={beams.get('max_tension', {}).get('value', 0):.2f} kip, "
                f"Compression={beams.get('max_compression', {}).get('value', 0):.2f} kip, "
                f"Moment={beams.get('max_bending_moment', {}).get('value', 0):.2f} kip*in\n"
                f"TRUSSES: Tension={trusses.get('max_tension', {}).get('value', 0):.2f} kip, "
                f"Compression={trusses.get('max_compression', {}).get('value', 0):.2f} kip\n"
                f"Analysis results saved to memory."
            )
            
            return summary
            
        except Exception as e:
            print(f"❌ OpenSees analysis failed: {str(e)}")
            return f"OpenSees analysis failed: {str(e)}"
    
    def generate_structural_model(self, description: str) -> Dict[str, Any]:
        """Generate structural model - accessible as class method"""
        try:
            if description in ["SAA_input_update", "SAA_input"]:
                actual_description = self.memory_manager.get_memory(description)
                if not actual_description:
                    actual_description = self.memory_manager.get_memory('SAA_input')
            else:
                actual_description = description
            
            if not actual_description:
                error_msg = "No structural description available for model generation"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            
            result = self.data_interface.generate_structural_model(actual_description)
            
            # Validate the generated model
            self._validate_structural_model(result, actual_description)
            
            self.memory_manager.update_memory("structural_model", result)
            
            # Report success with model statistics
            nodes = result.get("nodes", [])
            elements = result.get("elements", [])
            print(f"✅ Structural model generated successfully: {len(nodes)} nodes, {len(elements)} elements")
            
            return result
            
        except Exception as e:
            error_msg = f"Structural model generation failed: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _validate_structural_model(self, model: Dict[str, Any], description: str):
        """Validate generated structural model against requirements"""
        # Basic validation
        if not model or not isinstance(model, dict):
            raise RuntimeError("Generated model is invalid or empty")
        
        required_keys = ["nodes", "elements", "supports", "loads"]
        for key in required_keys:
            if key not in model:
                raise RuntimeError(f"Generated model missing required key: {key}")
        
        nodes = model.get("nodes", [])
        elements = model.get("elements", [])
        
        if len(nodes) < 4:  # At least 4 nodes needed for basic frame
            raise RuntimeError(f"Generated model has insufficient nodes: {len(nodes)} (minimum 4 required)")
        
        if len(elements) < 3:  # At least 2 columns + 1 brace
            raise RuntimeError(f"Generated model has insufficient elements: {len(elements)} (minimum 3 required)")
        
        # Validate element types
        beam_elements = [e for e in elements if e.get("type") == "elasticBeamColumn"]
        truss_elements = [e for e in elements if e.get("type") == "truss"]
        
        if len(beam_elements) < 2:
            raise RuntimeError(f"Generated model missing columns: found {len(beam_elements)}, expected at least 2")
        
        if len(truss_elements) < 1:
            raise RuntimeError(f"Generated model missing braces: found {len(truss_elements)}, expected at least 1")
    
    def register_all_functions(self, agents: Dict[str, ConversableAgent], user_proxy: UserProxyAgent):
        """Register all functions to agents and user_proxy - one-time setup"""
        
        # =====================
        # Step 1: Problem decomposition
        # =====================
        def split_problem_description(description: str) -> str:
            """Split problem description into components - efficient single call"""
            result = self.data_interface.decompose_problem(description)
            
            # Store all components in memory at once
            self.memory_manager.update_memory("SAA_input", result.get("SAA_input", ""))
            self.memory_manager.update_memory("SDA_input", result.get("SDA_input", ""))
            self.memory_manager.update_memory("LA_input", result.get("LA_input", ""))
            self.memory_manager.update_memory("number_of_bays", result.get("number_of_bays", 2))
            self.memory_manager.update_memory("number_of_pallets", result.get("number_of_pallets", 2))
            
            return f"Problem split into SAA, SDA, LA inputs. Bays: {result.get('number_of_bays', 2)}, Pallets: {result.get('number_of_pallets', 2)}"
        
        def adjust_pallet_weights(la_input: str, num_bays: int, num_pallets: int) -> str:
            """Adjust pallet weights based on configuration"""
            # Auto-fetch LA_input from memory if parameter is just a key name
            if la_input in ["LA_input", ""]:
                actual_la_input = self.memory_manager.get_memory('LA_input')
                if not actual_la_input:
                    error_msg = "Pallet weight adjustment failed: No LA_input available in memory"
                    print(f"❌ SYSTEM FAILURE: {error_msg}")
                    raise RuntimeError(error_msg)
            else:
                actual_la_input = la_input
            
            result = self.data_interface.adjust_pallet_weights(actual_la_input, num_bays, num_pallets)
            self.memory_manager.update_memory("LA_input_adjusted", result)
            
            # Extract adjusted loads from LA_input_adjusted and update loads_lbs in memory
            import re
            weight_pattern = r'(\d+)\s*lbs?\s*at\s*([\d.]+)\s*ft'
            matches = re.findall(weight_pattern, result)
            if matches:
                # Sort by elevation and extract weights
                elevation_weights = [(float(match[1]), int(match[0])) for match in matches]
                elevation_weights.sort(key=lambda x: x[0])  # Sort by elevation
                adjusted_loads_lbs = [weight for _, weight in elevation_weights]
                
                # Debug: Check current loads_lbs before update
                current_loads = self.memory_manager.get_memory("loads_lbs")
                print(f"DEBUG: Current loads_lbs before update: {current_loads}")
                
                # Update the loads_lbs in memory with adjusted values
                try:
                    self.memory_manager.update_memory("loads_lbs", adjusted_loads_lbs)
                    # Verify the update was successful
                    updated_loads = self.memory_manager.get_memory("loads_lbs")
                    print(f"DEBUG: loads_lbs after update: {updated_loads}")
                except Exception as e:
                    print(f"DEBUG: Failed to update loads_lbs: {e}")
                    return f"Error updating loads_lbs: {e}"
                
                # Also ensure floor_elevations_ft is updated if needed
                floor_elevations = [elevation for elevation, _ in elevation_weights]
                self.memory_manager.update_memory("floor_elevations_ft", floor_elevations)
                
                return f"Pallet weights adjusted for {num_bays} bays, {num_pallets} pallets. Updated loads_lbs: {adjusted_loads_lbs}"
            else:
                return f"Pallet weights adjusted for {num_bays} bays, {num_pallets} pallets (no weight extraction)"
        
        # =====================
        # Step 2: Section design (auto-fetch from memory)
        # =====================
        def extract_section_info(description: str = "") -> str:
            """Extract section info - auto-fetch from SDA_input if empty"""
            if not description:
                description = self.memory_manager.get_memory('SDA_input')
                if not description:
                    error_msg = "Section info extraction failed: No SDA_input available in memory"
                    print(f"❌ SYSTEM FAILURE: {error_msg}")
                    raise RuntimeError(error_msg)
            
            result = self.data_interface.extract_section_info(description)
            
            # Store all relevant data to memory
            self.memory_manager.update_memory("section_info", result)
            self.memory_manager.update_memory("section_data", result)  # Alternative key
            
            # Extract summary info
            columns = len(result.get("columns", []))
            braces = len(result.get("braces", []))
            beams = len(result.get("beams", []))
            
            return f"Section info extracted: {columns} columns, {braces} braces, {beams} beams"
        
        def calculate_section_capacities(section_info: str = "") -> str:
            """Calculate section capacities - auto-fetch if needed"""
            if not section_info:
                section_info = self.memory_manager.get_memory('section_info')
                if not section_info:
                    error_msg = "Section capacity calculation failed: No section information available in memory"
                    print(f"❌ SYSTEM FAILURE: {error_msg}")
                    raise RuntimeError(error_msg)
            
            if isinstance(section_info, str) and section_info.startswith('{'):
                section_info = json.loads(section_info)
            
            result = self.data_interface.calculate_section_capacities(section_info)
            self.memory_manager.update_memory("section_data", result)
            return f"Section capacities calculated successfully"
        
        # =====================
        # Step 3: Building info extraction (auto-fetch from memory)
        # =====================
        def extract_building_info(description: str = "") -> str:
            """Extract building info - auto-fetch from LA_input if empty"""
            if not description:
                description = self.memory_manager.get_memory('LA_input')
                if not description:
                    error_msg = "Building info extraction failed: No LA_input available in memory"
                    print(f"❌ SYSTEM FAILURE: {error_msg}")
                    raise RuntimeError(error_msg)
            
            result = self.data_interface.extract_building_info(description)
            
            # Store all relevant data to memory
            self.memory_manager.update_memory("building_info", result)
            
            # Store individual components that other functions need
            if "floor_elevations_ft" in result:
                self.memory_manager.update_memory("floor_elevations_ft", result["floor_elevations_ft"])
            
            # Only update loads_lbs if it doesn't already exist (to preserve adjusted values)
            existing_loads = self.memory_manager.get_memory("loads_lbs")
            if "loads_lbs" in result and existing_loads is None:
                self.memory_manager.update_memory("loads_lbs", result["loads_lbs"])
            
            # Extract summary info
            building_type = result.get("building_type", "unknown")
            floors = len(result.get("floor_elevations_ft", []))
            loads = len(result.get("loads_lbs", []))
            
            return f"Building info extracted: {building_type}, {floors} floors, {loads} load points"
        
        # =====================
        # Step 4: Seismic parameters
        # =====================
        def get_seismic_parameters(location: str) -> str:
            """Get seismic parameters using RAG"""
            result = self.data_interface.get_seismic_parameters_rag(location)
            self.memory_manager.update_memory("seismic_parameters", result)
            
            # Return key parameters
            sa_02 = result.get("Sa_02", 0)
            pga = result.get("PGA", 0)
            return f"Seismic parameters for {location}: Sa_02={sa_02}, PGA={pga}"
        
        # =====================
        # Step 5: Load calculations (fully automatic)
        # =====================
        def calculate_seismic_loads(floor_elevations_ft: List[float] = None, 
                                  loads_lbs: List[float] = None, 
                                  seismic_parameters: Dict[str, float] = None) -> str:
            """Calculate seismic loads - auto-fetch from memory if parameters missing"""
            # Auto-fetch missing parameters from memory
            if not floor_elevations_ft:
                floor_elevations_ft = self.memory_manager.get_memory('floor_elevations_ft')
            if not loads_lbs:
                # First check if we have adjusted loads from LA_input_adjusted
                la_input_adjusted = self.memory_manager.get_memory('LA_input_adjusted')
                if la_input_adjusted:
                    # Extract adjusted loads from LA_input_adjusted text
                    import re
                    weight_pattern = r'(\d+)\s*lbs?\s*at\s*([\d.]+)\s*ft'
                    matches = re.findall(weight_pattern, la_input_adjusted)
                    if matches:
                        # Sort by elevation and extract weights
                        elevation_weights = [(float(match[1]), int(match[0])) for match in matches]
                        elevation_weights.sort(key=lambda x: x[0])  # Sort by elevation
                        loads_lbs = [weight for _, weight in elevation_weights]
                
                # Fallback to original loads_lbs if no adjusted loads found
                if not loads_lbs:
                    loads_lbs = self.memory_manager.get_memory('loads_lbs')
                    
            if not seismic_parameters:
                seismic_parameters = self.memory_manager.get_memory('seismic_parameters')
            
            # Check if required parameters are available
            if not floor_elevations_ft:
                error_msg = "Seismic load calculation failed: No floor elevations available in memory"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            if not loads_lbs:
                error_msg = "Seismic load calculation failed: No load data available in memory"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            if not seismic_parameters:
                error_msg = "Seismic load calculation failed: No seismic parameters available in memory"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Calculate loads
            result = self.data_interface.calculate_seismic_loads(
                floor_elevations_ft, loads_lbs, seismic_parameters
            )
            
            # Save to memory with correct key
            self.memory_manager.update_memory("load_data", result)
            
            # Extract force values for summary
            seismic_forces = result.get("load_cases", {}).get("seismic", {})
            force_summary = ", ".join([f"{k}={v} kip" for k, v in seismic_forces.items()])
            
            return f"Seismic loads calculated using weights {loads_lbs}: {force_summary}"
        
        # =====================
        # Step 6: SAA input update
        # =====================
        def update_saa_input(saa_input: str = "", section_data: str = "", load_data: str = "") -> str:
            """Update SAA input - auto-fetch from memory"""
            if not saa_input:
                saa_input = self.memory_manager.get_memory("SAA_input") or ""
            if not section_data:
                section_data = self.memory_manager.get_memory("section_data") or {}
            if not load_data:
                load_data = self.memory_manager.get_memory("load_data") or {}
            
            if isinstance(section_data, str) and section_data.startswith('{'):
                section_data = json.loads(section_data)
            if isinstance(load_data, str) and load_data.startswith('{'):
                load_data = json.loads(load_data)
            
            result = self.data_interface.update_saa_input(saa_input, section_data, load_data)
            self.memory_manager.update_memory("SAA_input_update", result)
            return "SAA input updated successfully with section and load data"
        
        # =====================
        # Step 7: Structural model generation
        # =====================
        def generate_structural_model(description: str) -> str:
            """Generate structural model - auto-fetch SAA_input_update if needed"""
            if description in ["SAA_input_update", "SAA_input"]:
                actual_description = self.memory_manager.get_memory(description)
                if not actual_description:
                    actual_description = self.memory_manager.get_memory('SAA_input')
            else:
                actual_description = description
            
            if not actual_description:
                actual_description = "Default racking system model"
            
            result = self.data_interface.generate_structural_model(actual_description)
            self.memory_manager.update_memory("structural_model", result)
            
            nodes = len(result.get('nodes', []))
            elements = len(result.get('elements', []))
            return f"Structural model generated: {nodes} nodes, {elements} elements"
        
        # =====================
        # Step 8: OpenSees analysis (complete workflow)
        # =====================
        # =====================
        # Step 8: OpenSees analysis (wrapper function)
        # =====================
        def run_complete_opensees_analysis_wrapper(model_data: str = "") -> str:
            """Wrapper for class method to work with AutoGen"""
            return self.run_complete_opensees_analysis(model_data)
        
        # =====================
        # Step 9: Safety verification
        # =====================
        def verify_structural_safety(capacities: str = "", demands: str = "") -> str:
            """Verify structural safety - auto-fetch from memory if parameters missing"""
            # Note: Parameters not needed as the interface method now gets data from memory directly
            try:
                result = self.data_interface.verify_structural_safety({}, {})
                
                # Extract key information for return message
                status = result.get("result", "UNKNOWN")
                safety_status = result.get("safety_status", "UNKNOWN")
                failed_reasons = result.get("failed_reasons", [])
                
                if status == "STRUCTURALLY ADEQUATE":
                    return "Safety verification complete: PASS"
                elif status == "STRUCTURALLY INADEQUATE":
                    reasons_str = "; ".join(failed_reasons) if failed_reasons else "Multiple failures"
                    return f"Safety verification complete: FAIL - {reasons_str}"
                else:
                    return f"Safety verification status: {status}"
                    
            except Exception as e:
                return f"Safety verification failed: {str(e)}"
        
        # =====================
        # Memory and context functions
        # =====================
        def get_memory_summary() -> str:
            """Get memory summary"""
            memory_data = self.memory_manager.get_all_memory()
            non_null_count = sum(1 for v in memory_data.values() if v is not None)
            keys = [k for k, v in memory_data.items() if v is not None]
            return f"Memory contains {non_null_count} non-null entries:\n  - " + "\n  - ".join(keys)
        
        def get_memory_data(key: str) -> str:
            """Get specific memory data by key"""
            data = self.memory_manager.get_memory(key)
            if data is None:
                return f"No data found for key: {key}"
            
            if isinstance(data, (dict, list)):
                return json.dumps(data, indent=2)
            else:
                return str(data)
        
        def get_analysis_context() -> str:
            """Get complete analysis context in one call"""
            context = {}
            
            # Collect all relevant data
            keys = ["seismic_parameters", "structural_model", "processed_forces", "safety_evaluation", 
                   "SAA_input", "SDA_input", "LA_input_adjusted"]
            
            for key in keys:
                data = self.memory_manager.get_memory(key)
                if data:
                    context[key] = data
            
            return json.dumps(context, indent=2)
        
        def save_analysis_results(filepath: str = "analysis_results.json") -> str:
            """Save analysis results to file"""
            try:
                # Determine file path
                if not os.path.isabs(filepath):
                    if self.session_dir:
                        filepath = os.path.join(self.session_dir, filepath)
                    # Otherwise use default behavior with relative path
                
                context = {}
                keys = ["seismic_parameters", "structural_model", "processed_forces", "safety_evaluation"]
                
                for key in keys:
                    data = self.memory_manager.get_memory(key)
                    if data:
                        context[key] = data
                
                with open(filepath, 'w') as f:
                    json.dump(context, f, indent=2)
                
                return f"Analysis results saved to {filepath}"
            except Exception as e:
                return f"Error saving results: {str(e)}"
        
        # =====================
        # Function registration mapping
        # =====================
        function_mappings = [
            # Step 1: Problem decomposition
            ("split_problem_description", split_problem_description, "Split problem description into sub-tasks", ["ProjectManager"]),
            ("adjust_pallet_weights", adjust_pallet_weights, "Adjust pallet weights based on number of bays and pallets", ["ProjectManager"]),
            
            # Step 2: Section design
            ("extract_section_info", extract_section_info, "Extract section information from SDA_input", ["DesignEngineer"]),
            ("calculate_section_capacities", calculate_section_capacities, "Calculate section capacities and properties", ["DesignEngineer"]),
            
            # Step 3: Building info
            ("extract_building_info", extract_building_info, "Extract building information from LA_input", ["LoadingAnalyst"]),
            
            # Step 4: Seismic parameters
            ("get_seismic_parameters", get_seismic_parameters, "Get seismic parameters for a given location", ["SeismicAnalyst"]),
            
            # Step 5: Load calculations
            ("calculate_seismic_loads", calculate_seismic_loads, "Calculate seismic loads based on building info and seismic parameters", ["DynamicAnalyst"]),
            
            # Step 6: SAA update
            ("update_saa_input", update_saa_input, "Update SAA input with section and load data", ["ProjectManager"]),
            
            # Step 7: Model generation
            ("generate_structural_model", generate_structural_model, "Generate structural model JSON from description", ["StructuralAnalyst"]),
            
            # Step 8: OpenSees analysis
            ("run_complete_opensees_analysis", run_complete_opensees_analysis_wrapper, "Complete OpenSees analysis workflow in one step", ["ModelEngineer"]),
            
            # Step 9: Safety verification
            ("verify_structural_safety", verify_structural_safety, "Verify structural safety by comparing capacities and demands", ["VerificationEngineer"]),
            
            # Memory and context
            ("get_memory_summary", get_memory_summary, "Get memory summary", list(agents.keys())),
            ("get_memory_data", get_memory_data, "Get specific memory data by key", list(agents.keys())),
            ("get_analysis_context", get_analysis_context, "Get all analysis data in one call", ["VerificationEngineer", "SafetyManager"]),
            ("save_analysis_results", save_analysis_results, "Save analysis results to file", ["ProjectManager", "SafetyManager"])
        ]
        
        # Register functions to user_proxy for execution
        for func_name, func, description, agent_list in function_mappings:
            user_proxy.register_for_execution(name=func_name)(func)
        
        # Register functions to corresponding agents for LLM calls
        for func_name, func, description, agent_list in function_mappings:
            for agent_name in agent_list:
                if agent_name in agents:
                    agents[agent_name].register_for_llm(
                        name=func_name,
                        description=description
                    )(func)
        
        print(f"✅ Successfully registered {len(function_mappings)} functions to agents and user_proxy")


def register_functions(agents, user_proxy, memory_manager):
    """Main entry point for function registration"""
    config = get_default_config()
    registry = FunctionRegistry(config, memory_manager)
    registry.register_all_functions(agents, user_proxy) 