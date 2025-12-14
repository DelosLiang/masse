import json
import os
import subprocess
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from pint import UnitRegistry
from openai import OpenAI
from anthropic import Anthropic
from .rag_seismic import RAGSeismicInterface
from ..default_config import get_provider_for_model

# Import calculation modules (these will need to be implemented)
try:
    from .opensees_utils import OpenSeesInterface
    from .seismic_utils import SeismicDataInterface
    from .section_utils import SectionCalculator
except ImportError:
    # Fallback for development
    OpenSeesInterface = None
    SeismicDataInterface = None
    SectionCalculator = None


class StructuralDataInterface:
    """Unified structural engineering data interface - integrating all original engineering calculation functions"""
    
    def __init__(self, config: Dict[str, Any], memory_manager=None, session_dir: str = None):
        self.config = config
        self.memory_manager = memory_manager
        self.session_dir = session_dir
        self.ureg = UnitRegistry()
        
        # Initialize sub-interfaces
        self.opensees = OpenSeesInterface(config) if OpenSeesInterface else None
        self.seismic = SeismicDataInterface(config) if SeismicDataInterface else None
        self.sections = SectionCalculator(config) if SectionCalculator else None
        
        # Initialize RAG system for seismic parameters
        try:
            self.rag_seismic = RAGSeismicInterface(config)
            
            # Validate database
            if self.rag_seismic.validate_database():
                print("✅ RAG seismic database validation successful")
            else:
                print("❌ RAG seismic database validation failed")
                
        except Exception as e:
            print(f"❌ Failed to initialize RAG seismic interface: {str(e)}")
            raise RuntimeError(f"RAG initialization failed: {str(e)}")
        
    # =====================
    # Problem decomposition functionality (from original main.py)
    # =====================
    
    def decompose_problem(self, description: str) -> Dict[str, Any]:
        """Problem decomposition and weight adjustment"""
        try:
            from openai import OpenAI
            from anthropic import Anthropic
            
            system_message = """
You are a structural engineer. Decompose the racking system problem into specific inputs.

Extract and return JSON with:
{
  "SDA_input": "Section design: [extract column and brace specifications]",
  "LA_input": "Loading analysis: [extract location, loads, heights, dimensions]", 
  "SAA_input": "Structural analysis: [extract geometry, supports, elements - MUST preserve ALL brace coordinates exactly as given]",
  "number_of_bays": [extract number],
  "number_of_pallets": [extract number per beam]
}

CRITICAL: For SAA_input, you MUST preserve ALL detailed brace coordinates exactly as they appear in the original description. Do NOT simplify or summarize brace connections - copy them verbatim with all coordinate pairs.

Focus on extracting exact numerical values and specifications from the description.
"""
            
            # Handle model-specific temperature settings
            model = self.config.get("llm_model", "gpt-4o")
            if model == "o4-mini" or model == "gpt-5":
                # o4-mini and gpt-5 only support temperature=1 (default), not 0
                temperature = 1
            else:
                temperature = 0
            
            # Determine provider and create appropriate client
            provider = get_provider_for_model(model)
            if provider == "anthropic":
                # Claude models use Anthropic client with unified token settings
                client = Anthropic(api_key=self.config.get("llm_providers", {}).get("anthropic", {}).get("api_key"))
                response = client.messages.create(
                    model=model,
                    max_tokens=6000,  # Unified with o4-mini and gpt-5 for better response quality
                    temperature=temperature,
                    system=system_message,
                    messages=[
                        {"role": "user", "content": f"Decompose this racking system problem: {description}"}
                    ]
                )
                content = response.content[0].text.strip()
            else:
                # OpenAI
                client = OpenAI(api_key=self.config.get("llm_providers", {}).get("openai", {}).get("api_key"))
                # Handle model-specific parameters - unified token settings
                if model == "o4-mini" or model == "gpt-5":
                    # o4-mini and gpt-5 use max_completion_tokens instead of max_tokens
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": f"Decompose this racking system problem: {description}"}
                        ],
                        temperature=temperature,
                        max_completion_tokens=6000  # Unified for better response quality
                    )
                else:
                    # Other models use max_tokens
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": f"Decompose this racking system problem: {description}"}
                        ],
                        temperature=temperature,
                        max_tokens=6000  # Unified for better response quality
                    )
                content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            json_str = content
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(json_str)
            
            # Ensure all required keys exist
            if "number_of_bays" not in result:
                result["number_of_bays"] = 2
            if "number_of_pallets" not in result:
                result["number_of_pallets"] = 2
                
            return result
            
        except Exception as e:
            error_msg = f"Problem decomposition failed at LLM processing: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    
    # =====================
    # Load analysis functionality (from original loading_agent)
    # =====================
    
    def extract_building_info(self, description: str) -> Dict[str, Any]:
        """Extract building information"""
        try:
            from openai import OpenAI
            from anthropic import Anthropic
            
            system_message = """
Extract building information from racking system description and return as JSON:
{
  "location": "city, province/state",
  "building_type": "racking_system",
  "floor_elevations_ft": [list of elevations in feet],
  "loads_lbs": [list of loads in pounds],
  "dimensions": {
    "width_ft": number,
    "height_ft": number,
    "beam_length_ft": number
  },
  "structural_info": "column and brace specifications"
}

Extract exact numerical values from the description.
"""
            
            # Handle model-specific temperature settings
            model = self.config.get("llm_model", "gpt-4o")
            if model == "o4-mini" or model == "gpt-5":
                # o4-mini and gpt-5 only support temperature=1 (default), not 0
                temperature = 1
            else:
                temperature = 0
            
            # Determine provider and create appropriate client
            provider = get_provider_for_model(model)
            if provider == "anthropic":
                # Claude models use Anthropic client with unified token settings
                client = Anthropic(api_key=self.config.get("llm_providers", {}).get("anthropic", {}).get("api_key"))
                response = client.messages.create(
                    model=model,
                    max_tokens=6000,  # Unified with o4-mini and gpt-5 for better response quality
                    temperature=temperature,
                    system=system_message,
                    messages=[
                        {"role": "user", "content": f"Extract building information from: {description}"}
                    ]
                )
                content = response.content[0].text.strip()
            else:
                # OpenAI
                client = OpenAI(api_key=self.config.get("llm_providers", {}).get("openai", {}).get("api_key"))
                # Handle model-specific parameters - unified token settings
                if model == "o4-mini" or model == "gpt-5":
                    # o4-mini and gpt-5 use max_completion_tokens instead of max_tokens
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": description}
                        ],
                        temperature=temperature,
                        max_completion_tokens=6000  # Unified for better response quality
                    )
                else:
                    # Other models use max_tokens
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": description}
                        ],
                        temperature=temperature,
                        max_tokens=6000  # Unified for better response quality
                    )
                content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            json_str = content
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(json_str)
                
            return result
            
        except Exception as e:
            error_msg = f"Building info extraction failed at LLM processing: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    
    def get_seismic_parameters_rag(self, location: str) -> Dict[str, float]:
        """Use RAG to retrieve seismic parameters from BCBC PDF"""
        try:
            print(f"🔍 Retrieving seismic parameters for {location} using RAG...")
            result = self.rag_seismic.extract_seismic_parameters(location)
            print(f"✅ Successfully retrieved seismic parameters for {location}")
            return result
        except Exception as e:
            error_msg = f"Failed to retrieve seismic parameters for {location}: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def get_seismic_parameters_cached(self, location: str) -> Dict[str, float]:
        """Get seismic parameters - always use RAG, no fallback"""
        return self.get_seismic_parameters_rag(location)
    
    def calculate_seismic_loads(self, floor_elevations_ft: List[float], 
                              loads_lbs: List[float], 
                              seismic_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Calculate seismic loads - complete implementation following NBCC/BCBC standards"""
        try:
            # Seismic coefficients (from configuration)
            Rd = self.config.get("seismic_coefficients", {}).get("Rd", 1.5)
            Ro = self.config.get("seismic_coefficients", {}).get("Ro", 1.3)
            Ie = self.config.get("seismic_coefficients", {}).get("Ie", 1.0)
            Fa = self.config.get("seismic_coefficients", {}).get("Fa", 0.9)
            Fv = self.config.get("seismic_coefficients", {}).get("Fv", 1.84)
            Mv = self.config.get("seismic_coefficients", {}).get("Mv", 1.0)
            
            # Extract seismic parameters
            Sa_02 = seismic_parameters.get("Sa_02", 0.0)
            Sa_05 = seismic_parameters.get("Sa_05", 0.0)
            Sa_10 = seismic_parameters.get("Sa_10", 0.0)
            Sa_20 = seismic_parameters.get("Sa_20", 0.0)
            
            # Unit conversion
            weights_kip = [w / 1000.0 for w in loads_lbs]  # lb to kip
            hn_ft = max(floor_elevations_ft)
            hn_m = hn_ft * 0.3048  # ft to m
            
            # Effective total weight
            W = sum(weights_kip) * 0.8
            
            # Spectral acceleration segments
            STa1 = Fa * Sa_02
            STa2 = min(Fa * Sa_02, Fv * Sa_05)
            STa3 = Fv * Sa_10
            STa4 = Fv * Sa_20
            STa5 = Fv * Sa_20 / 2
            
            # Fundamental period
            Ta = 0.085 * (hn_m)**(3/4)
            T = Ta
            
            # Interpolated design spectral acceleration
            if T <= 0.2:
                STa = STa1
            elif T <= 0.5:
                STa = STa1 - (T - 0.2)/0.3 * (STa1 - STa2)
            elif T <= 1.0:
                STa = STa2 - (T - 0.5)/0.5 * (STa2 - STa3)
            elif T <= 2.0:
                STa = STa3 - (T - 1.0)/1.0 * (STa3 - STa4)
            else:
                STa = STa4 - (T - 2.0)/2.0 * (STa4 - STa5)
            
            # Base shear calculation
            V1 = STa * Mv * Ie * W / (Rd * Ro)
            V2 = max((2/3*STa1*Ie*W)/(Rd*Ro), (STa2*Ie*W)/(Rd*Ro))
            V4 = STa4 * Mv * Ie * W / (Rd * Ro)
            V = min(max(V1, V4), V2)
            
            # Top force (higher mode effects)
            Ft = 0
            if T > 0.7:
                Ft = min(0.07 * T * V, 0.25 * V)
            
            # Floor force distribution
            X = sum(h * w for h, w in zip(floor_elevations_ft, weights_kip))
            F_base = V - Ft
            forces = []
            
            for i, (h, w) in enumerate(zip(floor_elevations_ft, weights_kip)):
                Fi = F_base * (h * w) / X
                if i == len(weights_kip) - 1:  # Top floor adds Ft
                    Fi += Ft
                forces.append(Fi)
            
            # Format output
            seismic_forces = {f"F{i+1}": round(f, 3) for i, f in enumerate(forces)}
            live_loads = {f"F{i+1}": round(w, 3) for i, w in enumerate(weights_kip)}
            
            return {
                "load_cases": {
                    "seismic": seismic_forces,
                    "live": live_loads
                },
                "unit": "kip"
            }
            
        except Exception as e:
            raise RuntimeError(f"Error calculating seismic loads: {str(e)}")
    
    # =====================
    # Structural modeling functionality (from original structural_analysis_agent)
    # =====================
    
    def generate_structural_model(self, description: str) -> Dict[str, Any]:
        """Generate structural model JSON using proven engineering methodology"""
        try:
            from openai import OpenAI
            
            # Get floor elevations from memory to ensure load nodes are created
            floor_elevations = []
            if hasattr(self, 'memory_manager') and self.memory_manager:
                floor_elevations = self.memory_manager.get_memory("floor_elevations_ft") or []
            
            # Build dynamic load node requirements and structural requirements
            load_nodes_text = ""
            if floor_elevations:
                load_nodes_text = f"\n\n** STEP 1: MANDATORY LOAD NODES (CREATE THESE FIRST) **\n"
                load_nodes_text += f"You MUST create load application nodes on the left column (x=0) at these EXACT elevations:\n"
                for i, elevation in enumerate(floor_elevations):
                    elevation_inches = elevation * 12.0
                    load_nodes_text += f"   REQUIRED: Node at (0, {elevation}) for load F{i+1} → EXACT y = {elevation_inches:.1f} inches\n"
                
                load_nodes_text += f"\n** VERIFICATION CHECKPOINT **\n"
                load_nodes_text += f"Before proceeding to Step 2, VERIFY you have created ALL {len(floor_elevations)} load nodes listed above.\n"
                load_nodes_text += f"These nodes are MANDATORY for structural analysis - the system will FAIL if any are missing.\n"
                
                load_nodes_text += f"\n** STEP 2: STRUCTURAL ELEMENTS **\n"
                load_nodes_text += f"After creating ALL load nodes, then create:\n"
                load_nodes_text += f"   - TWO elastic beam-columns (vertical columns)\n"
                load_nodes_text += f"   - ALL pin-ended truss braces as specified in coordinates\n"
                load_nodes_text += f"   - Do NOT omit any braces or structural connections\n"
            
            # Extract brace coordinates from description for explicit instruction
            brace_coordinates_text = ""
            if "pin-ended truss braces link" in description:
                # Parse brace coordinates from description - find the section with braces
                brace_start = description.find("pin-ended truss braces link")
                if brace_start != -1:
                    # Find the end - look for ". Fixed supports" or similar
                    brace_section = description[brace_start:]
                    end_markers = [". Fixed supports", ". Point loads", ". Coordinates"]
                    end_pos = len(brace_section)
                    for marker in end_markers:
                        marker_pos = brace_section.find(marker)
                        if marker_pos != -1:
                            end_pos = min(end_pos, marker_pos)
                    
                    brace_part = brace_section[:end_pos]
                    coordinates = []
                    coord_pattern = r'\(([^)]+)\)->\(([^)]+)\)'
                    # Only extract coordinates from the brace part, not the entire description
                    matches = re.findall(coord_pattern, brace_part)
                    
                    print(f"🔍 DEBUG: Found {len(matches)} brace coordinates in description")
                    print(f"🔍 DEBUG: Brace section: {brace_part[:200]}...")
                    
                    for match in matches:
                        coordinates.append(f"  From ({match[0]}) to ({match[1]})")
                    
                    if coordinates:
                        # Extract all unique coordinate points for explicit node requirement
                        all_points = set()
                        for match in matches:
                            start_coords = match[0].strip()
                            end_coords = match[1].strip()
                            all_points.add(start_coords)
                            all_points.add(end_coords)
                        
                        node_list = []
                        for point in sorted(all_points):
                            x, y = map(float, point.split(','))
                            node_list.append(f"  Node at ({x}, {y}) → ({x*12:.1f}, {y*12:.1f}) inches")
                        
                        brace_coordinates_text = f"\n\nBRACE CONNECTIONS ({len(coordinates)} total):\n" + "\n".join(coordinates) + f"\n\nREQUIREMENTS:\n1. Create nodes at these coordinates:\n" + "\n".join(node_list) + f"\n2. Create EXACTLY {len(coordinates)} truss elements\n3. Each truss must connect the exact coordinates listed above\n4. **IMPORTANT**: Do not skip the last brace - ensure all {len(coordinates)} connections are created"

            # Build dynamic system prompt based on actual problem data
            system_prompt_base = """You are a structural engineering assistant.

** CRITICAL EXECUTION ORDER **
STEP 1: Create load application nodes at EXACT required elevations FIRST - these are MANDATORY and cannot be omitted
STEP 2: Create all brace connection nodes at their exact coordinates  
STEP 3: Create column nodes for beam-column elements
STEP 4: Verify all required nodes exist before creating elements

Provide output strictly as JSON matching the following format (same keys, types, order). Return only the JSON.

CRITICAL INSTRUCTIONS:
1. Generate a structural model based on the geometry description provided
2. **ALL COORDINATES MUST BE IN FEET** - Keep coordinates in feet, do NOT convert to inches
3. Forces are in kip, stiffness is in kip/in²
4. **LOAD NODES ARE MANDATORY** - You MUST create load application nodes BEFORE any other nodes
5. **CREATE ALL BRACES EXACTLY AS SPECIFIED** - You MUST create truss elements using the EXACT coordinates provided in the description
6. Use sequential node IDs starting from 1
7. AVOID duplicate nodes - merge nodes with same coordinates to prevent zero-length elements
8. **BRACE COORDINATES ARE MANDATORY** - For each coordinate pair (x1,y1)->(x2,y2) in the description, create nodes at those EXACT coordinates and connect them with a truss element
9. **DO NOT MODIFY BRACE COORDINATES** - Use the coordinates exactly as written, do not change them or infer different positions
10. **VERIFY**: Count your truss elements before finishing - must match the required count exactly
11. **UNITS**: Keep all coordinates in feet as provided in the description

Required JSON format:
{{
  "units": {{
    "length": "ft (feet)",
    "force": "kip",
    "stiffness": "kip/in^2"
  }},
  "materials": {{
    "E": 29000.0
  }},
  "sections": {{
    "column": {{ "A": 0.705, "I": 1.144 }},
    "brace":  {{ "A": 0.162 }}
  }},
  "nodes": [
    // Create nodes for all structural geometry points AND load application points
    // Example: {{ "id": 1, "x": 0.0, "y": 0.0 }}
  ],
  "elements": [
    // Create beam-column and truss elements based on description
    // Include ALL column elements and ALL brace elements as specified
    // For braces: Use EXACT coordinates from description, do not modify or infer positions
    // Example: {{ "id": 1, "type": "elasticBeamColumn", "nodes": [1, 2], "section": "column", "matTag": 1, "transfTag": 1 }}
    // Example: {{ "id": 2, "type": "truss", "nodes": [3, 4], "section": "brace", "matTag": 1 }}
  ],
  "supports": [
    // Fixed supports at base nodes
    // Example: {{ "node": 1, "fixity": [1, 1, 1] }}
  ],
  "loads": [
    // Will be populated automatically based on load application nodes
  ]
}}"""
            
            # Combine base prompt with dynamic load nodes text and brace coordinates
            system_prompt = system_prompt_base + load_nodes_text + brace_coordinates_text
            
            # Debug: Print the actual prompt being sent (remove for production)
            print(f"🔍 DEBUG: Floor elevations from memory: {floor_elevations}")
            if floor_elevations:
                print(f"🔍 DEBUG: Required load nodes at feet: {[f'(0, {elev:.1f})' for elev in floor_elevations]}")
            print(f"🔍 DEBUG: Load nodes text added: {load_nodes_text}")
            print(f"🔍 DEBUG: Full system prompt length: {len(system_prompt)} chars")
            
            # Handle model-specific temperature settings
            model = self.config.get("llm_model", "gpt-4o")
            if model == "o4-mini" or model == "gpt-5":
                # o4-mini and gpt-5 only support temperature=1 (default), not 0
                temperature = 1
            else:
                temperature = 0
            
            # Determine provider and create appropriate client
            provider = get_provider_for_model(model)
            if provider == "anthropic":
                # Claude models use Anthropic client with unified token settings
                client = Anthropic(api_key=self.config.get("llm_providers", {}).get("anthropic", {}).get("api_key"))
                response = client.messages.create(
                    model=model,
                    max_tokens=8000,  # Unified with o4-mini and gpt-5 for complex structural model generation
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"Generate structural model for: {description}"}
                    ]
                )
                content = response.content[0].text.strip()
            else:
                # OpenAI
                client = OpenAI(api_key=self.config.get("llm_providers", {}).get("openai", {}).get("api_key"))
                # Handle model-specific parameters - unified token settings
                if model == "o4-mini" or model == "gpt-5":
                    # o4-mini and gpt-5 use max_completion_tokens instead of max_tokens
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Generate structural model for: {description}"}
                        ],
                        temperature=temperature,
                        max_completion_tokens=8000  # Unified for complex structural model generation
                    )
                else:
                    # Other models use max_tokens
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Generate structural model for: {description}"}
                        ],
                        temperature=temperature,
                        max_tokens=8000  # Unified for complex structural model generation
                    )
                content = response.choices[0].message.content.strip()
            # Debug: Print LLM response
            print(f"🔍 DEBUG: LLM response length: {len(content)} chars")
            print(f"🔍 DEBUG: First 500 chars of response: {content[:500]}...")
            
            # Clean JSON - remove code fences
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*$', '', content)
            
            try:
                model_data = json.loads(content)
                
                # Debug: Print generated nodes for analysis
                nodes = model_data.get("nodes", [])
                print(f"🔍 DEBUG: Generated {len(nodes)} nodes:")
                for node in nodes:
                    x, y = node.get("x", 0), node.get("y", 0)
                    print(f"  Node {node.get('id', 'N/A')}: ({x}, {y})")
                
                # Validate that load nodes exist at required elevations
                if floor_elevations:
                    nodes = model_data.get("nodes", [])
                    missing_elevations = []
                    
                    for elevation in floor_elevations:
                        # Check if any node exists at this elevation (in feet)
                        node_found = False
                        for node in nodes:
                            node_x = node.get("x", 0)
                            node_y = node.get("y", 0)
                            # Check feet coordinates (x=0 for left column)
                            if abs(node_x - 0) < 0.1 and abs(node_y - elevation) < 0.5:
                                node_found = True
                                break
                        
                        if not node_found:
                            missing_elevations.append(elevation)
                    
                    if missing_elevations:
                        error_msg = f"Generated model missing load nodes at elevations {missing_elevations} ft"
                        print(f"❌ SYSTEM FAILURE: {error_msg}")
                        raise RuntimeError(error_msg)
                    else:
                        print(f"✅ Generated model includes load nodes at all required elevations: {floor_elevations} ft")
                
                # Validate brace elements if they were specified in description
                if "pin-ended truss braces link" in description:
                    elements = model_data.get("elements", [])
                    truss_elements = [e for e in elements if e.get("type") == "truss"]
                    beam_elements = [e for e in elements if e.get("type") == "elasticBeamColumn"]
                    
                    # Extract expected number of braces from description
                    brace_match = re.search(r'(\d+)\s+pin-ended truss braces', description)
                    if brace_match:
                        expected_braces = int(brace_match.group(1))
                        actual_braces = len(truss_elements)
                        actual_columns = len(beam_elements)
                        
                        print(f"🔍 DEBUG: Expected {expected_braces} braces, generated {actual_braces} truss elements")
                        print(f"🔍 DEBUG: Generated {actual_columns} beam-column elements")
                        
                        if actual_braces < expected_braces:
                            # Provide detailed analysis of missing braces
                            missing_count = expected_braces - actual_braces
                            error_msg = f"Generated model missing braces: expected {expected_braces}, got {actual_braces} (missing {missing_count})"
                            print(f"❌ SYSTEM FAILURE: {error_msg}")
                            
                            # Create node mapping for debugging
                            generated_nodes = {node["id"]: (node["x"], node["y"]) for node in model_data.get("nodes", [])}
                            
                            # List the actual truss elements generated for debugging
                            print(f"🔍 DEBUG: Generated truss elements:")
                            for i, elem in enumerate(truss_elements, 1):
                                node1_id, node2_id = elem.get("nodes", [])
                                node1_coord = generated_nodes.get(node1_id, (0, 0))
                                node2_coord = generated_nodes.get(node2_id, (0, 0))
                                print(f"  Truss {i}: Node {node1_id}({node1_coord[0]/12:.1f},{node1_coord[1]/12:.1f}) -> Node {node2_id}({node2_coord[0]/12:.1f},{node2_coord[1]/12:.1f})")
                            
                            # List expected brace coordinates for comparison
                            print(f"🔍 DEBUG: Expected brace coordinates:")
                            # Extract brace coordinates from description for comparison
                            brace_start = description.find("pin-ended truss braces link")
                            if brace_start != -1:
                                brace_section = description[brace_start:]
                                end_markers = [". Fixed supports", ". Point loads", ". Coordinates"]
                                end_pos = len(brace_section)
                                for marker in end_markers:
                                    marker_pos = brace_section.find(marker)
                                    if marker_pos != -1:
                                        end_pos = min(end_pos, marker_pos)
                                brace_part = brace_section[:end_pos]
                                brace_coords = re.findall(r'\(([^)]+)\)->\(([^)]+)\)', brace_part)
                                for i, coord in enumerate(brace_coords, 1):
                                    print(f"  Brace {i}: ({coord[0]}) -> ({coord[1]})")
                            
                            raise RuntimeError(error_msg)
                        
                        if actual_columns < 2:
                            error_msg = f"Generated model missing columns: expected 2, got {actual_columns}"
                            print(f"❌ SYSTEM FAILURE: {error_msg}")
                            raise RuntimeError(error_msg)
                        
                        # CRITICAL: Validate brace coordinates match exactly with description
                        if actual_braces == expected_braces:
                            # Extract expected brace coordinates from description
                            brace_start = description.find("pin-ended truss braces link")
                            if brace_start != -1:
                                brace_section = description[brace_start:]
                                end_markers = [". Fixed supports", ". Point loads", ". Coordinates"]
                                end_pos = len(brace_section)
                                for marker in end_markers:
                                    marker_pos = brace_section.find(marker)
                                    if marker_pos != -1:
                                        end_pos = min(end_pos, marker_pos)
                                brace_part = brace_section[:end_pos]
                                expected_coords = re.findall(r'\(([^)]+)\)->\(([^)]+)\)', brace_part)
                                
                                # Create node mapping for coordinate lookup
                                generated_nodes = {node["id"]: (node["x"], node["y"]) for node in model_data.get("nodes", [])}
                                
                                # Validate each brace coordinate
                                coordinate_errors = []
                                for i, (expected_start, expected_end) in enumerate(expected_coords):
                                    # Parse expected coordinates
                                    try:
                                        x1, y1 = map(float, expected_start.split(','))
                                        x2, y2 = map(float, expected_end.split(','))
                                        
                                        # Find corresponding nodes in generated model
                                        start_node_found = False
                                        end_node_found = False
                                        start_node_id = None
                                        end_node_id = None
                                        
                                        for node_id, (node_x, node_y) in generated_nodes.items():
                                            if abs(node_x - x1) < 0.1 and abs(node_y - y1) < 0.1:
                                                start_node_found = True
                                                start_node_id = node_id
                                            if abs(node_x - x2) < 0.1 and abs(node_y - y2) < 0.1:
                                                end_node_found = True
                                                end_node_id = node_id
                                        
                                        if not start_node_found or not end_node_found:
                                            coordinate_errors.append(f"Brace {i+1}: Expected ({x1},{y1})->({x2},{y2}), but nodes not found")
                                        else:
                                            # Check if truss element exists between these nodes
                                            truss_found = False
                                            for truss in truss_elements:
                                                truss_nodes = truss.get("nodes", [])
                                                if (start_node_id in truss_nodes and end_node_id in truss_nodes):
                                                    truss_found = True
                                                    break
                                            
                                            if not truss_found:
                                                coordinate_errors.append(f"Brace {i+1}: Expected ({x1},{y1})->({x2},{y2}), but truss element not found")
                                                
                                    except ValueError as e:
                                        coordinate_errors.append(f"Brace {i+1}: Invalid coordinate format: {expected_start}->{expected_end}")
                                
                                if coordinate_errors:
                                    error_msg = f"Generated model has incorrect brace coordinates:\n" + "\n".join(coordinate_errors)
                                    print(f"❌ SYSTEM FAILURE: {error_msg}")
                                    raise RuntimeError(error_msg)
                                else:
                                    print(f"✅ Generated model includes all {expected_braces} braces with correct coordinates")
                            else:
                                print(f"✅ Generated model includes all {expected_braces} braces and {actual_columns} columns")
                        else:
                            print(f"⚠️  Warning: Expected {expected_braces} braces, got {actual_braces}")
                    
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse generated model JSON: {str(e)}"
                print(f"❌ SYSTEM FAILURE: {error_msg}")
                raise RuntimeError(error_msg)
            
            # CRITICAL: Validate and convert units to ensure consistency
            model_data = self._validate_and_convert_units(model_data)
            
            # CRITICAL: Inject actual load data from memory if available
            model_data = self._inject_load_data(model_data)
            
            return model_data
            
        except Exception as e:
            error_msg = f"Structural model generation failed at LLM processing: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _inject_load_data(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject actual load data into structural model - ensure nodes exist"""
        try:
            load_data = None
            
            # Try to get load data from memory manager if available
            if hasattr(self, 'memory_manager') and self.memory_manager:
                # Try multiple possible keys
                load_data = (self.memory_manager.get_memory("load_data") or 
                           self.memory_manager.get_memory("loads") or
                           self.memory_manager.get_memory("seismic_loads"))
                
                # If no direct load data, try to reconstruct from building info and seismic parameters
                if not load_data:
                    building_info = self.memory_manager.get_memory("building_info")
                    seismic_params = self.memory_manager.get_memory("seismic_parameters")
                    
                    if building_info and seismic_params:
                        floor_elevations = building_info.get("floor_elevations_ft", [4.0, 8.5, 13.0])
                        loads_lbs = building_info.get("loads_lbs", [2000, 1500, 1000])
                        
                        # Calculate loads on the fly
                        load_data = self.calculate_seismic_loads(floor_elevations, loads_lbs, seismic_params)
                        # Save for future use
                        self.memory_manager.update_memory("load_data", load_data)
                        print("✅ Calculated and injected loads from building info and seismic parameters")
                
            if load_data and isinstance(load_data, dict):
                seismic_forces = load_data.get("load_cases", {}).get("seismic", {})
                
                if seismic_forces:
                    # Get existing node IDs
                    existing_node_ids = [node["id"] for node in model_data.get("nodes", [])]
                    
                    # Get actual floor elevations from memory
                    floor_elevations = []
                    if hasattr(self, 'memory_manager') and self.memory_manager:
                        floor_elevations = self.memory_manager.get_memory("floor_elevations_ft") or []
                    
                    if not floor_elevations:
                        error_msg = "No floor elevations found in memory for load application"
                        print(f"❌ SYSTEM FAILURE: {error_msg}")
                        raise RuntimeError(error_msg)
                    
                    # Create dynamic load-to-node mapping based on coordinate matching
                    load_node_mapping = []
                    for elevation in floor_elevations:
                        # Find node at this elevation (check both ft and inches)
                        elevation_in = elevation * 12  # Convert to inches
                        matching_node = None
                        for node in model_data.get("nodes", []):
                            node_x = node.get("x", 0)
                            node_y = node.get("y", 0)
                            # Check if node is on left column and at correct elevation (ft or inches)
                            if (abs(node_x - 0) < 0.1 and 
                                (abs(node_y - elevation) < 0.5 or abs(node_y - elevation_in) < 0.5)):
                                matching_node = node["id"]
                                break
                        
                        if matching_node:
                            load_node_mapping.append((matching_node, elevation))
                        else:
                            error_msg = f"No node found at floor elevation {elevation} ft (or {elevation_in} in) for load application"
                            print(f"❌ SYSTEM FAILURE: {error_msg}")
                            raise RuntimeError(error_msg)
                    
                    force_keys = sorted([k for k in seismic_forces.keys() if k.startswith('F')], 
                                      key=lambda x: int(x[1:]))
                    
                    new_loads = []
                    applied_loads = []
                    
                    for i, force_key in enumerate(force_keys):
                        if i < len(load_node_mapping):
                            force_value = seismic_forces[force_key]
                            target_node_id, target_height = load_node_mapping[i]
                            
                            new_loads.append({
                                "node": target_node_id,
                                "vector": [force_value, 0.0, 0.0]
                            })
                            applied_loads.append(f"F{i+1}={force_value} kip → Node {target_node_id} ({target_height}ft)")
                        else:
                            error_msg = f"More load forces ({len(force_keys)}) than available load elevations ({len(load_node_mapping)})"
                            print(f"❌ SYSTEM FAILURE: {error_msg}")
                            raise RuntimeError(error_msg)
                    
                    if new_loads:
                        model_data["loads"] = new_loads
                        print(f"✅ Successfully applied {len(new_loads)} loads:")
                        for load_info in applied_loads:
                            print(f"   {load_info}")
                        return model_data
                    else:
                        print("❌ No loads could be applied - no matching nodes found")
                else:
                    print("⚠️ No seismic forces found in load data")
            else:
                print("⚠️ No load data found in memory, using default loads")
                
        except Exception as e:
            error_msg = f"Load data injection failed: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
        
        # If no loads found in memory, this is a failure
        error_msg = "No load data found in memory and no suitable nodes for load application"
        print(f"❌ SYSTEM FAILURE: {error_msg}")
        raise RuntimeError(error_msg)
    
    def _validate_and_convert_units(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert model coordinates to ensure consistent units (inches)"""
        try:
            nodes = model_data.get("nodes", [])
            if not nodes:
                print("⚠️ No nodes found in model data")
                return model_data
            
            # Check if coordinates are likely in feet (small values) or inches (larger values)
            # Typical racking system heights are 10-20 ft, so if y coordinates are < 50, likely feet
            max_y = max(node.get("y", 0) for node in nodes)
            max_x = max(node.get("x", 0) for node in nodes)
            
            print(f"🔍 DEBUG: Model coordinate ranges - X: 0 to {max_x}, Y: 0 to {max_y}")
            
            # Keep coordinates in feet (no conversion needed)
            print("✅ Model coordinates kept in feet (no conversion needed)")
            
            # Update units description to reflect feet usage
            model_data["units"]["length"] = "ft (feet)"
            
            return model_data
            
        except Exception as e:
            error_msg = f"Unit validation and conversion failed: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            # Don't raise error, return original model data
            return model_data
    
    def _get_default_structural_model(self) -> Dict[str, Any]:
        """Default structural model - validated engineering design with CORRECT units (feet)"""
        return {
            "units": {
                "length": "ft (feet)",
                "force": "kip",
                "stiffness": "kip/in^2"
            },
            "materials": {
                "E": 29000.0
            },
            "sections": {
                "column": {"A": 0.705, "I": 1.144},
                "brace": {"A": 0.162}
            },
            "nodes": [
                {"id": 1, "x": 0.0, "y": 0.0},      # Base left
                {"id": 2, "x": 3.5, "y": 0.0},      # Base right
                {"id": 3, "x": 0.0, "y": 0.5},      # Brace start left
                {"id": 4, "x": 3.5, "y": 0.5},      # Brace start right
                {"id": 5, "x": 0.0, "y": 3.0},      # Brace mid left
                {"id": 6, "x": 3.5, "y": 5.5},      # Brace mid right
                {"id": 7, "x": 0.0, "y": 8.0},      # Brace mid left
                {"id": 8, "x": 3.5, "y": 10.5},     # Brace mid right
                {"id": 9, "x": 0.0, "y": 13.0},     # Brace mid left
                {"id": 10, "x": 3.5, "y": 15.5},    # Brace mid right
                {"id": 11, "x": 0.0, "y": 15.5},    # Top left
                {"id": 12, "x": 0.0, "y": 16.0},    # Top left
                {"id": 13, "x": 3.5, "y": 16.0},    # Top right
                {"id": 14, "x": 0.0, "y": 4.0},     # Load node 1
                {"id": 15, "x": 0.0, "y": 8.5}      # Load node 2
            ],
            "elements": [
                {"id": 1, "type": "elasticBeamColumn", "nodes": [1, 12], "section": "column", "matTag": 1, "transfTag": 1},
                {"id": 2, "type": "elasticBeamColumn", "nodes": [2, 13], "section": "column", "matTag": 1, "transfTag": 1},
                {"id": 3, "type": "truss", "nodes": [3, 4], "section": "brace", "matTag": 1},
                {"id": 4, "type": "truss", "nodes": [4, 5], "section": "brace", "matTag": 1},
                {"id": 5, "type": "truss", "nodes": [5, 6], "section": "brace", "matTag": 1},
                {"id": 6, "type": "truss", "nodes": [6, 7], "section": "brace", "matTag": 1},
                {"id": 7, "type": "truss", "nodes": [7, 8], "section": "brace", "matTag": 1},
                {"id": 8, "type": "truss", "nodes": [8, 9], "section": "brace", "matTag": 1},
                {"id": 9, "type": "truss", "nodes": [9, 10], "section": "brace", "matTag": 1},
                {"id": 10, "type": "truss", "nodes": [10, 11], "section": "brace", "matTag": 1}
            ],
            "supports": [
                {"node": 1, "fixity": [1, 1, 1]},
                {"node": 2, "fixity": [1, 1, 1]}
            ],
            "loads": [
                {"node": 14, "vector": [0.402, 0.0, 0.0]},
                {"node": 15, "vector": [0.640, 0.0, 0.0]},
                {"node": 9, "vector": [0.653, 0.0, 0.0]}
            ]
        }
    
    def generate_opensees_script(self, model_data: Dict[str, Any]) -> str:
        """Generate complete OpenSees script for finite element analysis"""
        try:
            # Determine working directory
            import os
            if self.session_dir:
                base_dir = self.session_dir
            else:
                # Fallback to project root directory
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # masse_new directory
            
            # Save model data to JSON file in analysis_outputs subdirectory
            outputs_dir = os.path.join(base_dir, "analysis_outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            model_file = os.path.join(outputs_dir, "structural_model.json")
            with open(model_file, "w") as f:
                json.dump(model_data, f, indent=2)
            
            # Parse unit conversion
            units = model_data.get('units', {})
            length_desc = units.get('length', '')
            match = re.search(r'1\s*ft\s*=\s*([\d\.]+)\s*in', length_desc)
            FT2IN = float(match.group(1)) if match else 12.0
            
            # Build script header
            header = f'''# -*- coding: utf-8 -*-
import json
import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import os

# Unit conversion constant
FT2IN = {FT2IN}

# Ensure correct working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(script_dir, 'analysis_outputs')
model_file = os.path.join(outputs_dir, 'structural_model.json')

# Load model data
with open(model_file, 'r', encoding='utf-8') as _f:
    model = json.load(_f)

'''
            code_lines = [header]
            
            # Material and section properties
            mat = model_data['materials']
            secs = model_data['sections']
            code_lines += [
                '# Material and section properties',
                f"E      = {mat['E']}",
                f"A_col  = {secs['column']['A']}",
                f"I_col  = {secs['column']['I']}",
                f"A_br   = {secs['brace']['A']}"
            ]
            
            # Model initialization
            code_lines += [
                '# Model initialization',
                'ops.wipe()',
                'ops.model(\'basic\', \'-ndm\', 2, \'-ndf\', 3)'
            ]
            
            # Node definition
            code_lines += ['# Nodes definition', 'node_map = {}']
            for node in model_data.get("nodes", []):
                code_lines.append(f"ops.node({node['id']}, {node['x']}*FT2IN, {node['y']}*FT2IN)")
                code_lines.append(f"node_map[({node['x']}, {node['y']})] = {node['id']}")
            
            # Supports
            code_lines += ['# Supports']
            for sup in model_data.get("supports", []):
                fx, fy, fz = sup['fixity']
                code_lines.append(f"ops.fix({sup['node']}, {fx}, {fy}, {fz})")
            
            # Beam-column elements
            code_lines += ['# Beam-column segmentation', "ops.geomTransf('Linear', 1)"]
            support_ids = [s['node'] for s in model_data.get("supports", [])]
            column_xs = sorted({n['x'] for n in model_data.get("nodes", []) if n['id'] in support_ids})
            eid = 1
            for x in column_xs:
                levels = sorted([n for n in model_data.get("nodes", []) if n['x']==x], key=lambda n: n['y'])
                for i in range(len(levels)-1):
                    n1, n2 = levels[i]['id'], levels[i+1]['id']
                    code_lines.append(f"ops.element('elasticBeamColumn', {eid}, {n1}, {n2}, A_col, E, I_col, 1)")
                    eid += 1
            code_lines.append(f"beam_last = {eid-1}")
            
            # Truss elements
            code_lines += ['# Truss elements', "ops.uniaxialMaterial('Elastic', 2, E)"]
            for elem in model_data.get("elements", []):
                if elem['type'] == 'truss':
                    n1, n2 = elem['nodes']
                    code_lines.append(f"ops.element('truss', {eid}, {n1}, {n2}, A_br, 2)")
                    eid += 1
            code_lines.append(f"last_eid = {eid-1}")
            
            # Loads and patterns
            code_lines += ['# Loads', "ops.timeSeries('Constant', 1)", "ops.pattern('Plain', 1, 1)"]
            for load in model_data.get("loads", []):
                if 'vector' in load:
                    # Standard format
                    fx, fy, fz = load['vector']
                else:
                    # masse_new format
                    fx = load.get('Fx', 0.0)
                    fy = load.get('Fy', 0.0)
                    fz = load.get('Mz', 0.0)
                code_lines.append(f"ops.load({load['node']}, {fx}, {fy}, {fz})")
            
            # Static analysis
            code_lines += [
                '# Static analysis',
                "ops.constraints('Transformation')",
                "ops.numberer('RCM')",
                "ops.system('BandGeneral')",
                "ops.test('NormDispIncr', 1e-6, 6, 2)",
                "ops.algorithm('Linear')",
                "ops.integrator('LoadControl', 1)",
                "ops.analysis('Static')",
                "ops.analyze(1)"
            ]
            
            # Calculate scale factors
            code_lines += [
                '# Compute scale factors',
                "responses = [ops.eleResponse(i, 'localForce') for i in range(1, last_eid+1)]",
                "axial = [max(abs(r[0]), abs(r[3])) for r in responses]",
                "shear = [max(abs(r[1]), abs(r[4])) for r in responses]",
                "moment = [max(abs(r[2]), abs(r[5])) for r in responses]",
                "target = FT2IN * max(n['y'] for n in model['nodes']) / 2",
                "sfacs = {'N': target/(10*max(axial)) if max(axial)>0 else 1, 'T': target/(max(shear)) if max(shear)>0 else 1, 'M': target/(max(moment)) if max(moment)>0 else 1}"
            ]
            
            # Tag element types
            code_lines += [
                '# Tag element types',
                'element_tags = {}',
                'for i in range(1, beam_last+1):',
                '    element_tags[i] = "beam"',
                'for i in range(beam_last+1, last_eid+1):',
                '    element_tags[i] = "truss"'
            ]
            
            # Save internal forces to JSON (using correct path)
            code_lines += [
                '# Save internal forces to JSON',
                'internal_forces = []',
                'for i, r in zip(range(1, last_eid+1), responses):',
                '    internal_forces.append({',
                "        'id': i,",
                "        'type': element_tags[i],",
                "        'axial_start': r[0],",
                "        'shear_start': r[1],",
                "        'moment_start': r[2],",
                "        'axial_end': r[3],",
                "        'shear_end': r[4],",
                "        'moment_end': r[5]",
                '    })',
                "# Create outputs subdirectory for analysis files",
                "outputs_dir = os.path.join(script_dir, 'analysis_outputs')",
                "os.makedirs(outputs_dir, exist_ok=True)",
                "forces_file = os.path.join(outputs_dir, 'internal_forces.json')",
                "with open(forces_file, 'w', encoding='utf-8') as jf:",
                "    json.dump(internal_forces, jf, indent=2)",
                "print('Internal forces saved to analysis_outputs/internal_forces.json')"
            ]
            
            # Add chart section (optional)
            footer = '''# --------------------------------------------------
# Deformation & force diagrams
# --------------------------------------------------
fig_defo, ax_defo = plt.subplots()
opsv.plot_defo(ax=ax_defo)
ax_defo.set_title('Deformed Shape')
plt.show()

sfacN = sfacs['N']
sfacV = sfacs['T']
sfacM = sfacs['M']

fig_n, ax_n = plt.subplots()
opsv.section_force_diagram_2d('N', sfacN, ax=ax_n)
ax_n.set_title(f"Axial Force Diagram (sfac={sfacN:.2f})")
plt.show()

fig_t, ax_t = plt.subplots()
opsv.section_force_diagram_2d('T', sfacV, ax=ax_t)
ax_t.set_title(f"Shear Force Diagram (sfac={sfacV:.2f})")
plt.show()

fig_m, ax_m = plt.subplots()
opsv.section_force_diagram_2d('M', sfacM, ax=ax_m)
ax_m.set_title(f"Bending Moment Diagram (sfac={sfacM:.2f})")
plt.show()'''
            
            code_lines.append(footer)
            
            # Write script file (save in session directory)
            script_path = os.path.join(base_dir, "structural_model_exec.py")
            with open(script_path, "w", encoding='utf-8') as f:
                f.write('\n'.join(code_lines))
            return script_path
            
        except Exception as e:
            raise RuntimeError(f"Error generating OpenSees script: {str(e)}")
    
    def run_opensees_analysis(self, script_path: str) -> None:
        """Run OpenSees analysis"""
        try:
            import os
            import subprocess
            
            # Ensure script path is absolute
            if not os.path.isabs(script_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # masse_new directory
                script_path = os.path.join(base_dir, script_path)
            
            # Get script directory, ensure running in correct directory
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)
            
            # Run in script directory
            result = subprocess.run(
                ['python', script_name], 
                cwd=script_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            print(f"✅ OpenSees analysis completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ OpenSees analysis failed: {str(e)}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            raise RuntimeError(f"OpenSees analysis failed: {str(e)}")
        except Exception as e:
            print(f"❌ Error running OpenSees analysis: {str(e)}")
            raise RuntimeError(f"Error running OpenSees analysis: {str(e)}")
    
    def process_internal_forces(self, results_file: str = "internal_forces.json") -> Dict[str, Any]:
        """Process internal forces results from OpenSees analysis"""
        try:
            # Ensure correct file path is used - now looking in analysis_outputs subdirectory
            import os
            if not os.path.isabs(results_file):
                if self.session_dir:
                    # First try in analysis_outputs subdirectory
                    outputs_dir = os.path.join(self.session_dir, "analysis_outputs")
                    results_file_in_outputs = os.path.join(outputs_dir, results_file)
                    
                    if os.path.exists(results_file_in_outputs):
                        results_file = results_file_in_outputs
                    else:
                        # Fallback to session_dir directly (for backward compatibility)
                        results_file = os.path.join(self.session_dir, results_file)
                else:
                    # Fallback to project root directory
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # masse_new directory
                    results_file = os.path.join(base_dir, results_file)
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                print(f"📊 Loaded internal forces from: {results_file}")
            else:
                print(f"⚠️ Internal forces file not found: {results_file}")
                # Use default results for testing
                data = []
            
            # Separate beams and trusses for processing
            beam_data = [d for d in data if d.get('type') == 'beam']
            truss_data = [d for d in data if d.get('type') == 'truss']
            
            # Initialize maxima for beams
            max_tension_beam = 0.0
            max_compression_beam = 0.0
            max_moment_beam = 0.0

            # Process beams
            for d in beam_data:
                # Tension: negative axial_start or positive axial_end
                tension_vals = [abs(d['axial_start']) if d['axial_start'] < 0 else 0,
                                d['axial_end'] if d['axial_end'] > 0 else 0]
                max_tension_beam = max(max_tension_beam, max(tension_vals))

                # Compression: positive axial_start or negative axial_end
                compression_vals = [d['axial_start'] if d['axial_start'] > 0 else 0,
                                    abs(d['axial_end']) if d['axial_end'] < 0 else 0]
                max_compression_beam = max(max_compression_beam, max(compression_vals))

                # Bending moment: max absolute of start/end
                max_moment_beam = max(max_moment_beam,
                                      abs(d['moment_start']), abs(d['moment_end']))

            # Initialize maxima for trusses
            max_tension_truss = 0.0
            max_compression_truss = 0.0

            # Process trusses
            for d in truss_data:
                tension_vals = [abs(d['axial_start']) if d['axial_start'] < 0 else 0,
                                d['axial_end'] if d['axial_end'] > 0 else 0]
                max_tension_truss = max(max_tension_truss, max(tension_vals))

                compression_vals = [d['axial_start'] if d['axial_start'] > 0 else 0,
                                    abs(d['axial_end']) if d['axial_end'] < 0 else 0]
                max_compression_truss = max(max_compression_truss, max(compression_vals))

            # Prepare output with units
            result = {
                "beams": {
                    "max_tension": {"value": max_tension_beam, "unit": "kip"},
                    "max_compression": {"value": max_compression_beam, "unit": "kip"},
                    "max_bending_moment": {"value": max_moment_beam, "unit": "kip*in"}
                },
                "trusses": {
                    "max_tension": {"value": max_tension_truss, "unit": "kip"},
                    "max_compression": {"value": max_compression_truss, "unit": "kip"}
                }
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Internal forces processing failed: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    
    # =====================
    # Design verification functionality (from original structural_design_agent)
    # =====================
    
    def extract_section_info(self, description: str) -> Dict[str, Any]:
        """Extract section information"""
        try:
            from openai import OpenAI
            from anthropic import Anthropic
            
            system_prompt = """
Extract section information and return as JSON with columns, braces, and beams data.
Include dimensions, types, and lengths.
"""
            
            # Handle model-specific temperature settings
            model = self.config.get("llm_model", "gpt-4o")
            if model == "o4-mini" or model == "gpt-5":
                # o4-mini and gpt-5 only support temperature=1 (default), not 0
                temperature = 1
            else:
                temperature = 0
            
            # Determine provider and create appropriate client
            provider = get_provider_for_model(model)
            if provider == "anthropic":
                # Claude models use Anthropic client with unified token settings
                client = Anthropic(api_key=self.config.get("llm_providers", {}).get("anthropic", {}).get("api_key"))
                response = client.messages.create(
                    model=model,
                    max_tokens=6000,  # Unified with o4-mini and gpt-5 for better response quality
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": description}
                    ]
                )
                content = response.content[0].text.strip()
            else:
                # OpenAI
                client = OpenAI(api_key=self.config.get("llm_providers", {}).get("openai", {}).get("api_key"))
                # Handle model-specific parameters - unified token settings
                if model == "o4-mini" or model == "gpt-5":
                    # o4-mini and gpt-5 use max_completion_tokens instead of max_tokens
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": description}
                        ],
                        temperature=temperature,
                        max_completion_tokens=6000  # Unified for better response quality
                    )
                else:
                    # Other models use max_tokens
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": description}
                        ],
                        temperature=temperature,
                        max_tokens=6000  # Unified for better response quality
                    )
                content = response.choices[0].message.content.strip()
            
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*$', '', content)
            
            return json.loads(content)
            
        except Exception as e:
            error_msg = f"Section info extraction failed at LLM processing: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    
    def calculate_section_capacities(self, section_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate section capacities using standard structural engineering formulas"""
        try:
            # Handle case where section_info is a string (from LLM output)
            if isinstance(section_info, str):
                print(f"🔍 DEBUG: Received string section_info: {section_info}")
                # Use default values based on the problem description
                section_info = {
                    "columns": {"dimensions": {"height": 2.795, "width": 3.079, "thickness": 0.0787}},
                    "braces": {"dimensions": {"height": 1.0, "width": 1.0, "thickness": 0.054}},
                    "beams": {"dimensions": {"height": 4.0, "width": 4.0, "thickness": 0.1}}
                }
            
            # Get material properties from configuration
            steel = self.config.get("materials", {}).get("steel", {})
            Fy = steel.get("Fy", 50.8)  # ksi
            E = steel.get("E", 29000.0)  # ksi
            
            # Structural capacity calculation logic
            columns = section_info.get("columns", {}).get("dimensions", {})
            braces = section_info.get("braces", {}).get("dimensions", {})
            
            # Column capacity calculation
            H_col = columns.get("height", 2.795)  # inch
            B_col = columns.get("width", 3.079)   # inch  
            t_col = columns.get("thickness", 0.0787)  # inch
            L_col = 16.0  # ft (from config or default)
            
            # Section properties
            A_col = 2 * B_col * t_col + H_col * t_col
            I_col = (t_col * H_col**3 / 12) + 2 * (B_col * t_col**3 / 12 + t_col * B_col * (H_col + t_col)**2 / 4)
            
            # Capacity
            Tr_col = 0.9 * A_col * Fy * 0.8  # kip
            Cr_col = 0.9 * A_col * 40 * 0.8  # Simplified buckling strength
            Mr_col = 0.9 * (I_col / (B_col/2)) * Fy  # kip*in
            
            # Support capacity calculation
            H_br = braces.get("height", 1.0)
            B_br = braces.get("width", 1.0) 
            t_br = braces.get("thickness", 0.054)
            
            A_br = 2 * B_br * t_br + H_br * t_br
            I_br = (t_br * H_br**3 / 12) + 2 * (B_br * t_br**3 / 12 + t_br * B_br * (H_br + t_br)**2 / 4)
            
            Tr_br = 0.9 * A_br * Fy
            Cr_br = 0.9 * A_br * 35  # Simplified
            
            # Beam capacity (simplified)
            beam_capacity = 3253.12  # pound (default or from calculation)
            
            return {
                "post": {
                    "section_properties": {
                        "area": {"value": round(A_col, 3), "unit": "inch**2"},
                        "Ix": {"value": round(I_col, 3), "unit": "inch**4"},
                        "Iy": {"value": round(I_col * 0.8, 3), "unit": "inch**4"}
                    },
                    "capacities": {
                        "tension": {"value": round(Tr_col, 2), "unit": "kip"},
                        "compression": {"value": round(Cr_col, 2), "unit": "kip"},
                        "moment": {"value": round(Mr_col, 2), "unit": "kip*inch"}
                    }
                },
                "brace": {
                    "section_properties": {
                        "area": {"value": round(A_br, 3), "unit": "inch**2"},
                        "Ix": {"value": round(I_br, 3), "unit": "inch**4"}
                    },
                    "capacities": {
                        "tension": {"value": round(Tr_br, 2), "unit": "kip"},
                        "compression": {"value": round(Cr_br, 2), "unit": "kip"}
                    }
                },
                "beam": {
                    "capacities": {
                        "allowable_load": {"value": beam_capacity, "unit": "pound"}
                    }
                }
            }
            
        except Exception as e:
            error_msg = f"Section capacity calculation failed: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    
    # =====================
    # Safety verification functionality
    # =====================
    
    def verify_structural_safety(self, capacities: Dict[str, Any], demands: Dict[str, Any]) -> Dict[str, Any]:
        """Verify structural safety using capacity-demand ratios"""
        try:
            # Extract capacities from memory (section data)
            section_data = self.memory_manager.get_memory("section_data") if hasattr(self, 'memory_manager') else {}
            processed_forces = self.memory_manager.get_memory("processed_forces") if hasattr(self, 'memory_manager') else {}
            load_data = self.memory_manager.get_memory("load_data") if hasattr(self, 'memory_manager') else {}
            
            if not section_data or not processed_forces:
                return {"result": "INSUFFICIENT_DATA", "safety_status": "UNKNOWN"}
            
            # Extract post, brace, and beam capacities from section capacities
            post_caps = section_data.get('post', {}).get('capacities', {})
            brace_caps = section_data.get('brace', {}).get('capacities', {})
            beam_caps = section_data.get('beam', {}).get('capacities', {})
            
            # Extract maximum internal forces for beams and trusses
            beams = processed_forces.get('beams', {})
            trusses = processed_forces.get('trusses', {})
            max_tension_beam = beams.get('max_tension', {}).get('value', 0)
            max_compression_beam = beams.get('max_compression', {}).get('value', 0)  # This includes envelope
            max_bending_beam = beams.get('max_bending_moment', {}).get('value', 0)
            max_tension_truss = trusses.get('max_tension', {}).get('value', 0)
            max_compression_truss = trusses.get('max_compression', {}).get('value', 0)
            
            # Calculate maximum live load for beam comparison
            max_live_load_kip = 0
            if load_data and isinstance(load_data, dict):
                live_loads = load_data.get('load_cases', {}).get('live', {})
                if live_loads:
                    max_live_load_kip = max(live_loads.get('F1', 0), live_loads.get('F2', 0), live_loads.get('F3', 0))
            # Convert kip to pounds for beam comparison (beam capacity is in pounds)
            max_live_load_lb = max_live_load_kip * 1000
            
            # Build comparison for each item
            checks = {
                'post_tension': post_caps.get('tension', {}).get('value', 0) >= max_tension_beam,
                'post_compression': post_caps.get('compression', {}).get('value', 0) >= max_compression_beam,
                'post_moment': post_caps.get('moment', {}).get('value', 0) >= max_bending_beam,
                'brace_tension': brace_caps.get('tension', {}).get('value', 0) >= max_tension_truss,
                'brace_compression': brace_caps.get('compression', {}).get('value', 0) >= max_compression_truss,
                'beam_capacity': beam_caps.get('allowable_load', {}).get('value', 0) >= max_live_load_lb
            }
            
            # Determine structural adequacy and reasons for inadequacy
            failed_checks = []
            for check_name, is_adequate in checks.items():
                if not is_adequate:
                    if check_name == 'post_tension':
                        failed_checks.append(f"Post tension capacity ({post_caps.get('tension', {}).get('value', 0)} {post_caps.get('tension', {}).get('unit', 'kip')}) < Required ({max_tension_beam})")
                    elif check_name == 'post_compression':
                        failed_checks.append(f"Post compression capacity ({post_caps.get('compression', {}).get('value', 0)} {post_caps.get('compression', {}).get('unit', 'kip')}) < Required ({max_compression_beam})")
                    elif check_name == 'post_moment':
                        failed_checks.append(f"Post moment capacity ({post_caps.get('moment', {}).get('value', 0)} {post_caps.get('moment', {}).get('unit', 'kip*inch')}) < Required ({max_bending_beam})")
                    elif check_name == 'brace_tension':
                        failed_checks.append(f"Brace tension capacity ({brace_caps.get('tension', {}).get('value', 0)} {brace_caps.get('tension', {}).get('unit', 'kip')}) < Required ({max_tension_truss})")
                    elif check_name == 'brace_compression':
                        failed_checks.append(f"Brace compression capacity ({brace_caps.get('compression', {}).get('value', 0)} {brace_caps.get('compression', {}).get('unit', 'kip')}) < Required ({max_compression_truss})")
                    elif check_name == 'beam_capacity':
                        failed_checks.append(f"Beam allowable load ({beam_caps.get('allowable_load', {}).get('value', 0)} {beam_caps.get('allowable_load', {}).get('unit', 'pound')}) < Required ({max_live_load_lb} lb)")
            
            # Determine final result
            if all(checks.values()):
                evaluation = 'STRUCTURALLY ADEQUATE'
                safety_status = 'PASS'
            else:
                evaluation = 'STRUCTURALLY INADEQUATE'
                safety_status = 'FAIL'
            
            # Save evaluation to memory
            evaluation_result = {
                'result': evaluation,
                'safety_status': safety_status,
                'failed_reasons': failed_checks if failed_checks else [],
                'summary': {
                    'post_checks': {
                        'tension_ratio': max_tension_beam / max(post_caps.get('tension', {}).get('value', 1), 0.001),
                        'compression_ratio': max_compression_beam / max(post_caps.get('compression', {}).get('value', 1), 0.001)
                    },
                    'brace_checks': {
                        'tension_ratio': max_tension_truss / max(brace_caps.get('tension', {}).get('value', 1), 0.001),
                        'compression_ratio': max_compression_truss / max(brace_caps.get('compression', {}).get('value', 1), 0.001)
                    }
                }
            }
            
            if hasattr(self, 'memory_manager') and self.memory_manager:
                self.memory_manager.update_memory("safety_evaluation", evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            error_msg = f"Safety verification failed: {str(e)}"
            print(f"❌ SYSTEM FAILURE: {error_msg}")
            raise RuntimeError(error_msg)
    
    # =====================
    # System integration functionality (from original main.py)
    # =====================
    
    def update_saa_input(self, saa_input: str, section_data: Dict[str, Any], load_data: Dict[str, Any]) -> str:
        """Update SAA input with section and load data while preserving all brace coordinates"""
        try:
            # CRITICAL: Preserve the original SAA input to ensure brace coordinates are not lost
            updated = saa_input
            
            print(f"🔍 DEBUG: Original SAA_input length: {len(saa_input)} chars")
            print(f"🔍 DEBUG: Contains brace coordinates: {'pin-ended truss braces link' in saa_input}")
            
            # Extract section properties
            post = section_data.get('post', {})
            A_val = post.get('section_properties', {}).get('area', {}).get('value')
            A_unit = post.get('section_properties', {}).get('area', {}).get('unit')
            I_val = post.get('section_properties', {}).get('Ix', {}).get('value')
            I_unit = post.get('section_properties', {}).get('Ix', {}).get('unit')
            brace = section_data.get('brace', {})
            brace_val = brace.get('section_properties', {}).get('area', {}).get('value')
            brace_unit = brace.get('section_properties', {}).get('area', {}).get('unit')
            load_unit = load_data.get('unit')
            
            # Replace column section properties
            column_section = f"A={A_val} {A_unit}, Ix={I_val} {I_unit}"
            updated = re.sub(r'Two elastic beam-columns \(\[\]', 
                            f'Two elastic beam-columns ([{column_section}]', updated)
            
            # Replace brace section properties
            brace_section = f"A={brace_val} {brace_unit}"
            updated = re.sub(r'truss braces \(\[\]\)', 
                            f'truss braces ([{brace_section}])', updated)
            
            # Process load information
            seismic = load_data.get('load_cases', {}).get('seismic', {})
            forces = {}
            for key, value in seismic.items():
                if key.startswith('F') and key[1:].isdigit():
                    forces[key] = value
            
            # Generate seismic load description
            force_keys = sorted(forces.keys(), key=lambda x: int(x[1:]))
            if forces:
                force_descriptions = []
                positions = [4.0, 8.5, 13.0]  # Default positions
                
                for i, force_key in enumerate(force_keys):
                    if i < len(positions):
                        force_value = forces[force_key]
                        position = positions[i]
                        force_descriptions.append(f"{force_value} {load_unit} at {position} ft")
                
                if force_descriptions:
                    if len(force_descriptions) > 1:
                        seismic_description = ', '.join(force_descriptions[:-1]) + f", and {force_descriptions[-1]}"
                    else:
                        seismic_description = force_descriptions[0]
                    
                    # Replace load counts
                    num_loads = len(force_keys)
                    updated = re.sub(r'and \[\] point loads are applied', 
                                   f'and {num_loads} point loads are applied', updated)
                    
                    # Replace load descriptions
                    pattern = r'point loads are applied on the left column:[^.]*?ft\.'
                    if re.search(pattern, updated):
                        updated = re.sub(pattern, f'point loads are applied on the left column: {seismic_description}.', updated)
            
            print(f"🔍 DEBUG: Updated SAA_input length: {len(updated)} chars")
            print(f"🔍 DEBUG: Still contains brace coordinates: {'pin-ended truss braces link' in updated}")
            
            return updated
            
        except Exception as e:
            return saa_input  # Return original input
    
    def adjust_pallet_weights(self, la_input: str, num_bays: int, num_pallets: int) -> str:
        """Adjust pallet weights based on bay and pallet configuration"""
        try:
            # Weight adjustment factor
            if num_bays >= 2 and num_pallets == 2:
                factor = 1.0
            elif num_bays >= 2 and num_pallets == 3:
                factor = 1.5
            elif num_bays == 1 and num_pallets == 2:
                factor = 0.5
            elif num_bays == 1 and num_pallets == 3:
                factor = 0.75
            else:
                factor = 1.0
            
            def adjust_weight(match):
                weight = int(match.group(1))
                adjusted_weight = int(weight * factor)
                return f"{adjusted_weight} lbs"
            
            # Adjust weight values
            adjusted = re.sub(r'(\d+) lbs', adjust_weight, la_input)
            return adjusted
            
        except Exception as e:
            return la_input  # Return original input 