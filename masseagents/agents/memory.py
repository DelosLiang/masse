import json
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import os


class StructuralMemoryManager:
    """Unified structural engineering memory manager, replacing the original memory.json mechanism"""
    
    def __init__(self, config: Dict[str, Any], session_dir: Optional[str] = None):
        self.config = config
        self.session_dir = session_dir
        self._lock = threading.Lock()
        self.memory = {
            # Input information
            "original_description": None,
            "location": None,
            
            # Decomposed inputs
            "SAA_input": None,  # Structural Analysis Agent input
            "SDA_input": None,  # Structural Design Agent input  
            "LA_input": None,   # Loading Agent input
            "LA_input_adjusted": None,  # Adjusted load input
            "SAA_input_update": None,   # Updated structural analysis input
            
            # Analysis results
            "building_info": None,
            "seismic_parameters": None,
            "section_properties": None,
            "loads": None,
            
            # Engineering calculation results
            "structural_model": None,
            "internal_forces": None,
            "processed_forces": None,
            "capacities": None,
            
            # Final assessment
            "safety_evaluation": None,
            "final_decision": None,
            
            # Metadata
            "number_of_bays": None,
            "number_of_pallets": None,
            "analysis_timestamp": None,
            "problem_id": None,
            
            # Additional keys used by function registry
            "section_info": None,
            "floor_elevations_ft": None,
            "loads_lbs": None,
            "load_data": None,
            "section_data": None,
            "section_capacities": None,
        }
        
    def update_memory(self, key: str, value: Any) -> None:
        """Thread-safe memory update"""
        with self._lock:
            if key in self.memory:
                self.memory[key] = value
            else:
                raise KeyError(f"Memory key '{key}' not recognized")
                
    def get_memory(self, key: str) -> Any:
        """Get memory data"""
        with self._lock:
            return self.memory.get(key)
            
    def get_all_memory(self) -> Dict[str, Any]:
        """Get copy of all memory data"""
        with self._lock:
            return self.memory.copy()
            
    def clear_memory(self) -> None:
        """Clear memory"""
        with self._lock:
            for key in self.memory:
                self.memory[key] = None
                
    def save_to_file(self, filename: str = "analysis_results.json") -> None:
        """Save memory data to file"""
        try:
            # Determine save location
            if not os.path.isabs(filename):
                if self.session_dir:
                    # Save in session directory
                    filename = os.path.join(self.session_dir, filename)
                else:
                    # Fallback to project root directory
                    current_dir = os.path.dirname(__file__)  # agents directory
                    masseagents_dir = os.path.dirname(current_dir)  # masseagents directory
                    masse_new_dir = os.path.dirname(masseagents_dir)  # masse_new directory
                    filename = os.path.join(masse_new_dir, filename)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
            print(f"✅ Memory data saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving memory to file: {e}")
            raise
            
    def load_from_file(self, filepath: str = "memory.json") -> None:
        """Load from JSON file (compatibility support)"""
        try:
            if not Path(filepath).exists():
                print(f"Memory file {filepath} does not exist")
                return
                
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Map compatible format to new memory structure
            if "SAA_input" in data:
                self.update_memory("SAA_input", data["SAA_input"])
            if "SDA_input" in data:
                self.update_memory("SDA_input", data["SDA_input"])
            if "LA_input" in data:
                self.update_memory("LA_input", data["LA_input"])
            if "LA_input_adjusted" in data:
                self.update_memory("LA_input_adjusted", data["LA_input_adjusted"])
            if "SAA_input_update" in data:
                self.update_memory("SAA_input_update", data["SAA_input_update"])
                
            if "number_of_bays" in data:
                self.update_memory("number_of_bays", data["number_of_bays"])
            if "number_of_pallets" in data:
                self.update_memory("number_of_pallets", data["number_of_pallets"])
                
            if "section" in data:
                self.update_memory("section_properties", data["section"])
            if "load" in data:
                self.update_memory("loads", data["load"])
            if "processed_forces" in data:
                self.update_memory("processed_forces", data["processed_forces"])
            if "evaluation" in data:
                self.update_memory("safety_evaluation", data["evaluation"])
                
        except Exception as e:
            print(f"Error loading memory from file: {e}")
            
    def get_summary(self) -> str:
        """Get memory status summary"""
        summary = []
        non_null_keys = [k for k, v in self.memory.items() if v is not None]
        summary.append(f"Memory contains {len(non_null_keys)} non-null entries:")
        for key in non_null_keys:
            summary.append(f"  - {key}")
        return "\n".join(summary)
        
    def has_complete_analysis(self) -> bool:
        """Check if complete analysis results are available"""
        required_keys = [
            "loads", "section_properties", 
            "processed_forces", "safety_evaluation"
        ]
        return all(self.memory.get(key) is not None for key in required_keys) 