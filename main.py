#!/usr/bin/env python3
"""
MASSE (Multi-Agent System for Structural Engineering) - Main Entry Point

Multi-agent structural engineering analysis system based on AutoGen framework

Usage:
python main.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import json
import os
import time
from datetime import datetime
import sys
from io import StringIO
from dotenv import load_dotenv

from masseagents.workflows.structural_workflow import StructuralAnalysisWorkflow
from masseagents.default_config import get_default_config, get_provider_for_model

# Load environment variables
load_dotenv()

class OutputRedirector:
    """Redirect stdout to console or file"""
    def __init__(self, callback=None):
        self.callback = callback
        
    def write(self, string):
        if self.callback:
            self.callback(string)
        
    def flush(self):
        pass

class MASSEInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("MASSE - Multi-Agent System for Structural Engineering")
        self.root.geometry("1400x800")
        
        # Load problem data
        self.problem_data = self.load_problem_data()
        
        # Variables
        self.problem_id = tk.StringVar(value="1")
        self.selected_model = tk.StringVar(value="gpt-4o")
        self.problem_description = tk.StringVar(value="")
        self.is_running = False
        
        # Matplotlib variables
        self.fig = None
        self.canvas = None
        
        # Create UI
        self.create_widgets()
        
    def load_problem_data(self):
        """Load problem descriptions from dataset folder"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(script_dir, "dataset", "problem_descriptions.json")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load problem data: {e}")
            return {}
    
    def create_widgets(self):
        """Create the main UI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏗️ MASSE - Multi-Agent System for Structural Engineering", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Problem ID input
        ttk.Label(control_frame, text="Problem ID (1-100):").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        id_frame = ttk.Frame(control_frame)
        id_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        id_frame.columnconfigure(1, weight=1)
        
        problem_spinbox = ttk.Spinbox(id_frame, from_=1, to=100, width=10, textvariable=self.problem_id)
        problem_spinbox.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        self.load_button = ttk.Button(id_frame, text="Load Problem", command=self.load_problem)
        self.load_button.grid(row=0, column=1, sticky=tk.W)
        
        # Model selection
        ttk.Label(control_frame, text="LLM Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        model_options = [
            "gpt-4o",
            "claude-3-5-sonnet-latest",
            "o4-mini",
            "gpt-5",
            "gpt-3.5-turbo"
        ]
        model_combo = ttk.Combobox(control_frame, textvariable=self.selected_model, 
                                  values=model_options, state="readonly", width=30)
        model_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Problem description display
        ttk.Label(control_frame, text="Problem Description:").grid(row=2, column=0, sticky=(tk.W, tk.N), pady=5)
        
        desc_frame = ttk.Frame(control_frame)
        desc_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        desc_frame.columnconfigure(0, weight=1)
        desc_frame.rowconfigure(0, weight=1)
        control_frame.rowconfigure(2, weight=1)
        
        self.description_text = tk.Text(desc_frame, height=15, wrap=tk.WORD, 
                                       font=("Arial", 9), state=tk.DISABLED)
        desc_scrollbar = ttk.Scrollbar(desc_frame, orient=tk.VERTICAL, command=self.description_text.yview)
        self.description_text.configure(yscrollcommand=desc_scrollbar.set)
        
        self.description_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        desc_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Run button
        self.run_button = ttk.Button(control_frame, text="🚀 Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Right panel - Geometry Visualization
        vis_frame = ttk.LabelFrame(main_frame, text="Geometry Visualization", padding="10")
        vis_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        vis_frame.columnconfigure(0, weight=1)
        vis_frame.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initialize with empty visualization
        self.create_empty_visualization(vis_frame)
        
        # Bind events
        self.problem_id.trace('w', self.on_problem_id_changed)
    
    def create_empty_visualization(self, parent):
        """Create empty visualization placeholder"""
        self.fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Load a problem to see geometry visualization', 
                ha='center', va='center', fontsize=14, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def on_problem_id_changed(self, *args):
        """Handle problem ID change"""
        # Clear description when ID changes
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        self.description_text.config(state=tk.DISABLED)
        
        # Clear visualization
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        self.create_empty_visualization(self.canvas.get_tk_widget().master)
    
    def load_problem(self):
        """Load problem description and create visualization"""
        try:
            pid = self.problem_id.get()
            if pid not in self.problem_data:
                messagebox.showerror("Error", f"Problem {pid} not found in database")
                return
            
            problem = self.problem_data[pid]
            
            # Load description
            description = problem.get("generated_description", "No description available")
            self.description_text.config(state=tk.NORMAL)
            self.description_text.delete(1.0, tk.END)
            self.description_text.insert(1.0, description)
            self.description_text.config(state=tk.DISABLED)
            
            # Create visualization
            self.create_visualization(problem)
            
            self.status_var.set(f"Problem {pid} loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load problem: {e}")
    
    def create_visualization(self, problem):
        """Create geometry visualization from problem data"""
        try:
            # Clear previous plot
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            
            # Extract geometry data
            width = float(problem.get("width", 3.5))
            height = float(problem.get("height", 16))
            floor_elevations = [float(x) for x in problem.get("floor_elevations", [])]
            floor_loads = [float(x) for x in problem.get("floor_loads", [])]
            brace_elements = problem.get("brace_elements", [])
            
            # Calculate bounds
            all_x_coords = [0, width]
            all_y_coords = [0, height]
            
            # Add brace coordinates to bounds
            for brace in brace_elements:
                if len(brace) >= 2:
                    try:
                        from_coord = brace[0].strip('()')
                        to_coord = brace[1].strip('()')
                        from_x, from_y = map(float, from_coord.split(','))
                        to_x, to_y = map(float, to_coord.split(','))
                        all_x_coords.extend([from_x, to_x])
                        all_y_coords.extend([from_y, to_y])
                    except:
                        pass
            
            # Add floor elevations
            all_y_coords.extend(floor_elevations)
            
            # Calculate display bounds with padding
            x_min, x_max = min(all_x_coords), max(all_x_coords)
            y_min, y_max = min(all_y_coords), max(all_y_coords)
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = max(0.5, x_range * 0.15)
            y_padding = max(0.5, y_range * 0.15)
            
            # Create new figure
            self.fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            ax.set_xlabel('Width (ft)')
            ax.set_ylabel('Height (ft)')
            ax.set_title(f'Racking System Visualization - Problem {problem.get("problem_id", "Unknown")}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Draw columns in red
            ax.plot([0, 0], [0, height], color='red', linewidth=4, label='Columns')
            ax.plot([width, width], [0, height], color='red', linewidth=4)
            
            # Draw braces in blue
            for i, brace in enumerate(brace_elements):
                if len(brace) >= 2:
                    try:
                        from_coord = brace[0].strip('()')
                        to_coord = brace[1].strip('()')
                        from_x, from_y = map(float, from_coord.split(','))
                        to_x, to_y = map(float, to_coord.split(','))
                        ax.plot([from_x, to_x], [from_y, to_y], color='blue', linewidth=2, 
                               label='Braces' if i == 0 else '')
                    except:
                        pass
            
            # Draw supports at base
            ax.plot(0, 0, 'ks', markersize=8, label='Fixed Supports')
            ax.plot(width, 0, 'ks', markersize=8)
            
            # Add floor loads
            if floor_loads and floor_elevations and len(floor_loads) == len(floor_elevations):
                center_x = width / 2
                for i, (elev, load) in enumerate(zip(floor_elevations, floor_loads)):
                    ax.plot(center_x, elev, 'go', markersize=8, label='Loads' if i == 0 else '')
                    ax.annotate(f'{int(load)} lbs', (center_x, elev), 
                               xytext=(center_x, elev + 0.3), 
                               fontsize=9, ha='center', va='bottom', fontweight='bold')
            
            # Place legend outside the plot area to avoid covering the structure
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
            
            # Adjust layout to make room for the external legend
            plt.tight_layout()
            plt.subplots_adjust(right=0.8)  # Make space for legend on the right
            
            # Update canvas
            parent = self.canvas.get_tk_widget().master if self.canvas else None
            if parent:
                self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
                self.canvas.draw()
                self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
        except Exception as e:
            print(f"Visualization error: {e}")
            # Show error message in visualization area
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            
            self.fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Visualization Error:\n{str(e)}', 
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            parent = self.canvas.get_tk_widget().master if self.canvas else None
            if parent:
                self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
                self.canvas.draw()
                self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def run_analysis(self):
        """Run the structural analysis in a separate thread"""
        if self.is_running:
            return
        
        # Validate inputs
        pid = self.problem_id.get()
        if pid not in self.problem_data:
            messagebox.showerror("Error", f"Problem {pid} not found in database")
            return
        
        # Check if description is loaded
        description = self.description_text.get(1.0, tk.END).strip()
        if not description or description == "No description available":
            messagebox.showerror("Error", "Please load a problem description first")
            return
        
        # Check if model is available
        model = self.selected_model.get()
        if model in ["qwen3-32B-thinking", "qwen3-32B-non-thinking"]:
            messagebox.showwarning("Warning", f"Model {model} is not yet available. This is a placeholder.")
            return
        
        # Get provider for confirmation dialog
        provider = get_provider_for_model(model)
        
        # Show confirmation dialog
        result = messagebox.askyesno("Confirm Analysis", 
                                   f"Run analysis for Problem {pid} using {model}?\n\n"
                                   f"Provider: {provider.upper()}\n"
                                   f"This will create a new folder: logs/{model}/problem_{pid}")
        if not result:
            return
        
        # Disable run button and show progress
        self.is_running = True
        self.run_button.config(state="disabled", text="Running Analysis...")
        
        # Start analysis in separate thread
        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def _run_analysis_thread(self):
        """Run analysis in separate thread"""
        start_time = time.time()
        
        try:
            # Get problem data
            pid = self.problem_id.get()
            problem_description = self.problem_data[pid]["generated_description"]
            model = self.selected_model.get()
            
            # Update status
            self.status_var.set(f"Running analysis for Problem {pid} using {model}...")
            
            # Create custom config with selected model
            config = get_default_config()
            config["llm_model"] = model
            
            # Ensure model-specific base directory exists and migrate legacy logs
            model_dir = os.path.join("logs", model)
            os.makedirs(model_dir, exist_ok=True)
            try:
                self._migrate_legacy_logs_to_model_folder(model)
            except Exception as _:
                pass

            # Create problem folder first under model directory
            problem_folder = os.path.join(model_dir, f"problem_{pid}")
            self._manage_problem_folder(problem_folder)
            
            # Initialize workflow with specific log directory
            workflow = StructuralAnalysisWorkflow(config, log_dir=problem_folder)
            
            # Run analysis
            result = workflow.run_full_analysis(problem_description, problem_id=pid)
            
            # Calculate runtime
            end_time = time.time()
            runtime_seconds = end_time - start_time
            
            # Add runtime to result
            if isinstance(result, dict):
                result["runtime_info"] = {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(end_time).isoformat(),
                    "duration_seconds": runtime_seconds,
                    "duration_formatted": f"{int(runtime_seconds // 60)}m {int(runtime_seconds % 60)}s"
                }
                result["model_used"] = model
                result["problem_id"] = pid
            
            # Save results
            self._save_results(result, workflow, pid)
            
            # Update status
            self.status_var.set(f"Analysis completed for Problem {pid}")
            
            # Show completion message
            final_result = result.get("final_result", "UNKNOWN") if isinstance(result, dict) else "UNKNOWN"
            runtime_str = result.get("runtime_info", {}).get("duration_formatted", "Unknown") if isinstance(result, dict) else "Unknown"
            
            messagebox.showinfo("Analysis Complete", 
                              f"Analysis completed for Problem {pid}\n\n"
                              f"Result: {final_result}\n"
                              f"Runtime: {runtime_str}\n\n"
                              f"Results saved to: {problem_folder}/")
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.status_var.set("Analysis failed")
            messagebox.showerror("Analysis Error", error_msg)
            
        finally:
            # Re-enable run button
            self.root.after(0, self._enable_run_button)
    
    def _save_results(self, result, workflow, problem_id):
        """Save analysis results to problem-specific folder"""
        try:
            # Use the workflow's logger session directory (already model-aware)
            problem_folder = workflow.logger.get_session_dir()
            
            # Save analysis results
            results_file = os.path.join(problem_folder, "analysis_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Close workflow logger
            workflow.logger.close()
            
            print(f"✅ Results saved to: {problem_folder}")
            
        except Exception as e:
            print(f"Error saving results: {e}")

    def _migrate_legacy_logs_to_model_folder(self, model_name: str):
        """Move existing logs/problem_* folders into logs/<model_name>/ once."""
        base_logs = "logs"
        model_dir = os.path.join(base_logs, model_name)
        if not os.path.isdir(base_logs):
            return
        os.makedirs(model_dir, exist_ok=True)
        for item in os.listdir(base_logs):
            src = os.path.join(base_logs, item)
            if os.path.isdir(src) and item.startswith("problem_"):
                dst = os.path.join(model_dir, item)
                if not os.path.exists(dst):
                    try:
                        import shutil
                        shutil.move(src, dst)
                        print(f"Moved {src} -> {dst}")
                    except Exception as e:
                        print(f"Log migration warning: {e}")
    
    def _manage_problem_folder(self, problem_folder):
        """Manage problem folder with version control"""
        if os.path.exists(problem_folder):
            # Create previous_records folder if it doesn't exist
            previous_folder = os.path.join(problem_folder, "previous_records")
            os.makedirs(previous_folder, exist_ok=True)
            
            # Find next version number
            version = 1
            while os.path.exists(os.path.join(previous_folder, str(version))):
                version += 1
            
            # Move existing files to previous_records
            version_folder = os.path.join(previous_folder, str(version))
            os.makedirs(version_folder, exist_ok=True)
            
            for item in os.listdir(problem_folder):
                if item != "previous_records":
                    src = os.path.join(problem_folder, item)
                    dst = os.path.join(version_folder, item)
                    if os.path.isfile(src):
                        import shutil
                        shutil.move(src, dst)
        else:
            os.makedirs(problem_folder, exist_ok=True)
    
    def _enable_run_button(self):
        """Re-enable the run button"""
        self.is_running = False
        self.run_button.config(state="normal", text="🚀 Run Analysis")

def main():
    """Main program entry point - launches UI"""
    # Check for API keys (optional warnings only)
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Just show informational messages, don't block startup
    if openai_key:
        print("✅ OpenAI API key found")
    else:
        print("ℹ️ OpenAI API key not found (GPT models may not work)")
        
    if anthropic_key:
        print("✅ Anthropic API key found")
    else:
        print("ℹ️ Anthropic API key not found (Claude models may not work)")
    
    print(f"🚀 Starting MASSE interface...")
    
    # Launch the UI interface
    root = tk.Tk()
    app = MASSEInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main() 