import logging
import os
import sys
from datetime import datetime
from typing import Optional
from io import StringIO

class PrintCapture:
    """Class for capturing print output"""
    
    def __init__(self, original_stdout, log_file_path):
        self.original_stdout = original_stdout
        self.log_file_path = log_file_path
        self.buffer = StringIO()
        
    def write(self, text):
        # Write to original terminal
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        # Write to log file
        if text.strip():  # Only record non-empty content
            try:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"[{timestamp}] {text}")
                    if not text.endswith('\n'):
                        f.write('\n')
            except Exception as e:
                # If log writing fails, at least show error in terminal
                self.original_stdout.write(f"[LOG ERROR] {e}\n")
    
    def flush(self):
        self.original_stdout.flush()
    
    def isatty(self):
        return self.original_stdout.isatty()

class AnalysisLogger:
    """Structural Analysis Logger - Capture all terminal output"""
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO, session_dir: Optional[str] = None):
        self.base_log_dir = log_dir
        self.session_dir = session_dir  # If provided, use this specific directory
        self.log_level = log_level
        self.logger = None
        self.log_file_path = None
        self.original_stdout = None
        self.print_capture = None
        self._setup_logger()
        self._start_print_capture()
    
    def _setup_logger(self):
        """Setup logger"""
        # Determine the actual log directory
        if self.session_dir:
            # Use the provided session directory
            self.log_dir = self.session_dir
        else:
            # Create a new timestamped session directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(self.base_log_dir, f"session_{timestamp}")
        
        # Create log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Generate log file name
        self.log_file_path = os.path.join(self.log_dir, "analysis.log")
        
        # Create logger
        self.logger = logging.getLogger('StructuralAnalysis')
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create file handler (for structured logging)
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # Set format for structured logs
        formatter = logging.Formatter(
            '[%(asctime)s] [LOGGER] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Only add file handler (no console handler since we're capturing prints)
        self.logger.addHandler(file_handler)
        
        # Write initial log header
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========== MASSE ANALYSIS LOG START ==========\n")
    
    def _start_print_capture(self):
        """Start capturing print output"""
        if sys.stdout != sys.__stdout__:  # If already redirected, restore first
            sys.stdout = sys.__stdout__
        
        self.original_stdout = sys.stdout
        self.print_capture = PrintCapture(self.original_stdout, self.log_file_path)
        sys.stdout = self.print_capture
        
        print("📝 Logger initialized - All terminal output will be saved to log file")
        print(f"📁 Log file: {self.log_file_path}")
    
    def _stop_print_capture(self):
        """Stop capturing print output"""
        if self.original_stdout:
            sys.stdout = self.original_stdout
            print(f"📝 Log capture stopped - Log saved to: {self.log_file_path}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical error message"""
        self.logger.critical(message)
    
    def log_step(self, step_name: str, status: str, details: str = ""):
        """Log analysis step"""
        message = f"📝 {step_name}: {status}"
        if details:
            message += f" - {details}"
        print(message)  # Use print instead of logger so it gets captured
    
    def log_memory_update(self, key: str, data_type: str):
        """Log memory update"""
        print(f"💾 Memory updated: {key} ({data_type})")
    
    def log_agent_chat(self, agent_name: str, message_preview: str, turns: int):
        """Log agent conversation"""
        preview = message_preview[:50] + "..." if len(message_preview) > 50 else message_preview
        print(f"🤖 Chat with {agent_name}: '{preview}' ({turns} turns)")
    
    def log_function_call(self, function_name: str, result_preview: str):
        """Log function call"""
        preview = result_preview[:100] + "..." if len(result_preview) > 100 else result_preview
        print(f"⚙️ Function call: {function_name} -> {preview}")
    
    def log_analysis_start(self, problem_description: str, location: str):
        """Log analysis start"""
        print("=" * 80)
        print("🚀 STRUCTURAL ANALYSIS STARTED")
        print(f"📍 Location: {location}")
        print(f"📋 Problem: {problem_description[:200]}...")
        print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def log_analysis_end(self, final_result: str, duration: float):
        """Log analysis end"""
        print("=" * 80)
        print("🏁 STRUCTURAL ANALYSIS COMPLETED")
        print(f"🎯 Final result: {final_result}")
        print(f"⏱️ Duration: {duration:.2f} seconds")
        print(f"📁 Log saved to: {self.log_file_path}")
        print("=" * 80)
    
    def get_log_file_path(self) -> str:
        """Get log file path"""
        return self.log_file_path
    
    def get_session_dir(self) -> str:
        """Get session directory path"""
        return self.log_dir
    
    def close(self):
        """Close logger"""
        # Write end marker
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ========== MASSE ANALYSIS LOG END ==========\n")
        except Exception:
            pass
        
        # Stop print capture
        self._stop_print_capture()
        
        # Close logger handlers
        if self.logger:
            for handler in self.logger.handlers:
                handler.close()
                self.logger.removeHandler(handler) 