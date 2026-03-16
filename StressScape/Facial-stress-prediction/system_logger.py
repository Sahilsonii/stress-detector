import logging
import sys
from pathlib import Path
from datetime import datetime
import traceback

class SystemLogger:
    """Comprehensive logging system for all operations"""
    
    def __init__(self, log_name="system", log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{log_name}_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler - captures everything
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler - shows important info
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log session start
        self.logger.info("="*70)
        self.logger.info(f"LOGGING SESSION STARTED: {log_name}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("="*70)
    
    def debug(self, message):
        """Log debug information"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log general information"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning"""
        self.logger.warning(message)
    
    def error(self, message, exc_info=False):
        """Log error with optional exception info"""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message, exc_info=False):
        """Log critical error"""
        self.logger.critical(message, exc_info=exc_info)
    
    def exception(self, message):
        """Log exception with full traceback"""
        self.logger.exception(message)
    
    def section(self, title):
        """Log section header"""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(title)
        self.logger.info("="*70)
    
    def subsection(self, title):
        """Log subsection header"""
        self.logger.info("")
        self.logger.info(f"--- {title} ---")
    
    def metric(self, name, value):
        """Log metric"""
        self.logger.info(f"  {name}: {value}")
    
    def progress(self, current, total, message=""):
        """Log progress"""
        percentage = (current / total) * 100
        self.logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) {message}")
    
    def close(self):
        """Close logging session"""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("LOGGING SESSION ENDED")
        self.logger.info("="*70)
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
    
    def get_log_path(self):
        """Return path to log file"""
        return str(self.log_file)

# Global logger instance
_global_logger = None

def get_logger(log_name="system", log_dir="logs"):
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SystemLogger(log_name, log_dir)
    return _global_logger

def close_logger():
    """Close global logger"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None
