"""
Logging utility for ReViision Pi Test Bench
"""

import logging
import logging.handlers
from pathlib import Path
import colorlog
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/pi_testbench.log",
    max_size: str = "10MB",
    backup_count: int = 5,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the Pi test bench
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        max_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        format_str: Custom format string
    
    Returns:
        Configured logger instance
    """
    
    # Convert max_size string to bytes
    if max_size.endswith('MB'):
        max_bytes = int(max_size[:-2]) * 1024 * 1024
    elif max_size.endswith('KB'):
        max_bytes = int(max_size[:-2]) * 1024
    else:
        max_bytes = int(max_size)
    
    # Default format
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Color format for console
    color_format = (
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s'
    )
    
    # Create logger
    logger = logging.getLogger('reviision_pi')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create log directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_formatter = logging.Formatter(format_str)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        color_format,
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a child logger with the specified name"""
    return logging.getLogger(f'reviision_pi.{name}') 