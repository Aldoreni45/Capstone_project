import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Formatting pattern
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

def setup_logger(name: str = "app") -> logging.Logger:
    """Configures and returns a multi-level structured logger."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)
    
    # 1. Console Handler (Default to INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Helper to add a rotating file handler for a specific level
    def add_file_handler(filename: str, level: int):
        handler = RotatingFileHandler(
            LOG_DIR / filename,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        
        # Add filter so files only contain logs corresponding to their specific level or higher
        # except debug which catches everything
        if level != logging.DEBUG:
            class LevelFilter(logging.Filter):
                def __init__(self, target_level):
                    super().__init__()
                    self.target_level = target_level
                def filter(self, record):
                    return record.levelno == self.target_level
            handler.addFilter(LevelFilter(level))
            
        logger.addHandler(handler)

    # 2. Add Rotating File Handlers
    add_file_handler("info.log", logging.INFO)
    add_file_handler("warning.log", logging.WARNING)
    add_file_handler("error.log", logging.ERROR)
    add_file_handler("debug.log", logging.DEBUG)
    
    return logger

# Root application logger
app_logger = setup_logger("app_root")
