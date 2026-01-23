import logging
import os

def get_logger(name, log_filename):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if the logger is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File Handler
        file_h = logging.FileHandler(f"logs/{log_filename}")
        file_h.setFormatter(formatter)
        
        # Console Handler
        stream_h = logging.StreamHandler()
        stream_h.setFormatter(formatter)
        
        logger.addHandler(file_h)
        logger.addHandler(stream_h)
        
    return logger