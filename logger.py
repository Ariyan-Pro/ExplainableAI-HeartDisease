# healthcare_model/logger.py
import logging
import sys
from datetime import datetime

def setup_logger():
    """Simple logging setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'healthcare_model/logs/api_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    
    # Create logs directory
    from pathlib import Path
    Path('healthcare_model/logs').mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info("🎯 Logging system initialized")
    return logger

logger = setup_logger()