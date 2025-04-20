# auto_retrain.py

import time
import logging
import os
import schedule
from datetime import datetime
from buffer_system import PredictionBuffer
from online_learning import OnlineLearning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='auto_retrain.log'
)
logger = logging.getLogger('auto_retrain')

def check_and_retrain():
    """Check buffer size and retrain if needed"""
    try:
        logger.info("Checking buffer for retraining...")
        buffer = PredictionBuffer()
        current_size = buffer.get_buffer_size()
        
        logger.info(f"Current buffer size: {current_size}")
        if current_size >= buffer.buffer_size:
            logger.info(f"Buffer size ({current_size}) exceeds threshold ({buffer.buffer_size}). Initiating retraining.")
            
            # Retrain model
            online_learner = OnlineLearning()
            success = online_learner.update_model()
            
            if success:
                # Clear buffer after successful retraining
                buffer.clear_buffer()
                logger.info("Model successfully retrained and buffer cleared.")
            else:
                logger.warning("Model retraining failed or wasn't necessary.")
        else:
            logger.info(f"Buffer size ({current_size}) below threshold ({buffer.buffer_size}). No retraining needed.")
    
    except Exception as e:
        logger.error(f"Error during auto-retraining: {e}")

def main():
    logger.info("Starting automated retraining service")
    
    # Schedule checking every hour
    schedule.every(1).hours.do(check_and_retrain)
    
    # Also run immediately at startup
    check_and_retrain()
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Sleep for a minute between checks

if __name__ == "__main__":
    main()