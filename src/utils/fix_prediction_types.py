import pandas as pd
from pathlib import Path
import logging

"""Converts float values to integers in the GPT-3.5-turbo predictions file.

Ensures prediction labels are stored as integers rather than floats.
Creates a backup of the original file before modifying.

Usage: python -m src.utils.fix_prediction_types
"""

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "fix_prediction_types.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def fix_prediction_types():
    logger = setup_logging()
    
    try:
        # File paths
        predictions_path = Path("data/predictions/predicted_labels_gpt-3.5-turbo.csv")
        backup_path = predictions_path.with_suffix('.bak.csv')
        
        # Load predictions
        logger.info(f"Loading predictions from {predictions_path}")
        df = pd.read_csv(predictions_path)
        
        # Create backup
        logger.info(f"Creating backup at {backup_path}")
        df.to_csv(backup_path, index=False)
        
        # Check current types
        original_type = df['predicted_label'].dtype
        logger.info(f"Original predicted_label type: {original_type}")
        
        # Convert floats to integers
        df['predicted_label'] = df['predicted_label'].fillna(-1).astype(int)
        
        # Replace any -1 back to NaN
        df['predicted_label'] = df['predicted_label'].replace(-1, pd.NA)
        
        # Save modified file
        logger.info("Saving modified predictions with integer types")
        df.to_csv(predictions_path, index=False)
        
        # Verify changes
        new_df = pd.read_csv(predictions_path)
        new_type = new_df['predicted_label'].dtype
        logger.info(f"New predicted_label type: {new_type}")
        
        # Count modifications
        modified_count = len(df[df['predicted_label'].notna()])
        logger.info(f"Modified {modified_count} prediction values to integer type")
        
        logger.info("Conversion completed successfully")
        
    except Exception as e:
        logger.error(f"Error during type conversion: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    fix_prediction_types() 