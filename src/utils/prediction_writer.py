import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class PredictionWriter:
    def __init__(self, output_dir: str = "results/predictions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_predictions(self, 
                        predictions: List[Dict], 
                        model_name: str,
                        timestamp: str) -> Path:
        """
        Save predictions in a format suitable for evaluation.
        
        Args:
            predictions: List of prediction dictionaries
            model_name: Name of the model used
            timestamp: Timestamp of the evaluation run
        
        Returns:
            Path to saved file
        """
        # Create DataFrame with required columns
        df = pd.DataFrame(predictions)
        
        # Restructure to match gold label format
        eval_df = pd.DataFrame({
            'talemaade_udtryk': df['idiom'],
            'predicted_label': df['prediction'],
            'model': model_name,
            'timestamp': timestamp,
            'confidence': 1.0  # Could be added later if models provide confidence
        })
        
        # Save both detailed and evaluation formats
        detailed_path = self.output_dir / f"{model_name}_detailed_{timestamp}.csv"
        eval_path = self.output_dir / f"{model_name}_eval_{timestamp}.csv"
        
        df.to_csv(detailed_path, index=False)
        eval_df.to_csv(eval_path, index=False)
        
        logger.info(f"Saved predictions to {eval_path}")
        return eval_path 