import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class TalemaaderDataLoader:
    """Data loader for the talemåder dataset."""
    
    def __init__(self, data_dir: str = "data/talemaader"):
        self.data_dir = Path(data_dir)
        self.validate_data_directory()
        
    def validate_data_directory(self) -> None:
        """Check if required files exist."""
        required_files = [
            "talemaader_leverance_1.csv",
            "talemaader_leverance_2_kun_labels.csv",
            "talemaader_leverance_2_uden_labels.csv"
        ]
        
        for file in required_files:
            if not (self.data_dir / file).exists():
                raise FileNotFoundError(f"Required file {file} not found in {self.data_dir}")
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all dataset files."""
        try:
            data = {
                "expressions": pd.read_csv(self.data_dir / "talemaader_leverance_1.csv"),
                "labels": pd.read_csv(self.data_dir / "talemaader_leverance_2_kun_labels.csv"),
                "options": pd.read_csv(self.data_dir / "talemaader_leverance_2_uden_labels.csv")
            }
            
            logger.info(f"Loaded {len(data['expressions'])} expressions")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_evaluation_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the data needed for model evaluation.
        
        Returns:
            Tuple containing:
            - DataFrame with expressions and definition options
            - Series with correct labels
        """
        try:
            # Load options and labels
            options_df = pd.read_csv(self.data_dir / "talemaader_leverance_2_uden_labels.csv")
            labels_df = pd.read_csv(self.data_dir / "talemaader_leverance_2_kun_labels.csv")
            
            # Extract correct labels
            correct_labels = labels_df['korrekt_def'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})
            
            return options_df, correct_labels
            
        except Exception as e:
            logger.error(f"Error loading evaluation data: {str(e)}")
            raise
    
    def save_predictions(self, 
                        predictions: pd.DataFrame, 
                        model_name: str,
                        output_dir: str = "data/predictions") -> None:
        """
        Save model predictions to file.
        
        Args:
            predictions: DataFrame with predictions
            model_name: Name of the model used
            output_dir: Directory to save predictions
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        filename = f"talemaader_{model_name}_predicted_labels.csv"
        predictions.to_csv(output_path / filename, index=False)
        logger.info(f"Saved predictions to {output_path / filename}")