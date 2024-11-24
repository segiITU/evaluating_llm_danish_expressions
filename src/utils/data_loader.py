import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple


"""Handles loading and validation of Danish expressions dataset.

Manages data loading from raw files, performs validation checks, and prepares
data structure for model evaluation. Handles multiple file formats and encodings.

Usage: Import TalemaaderDataLoader class
Required files: In data/talemaader/raw/
- talemaader_leverance_1.csv
- talemaader_leverance_2_kun_labels.csv
- talemaader_leverance_2_uden_labels.csv
"""


logger = logging.getLogger(__name__)

class TalemaaderDataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.logger = logger
        
    def validate_files_exist(self) -> bool:
        """Check if all required files exist."""
        required_files = [
            "talemaader_leverance_1.csv",
            "talemaader_leverance_2_kun_labels.csv",
            "talemaader_leverance_2_uden_labels.csv"
        ]
        
        for file in required_files:
            if not (self.data_dir / file).exists():
                self.logger.error(f"Missing required file: {file}")
                return False
        return True
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three CSV files and validate their contents.
        
        Returns:
            Tuple containing:
            - Gold standard definitions DataFrame
            - Label mappings DataFrame
            - Test data DataFrame
        """
        if not self.validate_files_exist():
            raise FileNotFoundError("Required data files are missing")
            
        try:
            # Load all three files
            gold_df = pd.read_csv(self.data_dir / "talemaader_leverance_1.csv")
            labels_df = pd.read_csv(self.data_dir / "talemaader_leverance_2_kun_labels.csv")
            test_df = pd.read_csv(self.data_dir / "talemaader_leverance_2_uden_labels.csv")
            
            # Validate data structure
            self._validate_data_structure(gold_df, labels_df, test_df)
            
            return gold_df, labels_df, test_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _validate_data_structure(self, gold_df: pd.DataFrame, labels_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Validate the structure and content of the dataframes."""
        # Check required columns
        if not all(col in gold_df.columns for col in ['udtryk_id', 'talemaade_udtryk', 'ddo_definition']):
            raise ValueError("Gold standard file missing required columns")
            
        if not all(col in labels_df.columns for col in ['udtryk_id', 'talemaade_udtryk', 'korrekt_def']):
            raise ValueError("Label mappings file missing required columns")
            
        if not all(col in test_df.columns for col in ['talemaade_udtryk', 'A', 'B', 'C', 'D']):
            raise ValueError("Test data file missing required columns")
            
        # Check for missing values
        for df, name in [(gold_df, 'gold'), (labels_df, 'labels'), (test_df, 'test')]:
            if df.isnull().any().any():
                self.logger.warning(f"Missing values found in {name} dataset")
                
    def prepare_evaluation_data(self) -> Dict[str, List]:
        """
        Prepare data for model evaluation.
        
        Returns:
            Dictionary containing:
            - expressions: List of idioms to evaluate
            - definitions: List of definition dictionaries
            - correct_labels: List of correct answers (A, B, C, D)
        """
        _, labels_df, test_df = self.load_data()
        
        # Convert numeric indices to letter labels
        label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        correct_labels = labels_df['korrekt_def'].map(label_map).tolist()
        
        # Prepare data in the format expected by models
        evaluation_data = []
        for _, row in test_df.iterrows():
            evaluation_data.append({
                'expression': row['talemaade_udtryk'],
                'definition_a': row['A'],
                'definition_b': row['B'],
                'definition_c': row['C'],
                'definition_d': row['D']
            })
            
        return {
            'data': evaluation_data,
            'correct_labels': correct_labels
        }
        
    def load_evaluation_data(self) -> pd.DataFrame:
        """Load the test data file with idioms and their multiple choice options."""
        try:
            # This line needs to be updated with all parameters
            options_df = pd.read_csv(
                self.data_dir / "talemaader_leverance_2_uden_labels.csv",
                sep='\t',                    # Tab separator
                encoding='utf-8',            # UTF-8 encoding
                on_bad_lines='skip',         # Skip problematic lines
                engine='python'              # Use python engine instead of C
            )
            
            required_cols = ['talemaade_udtryk', 'A', 'B', 'C', 'D']
            if not all(col in options_df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Expected: {required_cols}")
                
            return options_df
            
        except Exception as e:
            self.logger.error(f"Error loading evaluation data: {str(e)}")
            raise