from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging
from pathlib import Path
import sys
sys.path.append("src/config")
from model_configs import PROMPT_TEMPLATE

class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = self._setup_logging()
        self.predictions_file = Path("data/predictions") / f"talemaader_{model_name}_predicted_labels.csv"
    
    @abstractmethod
    def predict(self, expression: str, definitions: Dict[str, str]) -> str:
        """
        Make a single prediction for given expression and definitions.
        
        Args:
            expression: The idiom to classify
            definitions: Dictionary with keys 'definition_a' through 'definition_d'
        
        Returns:
            str: Single letter prediction (A, B, C, or D)
        """
        pass
    
    @abstractmethod
    def batch_predict(self, data: List[Dict[str, str]], batch_size: int = 10) -> List[str]:
        """
        Make predictions for a batch of expressions.
        
        Args:
            data: List of dictionaries containing expressions and definitions
            batch_size: Number of predictions to process at once
            
        Returns:
            List[str]: List of predictions (A, B, C, or D)
        """
        pass

    def format_prompt(self, expression: str, definitions: Dict[str, str]) -> str:
        """Format the prompt using the template."""
        return PROMPT_TEMPLATE.format(
            metaphorical_expression=expression,
            definition_a=definitions['definition_a'],
            definition_b=definitions['definition_b'],
            definition_c=definitions['definition_c'],
            definition_d=definitions['definition_d']
        )
    
    def validate_prediction(self, prediction: str) -> bool:
        """Validate that prediction is a single letter A-D."""
        return prediction.strip() in ['A', 'B', 'C', 'D']
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the model."""
        logger = logging.getLogger(f"{self.model_name}_model")
        logger.setLevel(logging.INFO)
        
        Path("logs").mkdir(exist_ok=True)
        fh = logging.FileHandler(f"logs/{self.model_name}.log")
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def save_predictions(self, predictions: List[str]) -> None:
        """Save predictions to file."""
        self.predictions_file.parent.mkdir(exist_ok=True)
        with open(self.predictions_file, 'w') as f:
            f.write('\n'.join(predictions))
        self.logger.info(f"Saved predictions to {self.predictions_file}")