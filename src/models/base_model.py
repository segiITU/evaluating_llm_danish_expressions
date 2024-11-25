from abc import ABC, abstractmethod
from typing import Dict

class BaseModel(ABC):
    @abstractmethod
    def predict(self, expression: str, options: Dict[str, str]) -> str:
        """
        Predict the correct definition for a Danish expression.
        
        Args:
            expression: The Danish expression
            options: Dictionary with keys 'A', 'B', 'C', 'D' containing possible definitions
            
        Returns:
            str: Predicted label ('A', 'B', 'C', or 'D')
        """
        pass