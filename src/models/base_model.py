from abc import ABC, abstractmethod
from typing import Dict

class BaseModel(ABC):
    @abstractmethod
    def predict(self, idiom: str, options: Dict[str, str]) -> str:
        """
        Predict the correct definition for a Danish idiom.
        
        Args:
            idiom: The Danish idiom
            options: Dictionary with keys 'A', 'B', 'C', 'D' containing possible definitions
            
        Returns:
            str: Predicted label ('A', 'B', 'C', or 'D')
        """
        pass