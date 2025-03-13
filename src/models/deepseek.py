from src.models.base_model import BaseModel
from typing import Dict
import logging
import os
from openai import OpenAI
from src.config.model_configs import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class DeepseekModel(BaseModel):
    def __init__(self, model_name: str = "deepseek-chat", api_key: str = None):
        """Initialize DeepSeek model with API key and model config."""
        try:
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not found or empty")
                
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            self.model = model_name
            self.logger = logging.getLogger(f"{model_name}_model")
        except Exception as e:
            logger.error(f"Error initializing DeepSeek model: {str(e)}")
            raise

    def predict(self, expression: str, options: Dict[str, str]) -> str:
        """
        Predict the correct definition for a Danish expression.
        
        Args:
            expression: The Danish expression
            options: Dictionary with keys 'A', 'B', 'C', 'D' containing possible definitions
            
        Returns:
            str: Predicted label ('A', 'B', 'C', or 'D')
        """
        prompt = PROMPT_TEMPLATE.format(
            metaphorical_expression=expression,
            definition_a=options['A'],
            definition_b=options['B'],
            definition_c=options['C'],
            definition_d=options['D']
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0
            )
            
            prediction = response.choices[0].message.content.strip().upper()
            
            # Validate prediction
            if prediction not in ['A', 'B', 'C', 'D']:
                logger.warning(f"Invalid prediction from DeepSeek: {prediction}")
                raise ValueError(f"Invalid prediction: {prediction}")
                
            return prediction
            
        except Exception as e:
            logger.error(f"DeepSeek prediction error: {str(e)}")
            raise