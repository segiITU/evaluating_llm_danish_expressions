from src.models.base_model import BaseModel
import google.generativeai as genai
import os
import logging
from typing import Dict
from src.config.model_configs import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class GeminiModel(BaseModel):
    def __init__(self):
        """Initialize Gemini model with API key and model config."""
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
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
        # Format the prompt using the template
        prompt = PROMPT_TEMPLATE.format(
            metaphorical_expression=expression,
            definition_a=options['A'],
            definition_b=options['B'],
            definition_c=options['C'],
            definition_d=options['D']
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=1,
                    candidate_count=1
                )
            )
            prediction = response.text.strip().upper()
            
            # Validate prediction
            if prediction not in ['A', 'B', 'C', 'D']:
                logger.warning(f"Invalid prediction from Gemini: {prediction}")
                raise ValueError(f"Invalid prediction: {prediction}")
                
            return prediction
            
        except Exception as e:
            logger.error(f"Gemini prediction error: {str(e)}")
            raise