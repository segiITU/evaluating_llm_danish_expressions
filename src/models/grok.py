from src.models.base_model import BaseModel
import logging
from typing import Dict
import requests
import os
from src.config.model_configs import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class GrokModel(BaseModel):
    def __init__(self):
        """Initialize Grok model with API key and model config."""
        try:
            # Print all environment variables for debugging
            logger.info("Environment variables:")
            for key, value in os.environ.items():
                logger.info(f"{key}: {value[:5]}..." if value else f"{key}: {value}")
            
            self.api_key = os.getenv("XAI_API_KEY")
            if not self.api_key:
                raise ValueError("XAI_API_KEY environment variable not found")
            
            self.api_url = "https://api.x.ai/v1/chat/completions"
            self.model = "grok-2-1212"
        except Exception as e:
            logger.error(f"Error initializing Grok model: {str(e)}")
            raise

    def predict(self, expression: str, options: Dict[str, str]) -> str:
        prompt = PROMPT_TEMPLATE.format(
            metaphorical_expression=expression,
            definition_a=options['A'],
            definition_b=options['B'],
            definition_c=options['C'],
            definition_d=options['D']
        )
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"API Response: {result}")
            
            if 'choices' not in result:
                logger.error(f"Unexpected API response structure: {result}")
                raise ValueError(f"API response missing 'choices' field")
                
            prediction = result["choices"][0]["message"]["content"].strip().upper()
            
            if prediction not in ['A', 'B', 'C', 'D']:
                logger.warning(f"Invalid prediction from Grok: {prediction}")
                raise ValueError(f"Invalid prediction: {prediction}")
                
            return prediction
            
        except Exception as e:
            logger.error(f"Grok prediction error: {str(e)}")
            if isinstance(e, requests.exceptions.RequestException):
                logger.error(f"Request error details: {e.response.text if hasattr(e, 'response') else 'No response details'}")
            raise