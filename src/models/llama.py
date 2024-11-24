from src.models.base_model import BaseModel
from typing import Dict
import logging
from llamaapi import LlamaAPI
import os
import json
from src.config.model_configs import PROMPT_TEMPLATE, MODEL_CONFIGS

logger = logging.getLogger(__name__)

class LlamaModel(BaseModel):
    def __init__(self):
        """Initialize Llama model with API key and model config."""
        try:
            api_token = os.getenv("LLAMA_API_KEY")
            if not api_token:
                raise ValueError("LLAMA_API_KEY environment variable not found")
                
            self.llama = LlamaAPI(api_token)
            self.config = MODEL_CONFIGS["llama-3.1"]
            
        except Exception as e:
            logger.error(f"Error initializing Llama model: {str(e)}")
            raise

    def predict(self, expression: str, options: Dict[str, str]) -> str:
        """
        Predict the correct definition for a Danish expression.
        """
        prompt = PROMPT_TEMPLATE.format(
            metaphorical_expression=expression,
            definition_a=options['A'],
            definition_b=options['B'],
            definition_c=options['C'],
            definition_d=options['D']
        )
        
        try:
            api_request = {
                "model": "llama3.1-70b",  # Correct model name
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "max_tokens": 1,
                "temperature": 0
            }
            
            # Log the request
            logger.info(f"Sending request to Llama API: {json.dumps(api_request, indent=2)}")
            
            # Get response
            response = self.llama.run(api_request)
            response_json = response.json()
            
            # Log the response
            logger.info(f"API Response: {json.dumps(response_json, indent=2)}")
            
            # Get prediction from response
            prediction = response_json.get("choices", [])[0].get("message", {}).get("content", "").strip().upper()
            
            # Validate prediction
            if prediction not in ['A', 'B', 'C', 'D']:
                logger.warning(f"Invalid prediction from Llama: {prediction}")
                raise ValueError(f"Invalid prediction: {prediction}")
                
            return prediction
            
        except Exception as e:
            logger.error(f"Llama prediction error: {str(e)}")
            raise