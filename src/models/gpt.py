import logging
import os
import requests
import re
from typing import Dict
from src.models.base_model import BaseModel
from src.config.model_configs import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class GPTModel(BaseModel):
    def __init__(self, model_name: str = "gpt-4-0125-preview"):
        self.model = model_name
        self.logger = logging.getLogger(f"{model_name}_model")
        
    def predict(self, expression: str, options: Dict[str, str]) -> str:
        """Predict the correct definition for a Danish expression."""
        try:
            prompt = PROMPT_TEMPLATE.format(
                metaphorical_expression=expression,
                definition_a=options['A'],
                definition_b=options['B'],
                definition_c=options['C'],
                definition_d=options['D']
            )

            if self.model == "gpt-3.5-turbo" or self.model == "gpt-3.5-turbo-0125":
                return self._predict_with_requests(prompt)
            
            params = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1,
                'temperature': 0
            }
            
            self.logger.info(f"Making API call with params: {params}")
            
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(**params)
            
            prediction = response.choices[0].message.content.strip().upper()
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def get_single_response(self, expression: str, definition: str) -> int:
        """
        Get a binary response (1 for yes, 0 for no) for a single definition.
        
        Args:
            expression: The Danish expression
            definition: A possible definition
            
        Returns:
            int: 1 for "yes", 0 for "no"
        """
        prompt = PROMPT_TEMPLATE.format(
            idiom=expression,
            definition=definition
        )
        
        try:
            self.logger.info(f"Sending prompt: {prompt}")
            
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0
            )
            
            response_text = response.choices[0].message.content.strip().lower()
            self.logger.info(f"Response: '{response_text}'")
            
            # Check for "ja" response (Danish for "yes")
            if (response_text.startswith("ja") or 
                re.search(r'\bja\b', response_text) or 
                "ja." in response_text):
                return 1
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"Error getting response: {str(e)}")
            return 0
            
    def _predict_with_requests(self, prompt: str) -> str:
        """Use direct requests to OpenAI API for GPT-3.5-turbo to avoid client issues."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": "gpt-3.5-turbo-0125",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0
        }
        
        self.logger.info(f"Making direct API call with params: {data}")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=data,
            headers=headers
        )
        
        response.raise_for_status()
        result = response.json()
        
        prediction = result["choices"][0]["message"]["content"].strip().upper()
        
        if prediction not in ['A', 'B', 'C', 'D']:
            self.logger.warning(f"Invalid prediction: {prediction}")
            raise ValueError(f"Invalid prediction: {prediction}")
            
        return prediction