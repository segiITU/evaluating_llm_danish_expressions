import openai
import logging
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

            params = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_completion_tokens': 1,
                'stream': False,
                'seed': 42
            }
            
            self.logger.info(f"Making API call with params: {params}")
            
            client = openai.OpenAI()
            response = client.chat.completions.create(**params)
            
            prediction = response.choices[0].message.content.strip().upper()
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

if __name__ == "__main__":
    test_data = {
        "expression": "have sommerfugle i maven",
        "A": "føle sig dårligt tilpas",
        "B": "være nervøs eller anspændt før en vigtig begivenhed",
        "C": "have spist for meget",
        "D": "være sulten"
    }
    
    model = GPTModel()
    try:
        prediction = model.predict(test_data["expression"], test_data)
        print(f"Expression: {test_data['expression']}")
        print(f"Prediction: {prediction}")
        print(f"Meaning: {test_data[prediction]}")
    except Exception as e:
        print(f"Error predicting for expression '{test_data['expression']}': {str(e)}")