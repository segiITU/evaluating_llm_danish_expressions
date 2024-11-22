import openai
import logging
from typing import Dict
from src.models.base_model import BaseModel
from src.utils.rate_limiter import rate_limit

logger = logging.getLogger(__name__)

class GPTModel(BaseModel):
    def __init__(self, model_name: str = "gpt-4-0125-preview"):
        self.model = model_name
        self.logger = logging.getLogger(f"{model_name}_model")
        
    @rate_limit(calls_per_minute=60)
    def predict(self, idiom: str, options: Dict[str, str]) -> str:
        """Predict the correct definition for a Danish idiom."""
        try:
            # Format the prompt exactly as in the working logs
            prompt = (
                "Choose the correct definition for the given metaphorical expression "
                "by responding with only a single letter representing your choice (A, B, C, or D).\n"
                f"Sentence: {idiom}\n"
                f"Option A: {options['A']}\n"
                f"Option B: {options['B']}\n"
                f"Option C: {options['C']}\n"
                f"Option D: {options['D']}\n"
                "Your response should be exactly one letter: A, B, C, or D."
            )

            # Use the exact parameters from the working logs
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
    # Use the same test case from the logs
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
        print(f"Idiom: {test_data['expression']}")
        print(f"Prediction: {prediction}")
        print(f"Meaning: {test_data[prediction]}")
    except Exception as e:
        print(f"Error predicting for idiom '{test_data['expression']}': {str(e)}")