import logging
from typing import Dict
import anthropic
from src.models.base_model import BaseModel
from src.config.model_configs import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class ClaudeModel(BaseModel):
    def __init__(self, model_name: str):
        """Initialize Claude model with API key and model config."""
        try:
            self.client = anthropic.Client()
            self.model = model_name
        except Exception as e:
            logger.error(f"Error initializing Claude model: {str(e)}")
            raise

    def predict(self, expression: str, options: Dict[str, str]) -> str:
        """Predict the correct definition for a Danish expression."""
        prompt = PROMPT_TEMPLATE.format(
            metaphorical_expression=expression,
            definition_a=options['A'],
            definition_b=options['B'],
            definition_c=options['C'],
            definition_d=options['D']
        )
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            prediction = message.content[0].text.strip().upper()
            if prediction not in ['A', 'B', 'C', 'D']:
                logger.warning(f"Invalid prediction from Claude: {prediction}")
                raise ValueError(f"Invalid prediction: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Claude prediction error: {str(e)}")
            raise

class Claude35Sonnet20241022(ClaudeModel):
    def __init__(self):
        super().__init__(model_name="claude-3-5-sonnet-20241022")

class Claude3Sonnet20240229(ClaudeModel):
    def __init__(self):
        super().__init__(model_name="claude-3-sonnet-20240229")
