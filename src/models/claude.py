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
        """
        Predict the correct definition for a Danish expression by asking yes/no questions
        for each definition option.
        
        Args:
            expression: The Danish expression
            options: Dictionary with keys 'A', 'B', 'C', 'D' containing possible definitions
            
        Returns:
            str: Predicted label ('A', 'B', 'C', or 'D')
        """
        predictions = {}
        
        # Ask about each definition option
        for option_key, definition in options.items():
            prompt = PROMPT_TEMPLATE.format(
                metaphorical_expression=expression,
                definition_a=definition
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
                
                response = message.content[0].text.strip().lower()
                
                # Map response to a binary value
                if 'ja' in response:
                    predictions[option_key] = 1
                else:
                    predictions[option_key] = 0
                
                logger.info(f"Option {option_key} response: {response} -> {predictions[option_key]}")
                
            except Exception as e:
                logger.error(f"Claude prediction error for option {option_key}: {str(e)}")
                predictions[option_key] = 0
        
        # Find the option with a "ja" response (should only be one)
        yes_responses = [k for k, v in predictions.items() if v == 1]
        
        if len(yes_responses) == 1:
            # Return the one option that got a "yes"
            return yes_responses[0]
        elif len(yes_responses) > 1:
            # If multiple "yes" responses, log a warning and return the first one
            logger.warning(f"Multiple 'yes' responses for expression '{expression}': {yes_responses}")
            return yes_responses[0]
        else:
            # If no "yes" responses, log error and default to first option
            logger.error(f"No 'yes' responses for expression '{expression}'")
            return 'A'  # Default to first option

class Claude35Sonnet20241022(ClaudeModel):
    def __init__(self):
        super().__init__(model_name="claude-3-5-sonnet-20241022")

class Claude3Sonnet20240229(ClaudeModel):
    def __init__(self):
        super().__init__(model_name="claude-3-sonnet-20240229")