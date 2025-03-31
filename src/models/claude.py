import logging
from typing import Dict
import anthropic
from src.models.base_model import BaseModel
from src.config.model_configs import PROMPT_TEMPLATE
import os
import json

logger = logging.getLogger(__name__)

class ClaudeModel(BaseModel):
    def __init__(self, model_name: str):
        """Initialize Claude model with API key and model config."""
        try:
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model_name
            logger.info(f"Initialized Claude model: {model_name}")
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
                idiom=expression,
                definition=definition
            )
            
            try:
                # Log the full prompt for debugging
                logger.info(f"Sending prompt for option {option_key}: {prompt}")
                
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,  # Increased from 1 to ensure we get a response
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                # Log the full message response for debugging
                logger.info(f"Raw message response: {message}")
                
                # Handle different response formats
                response_text = ""
                if hasattr(message, 'content') and message.content:
                    if isinstance(message.content, list) and len(message.content) > 0:
                        response_text = message.content[0].text
                    elif isinstance(message.content, str):
                        response_text = message.content
                
                response_text = response_text.strip().lower()
                logger.info(f"Response for '{expression}' option {option_key}: '{response_text}'")
                
                # Map response to a binary value
                if 'ja' in response_text:
                    predictions[option_key] = 1
                else:
                    predictions[option_key] = 0
                
                logger.info(f"Option {option_key} response: {response_text} -> {predictions[option_key]}")
                
            except Exception as e:
                logger.error(f"Claude prediction error for option {option_key}: {str(e)}")
                try:
                    # Try to log the full message structure if it exists
                    if 'message' in locals():
                        logger.error(f"Message structure: {vars(message)}")
                except:
                    pass
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
            # Log the prompt for debugging
            logger.info(f"Sending prompt: {prompt}")
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=5,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract the response text
            response_text = ""
            if hasattr(message, 'content') and message.content:
                if isinstance(message.content, list) and len(message.content) > 0:
                    for content_item in message.content:
                        if hasattr(content_item, 'text'):
                            response_text += content_item.text
                elif isinstance(message.content, str):
                    response_text = message.content
            
            response_text = response_text.strip().lower()
            logger.info(f"Response: '{response_text}'")
            
            # Check for "ja" response
            if (response_text.startswith("ja") or 
                re.search(r'\bja\b', response_text) or 
                "ja." in response_text):
                return 1
            else:
                return 0
            
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return 0

class Claude35Sonnet20241022(ClaudeModel):
    def __init__(self):
        super().__init__(model_name="claude-3-5-sonnet-20241022")

class Claude3Sonnet20240229(ClaudeModel):
    def __init__(self):
        super().__init__(model_name="claude-3-sonnet-20240229")