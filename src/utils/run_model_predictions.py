import pandas as pd
from pathlib import Path
import logging
from src.utils.data_loader import TalemaaderDataLoader
from src.models.gpt import GPTModel
from src.models.gemini import GeminiModel
from src.models.llama import LlamaModel
from src.models.claude import ClaudeModel
from src.models.grok import GrokModel
from src.models.deepseek import DeepseekModel
import time
from typing import Set
import argparse
import os

def setup_logging(model_name: str, batch_size: int):
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"batch_yesno_{model_name}_{batch_size}_{run_number}.log"
        if not log_file.exists():
            break
        run_number += 1
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_name: str = "gpt-4", batch_size: int = 5):
        self.model_name = model_name
        self.batch_size = batch_size
        self.logger = setup_logging(model_name, batch_size)
        self.data_loader = TalemaaderDataLoader()
        self.pred_dir = Path("data/predictions/yesno")
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.pred_dir / f"predicted_labels_{model_name}.csv"
        
        if model_name == 'gemini':
            self.model = GeminiModel()
        elif model_name == 'llama':
            self.model = LlamaModel()
        elif model_name == 'grok-2':
            self.model = GrokModel()
        elif model_name in ['claude', 'claude-3-5-sonnet', 'claude-3-7-sonnet']:
            model_string = "claude-3-sonnet-20240229"  # Default
            if model_name == 'claude-3-5-sonnet':
                model_string = "claude-3-5-sonnet-20241022"
            elif model_name == 'claude-3-7-sonnet':
                model_string = "claude-3-7-sonnet-20250219"
            self.model = ClaudeModel(model_name=model_string)
        elif model_name == 'deepseek':
            self.model = DeepseekModel()
        else:
            self.model = GPTModel(model_name=model_name)
        
    def get_processed_idioms(self) -> Set[str]:
        """Get set of already processed idioms for this model"""
        processed = set()
        if self.output_file.exists():
            try:
                previous_preds = pd.read_csv(self.output_file)
                processed = set(previous_preds['talemaade_udtryk'].unique())
                self.logger.info(f"Found {len(processed)} previously processed idioms for {self.model_name}")
            except Exception as e:
                self.logger.error(f"Error loading previous predictions: {str(e)}")
        return processed

    def save_predictions(self, new_predictions: list):
        """Save predictions, appending to existing file if it exists"""
        new_df = pd.DataFrame(new_predictions)
        
        if self.output_file.exists():
            try:
                existing_df = pd.read_csv(self.output_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(self.output_file, index=False)
            except Exception as e:
                self.logger.error(f"Error appending predictions: {str(e)}")
                new_df.to_csv(self.output_file.with_suffix('.new.csv'), index=False)
        else:
            new_df.to_csv(self.output_file, index=False)

    def run_predictions(self):
        """Run predictions on unprocessed idioms using yes/no questions for each definition"""
        try:
            test_df = self.data_loader.load_evaluation_data()
            
            processed_idioms = self.get_processed_idioms()
            
            unprocessed_df = test_df[~test_df['talemaade_udtryk'].isin(processed_idioms)]
            
            if len(unprocessed_df) == 0:
                self.logger.info(f"No new idioms to process for {self.model_name}!")
                return
                
            batch_df = unprocessed_df.head(self.batch_size)
            self.logger.info(f"Processing batch of {len(batch_df)} idioms using {self.model_name}")
            
            new_predictions = []
            for idx, row in batch_df.iterrows():
                expression = row['talemaade_udtryk']
                self.logger.info(f"Processing expression {idx+1}/{len(batch_df)}: {expression}")
                
                options = {
                    'A': row['A'],
                    'B': row['B'],
                    'C': row['C'],
                    'D': row['D']
                }
                
                try:
                    # For yes/no approach, query each option
                    responses = {}
                    for opt_key, definition in options.items():
                        # This would need to be added to the model class
                        response = self.model.get_single_response(expression, definition)
                        responses[opt_key] = response
                    
                    # Find which options got "yes"
                    yes_responses = [k for k, v in responses.items() if v == 1]
                    
                    if len(yes_responses) == 1:
                        pred_letter = yes_responses[0]
                    elif len(yes_responses) > 1:
                        self.logger.warning(f"Multiple 'yes' responses for {expression}: {yes_responses}")
                        pred_letter = yes_responses[0]  # Default to first one
                    else:
                        self.logger.warning(f"No 'yes' responses for {expression}")
                        pred_letter = 'A'  # Default
                    
                    pred_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[pred_letter]
                    
                    # Save the prediction with all response data
                    new_predictions.append({
                        'talemaade_udtryk': expression,
                        'predicted_label': pred_num,
                        'yes_count': len(yes_responses),
                        'yes_options': ','.join(yes_responses) if yes_responses else 'None',
                        'A_response': responses.get('A', 0),
                        'B_response': responses.get('B', 0),
                        'C_response': responses.get('C', 0),
                        'D_response': responses.get('D', 0)
                    })
                    
                    # Save after each prediction to avoid losing progress
                    if new_predictions:
                        self.save_predictions(new_predictions)
                        new_predictions = []
                        
                except Exception as e:
                    self.logger.error(f"Error processing expression {expression}: {str(e)}")
                    continue
                
                time.sleep(1)  # Courtesy delay between API calls
                
            self.logger.info(f"Batch processing complete. Processed {len(batch_df)} idioms with {self.model_name}.")
            
            if new_predictions:
                self.save_predictions(new_predictions)
                
        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run predictions for a specific model using yes/no prompt approach')
    parser.add_argument('--model', type=str, default="gpt-4", 
              choices=['gpt-4', 'gpt-4o', 'gemini', 'llama', 'claude', 'claude-3-5-sonnet', 
                       'claude-3-7-sonnet', 'grok-2', 'deepseek', 'gpt-3.5-turbo'],
              help='Model name to use for predictions')
    parser.add_argument('--batch-size', type=int, default=5,
                      help='Number of idioms to process in this batch (default: 5)')
    
    args = parser.parse_args()
    predictor = ModelPredictor(model_name=args.model, batch_size=args.batch_size)
    predictor.run_predictions()