import pandas as pd
from pathlib import Path
import logging
from src.utils.data_loader import TalemaaderDataLoader
from src.models.gpt import GPTModel
from src.models.gemini import GeminiModel
from src.models.llama import LlamaModel
from src.models.claude import ClaudeModel
import time
from typing import Set
import argparse
import os


"""Generates predictions for Danish expressions using selected LLM model.

Processes batches of expressions from raw data through specified model (GPT-4, GPT-4o, Gemini, Llama, Claude).
Saves predictions and handles partial completions through batch processing.

Usage: python -m src.utils.run_model_predictions --model [model_name] --batch-size [size]
Output: Saves predictions to data/predictions/predicted_labels_[model_name].csv
"""


def setup_logging(model_name: str, batch_size: int):
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"batch_{model_name}_{batch_size}_{run_number}.log"
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
        self.pred_dir = Path("data/predictions")
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.pred_dir / f"predicted_labels_{model_name}.csv"
        
        if model_name == 'gemini':
            self.model = GeminiModel()
        elif model_name == 'llama':
            self.model = LlamaModel()
        elif model_name == 'claude':
            self.model = ClaudeModel()
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
        """Run predictions on unprocessed idioms"""
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
                    pred_letter = self.model.predict(expression, options)
                    pred_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[pred_letter]
                    
                    new_predictions.append({
                        'talemaade_udtryk': expression,
                        'predicted_label': pred_num
                    })
                    
                    if new_predictions:
                        self.save_predictions(new_predictions)
                        new_predictions = []
                        
                except Exception as e:
                    self.logger.error(f"Error processing expression {expression}: {str(e)}")
                    continue
                
                time.sleep(1)  
                
            self.logger.info(f"Batch processing complete. Processed {len(batch_df)} idioms with {self.model_name}.")
            
            if new_predictions:
                self.save_predictions(new_predictions)
                
        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run predictions for a specific model')
    parser.add_argument('--model', type=str, default="gpt-4", 
                      choices=['gpt-4', 'gpt-4o', 'gemini', 'llama', 'claude'],
                      help='Model name to use for predictions')
    parser.add_argument('--batch-size', type=int, default=5,
                      help='Number of idioms to process in this batch (default: 5)')
    
    args = parser.parse_args()
    predictor = ModelPredictor(model_name=args.model, batch_size=args.batch_size)
    predictor.run_predictions()