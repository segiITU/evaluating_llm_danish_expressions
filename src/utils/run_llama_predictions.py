import pandas as pd
from pathlib import Path
import logging
from src.utils.data_loader import TalemaaderDataLoader
from llamaapi import LlamaAPI
import time
from typing import Set, Dict
import argparse
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/llama_batch_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LlamaModel:
    def __init__(self):
        self.api = LlamaAPI(api_key=os.getenv("LLAMA_API_KEY"))
        
    def predict(self, idiom: str, options: Dict[str, str]) -> str:
        """Predict the correct definition for a Danish idiom."""
        prompt = ("""Choose the correct definition for the given metaphorical expression by responding with only a single letter representing your choice (A, B, C, or D).
Sentence: {metaphorical_expression}
Option A: {definition_a}
Option B: {definition_b}
Option C: {definition_c}
Option D: {definition_d}
Your response should be exactly one letter: A, B, C, or D."""
        )
        
        try:
            response = self.api.generate(prompt, max_tokens=1, temperature=0)
            prediction = response.strip().upper()
            
            # Validate prediction
            if prediction not in ['A', 'B', 'C', 'D']:
                logger.warning(f"Invalid prediction from Llama: {prediction}")
                raise ValueError(f"Invalid prediction: {prediction}")
                
            return prediction
            
        except Exception as e:
            logger.error(f"Llama prediction error: {str(e)}")
            raise

class ModelPredictor:
    def __init__(self, model_name: str = "llama"):
        self.model_name = model_name
        self.data_loader = TalemaaderDataLoader()
        self.pred_dir = Path("data/predictions")
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.pred_dir / f"predicted_labels_{model_name}.csv"
        self.batch_size = 5  # Default 5 for all models
        
        # Initialize Llama model
        self.model = LlamaModel()
        
    def get_processed_idioms(self) -> Set[str]:
        """Get set of already processed idioms for this model"""
        processed = set()
        if self.output_file.exists():
            try:
                previous_preds = pd.read_csv(self.output_file)
                processed = set(previous_preds['talemaade_udtryk'].unique())
                logger.info(f"Found {len(processed)} previously processed idioms for {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading previous predictions: {str(e)}")
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
                logger.error(f"Error appending predictions: {str(e)}")
                new_df.to_csv(self.output_file.with_suffix('.new.csv'), index=False)
        else:
            new_df.to_csv(self.output_file, index=False)

    def run_predictions(self):
        """Run predictions on unprocessed idioms"""
        try:
            # Load test data only (no labels)
            test_df = self.data_loader.load_evaluation_data()
            
            # Get already processed idioms
            processed_idioms = self.get_processed_idioms()
            
            # Filter for unprocessed idioms
            unprocessed_df = test_df[~test_df['talemaade_udtryk'].isin(processed_idioms)]
            
            if len(unprocessed_df) == 0:
                logger.info(f"No new idioms to process for {self.model_name}!")
                return
                
            # Take next batch
            batch_df = unprocessed_df.head(self.batch_size)
            logger.info(f"Processing batch of {len(batch_df)} idioms using {self.model_name}")
            
            # Process batch
            new_predictions = []
            for idx, row in batch_df.iterrows():
                idiom = row['talemaade_udtryk']
                logger.info(f"Processing expression {idx+1}/{len(batch_df)}: {idiom}")
                
                options = {
                    'A': row['A'],
                    'B': row['B'],
                    'C': row['C'],
                    'D': row['D']
                }
                
                try:
                    pred_letter = self.model.predict(idiom, options)
                    pred_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[pred_letter]
                    
                    new_predictions.append({
                        'talemaade_udtryk': idiom,
                        'predicted_label': pred_num
                    })
                    
                    # Save after each prediction in case of interruption
                    if new_predictions:
                        self.save_predictions(new_predictions)
                        new_predictions = []
                        
                except Exception as e:
                    logger.error(f"Error processing idiom {idiom}: {str(e)}")
                    continue
                
                time.sleep(1)  # Extra safety on top of rate limiter
                
            logger.info(f"Batch processing complete. Processed {len(batch_df)} idioms with {self.model_name}.")
            
            # Final save in case any predictions remain
            if new_predictions:
                self.save_predictions(new_predictions)
                
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run predictions for the Llama model')
    parser.add_argument('--batch-size', type=int, default=5,
                      help='Number of idioms to process in this batch (default: 5)')
    
    args = parser.parse_args()
    predictor = ModelPredictor(model_name="llama")
    predictor.batch_size = args.batch_size
    predictor.run_predictions() 