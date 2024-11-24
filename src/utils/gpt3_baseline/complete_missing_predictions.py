import pandas as pd
from pathlib import Path
import logging
import time
import openai
from typing import Dict
import sys

"""Script to complete missing predictions in GPT-3.5-turbo predictions file.

1. Identifies missing expressions using validate_expressions logic
2. Runs GPT-3.5-turbo predictions only for missing items
3. Merges new predictions with existing ones
4. Saves complete prediction set

Usage: python -m src.utils.complete_missing_predictions
"""

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"complete_predictions_{run_number}.log"
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

class PredictionCompleter:
    def __init__(self):
        self.logger = setup_logging()
        self.model = "gpt-3.5-turbo-0125"
        self.predictions_file = Path("data/predictions/predicted_labels_gpt-3.5-turbo.csv")
        self.true_labels_file = Path("data/raw/talemaader_leverance_2_kun_labels.csv")
        self.test_data_file = Path("data/raw/talemaader_leverance_2_uden_labels.csv")
        
    def load_existing_predictions(self) -> pd.DataFrame:
        """Load existing predictions if file exists, else return empty DataFrame"""
        if self.predictions_file.exists():
            return pd.read_csv(self.predictions_file)
        return pd.DataFrame(columns=['talemaade_udtryk', 'predicted_label'])
        
    def get_missing_expressions(self) -> set:
        """Identify missing expressions by comparing with true labels"""
        true_df = pd.read_csv(self.true_labels_file, sep='\t')
        pred_df = self.load_existing_predictions()
        
        true_expressions = set(true_df['talemaade_udtryk'].str.strip())
        pred_expressions = set(pred_df['talemaade_udtryk'].str.strip())
        
        return true_expressions - pred_expressions
        
    def predict(self, expression: str, options: Dict[str, str]) -> str:
        """Make a single prediction using GPT-3.5-turbo"""
        prompt = (
            "Choose the correct definition for the given metaphorical expression by responding with only "
            "a single letter representing your choice (A, B, C, or D).\n"
            f"Sentence: {expression}\n"
            f"Option A: {options['A']}\n"
            f"Option B: {options['B']}\n"
            f"Option C: {options['C']}\n"
            f"Option D: {options['D']}\n"
            "Your response should be exactly one letter: A, B, C, or D."
        )
        
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0
            )
            
            prediction = response.choices[0].message.content.strip().upper()
            
            if prediction not in ['A', 'B', 'C', 'D']:
                raise ValueError(f"Invalid prediction: {prediction}")
                
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
            
    def complete_predictions(self):
        """Complete missing predictions and merge with existing ones"""
        try:
            # Load true labels to maintain order
            true_df = pd.read_csv(self.true_labels_file, sep='\t')
            existing_predictions = self.load_existing_predictions()
            
            # Create a DataFrame with the correct order
            ordered_predictions = pd.DataFrame()
            ordered_predictions['talemaade_udtryk'] = true_df['talemaade_udtryk']
            
            # Merge existing predictions while maintaining order
            if not existing_predictions.empty:
                ordered_predictions = ordered_predictions.merge(
                    existing_predictions,
                    on='talemaade_udtryk',
                    how='left'
                )
            
            # Identify missing expressions while preserving order
            missing_mask = ordered_predictions['predicted_label'].isna()
            missing_expressions = ordered_predictions[missing_mask]['talemaade_udtryk'].tolist()
            
            if not missing_expressions:
                self.logger.info("No missing expressions found. Predictions are complete!")
                return
                
            self.logger.info(f"Found {len(missing_expressions)} missing expressions to predict")
            
            # Load test data with options
            test_df = pd.read_csv(self.test_data_file, sep='\t')
            
            # Process missing expressions in order
            for idx, expr in enumerate(missing_expressions, 1):
                self.logger.info(f"Processing missing expression {idx}/{len(missing_expressions)}: {expr}")
                
                # Get options for this expression
                options_row = test_df[test_df['talemaade_udtryk'] == expr]
                if len(options_row) == 0:
                    self.logger.error(f"Could not find options for expression: {expr}")
                    continue
                    
                options = {
                    'A': options_row.iloc[0]['A'],
                    'B': options_row.iloc[0]['B'],
                    'C': options_row.iloc[0]['C'],
                    'D': options_row.iloc[0]['D']
                }
                
                try:
                    pred_letter = self.predict(expr, options)
                    pred_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[pred_letter]
                    
                    # Update the prediction in the ordered DataFrame
                    ordered_predictions.loc[
                        ordered_predictions['talemaade_udtryk'] == expr,
                        'predicted_label'
                    ] = pred_num
                    
                    # Save progress every 10 predictions
                    if idx % 10 == 0:
                        ordered_predictions.to_csv(self.predictions_file, index=False)
                        self.logger.info(f"Progress saved: {idx}/{len(missing_expressions)}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to predict for expression '{expr}': {str(e)}")
                    continue
                    
                time.sleep(1)  # Rate limiting
                
            # Final save
            ordered_predictions.to_csv(self.predictions_file, index=False)
            self.logger.info("All missing predictions completed and saved!")
            
            # Validate final state
            missing_after = ordered_predictions['predicted_label'].isna().sum()
            if missing_after > 0:
                self.logger.error(f"Still missing {missing_after} predictions after completion!")
                missing_exprs = ordered_predictions[ordered_predictions['predicted_label'].isna()]['talemaade_udtryk'].tolist()
                for expr in missing_exprs:
                    self.logger.error(f"Still missing prediction for: {expr}")
            else:
                self.logger.info("All expressions successfully predicted!")
                
        except Exception as e:
            self.logger.error(f"Error during completion process: {str(e)}")
            self.logger.exception("Detailed traceback:")
            raise
            
    def save_progress(self, existing_df: pd.DataFrame, new_predictions: list):
        """Save combined predictions to file"""
        if new_predictions:
            new_df = pd.DataFrame(new_predictions)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(self.predictions_file, index=False)
            self.logger.info(f"Saved {len(combined_df)} total predictions")
            
    def validate_final_predictions(self):
        """Validate that all expressions are now present"""
        true_df = pd.read_csv(self.true_labels_file, sep='\t')
        final_pred_df = pd.read_csv(self.predictions_file)
        
        true_expressions = set(true_df['talemaade_udtryk'].str.strip())
        final_expressions = set(final_pred_df['talemaade_udtryk'].str.strip())
        
        missing = true_expressions - final_expressions
        
        if missing:
            self.logger.warning(f"Still missing {len(missing)} expressions after completion!")
            for expr in missing:
                self.logger.warning(f"Missing: {expr}")
        else:
            self.logger.info("All expressions successfully predicted!")

def main():
    completer = PredictionCompleter()
    completer.complete_predictions()
    completer.validate_final_predictions()

if __name__ == "__main__":
    main() 