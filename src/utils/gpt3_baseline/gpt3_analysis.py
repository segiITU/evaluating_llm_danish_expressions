import pandas as pd
from pathlib import Path
import logging
import time
import openai
from typing import Dict, List
import os

"""All-in-one script for running GPT-3.5-turbo predictions on Danish expressions.

Handles complete workflow:
1. Generates predictions using GPT-3.5-turbo
2. Adds true labels and calculates accuracy
3. Analyzes and saves discrepancies

Usage: python src/utils/gpt3_analysis.py
Output: Generates all analysis files with prefix 'gpt-3.5-turbo'
"""

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"batch_gpt-3.5-turbo_{run_number}.log"
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

class GPT35Predictor:
    def __init__(self):
        self.logger = setup_logging()
        self.model = "gpt-3.5-turbo-0125"
        self.data_dir = Path("data")
        self.predictions_file = self.data_dir / "predictions" / f"predicted_labels_gpt-3.5-turbo.csv"
        self.analyzed_file = self.data_dir / "predictions" / f"predicted_and_gold_labels_gpt-3.5-turbo.csv"
        self.discrepancies_file = self.data_dir / "processed" / f"only_discrepancies_gpt-3.5-turbo.csv"
        
        # Create necessary directories
        self.data_dir.joinpath("predictions").mkdir(parents=True, exist_ok=True)
        self.data_dir.joinpath("processed").mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized with paths:")
        self.logger.info(f"Predictions: {self.predictions_file}")
        self.logger.info(f"Analyzed results: {self.analyzed_file}")
        self.logger.info(f"Discrepancies: {self.discrepancies_file}")
        
    def load_data(self):
        self.logger.info("Loading test data...")
        test_data = pd.read_csv(
            "data/raw/talemaader_leverance_2_uden_labels.csv", 
            sep='\t'
        )
        self.logger.info(f"Loaded {len(test_data)} expressions for testing")
        return test_data
        
    def predict(self, expression: str, options: Dict[str, str]) -> str:
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
            self.logger.info(f"Received prediction: {prediction}")
            
            if prediction not in ['A', 'B', 'C', 'D']:
                raise ValueError(f"Invalid prediction: {prediction}")
                
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
            
    def run_predictions(self):
        self.logger.info("Starting predictions phase...")
        self.logger.info(f"Will save predictions to: {self.predictions_file}")
        
        test_df = self.load_data()
        predictions = []
        total_count = len(test_df)
        successful_predictions = 0
        failed_predictions = 0
        
        # Add validation check for empty dataset
        if total_count == 0:
            self.logger.error("No test data found. Aborting predictions.")
            raise ValueError("Empty test dataset")

        for idx, row in test_df.iterrows():
            expression = row['talemaade_udtryk']
            self.logger.info(f"Processing expression {idx+1}/{total_count}: {expression}")
            
            options = {
                'A': row['A'],
                'B': row['B'],
                'C': row['C'],
                'D': row['D']
            }
            
            # Validate options
            if any(not option for option in options.values()):
                self.logger.error(f"Invalid options for expression {expression}: {options}")
                failed_predictions += 1
                continue
            
            try:
                pred_letter = self.predict(expression, options)
                pred_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[pred_letter]
                
                prediction = {
                    'talemaade_udtryk': expression,
                    'predicted_label': pred_num
                }
                predictions.append(prediction)
                successful_predictions += 1
                
                # Enhanced batch saving logging
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Progress: {idx+1}/{total_count} ({(idx+1)/total_count:.1%})")
                    self.logger.info(f"Successful predictions: {successful_predictions}")
                    self.logger.info(f"Failed predictions: {failed_predictions}")
                    temp_df = pd.DataFrame(predictions)
                    temp_df.to_csv(self.predictions_file, index=False)
                    
            except Exception as e:
                failed_predictions += 1
                self.logger.error(f"Error processing expression '{expression}': {str(e)}")
                self.logger.exception("Detailed traceback:")
                continue
                
            time.sleep(1)
        
        # Final validation and statistics
        self.logger.info("\n=== Final Prediction Statistics ===")
        self.logger.info(f"Total expressions processed: {total_count}")
        self.logger.info(f"Successful predictions: {successful_predictions}")
        self.logger.info(f"Failed predictions: {failed_predictions}")
        self.logger.info(f"Success rate: {successful_predictions/total_count:.2%}")
        
        if successful_predictions == 0:
            self.logger.error("No successful predictions were made!")
            raise RuntimeError("Prediction process failed completely")
        
        # Final save with validation
        if predictions:
            pred_df = pd.DataFrame(predictions)
            if len(pred_df) != successful_predictions:
                self.logger.error(f"Prediction count mismatch: DataFrame has {len(pred_df)} rows but recorded {successful_predictions} successes")
            pred_df.to_csv(self.predictions_file, index=False)
            self.logger.info(f"All predictions saved to {self.predictions_file}")
        
        return pd.DataFrame(predictions)
        
    def add_true_labels(self, predictions_df):
        self.logger.info("Adding true labels...")
        
        # Validate predictions DataFrame
        if len(predictions_df) == 0:
            self.logger.error("Empty predictions DataFrame")
            raise ValueError("No predictions to analyze")
        
        labels_path = Path("data/raw/talemaader_leverance_2_kun_labels.csv")
        if not labels_path.exists():
            self.logger.error(f"True labels file not found: {labels_path}")
            raise FileNotFoundError(f"Missing true labels file: {labels_path}")
        
        true_labels_df = pd.read_csv(labels_path, sep='\t')
        self.logger.info(f"Loaded {len(true_labels_df)} true labels")
        
        # Validate true labels DataFrame
        if len(true_labels_df) == 0:
            self.logger.error("Empty true labels DataFrame")
            raise ValueError("No true labels loaded")
        
        merged_df = pd.merge(
            predictions_df,
            true_labels_df[['talemaade_udtryk', 'korrekt_def']],
            on='talemaade_udtryk',
            how='left'
        )
        
        # Check for missing matches
        missing_matches = merged_df['korrekt_def'].isna().sum()
        if missing_matches > 0:
            self.logger.warning(f"Found {missing_matches} predictions without matching true labels")
        
        merged_df = merged_df.rename(columns={'korrekt_def': 'true_label'})
        merged_df['Discrepancy'] = ''
        mask = merged_df['predicted_label'].astype(int) != merged_df['true_label'].astype(int)
        merged_df.loc[mask, 'Discrepancy'] = 'DISCREPANCY'
        
        total = len(merged_df)
        discrepancies = (merged_df['Discrepancy'] == 'DISCREPANCY').sum()
        accuracy = (total - discrepancies) / total
        
        self.logger.info(f"Accuracy: {accuracy:.2%}")
        
        self.logger.info(f"Saving analyzed results to {self.analyzed_file}")
        merged_df.to_csv(self.analyzed_file, index=False)
        self.logger.info("Analysis saved successfully")
        
        return merged_df
        
    def analyze_discrepancies(self, analyzed_df):
        self.logger.info("Analyzing discrepancies...")
        
        discrepancy_df = analyzed_df[analyzed_df['Discrepancy'] == 'DISCREPANCY'].copy()
        options_df = pd.read_csv("data/raw/talemaader_leverance_2_uden_labels.csv", sep='\t')
        true_defs_df = pd.read_csv("data/raw/talemaader_leverance_1.csv", sep='\t')
        
        def get_predicted_definition(row):
            options_row = options_df[options_df['talemaade_udtryk'] == row['talemaade_udtryk']].iloc[0]
            pred_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[row['predicted_label']]
            return options_row[pred_letter]
        
        discrepancy_df['predicted_definition'] = discrepancy_df.apply(get_predicted_definition, axis=1)
        
        discrepancy_df = pd.merge(
            discrepancy_df,
            true_defs_df[['talemaade_udtryk', 'ddo_definition']],
            on='talemaade_udtryk',
            how='left'
        )
        
        discrepancy_df = discrepancy_df.rename(columns={'ddo_definition': 'true_definition'})
        
        columns_order = [
            'talemaade_udtryk',
            'predicted_label',
            'true_label',
            'predicted_definition',
            'true_definition',
            'Discrepancy'
        ]
        
        discrepancy_df = discrepancy_df[columns_order]
        
        self.logger.info(f"Saving discrepancies to {self.discrepancies_file}")
        discrepancy_df.to_csv(self.discrepancies_file, index=False)
        self.logger.info("Discrepancies saved successfully")
        
        self.logger.info(f"Found {len(discrepancy_df)} discrepancies")
        if len(discrepancy_df) > 0:
            sample = discrepancy_df.iloc[0]
            self.logger.info(f"\nSample discrepancy (first row):")
            self.logger.info(f"Expression: {sample['talemaade_udtryk']}")
            self.logger.info(f"Predicted (Label {sample['predicted_label']}): {sample['predicted_definition']}")
            self.logger.info(f"True (Label {sample['true_label']}): {sample['true_definition']}")
            
    def run_complete_analysis(self):
        self.logger.info("Starting complete GPT-3.5-turbo analysis...")
        
        predictions_df = self.run_predictions()
        analyzed_df = self.add_true_labels(predictions_df)
        self.analyze_discrepancies(analyzed_df)
        
        self.logger.info("Analysis complete!")

if __name__ == "__main__":
    analyzer = GPT35Predictor()
    analyzer.run_complete_analysis()