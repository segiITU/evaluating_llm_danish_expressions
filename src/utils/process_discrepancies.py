import pandas as pd
from pathlib import Path
import logging
import argparse
import os

"""Analyzes prediction discrepancies between model outputs and true labels.

Takes predicted vs. true label comparisons and generates detailed analysis of incorrect predictions,
including both numerical labels and full definitions for easier analysis.

Usage: python -m src.utils.process_discrepancies --model [model_name]
Output: Saves analysis to data/processed/only_discrepancies_[model_name].csv
"""


def setup_logging(model_name: str):
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"discrepancies_{model_name}_{run_number}.log"
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

def process_discrepancies(model_name: str = "gpt-4"):
    logger = setup_logging(model_name)
    
    try:
        analyzed_path = Path(f"data/predictions/predicted_and_gold_labels_{model_name}.csv")
        output_path = Path(f"data/processed/only_discrepancies_{model_name}.csv")
        
        analyzed_df = pd.read_csv(analyzed_path)
        logger.info(f"Loaded {len(analyzed_df)} analyzed predictions for {model_name}")
        
        discrepancy_df = analyzed_df[analyzed_df['Discrepancy'] == 'DISCREPANCY'].copy()
        logger.info(f"Found {len(discrepancy_df)} discrepancies")
        
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
        
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        discrepancy_df.to_csv(output_path, index=False)
        
        logger.info(f"\nProcessing complete for {model_name}:")
        logger.info(f"Total discrepancies processed: {len(discrepancy_df)}")
        logger.info(f"Output saved to: {output_path}")
        
        if len(discrepancy_df) > 0:
            logger.info(f"\nSample discrepancy for {model_name} (first row):")
            sample = discrepancy_df.iloc[0]
            logger.info(f"Expression: {sample['talemaade_udtryk']}")
            logger.info(f"Predicted (Label {sample['predicted_label']}): {sample['predicted_definition']}")
            logger.info(f"True (Label {sample['true_label']}): {sample['true_definition']}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process prediction discrepancies for a specific model.')
    parser.add_argument('--model', type=str, default="gpt-4", 
                  choices=['gpt-4', 'gpt-4o', 'gpt-4o-smaller-prompt', 'gemini', 'llama', 'claude', 
                          'gpt-3.5-one_shot', 'claude-3-5-sonnet', 'claude-3-7-sonnet', 'grok-2'],
                  help='Model name to process (default: gpt-4)')
    
    args = parser.parse_args()
    process_discrepancies(model_name=args.model)