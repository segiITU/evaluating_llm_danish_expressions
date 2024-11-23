import pandas as pd
from pathlib import Path
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_discrepancies(model_name: str = "gpt-4"):
    try:
        # Define file paths with model name
        analyzed_path = Path(f"data/predictions/predicted_labels_{model_name}_analyzed.csv")
        output_path = Path(f"data/talemaader/processed/predicted_labels_{model_name}_discrepancies.csv")
        
        # Load the analyzed predictions
        analyzed_df = pd.read_csv(analyzed_path)
        logger.info(f"Loaded {len(analyzed_df)} analyzed predictions for {model_name}")
        
        # Filter for discrepancies only
        discrepancy_df = analyzed_df[analyzed_df['Discrepancy'] == 'DISCREPANCY'].copy()
        logger.info(f"Found {len(discrepancy_df)} discrepancies")
        
        # Load the multiple choice options
        options_df = pd.read_csv("data/talemaader/raw/talemaader_leverance_2_uden_labels.csv", sep='\t')
        
        # Load the true definitions
        true_defs_df = pd.read_csv("data/talemaader/raw/talemaader_leverance_1.csv", sep='\t')
        
        # Add predicted definition
        def get_predicted_definition(row):
            options_row = options_df[options_df['talemaade_udtryk'] == row['talemaade_udtryk']].iloc[0]
            pred_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[row['predicted_label']]
            return options_row[pred_letter]
            
        discrepancy_df['predicted_definition'] = discrepancy_df.apply(get_predicted_definition, axis=1)
        
        # Add true definition
        discrepancy_df = pd.merge(
            discrepancy_df,
            true_defs_df[['talemaade_udtryk', 'ddo_definition']],
            on='talemaade_udtryk',
            how='left'
        )
        
        # Rename for clarity
        discrepancy_df = discrepancy_df.rename(columns={'ddo_definition': 'true_definition'})
        
        # Reorder columns for better readability
        columns_order = [
            'talemaade_udtryk',
            'predicted_label',
            'true_label',
            'predicted_definition',
            'true_definition',
            'Discrepancy'
        ]
        discrepancy_df = discrepancy_df[columns_order]
        
        # Create processed directory if it doesn't exist
        output_dir = Path("data/talemaader/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the processed discrepancies
        discrepancy_df.to_csv(output_path, index=False)
        
        logger.info(f"\nProcessing complete for {model_name}:")
        logger.info(f"Total discrepancies processed: {len(discrepancy_df)}")
        logger.info(f"Output saved to: {output_path}")
        
        # Show sample of processed data
        if len(discrepancy_df) > 0:
            logger.info(f"\nSample discrepancy for {model_name} (first row):")
            sample = discrepancy_df.iloc[0]
            logger.info(f"Idiom: {sample['talemaade_udtryk']}")
            logger.info(f"Predicted (Label {sample['predicted_label']}): {sample['predicted_definition']}")
            logger.info(f"True (Label {sample['true_label']}): {sample['true_definition']}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process prediction discrepancies for a specific model.')
    parser.add_argument('--model', type=str, default="gpt-4", 
                      choices=['gpt-4', 'gpt-4o', 'gemini'],
                      help='Model name to process (default: gpt-4)')
    
    args = parser.parse_args()
    process_discrepancies(model_name=args.model)