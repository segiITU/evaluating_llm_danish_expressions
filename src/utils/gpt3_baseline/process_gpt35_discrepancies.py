import pandas as pd
from pathlib import Path
import logging

"""Analyzes prediction discrepancies between GPT-3.5-turbo outputs and true labels.

Takes predicted vs. true label comparisons and generates detailed analysis of incorrect predictions,
including both numerical labels and full definitions for easier analysis.

Usage: python -m src.utils.process_gpt35_discrepancies
Output: Saves analysis to data/processed/only_discrepancies_gpt-3.5-turbo.csv
"""

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "process_gpt35_discrepancies.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_discrepancies():
    logger = setup_logging()
    
    try:
        # Define file paths
        predictions_path = Path("data/predictions/predicted_labels_gpt-3.5-turbo.csv")
        true_labels_path = Path("data/raw/talemaader_leverance_2_kun_labels.csv")
        options_path = Path("data/raw/talemaader_leverance_2_uden_labels.csv")
        output_path = Path("data/processed/only_discrepancies_gpt-3.5-turbo.csv")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading predictions and true labels...")
        pred_df = pd.read_csv(predictions_path)
        true_df = pd.read_csv(true_labels_path, sep='\t')
        options_df = pd.read_csv(options_path, sep='\t')
        
        # Remove any rows with missing predictions
        pred_df = pred_df.dropna(subset=['predicted_label'])
        
        # Merge predictions with true labels
        logger.info("Merging predictions with true labels...")
        merged_df = pd.merge(
            pred_df,
            true_df[['talemaade_udtryk', 'korrekt_def']],
            on='talemaade_udtryk',
            how='inner'  # Changed to inner join to only keep matched predictions
        )
        
        # Identify discrepancies
        merged_df = merged_df.rename(columns={'korrekt_def': 'true_label'})
        merged_df['predicted_label'] = merged_df['predicted_label'].astype(int)
        merged_df['true_label'] = merged_df['true_label'].astype(int)
        merged_df['Discrepancy'] = merged_df['predicted_label'] != merged_df['true_label']
        
        # Filter for discrepancies only
        discrepancy_df = merged_df[merged_df['Discrepancy']].copy()
        
        # Get predicted and true definitions
        logger.info("Adding full definitions for discrepancies...")
        
        def get_definition(row, label_col):
            options_row = options_df[options_df['talemaade_udtryk'] == row['talemaade_udtryk']].iloc[0]
            label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            label = label_map[row[label_col]]
            return options_row[label]
        
        discrepancy_df['predicted_definition'] = discrepancy_df.apply(
            lambda row: get_definition(row, 'predicted_label'), axis=1
        )
        discrepancy_df['true_definition'] = discrepancy_df.apply(
            lambda row: get_definition(row, 'true_label'), axis=1
        )
        
        # Organize columns
        columns_order = [
            'talemaade_udtryk',
            'predicted_label',
            'true_label',
            'predicted_definition',
            'true_definition'
        ]
        discrepancy_df = discrepancy_df[columns_order]
        
        # Calculate statistics
        total_predictions = len(merged_df)
        total_discrepancies = len(discrepancy_df)
        accuracy = (total_predictions - total_discrepancies) / total_predictions
        
        # Log statistics
        logger.info("\n=== Discrepancy Analysis ===")
        logger.info(f"Total valid predictions analyzed: {total_predictions}")
        logger.info(f"Total discrepancies found: {total_discrepancies}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        
        # Save results
        discrepancy_df.to_csv(output_path, index=False)
        logger.info(f"\nDiscrepancy analysis saved to: {output_path}")
        
        # Log sample discrepancies
        if not discrepancy_df.empty:
            logger.info("\nSample Discrepancies (first 3):")
            for _, row in discrepancy_df.head(3).iterrows():
                logger.info(f"\nExpression: {row['talemaade_udtryk']}")
                logger.info(f"Predicted (Label {row['predicted_label']}): {row['predicted_definition']}")
                logger.info(f"True (Label {row['true_label']}): {row['true_definition']}")
        
    except Exception as e:
        logger.error(f"Error during discrepancy processing: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    process_discrepancies() 