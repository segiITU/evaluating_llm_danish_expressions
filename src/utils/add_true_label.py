import pandas as pd
from pathlib import Path
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_true_label(model_name: str = "gpt-4"):
    try:
        # Define paths with model name
        pred_path = Path(f"data/predictions/predicted_labels_{model_name}.csv")
        output_path = pred_path.parent / f'predicted_labels_{model_name}_analyzed.csv'
        
        # Load predictions
        predictions_df = pd.read_csv(pred_path)
        logger.info(f"Loaded {len(predictions_df)} predictions for {model_name}")

        # Load true labels
        labels_path = Path("data/talemaader/raw/talemaader_leverance_2_kun_labels.csv")
        true_labels_df = pd.read_csv(labels_path, sep='\t')
        logger.info(f"Loaded {len(true_labels_df)} true labels")

        # Drop existing true_label column if it exists
        if 'true_label' in predictions_df.columns:
            predictions_df = predictions_df.drop('true_label', axis=1)

        # Merge predictions with true labels
        merged_df = pd.merge(
            predictions_df,
            true_labels_df[['talemaade_udtryk', 'korrekt_def']],
            on='talemaade_udtryk',
            how='left'
        )

        # Add true_label column
        merged_df = merged_df.rename(columns={'korrekt_def': 'true_label'})

        # Add Discrepancy column
        merged_df['Discrepancy'] = ''
        mask = merged_df['predicted_label'].astype(int) != merged_df['true_label'].astype(int)
        merged_df.loc[mask, 'Discrepancy'] = 'DISCREPANCY'

        # Calculate accuracy
        total = len(merged_df)
        discrepancies = (merged_df['Discrepancy'] == 'DISCREPANCY').sum()
        accuracy = (total - discrepancies) / total

        # Save enriched predictions
        merged_df.to_csv(output_path, index=False)

        # Log results
        logger.info(f"\nAnalysis complete for {model_name}:")
        logger.info(f"Total predictions: {total}")
        logger.info(f"Discrepancies found: {discrepancies}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Enriched predictions saved to: {output_path}")

        # Display example discrepancies
        if discrepancies > 0:
            logger.info(f"\nExample discrepancies for {model_name}:")
            discrepancy_examples = merged_df[merged_df['Discrepancy'] == 'DISCREPANCY'].head(5)
            for _, row in discrepancy_examples.iterrows():
                logger.info(f"Idiom: {row['talemaade_udtryk']}")
                logger.info(f"Predicted: {int(row['predicted_label'])}, True: {int(row['true_label'])}")
                logger.info("---")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add true labels to predictions for a specific model.')
    parser.add_argument('--model', type=str, default="gpt-4", 
                      choices=['gpt-4', 'gpt-4o', 'gemini'],
                      help='Model name to process (default: gpt-4)')
    
    args = parser.parse_args()
    add_true_label(model_name=args.model)