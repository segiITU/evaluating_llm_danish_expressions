import pandas as pd
from pathlib import Path
import logging
import argparse
import os


"""Combines model predictions with gold standard labels for evaluation.

Merges model predictions with true labels, calculates accuracy metrics,
and identifies prediction discrepancies for further analysis.

Usage: python -m src.utils.add_true_label --model [model_name]
Output: Saves to data/predictions/predicted_and_gold_labels_[model_name].csv
"""


def setup_logging(model_name: str):
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"analysis_{model_name}_{run_number}.log"
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

def add_true_label(model_name: str = "gpt-4"):
    logger = setup_logging(model_name)
    
    try:
        pred_path = Path(f"data/predictions/predicted_labels_{model_name}.csv")
        output_path = pred_path.parent / f'predicted_and_gold_labels_{model_name}.csv'
        
        predictions_df = pd.read_csv(pred_path)
        logger.info(f"Loaded {len(predictions_df)} predictions for {model_name}")

        labels_path = Path("data/raw/talemaader_leverance_2_kun_labels.csv")
        true_labels_df = pd.read_csv(labels_path, sep='\t')
        logger.info(f"Loaded {len(true_labels_df)} true labels")

        if 'true_label' in predictions_df.columns:
            predictions_df = predictions_df.drop('true_label', axis=1)

        merged_df = pd.merge(
            predictions_df,
            true_labels_df[['talemaade_udtryk', 'korrekt_def']],
            on='talemaade_udtryk',
            how='left'
        )

        merged_df = merged_df.rename(columns={'korrekt_def': 'true_label'})

        merged_df['Discrepancy'] = ''
        mask = merged_df['predicted_label'].astype(int) != merged_df['true_label'].astype(int)
        merged_df.loc[mask, 'Discrepancy'] = 'DISCREPANCY'

        total = len(merged_df)
        discrepancies = (merged_df['Discrepancy'] == 'DISCREPANCY').sum()
        accuracy = (total - discrepancies) / total

        merged_df.to_csv(output_path, index=False)

        logger.info(f"\nAnalysis complete for {model_name}:")
        logger.info(f"Total predictions: {total}")
        logger.info(f"Discrepancies found: {discrepancies}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Enriched predictions saved to: {output_path}")

        if discrepancies > 0:
            logger.info(f"\nExample discrepancies for {model_name}:")
            discrepancy_examples = merged_df[merged_df['Discrepancy'] == 'DISCREPANCY'].head(5)
            for _, row in discrepancy_examples.iterrows():
                logger.info(f"Expression: {row['talemaade_udtryk']}")
                logger.info(f"Predicted: {int(row['predicted_label'])}, True: {int(row['true_label'])}")
                logger.info("---")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add true labels to predictions for a specific model.')
    parser.add_argument('--model', type=str, default="gpt-4", 
                  choices=['gpt-4', 'gpt-4o', 'gpt-4o-smaller-prompt', 'gemini', 'llama', 'claude', 
                           'gpt-3.5-one_shot', 'claude-3-5-sonnet'],
                  help='Model name to process (default: gpt-4)')
    
    args = parser.parse_args()
    add_true_label(model_name=args.model)