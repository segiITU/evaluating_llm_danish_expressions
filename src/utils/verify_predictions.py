import pandas as pd
from pathlib import Path
import logging
import os

"""Verifies ordering consistency across prediction files for Danish expressions.

Compares five prediction files (GPT-4, GPT-4o, Gemini, Llama, Claude) ensuring:
- Identical number of rows
- Same ordering of expressions
- No duplicates in 'talemaade_udtryk' column

Usage: python src/utils/verify_predictions.py
Output: Logs verification results to logs/verify_predictions_{run_number}.log
"""

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"verify_predictions_{run_number}.log"
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

def compare_prediction_files():
    logger = setup_logging()
    
    try:
        pred_dir = Path("data/predictions")
        models = ['gpt-4', 'gpt-4o', 'gemini', 'llama', 'claude']
        prediction_files = {model: pred_dir / f"predicted_labels_{model}.csv" for model in models}
        
        dataframes = {}
        expressions = {}
        row_counts = {}
        found_files = []
        
        logger.info("Starting verification of prediction files...")
        logger.info("\nChecking file existence and loading data...")
        for model, file_path in prediction_files.items():
            if not file_path.exists():
                logger.error(f"Missing prediction file: {file_path}")
                continue
                
            found_files.append(file_path.name)
            df = pd.read_csv(file_path)
            dataframes[model] = df
            expressions[model] = df['talemaade_udtryk'].tolist()
            row_counts[model] = len(df)
            logger.info(f"Successfully loaded {file_path.name}: {row_counts[model]} expressions")
            
        if not found_files:
            logger.error("No prediction files found!")
            return
            
        logger.info(f"\nFound {len(found_files)} prediction files to compare")
        
        logger.info("\nVerifying row counts...")
        reference_count = row_counts[models[0]]
        row_mismatches = False
        for model, count in row_counts.items():
            if count != reference_count:
                logger.error(f"Row count mismatch - {model}: {count} rows (expected {reference_count})")
                row_mismatches = True
                
        if not row_mismatches:
            logger.info(f"✓ All files contain {reference_count} expressions")
                
        logger.info("\nVerifying expression order...")
        reference_expressions = expressions[models[0]]
        order_mismatches = False
        for i, expr in enumerate(reference_expressions):
            mismatches = []
            for model in models[1:]:
                if model not in expressions:
                    continue
                if i >= len(expressions[model]):
                    mismatches.append(f"{model}:missing")
                elif expressions[model][i] != expr:
                    mismatches.append(f"{model}:{expressions[model][i]}")
            
            if mismatches:
                logger.error(f"Mismatch at position {i}:")
                logger.error(f"Reference ({models[0]}): {expr}")
                logger.error(f"Mismatches: {', '.join(mismatches)}")
                logger.error("---")
                order_mismatches = True
                
        if not order_mismatches:
            logger.info("✓ Expression order matches across all files")
                
        logger.info("\nChecking for duplicate expressions...")
        duplicate_found = False
        for model, expr_list in expressions.items():
            duplicates = pd.Series(expr_list).duplicated()
            if duplicates.any():
                duplicate_found = True
                dup_indices = duplicates[duplicates].index.tolist()
                logger.error(f"\nDuplicates found in {model}:")
                for idx in dup_indices:
                    logger.error(f"Position {idx}: {expr_list[idx]}")
        
        if not duplicate_found:
            logger.info("✓ No duplicate expressions found in any file")
                    
        logger.info("\nVerification Summary:")
        if not any([row_mismatches, order_mismatches, duplicate_found]):
            logger.info(f"✓ All checks passed successfully for:")
            for file_name in found_files:
                logger.info(f"  - {file_name}")
        else:
            logger.warning("⚠ Some checks failed - see details above")
            
        total_expressions = len(reference_expressions)
        logger.info(f"\nTotal expressions verified: {total_expressions}")
        logger.info("Verification complete.")
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        raise

if __name__ == "__main__":
    compare_prediction_files()