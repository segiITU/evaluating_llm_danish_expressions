import pandas as pd
from pathlib import Path
import logging
import sys

"""Verifies GPT-3.5-turbo predictions against true labels file.

Ensures:
1. All expressions from true labels are present in predictions
2. Order matches exactly with true labels file
3. No extra or missing expressions
4. No case/whitespace differences in expressions

Usage: python -m src.utils.verify_gpt35_predictions
Output: Logs verification results to logs/verify_gpt35_{run_number}.log
"""

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"verify_gpt35_{run_number}.log"
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

def verify_predictions():
    logger = setup_logging()
    
    try:
        # Define file paths
        true_labels_path = Path("data/raw/talemaader_leverance_2_kun_labels.csv")
        predictions_path = Path("data/predictions/predicted_labels_gpt-3.5-turbo.csv")
        
        # Check file existence
        if not true_labels_path.exists():
            logger.error(f"True labels file not found: {true_labels_path}")
            sys.exit(1)
        if not predictions_path.exists():
            logger.error(f"Predictions file not found: {predictions_path}")
            sys.exit(1)
            
        # Load DataFrames
        true_df = pd.read_csv(true_labels_path, sep='\t')
        pred_df = pd.read_csv(predictions_path)
        
        logger.info(f"Loaded {len(true_df)} true labels and {len(pred_df)} predictions")
        
        # Basic count verification
        if len(true_df) != len(pred_df):
            logger.error(f"Count mismatch: {len(true_df)} true labels vs {len(pred_df)} predictions")
            
        # Create ordered expression lists
        true_expressions = true_df['talemaade_udtryk'].tolist()
        pred_expressions = pred_df['talemaade_udtryk'].tolist()
        
        # Verify exact matches and order
        mismatches = []
        extra_predictions = []
        missing_predictions = []
        
        # Check each true expression against predictions
        for i, true_expr in enumerate(true_expressions):
            if i >= len(pred_expressions):
                missing_predictions.append(true_expr)
                continue
                
            pred_expr = pred_expressions[i]
            if true_expr != pred_expr:
                mismatches.append({
                    'position': i,
                    'true_expression': true_expr,
                    'pred_expression': pred_expr
                })
                
        # Check for extra predictions
        if len(pred_expressions) > len(true_expressions):
            extra_predictions = pred_expressions[len(true_expressions):]
            
        # Log results
        logger.info("\n=== Verification Results ===")
        
        if not any([mismatches, extra_predictions, missing_predictions]):
            logger.info("✓ All checks passed successfully!")
            logger.info(f"✓ {len(true_expressions)} expressions verified in correct order")
        else:
            logger.warning("⚠ Issues found during verification:")
            
            if mismatches:
                logger.error("\nOrder/Content Mismatches:")
                for mismatch in mismatches:
                    logger.error(f"Position {mismatch['position']}:")
                    logger.error(f"  Expected: {mismatch['true_expression']}")
                    logger.error(f"  Found: {mismatch['pred_expression']}")
                    
            if missing_predictions:
                logger.error("\nMissing Predictions:")
                for expr in missing_predictions:
                    logger.error(f"  - {expr}")
                    
            if extra_predictions:
                logger.error("\nExtra Predictions:")
                for expr in extra_predictions:
                    logger.error(f"  - {expr}")
                    
        # Check for case/whitespace differences
        case_differences = []
        for i, (true_expr, pred_expr) in enumerate(zip(true_expressions, pred_expressions)):
            if true_expr != pred_expr and true_expr.lower().strip() == pred_expr.lower().strip():
                case_differences.append({
                    'position': i,
                    'true_expression': true_expr,
                    'pred_expression': pred_expr
                })
                
        if case_differences:
            logger.warning("\nCase/Whitespace Differences:")
            for diff in case_differences:
                logger.warning(f"Position {diff['position']}:")
                logger.warning(f"  Expected: '{diff['true_expression']}'")
                logger.warning(f"  Found: '{diff['pred_expression']}'")
                
        # Final summary
        logger.info("\nVerification Summary:")
        logger.info(f"Total expressions checked: {len(true_expressions)}")
        logger.info(f"Mismatches found: {len(mismatches)}")
        logger.info(f"Missing predictions: {len(missing_predictions)}")
        logger.info(f"Extra predictions: {len(extra_predictions)}")
        logger.info(f"Case/whitespace differences: {len(case_differences)}")
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    verify_predictions() 