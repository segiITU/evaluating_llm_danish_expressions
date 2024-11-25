import pandas as pd
from pathlib import Path
import logging
import sys

"""Validates expressions between prediction file and true labels file.

Checks for:
1. Missing expressions in predictions
2. Extra expressions in predictions
3. Case or whitespace differences
4. Character encoding issues

Usage: python -m src.utils.validate_expressions
Output: Logs validation results and saves discrepancies to data/processed/expression_validation.csv
"""

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"expression_validation_{run_number}.log"
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

def validate_expressions():
    logger = setup_logging()
    
    try:
        # Load files
        predictions_path = Path("data/predictions/predicted_labels_gpt-3.5-turbo.csv")
        true_labels_path = Path("data/raw/talemaader_leverance_2_kun_labels.csv")
        output_path = Path("data/processed/expression_validation.csv")
        
        if not predictions_path.exists():
            logger.error(f"Predictions file not found: {predictions_path}")
            sys.exit(1)
            
        if not true_labels_path.exists():
            logger.error(f"True labels file not found: {true_labels_path}")
            sys.exit(1)
        
        # Load DataFrames
        pred_df = pd.read_csv(predictions_path)
        true_df = pd.read_csv(true_labels_path, sep='\t')
        
        logger.info(f"Loaded {len(pred_df)} predictions and {len(true_df)} true labels")
        
        # Create sets of expressions
        pred_expressions = set(pred_df['talemaade_udtryk'].str.strip())
        true_expressions = set(true_df['talemaade_udtryk'].str.strip())
        
        # Find differences
        missing_in_predictions = true_expressions - pred_expressions
        extra_in_predictions = pred_expressions - true_expressions
        
        # Check for case/whitespace differences
        case_differences = []
        for pred_expr in pred_df['talemaade_udtryk']:
            matches = true_df[true_df['talemaade_udtryk'].str.lower().str.strip() == 
                            pred_expr.lower().strip()]
            if len(matches) > 0 and pred_expr not in matches['talemaade_udtryk'].values:
                case_differences.append({
                    'prediction_expression': pred_expr,
                    'true_expression': matches.iloc[0]['talemaade_udtryk'],
                    'issue': 'Case/whitespace mismatch'
                })
        
        # Prepare validation results
        validation_results = []
        
        # Add missing expressions
        for expr in missing_in_predictions:
            validation_results.append({
                'expression': expr,
                'status': 'Missing in predictions',
                'found_in': 'True labels only'
            })
            
        # Add extra expressions
        for expr in extra_in_predictions:
            validation_results.append({
                'expression': expr,
                'status': 'Extra in predictions',
                'found_in': 'Predictions only'
            })
            
        # Add case differences
        for diff in case_differences:
            validation_results.append({
                'expression': diff['prediction_expression'],
                'status': diff['issue'],
                'found_in': f"True expression: {diff['true_expression']}"
            })
        
        # Log results
        logger.info("\n=== Validation Results ===")
        logger.info(f"Total expressions in predictions: {len(pred_expressions)}")
        logger.info(f"Total expressions in true labels: {len(true_expressions)}")
        logger.info(f"Missing in predictions: {len(missing_in_predictions)}")
        logger.info(f"Extra in predictions: {len(extra_in_predictions)}")
        logger.info(f"Case/whitespace differences: {len(case_differences)}")
        
        if validation_results:
            # Save results to CSV
            results_df = pd.DataFrame(validation_results)
            results_df.to_csv(output_path, index=False)
            logger.info(f"\nValidation issues found! Details saved to: {output_path}")
            
            # Log sample issues
            logger.info("\nSample validation issues:")
            for i, result in enumerate(validation_results[:5], 1):
                logger.info(f"\n{i}. Issue type: {result['status']}")
                logger.info(f"   Expression: {result['expression']}")
                logger.info(f"   Details: {result['found_in']}")
        else:
            logger.info("\nNo validation issues found! All expressions match perfectly.")
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    validate_expressions() 