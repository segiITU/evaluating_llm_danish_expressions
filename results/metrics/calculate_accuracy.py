import pandas as pd
import os
from pathlib import Path
import glob
import logging
from collections import Counter

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"calculate_accuracy_{run_number}.log"
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

def calculate_accuracy(model_name, total_samples=1000):
    """
    Calculate accuracy for a model based on its discrepancies or predictions file.
    
    Args:
        model_name (str): Name of the model
        total_samples (int): Total number of samples in original dataset
    
    Returns:
        dict: Dictionary with accuracy metrics
    """
    logger = logging.getLogger(__name__)
    
    # Try to find files in different locations and formats
    metrics = {}
    
    # Method 1: Original approach - count discrepancies
    discrepancies_path = Path(f'data/processed/only_discrepancies_{model_name}.csv')
    if discrepancies_path.exists():
        try:
            disc_df = pd.read_csv(discrepancies_path)
            wrong_predictions = len(disc_df)
            correct_predictions = total_samples - wrong_predictions
            metrics['accuracy'] = correct_predictions / total_samples
            metrics['approach'] = 'original'
            logger.info(f"Calculated accuracy for {model_name} using original approach: {metrics['accuracy']:.4f}")
            return metrics
        except Exception as e:
            logger.warning(f"Could not read {discrepancies_path}: {e}")
    
    # Method 2: New yes/no approach - from predicted_and_gold_labels
    yesno_path = Path(f'data/predictions/yesno/predicted_and_gold_labels_{model_name}.csv')
    if yesno_path.exists():
        try:
            pred_df = pd.read_csv(yesno_path)
            discrepancies = (pred_df['Discrepancy'] == 'DISCREPANCY').sum()
            total = len(pred_df)
            metrics['accuracy'] = (total - discrepancies) / total
            metrics['approach'] = 'yes/no'
            metrics['total_predictions'] = total
            
            # Check if we have enhanced data
            if 'yes_count' in pred_df.columns:
                yes_counts = Counter(pred_df['yes_count'])
                
                # Calculate yes count accuracy
                correct_yes_count = yes_counts.get(1, 0)
                metrics['yes_count_accuracy'] = correct_yes_count / total
                
                # Multiple yes responses
                multiple_yes = sum(yes_counts.get(i, 0) for i in range(2, 5))
                metrics['multiple_yes_percent'] = multiple_yes / total
                
                # No yes responses
                no_yes = yes_counts.get(0, 0)
                metrics['no_yes_percent'] = no_yes / total
                
                # Add raw counts
                for count in sorted(yes_counts.keys()):
                    metrics[f'yes_count_{count}'] = yes_counts[count]
                
                # Calculate accuracy by yes count
                for count in sorted(yes_counts.keys()):
                    count_df = pred_df[pred_df['yes_count'] == count]
                    count_correct = len(count_df[count_df['Discrepancy'] == ''])
                    metrics[f'accuracy_yes_count_{count}'] = count_correct / len(count_df) if len(count_df) > 0 else 0
            
            logger.info(f"Calculated accuracy for {model_name} using yes/no approach: {metrics['accuracy']:.4f}")
            return metrics
        except Exception as e:
            logger.warning(f"Could not read {yesno_path}: {e}")
    
    # If we get here, we couldn't find any data
    logger.warning(f"No data found for model {model_name}")
    metrics['accuracy'] = float('nan')
    metrics['approach'] = 'not found'
    return metrics

def main():
    logger = setup_logging()
    logger.info("Starting accuracy calculation...")
    
    # Create results directory if it doesn't exist
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    
    # Gather all model names from different sources
    models = set()
    
    # Check original approach
    orig_disc_files = glob.glob('data/processed/only_discrepancies_*.csv')
    for file_path in orig_disc_files:
        model_name = Path(file_path).stem.split('only_discrepancies_')[-1]
        models.add(model_name)
    
    # Check yes/no approach
    yesno_disc_files = glob.glob('data/processed/yesno/only_discrepancies_*.csv')
    for file_path in yesno_disc_files:
        model_name = Path(file_path).stem.split('only_discrepancies_')[-1]
        models.add(model_name)
    
    yesno_pred_files = glob.glob('data/predictions/yesno/predicted_and_gold_labels_*.csv')
    for file_path in yesno_pred_files:
        model_name = Path(file_path).stem.split('predicted_and_gold_labels_')[-1]
        models.add(model_name)
    
    logger.info(f"Found {len(models)} models to analyze: {', '.join(models)}")
    
    # Calculate accuracy for each model
    results = {}
    for model_name in models:
        results[model_name] = calculate_accuracy(model_name)
    
    # Create results DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Ensure accuracy is properly formatted
    if 'accuracy' in results_df.columns:
        basic_metrics = ['accuracy']
        enhanced_metrics = [col for col in results_df.columns if col.startswith('yes_') or col.startswith('accuracy_yes_') or col.startswith('multiple_') or col.startswith('no_')]
        
        # Format percentages
        for col in basic_metrics + enhanced_metrics:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: x * 100 if not pd.isna(x) else x)
    
    # Save detailed results
    results_df.to_csv('results/metrics/model_accuracy_detailed.csv')
    
    # Create a simplified version with just key metrics
    key_metrics = ['accuracy', 'approach', 'yes_count_accuracy', 'multiple_yes_percent', 'no_yes_percent']
    key_metrics = [col for col in key_metrics if col in results_df.columns]
    
    simple_df = results_df[key_metrics].copy()
    simple_df.to_csv('results/metrics/model_accuracy.csv')
    
    # Print results
    logger.info("\nModel Accuracy (%):")
    print("\nModel Accuracy (%):")
    print(simple_df.sort_values('accuracy', ascending=False))
    
    # Print formatted table
    logger.info("\nFormatted Results:")
    print("\nFormatted Results:")
    for model, metrics in results_df.sort_values('accuracy', ascending=False).iterrows():
        print(f"\n{model}:")
        print(f"Accuracy: {metrics.get('accuracy', float('nan')):.2f}%")
        if 'yes_count_accuracy' in metrics:
            print(f"Yes Count Accuracy: {metrics['yes_count_accuracy']:.2f}%")
        if 'multiple_yes_percent' in metrics:
            print(f"Multiple Yes: {metrics['multiple_yes_percent']:.2f}%")
        if 'no_yes_percent' in metrics:
            print(f"No Yes: {metrics['no_yes_percent']:.2f}%")
    
    logger.info("Accuracy calculation complete")

if __name__ == "__main__":
    main()