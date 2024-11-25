import pandas as pd
import os
from pathlib import Path
import glob

def calculate_accuracy(discrepancies_file, total_samples=1000):
    """
    Calculate accuracy for a model based on its discrepancies file.
    
    Args:
        discrepancies_file (str): Path to the discrepancies CSV file
        total_samples (int): Total number of samples in original dataset
    
    Returns:
        float: Accuracy score
    """
    # Read discrepancies file
    df = pd.read_csv(discrepancies_file)
    
    # Calculate wrong predictions (rows - 1 for header)
    wrong_predictions = len(df) - 1
    correct_predictions = total_samples - wrong_predictions
    
    # Calculate accuracy
    return correct_predictions / total_samples

def main():
    # Create results directory if it doesn't exist
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    
    # Get all discrepancies files
    discrepancies_files = glob.glob('data/processed/*_discrepancies_*.csv')
    
    # Calculate accuracy for each model
    results = {}
    for file_path in discrepancies_files:
        model_name = file_path.split('discrepancies_')[-1].replace('.csv', '')
        accuracy = calculate_accuracy(file_path)
        results[model_name] = {'accuracy': accuracy}
    
    # Create results DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Format metrics as percentages
    results_df = results_df.round(4) * 100
    
    # Save results
    results_df.to_csv('results/metrics/model_accuracy.csv')
    
    # Print results
    print("\nModel Accuracy (%):")
    print(results_df)
    
    # Print formatted table
    print("\nFormatted Results:")
    for model, metrics in results_df.iterrows():
        print(f"\n{model}:")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")

if __name__ == "__main__":
    main() 