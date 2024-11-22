import pandas as pd
from src.models.gpt import GPTModel
from pathlib import Path
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_sample_with_metrics(n_samples: int = 5):
    print("\nProcessing samples...")
    
    # Create predictions directory if it doesn't exist
    pred_dir = Path("data/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data with explicit encoding and separator
        test_data = pd.read_csv(
            "data/talemaader/raw/talemaader_leverance_2_uden_labels.csv", 
            sep='\t',
            encoding='utf-8'
        )
        
        gold_labels = pd.read_csv(
            "data/talemaader/raw/talemaader_leverance_2_kun_labels.csv",
            sep='\t',
            encoding='utf-8'
        )
        
        # Verify gold labels loaded correctly
        print(f"Loaded {len(gold_labels)} gold labels")
        
        # Take samples
        test_samples = test_data.head(n_samples)
        
        # Initialize model
        model = GPTModel()
        
        # Letter to number mapping
        letter_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
        # Store results
        results = []
        
        # Process each sample
        for idx, row in test_samples.iterrows():
            idiom = row['talemaade_udtryk']
            print(f"\nProcessing idiom {idx + 1}/{n_samples}: {idiom}")
            
            # Get true label first
            gold_row = gold_labels[gold_labels['talemaade_udtryk'] == idiom]
            if len(gold_row) == 0:
                print(f"Warning: No gold label found for {idiom}")
                continue
                
            true_label = gold_row['korrekt_def'].iloc[0]
            
            options = {
                'A': row['A'],
                'B': row['B'],
                'C': row['C'],
                'D': row['D']
            }
            
            try:
                # Get prediction and convert to numeric
                pred_letter = model.predict(idiom, options)
                pred_num = letter_to_num[pred_letter]
                
                # Store result
                results.append({
                    'talemaade_udtryk': idiom,
                    'predicted_label': pred_num,
                    'true_label': true_label
                })
                
                print(f"Prediction: {pred_num}, True label: {true_label}")
                
            except Exception as e:
                print(f"Error processing idiom: {str(e)}")
        
        # Create and save DataFrame
        if results:
            results_df = pd.DataFrame(results)
            output_path = pred_dir / "predicted_labels_gpt4.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            print("\nSample of results:")
            print(results_df)
        else:
            print("No results to save!")
            
    except Exception as e:
        print(f"Error in data processing: {str(e)}")

if __name__ == "__main__":
    test_sample_with_metrics() 