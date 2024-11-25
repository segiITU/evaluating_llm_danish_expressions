import pandas as pd
from pathlib import Path

def analyze_misinterpretations():
    # Read the labels file
    labels_df = pd.read_csv('data/raw/talemaader_leverance_2_kun_labels.csv', sep='\t')
    
    # Get list of discrepancy files
    processed_dir = Path('data/processed')
    discrepancy_files = list(processed_dir.glob('only_discrepancies_*.csv'))
    
    for file_path in discrepancy_files:
        # Extract LLM name from filename
        llm_name = file_path.stem.split('_')[-1]
        
        # Read discrepancy file
        disc_df = pd.read_csv(file_path)
        
        # Initialize misinterpretation column
        misinterpretations = []
        
        # For each discrepancy
        for _, row in disc_df.iterrows():
            # Find corresponding row in labels
            label_row = labels_df[labels_df['talemaade_udtryk'] == row['talemaade_udtryk']].iloc[0]
            pred_label = row['predicted_label']
            
            # Determine misinterpretation type
            if pred_label == label_row['falsk1']:
                mistype = 'concrete misinterpretation'
            elif pred_label == label_row['falsk2']:
                mistype = 'abstract misinterpretation'
            elif pred_label == label_row['falsk3']:
                mistype = 'random definition'
            else:
                mistype = 'unknown'
                
            misinterpretations.append(mistype)
        
        # Create output dataframe with required columns
        output_df = pd.DataFrame({
            'talemaade_udtryk': disc_df['talemaade_udtryk'],
            'predicted_label': disc_df['predicted_label'],
            'misinterpretation_type': misinterpretations,
            'true_label': disc_df['true_label']
        })
        
        # Save to results directory
        output_path = Path(f'results/predictions/misinterpretations_{llm_name}.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"Processed {llm_name}")

if __name__ == "__main__":
    analyze_misinterpretations()