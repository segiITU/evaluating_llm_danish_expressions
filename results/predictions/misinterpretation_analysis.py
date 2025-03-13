import pandas as pd
from pathlib import Path

def analyze_misinterpretations():
    labels_df = pd.read_csv('data/raw/talemaader_leverance_2_kun_labels.csv', sep='\t')
    
    processed_dir = Path('data/processed')
    discrepancy_files = list(processed_dir.glob('only_discrepancies_*.csv'))
    
    print(f"Found {len(discrepancy_files)} discrepancy files:")
    for file in discrepancy_files:
        print(f"- {file.name}")
    
    # Prepare overview data for all models
    overview_data = []
    
    for file_path in discrepancy_files:
        llm_name = file_path.stem.split('only_discrepancies_')[-1]
        
        disc_df = pd.read_csv(file_path)
        misinterpretations = []
        
        for _, row in disc_df.iterrows():
            label_rows = labels_df[labels_df['talemaade_udtryk'] == row['talemaade_udtryk']]
            if len(label_rows) == 0:
                print(f"Warning: Couldn't find '{row['talemaade_udtryk']}' in labels file")
                misinterpretations.append('unknown')
                continue
                
            label_row = label_rows.iloc[0]
            pred_label = row['predicted_label']
            
            try:
                if pred_label == label_row['falsk1']:
                    mistype = 'concrete misinterpretation'
                elif pred_label == label_row['falsk2']:
                    mistype = 'abstract misinterpretation'
                elif pred_label == label_row['falsk3']:
                    mistype = 'random definition'
                else:
                    mistype = 'unknown'
            except Exception as e:
                print(f"Error comparing labels: {e}")
                mistype = 'unknown'
                
            misinterpretations.append(mistype)
        
        output_df = pd.DataFrame({
            'talemaade_udtryk': disc_df['talemaade_udtryk'],
            'predicted_label': disc_df['predicted_label'],
            'misinterpretation_type': misinterpretations,
            'true_label': disc_df['true_label']
        })
        
        # Save individual model results
        output_path = Path(f'results/predictions/misinterpretations_{llm_name}.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"Processed {llm_name}: {len(disc_df)} discrepancies")
        
        # Calculate statistics for overview
        total = len(disc_df)
        concrete = sum(1 for m in misinterpretations if m == 'concrete misinterpretation')
        abstract = sum(1 for m in misinterpretations if m == 'abstract misinterpretation')
        random = sum(1 for m in misinterpretations if m == 'random definition')
        unknown = sum(1 for m in misinterpretations if m == 'unknown')
        
        overview_data.append({
            'model': llm_name,
            'total_misinterpretations': total,
            'concrete_misinterpretations': concrete,
            'concrete_percent': round(concrete/total*100, 2) if total > 0 else 0,
            'abstract_misinterpretations': abstract,
            'abstract_percent': round(abstract/total*100, 2) if total > 0 else 0,
            'random_definitions': random,
            'random_percent': round(random/total*100, 2) if total > 0 else 0,
            'unknown': unknown,
            'unknown_percent': round(unknown/total*100, 2) if total > 0 else 0
        })
    
    # Create and save overview
    if overview_data:
        overview_df = pd.DataFrame(overview_data)
        overview_path = Path('results/predictions/overview_misinterpretation.csv')
        overview_df.to_csv(overview_path, index=False)
        print(f"Overview saved to {overview_path}")

if __name__ == "__main__":
    analyze_misinterpretations()