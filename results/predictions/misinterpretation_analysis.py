import pandas as pd
from pathlib import Path
import os

def generate_misinterpretation_overview():
    results_dir = Path('results/predictions')
    output_path = results_dir / 'overview_misinterpretation.csv'
    
    misinterpretation_files = list(results_dir.glob('misinterpretations_*.csv'))
    
    if not misinterpretation_files:
        print("No misinterpretation files found!")
        return
        
    overview_data = []
    
    for file_path in misinterpretation_files:
        model_name = file_path.stem.replace('misinterpretations_', '')
        
        df = pd.read_csv(file_path)
        
        total_misinterpretations = len(df)
        
        type_counts = df['misinterpretation_type'].value_counts().to_dict()
        
        for category in ['concrete misinterpretation', 'abstract misinterpretation', 'random definition', 'unknown']:
            if category not in type_counts:
                type_counts[category] = a = 0
        
        model_data = {
            'model': model_name,
            'total_misinterpretations': total_misinterpretations,
            'concrete_misinterpretations': type_counts.get('concrete misinterpretation', 0),
            'abstract_misinterpretations': type_counts.get('abstract misinterpretation', 0),
            'random_definitions': type_counts.get('random definition', 0),
            'unknown': type_counts.get('unknown', 0)
        }
        
        if total_misinterpretations > 0:
            model_data['concrete_percent'] = round(100 * model_data['concrete_misinterpretations'] / total_misinterpretations, 2)
            model_data['abstract_percent'] = round(100 * model_data['abstract_misinterpretations'] / total_misinterpretations, 2)
            model_data['random_percent'] = round(100 * model_data['random_definitions'] / total_misinterpretations, 2)
            model_data['unknown_percent'] = round(100 * model_data['unknown'] / total_misinterpretations, 2)
        else:
            model_data['concrete_percent'] = 0
            model_data['abstract_percent'] = 0
            model_data['random_percent'] = 0
            model_data['unknown_percent'] = 0
        
        overview_data.append(model_data)
    
    overview_df = pd.DataFrame(overview_data)
    
    column_order = [
        'model', 'total_misinterpretations',
        'concrete_misinterpretations', 'concrete_percent',
        'abstract_misinterpretations', 'abstract_percent',
        'random_definitions', 'random_percent',
        'unknown', 'unknown_percent'
    ]
    overview_df = overview_df[column_order]
    
    overview_df = overview_df.sort_values('total_misinterpretations')
    
    overview_df.to_csv(output_path, index=False)
    
    print(f"Generated overview of misinterpretations for {len(overview_data)} models")
    print(f"Overview saved to: {output_path}")
    
    print("\nMisinterpretation Overview:")
    print(overview_df[['model', 'total_misinterpretations', 
                      'concrete_percent', 'abstract_percent', 'random_percent']].to_string(index=False))

if __name__ == "__main__":
    generate_misinterpretation_overview()