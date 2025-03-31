import pandas as pd
from pathlib import Path
import logging

def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"misinterpretation_analysis_{run_number}.log"
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

def analyze_misinterpretations():
    logger = setup_logging()
    
    logger.info("Starting misinterpretation analysis...")
    
    labels_df = pd.read_csv('data/raw/talemaader_leverance_2_kun_labels.csv', sep='\t')
    
    # For yesno approach
    yesno_dir = Path('data/processed/yesno')
    discrepancy_files = list(yesno_dir.glob('only_discrepancies_*.csv'))
    
    # For original approach
    orig_dir = Path('data/processed')
    orig_files = list(orig_dir.glob('only_discrepancies_*.csv'))
    
    # Combine both sets of files
    all_files = discrepancy_files + orig_files
    
    logger.info(f"Found {len(all_files)} discrepancy files:")
    for file in all_files:
        logger.info(f"- {file.name}")
    
    # Prepare overview data for all models
    overview_data = []
    
    for file_path in all_files:
        llm_name = file_path.stem.split('only_discrepancies_')[-1]
        
        logger.info(f"Processing {llm_name}...")
        
        try:
            disc_df = pd.read_csv(file_path)
            misinterpretations = []
            
            for _, row in disc_df.iterrows():
                label_rows = labels_df[labels_df['talemaade_udtryk'] == row['talemaade_udtryk']]
                if len(label_rows) == 0:
                    logger.warning(f"Couldn't find '{row['talemaade_udtryk']}' in labels file")
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
                    logger.error(f"Error comparing labels: {e}")
                    mistype = 'unknown'
                    
                misinterpretations.append(mistype)
            
            # Create simplified output DataFrame with only the required columns
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
            logger.info(f"Processed {llm_name}: {len(disc_df)} discrepancies")
            
            # Calculate statistics for overview
            total = len(disc_df)
            if total == 0:
                logger.info(f"No discrepancies found for {llm_name}, skipping")
                continue
                
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
            
        except Exception as e:
            logger.error(f"Error processing {llm_name}: {e}")
    
    # Create and save overview
    if overview_data:
        overview_df = pd.DataFrame(overview_data)
        overview_path = Path('results/predictions/overview_misinterpretation.csv')
        overview_df.to_csv(overview_path, index=False)
        logger.info(f"Overview saved to {overview_path}")
    
    logger.info("Misinterpretation analysis complete")

if __name__ == "__main__":
    analyze_misinterpretations()