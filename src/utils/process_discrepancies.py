import pandas as pd
from pathlib import Path
import logging
import argparse
import os
from collections import Counter

def setup_logging(model_name: str):
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    run_number = 1
    while True:
        log_file = log_dir / f"discrepancies_yesno_{model_name}_{run_number}.log"
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

def process_discrepancies(model_name: str = "gpt-4"):
    logger = setup_logging(model_name)
    
    try:
        analyzed_path = Path(f"data/predictions/yesno/predicted_and_gold_labels_{model_name}.csv")
        output_path = Path(f"data/processed/yesno/only_discrepancies_{model_name}.csv")
        mislabeled_path = Path(f"data/processed/yesno/{model_name}_mislabeled.csv")
        
        analyzed_df = pd.read_csv(analyzed_path)
        logger.info(f"Loaded {len(analyzed_df)} analyzed predictions for {model_name}")
        
        # Check if the enhanced columns exist
        has_enhanced_data = 'yes_count' in analyzed_df.columns
        
        # Analyze yes count if the column exists
        if has_enhanced_data:
            yes_counts = Counter(analyzed_df['yes_count'])
            
            logger.info("\nYes count analysis:")
            logger.info(f"Total predictions: {len(analyzed_df)}")
            
            for count in sorted(yes_counts.keys()):
                logger.info(f"Idioms with {count} yes: {yes_counts[count]} ({yes_counts[count]/len(analyzed_df):.2%})")
            
            # Calculate yes count accuracy (ideally should be 1 'yes' per idiom)
            correct_yes_count = yes_counts.get(1, 0)
            yes_count_accuracy = correct_yes_count / len(analyzed_df)
            logger.info(f"Yes count accuracy: {yes_count_accuracy:.2%}")
            
            # Multiple yes responses
            multiple_yes = sum(count for count_val, count in yes_counts.items() if count_val > 1)
            logger.info(f"Idioms with more than one yes: {multiple_yes} ({multiple_yes/len(analyzed_df):.2%})")
            
            # No yes responses
            no_yes = yes_counts.get(0, 0)
            logger.info(f"Idioms with no yes: {no_yes} ({no_yes/len(analyzed_df):.2%})")
            
            # Analyze idioms with multiple yes responses
            if multiple_yes > 0:
                # Load label data with category mappings
                labels_df = pd.read_csv("data/raw/talemaader_leverance_2_kun_labels.csv", sep='\t')
                options_df = pd.read_csv("data/raw/talemaader_leverance_2_uden_labels.csv", sep='\t')
                
                # Get idioms with multiple yes responses
                multi_yes_df = analyzed_df[analyzed_df['yes_count'] > 1].copy()
                logger.info(f"\nAnalyzing {len(multi_yes_df)} idioms with multiple yes responses")
                
                multi_yes_data = []
                
                for _, row in multi_yes_df.iterrows():
                    idiom = row['talemaade_udtryk']
                    label_row = labels_df[labels_df['talemaade_udtryk'] == idiom]
                    
                    if len(label_row) == 0:
                        logger.warning(f"Could not find idiom in labels file: {idiom}")
                        continue
                        
                    label_row = label_row.iloc[0]
                    correct_label = int(label_row['korrekt_def'])
                    
                    # Skip if we don't have the response columns
                    if not all(f"{opt}_response" in row for opt in ['A', 'B', 'C', 'D']):
                        logger.warning(f"Response columns missing for idiom: {idiom}")
                        continue
                    
                    # Get yes options
                    yes_options = []
                    for i, opt in enumerate(['A', 'B', 'C', 'D']):
                        if row[f"{opt}_response"] == 1:
                            yes_options.append(i)
                    
                    # Categorize each yes response
                    for option_idx in yes_options:
                        # Skip correct options - we only want mislabeled ones
                        if option_idx == correct_label:
                            continue
                            
                        category = "unknown"
                        
                        # Determine the category of this option
                        if option_idx == label_row['falsk1']:
                            category = "concrete misinterpretation"
                        elif option_idx == label_row['falsk2']:
                            category = "abstract misinterpretation"
                        elif option_idx == label_row['falsk3']:
                            category = "random definition"
                        
                        # Get the definition text for this option
                        option_letter = chr(65 + option_idx)  # Convert 0,1,2,3 to A,B,C,D
                        options_row = options_df[options_df['talemaade_udtryk'] == idiom]
                        definition = options_row.iloc[0][option_letter] if len(options_row) > 0 else "Definition not found"
                        
                        # Add to our analysis data
                        multi_yes_data.append({
                            'talemaade_udtryk': idiom,
                            'yes_count': row['yes_count'],
                            'option_idx': option_idx,
                            'option_letter': option_letter,
                            'category': category,
                            'definition': definition
                        })
                
                # Create and save the mislabeled dataframe
                if multi_yes_data:
                    mislabeled_df = pd.DataFrame(multi_yes_data)
                    
                    # Create the directory if it doesn't exist
                    mislabeled_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    mislabeled_df.to_csv(mislabeled_path, index=False)
                    logger.info(f"Mislabeled definitions saved to: {mislabeled_path}")
                    
                    # Log category distribution
                    categories = Counter(mislabeled_df['category'])
                    logger.info("\nCategory distribution in mislabeled definitions:")
                    total_responses = len(mislabeled_df)
                    for category, count in categories.items():
                        logger.info(f"{category}: {count} ({count/total_responses:.2%})")
        
        # Continue with original discrepancy analysis
        discrepancy_df = analyzed_df[analyzed_df['Discrepancy'] == 'DISCREPANCY'].copy()
        logger.info(f"\nFound {len(discrepancy_df)} discrepancies")
        
        # Skip further processing if there are no discrepancies
        if len(discrepancy_df) == 0:
            logger.info("No discrepancies to process, skipping further analysis")
            return
        
        options_df = pd.read_csv("data/raw/talemaader_leverance_2_uden_labels.csv", sep='\t')
        true_defs_df = pd.read_csv("data/raw/talemaader_leverance_1.csv", sep='\t')
        
        # Fix the predicted definition function
        def get_predicted_definition(row):
            try:
                expr = row['talemaade_udtryk']
                pred_label = int(row['predicted_label'])
                
                # Find the options row for this expression
                options_row = options_df[options_df['talemaade_udtryk'] == expr]
                if len(options_row) == 0:
                    return "Option not found"
                
                options_row = options_row.iloc[0]
                
                # Map numerical label to letter
                pred_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[pred_label]
                
                # Return the definition
                return options_row[pred_letter]
            except Exception as e:
                logger.error(f"Error getting definition: {e}")
                return "Error"
        
        # Apply the function one row at a time
        discrepancy_df['predicted_definition'] = discrepancy_df.apply(
            lambda row: get_predicted_definition(row), 
            axis=1
        )
        
        # Merge with true definitions
        discrepancy_df = pd.merge(
            discrepancy_df,
            true_defs_df[['talemaade_udtryk', 'ddo_definition']],
            on='talemaade_udtryk',
            how='left'
        )
        
        discrepancy_df = discrepancy_df.rename(columns={'ddo_definition': 'true_definition'})
        
        # Categorize discrepancies
        labels_df = pd.read_csv("data/raw/talemaader_leverance_2_kun_labels.csv", sep='\t')
        
        def categorize_error(row):
            try:
                idiom = row['talemaade_udtryk']
                pred_label = int(row['predicted_label'])
                
                label_row = labels_df[labels_df['talemaade_udtryk'] == idiom]
                if len(label_row) == 0:
                    return "unknown"
                
                label_row = label_row.iloc[0]
                
                if pred_label == label_row['falsk1']:
                    return "concrete misinterpretation"
                elif pred_label == label_row['falsk2']:
                    return "abstract misinterpretation"
                elif pred_label == label_row['falsk3']:
                    return "random definition"
                else:
                    return "unknown"
            except Exception as e:
                logger.error(f"Error categorizing: {e}")
                return "error"
        
        discrepancy_df['error_category'] = discrepancy_df.apply(categorize_error, axis=1)
        
        # Add enhanced columns if they exist
        columns_order = [
            'talemaade_udtryk',
            'predicted_label',
            'true_label',
            'error_category',
            'predicted_definition',
            'true_definition',
            'Discrepancy'
        ]
        
        if has_enhanced_data:
            enhanced_columns = ['yes_count', 'yes_options', 'A_response', 'B_response', 'C_response', 'D_response']
            for col in enhanced_columns:
                if col in discrepancy_df.columns:
                    columns_order.append(col)
        
        discrepancy_df = discrepancy_df[columns_order]
        
        output_dir = Path("data/processed/yesno")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        discrepancy_df.to_csv(output_path, index=False)
        
        # Summary statistics for discrepancies
        error_categories = Counter(discrepancy_df['error_category'])
        logger.info("\nError categories in discrepancies:")
        for category, count in error_categories.items():
            logger.info(f"{category}: {count} ({count/len(discrepancy_df):.2%})")
        
        logger.info(f"\nProcessing complete for {model_name} using yes/no approach:")
        logger.info(f"Total discrepancies processed: {len(discrepancy_df)}")
        logger.info(f"Output saved to: {output_path}")
        
        if len(discrepancy_df) > 0:
            logger.info(f"\nSample discrepancy for {model_name} (first row):")
            sample = discrepancy_df.iloc[0]
            logger.info(f"Expression: {sample['talemaade_udtryk']}")
            logger.info(f"Predicted (Label {sample['predicted_label']}): {sample['predicted_definition']}")
            logger.info(f"True (Label {sample['true_label']}): {sample['true_definition']}")
            logger.info(f"Error category: {sample['error_category']}")
            
            if has_enhanced_data:
                logger.info(f"Yes count: {sample['yes_count']}")
                logger.info(f"Yes options: {sample['yes_options']}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process prediction discrepancies for a specific model using yes/no approach.')
    parser.add_argument('--model', type=str, default="gpt-4", 
              choices=['gpt-4', 'gpt-4o', 'gpt-4o-smaller-prompt', 'gemini', 'llama', 'claude', 
                      'gpt-3.5-one_shot', 'claude-3-5-sonnet', 'claude-3-7-sonnet', 'grok-2', 'deepseek', 'gpt-3.5-turbo'],
              help='Model name to process (default: gpt-4)')
    
    args = parser.parse_args()
    process_discrepancies(model_name=args.model)