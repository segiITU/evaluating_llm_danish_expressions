import pandas as pd
from pathlib import Path

def compare_order():
    # Load the original file
    original_file = Path("data/raw/talemaader_leverance_2_uden_labels.csv")
    original_df = pd.read_csv(original_file, sep='\t')
    original_expressions = original_df['talemaade_udtryk'].tolist()
    
    # Load the predictions file
    predictions_file = Path("data/predictions/predicted_labels_claude-3-5-sonnet.csv")
    predictions_df = pd.read_csv(predictions_file)
    prediction_expressions = predictions_df['talemaade_udtryk'].tolist()
    
    # Compare lengths
    original_length = len(original_expressions)
    predictions_length = len(prediction_expressions)
    
    print(f"Original file: {original_length} expressions")
    print(f"Predictions file: {predictions_length} expressions")
    
    # Check if all original expressions are in predictions
    missing_expressions = [expr for expr in original_expressions if expr not in prediction_expressions]
    print(f"Missing expressions: {len(missing_expressions)}")
    if missing_expressions:
        print("First 5 missing expressions:")
        for expr in missing_expressions[:5]:
            print(f"  - {expr}")
    
    # Check if order matches
    order_matches = True
    mismatches = []
    
    for i, (orig, pred) in enumerate(zip(original_expressions, prediction_expressions[:original_length])):
        if orig != pred:
            order_matches = False
            mismatches.append((i, orig, pred))
            if len(mismatches) >= 5:
                break
    
    if order_matches:
        print("✓ Order matches between files")
    else:
        print("✗ Order does not match between files")
        print("First 5 mismatches:")
        for i, orig, pred in mismatches:
            print(f"  Position {i}:")
            print(f"    Original: {orig}")
            print(f"    Prediction: {pred}")
    
    # Save a file with correct ordering if needed
    if not order_matches and predictions_length > 0:
        print("\nCreating a correctly ordered prediction file...")
        # Create a mapping from expression to predicted label
        pred_map = dict(zip(prediction_expressions, predictions_df['predicted_label']))
        
        # Create new DataFrame with original order
        ordered_data = []
        for expr in original_expressions:
            if expr in pred_map:
                ordered_data.append({
                    'talemaade_udtryk': expr,
                    'predicted_label': pred_map[expr]
                })
            else:
                ordered_data.append({
                    'talemaade_udtryk': expr,
                    'predicted_label': None
                })
        
        ordered_df = pd.DataFrame(ordered_data)
        output_path = Path("data/predictions/predicted_labels_claude-3-5-sonnet_ordered.csv")
        ordered_df.to_csv(output_path, index=False)
        print(f"Ordered predictions saved to: {output_path}")

if __name__ == "__main__":
    compare_order()