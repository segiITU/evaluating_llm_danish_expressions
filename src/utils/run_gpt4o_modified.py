import os
from pathlib import Path
from src.utils.run_model_predictions import ModelPredictor
from src.config.model_configs import MODEL_CONFIGS
import pandas as pd

def run_modified_gpt4o():
    output_file = Path("data/predictions/predicted_labels_gpt-4o-smaller-prompt.csv")
    
    existing_predictions = None
    if output_file.exists():
        existing_predictions = pd.read_csv(output_file)
        print(f"Found {len(existing_predictions)} existing predictions")
    
    modified_prompt = """Choose the correct definition for the given expression by responding with only a single letter representing your choice (A, B, C, or D).
Sentence: {expression}
Option A: {definition_a}
Option B: {definition_b}
Option C: {definition_c}
Option D: {definition_d}
Your response should be exactly one letter: A, B, C, or D."""

    predictor = ModelPredictor(model_name="gpt-4o")
    predictor.output_file = output_file
    predictor.prompt_template = modified_prompt
    
    predictor.batch_size = 1000
    
    if existing_predictions is not None:
        predictor.skip_expressions = set(existing_predictions['talemaade_udtryk'].values)
        print(f"Will skip {len(predictor.skip_expressions)} already predicted expressions")
    
    predictor.run_predictions()

if __name__ == "__main__":
    run_modified_gpt4o() 