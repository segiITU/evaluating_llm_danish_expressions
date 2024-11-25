import logging
from pathlib import Path
from typing import List, Dict
from src.utils.data_loader import TalemaaderDataLoader
from src.models.gpt import GPTModel
import pandas as pd
import time
from datetime import datetime
from src.utils.prediction_writer import PredictionWriter
from src.evaluation.metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluation(model_names: List[str]):
    """
    Run evaluation for specified models and save predictions.
    """
    # Create output directory
    output_dir = Path("results/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = TalemaaderDataLoader()
    test_df = loader.load_evaluation_data()
    
    writer = PredictionWriter()
    metrics = EvaluationMetrics()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name in model_names:
        logger.info(f"Starting evaluation for {model_name}")
        start_time = time.time()
        
        # Initialize results storage
        predictions = []
        
        # Initialize model
        model = GPTModel(model_name)
        
        # Process each idiom
        for idx, row in test_df.iterrows():
            try:
                options = {
                    'A': row['A'],
                    'B': row['B'],
                    'C': row['C'],
                    'D': row['D']
                }
                
                prediction = model.predict(row['talemaade_udtryk'], options)
                predictions.append({
                    'idiom': row['talemaade_udtryk'],
                    'prediction': prediction,
                    'predicted_definition': options[prediction]
                })
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1} idioms")
                    
            except Exception as e:
                logger.error(f"Error processing idiom {row['talemaade_udtryk']}: {str(e)}")
                predictions.append({
                    'idiom': row['talemaade_udtryk'],
                    'prediction': 'ERROR',
                    'predicted_definition': str(e)
                })
        
        # Save predictions
        pred_path = writer.save_predictions(
            predictions=predictions,
            model_name=model_name,
            timestamp=timestamp
        )
        
        # Evaluate predictions
        results = metrics.evaluate_predictions(pred_path)
        
        # Log results
        logger.info(f"\nResults for {model_name}:")
        logger.info(f"Accuracy: {results['accuracy']:.2%}")
        logger.info(f"Total samples: {results['total_samples']}")
        logger.info("\nDetailed metrics saved to results/evaluation/")
        
        duration = time.time() - start_time
        logger.info(f"Completed {model_name} evaluation in {duration:.2f} seconds")
        logger.info(f"Results saved to {pred_path}")

if __name__ == "__main__":
    models_to_evaluate = ["gpt-4-0125-preview"]
    run_evaluation(models_to_evaluate)