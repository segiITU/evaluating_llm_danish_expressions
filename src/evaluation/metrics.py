import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    def __init__(self, gold_labels_path: str = "data/talemaader/raw/talemaader_leverance_2_kun_labels.csv"):
        self.gold_labels_path = Path(gold_labels_path)
        self._load_gold_labels()
        
    def _load_gold_labels(self):
        """Load gold standard labels."""
        try:
            self.gold_df = pd.read_csv(
                self.gold_labels_path,
                sep='\t',
                encoding='utf-8'
            )
            # Ensure we have required columns
            required_cols = ['talemaade_udtryk', 'correct_label']
            if not all(col in self.gold_df.columns for col in required_cols):
                raise ValueError(f"Gold labels file missing required columns: {required_cols}")
                
        except Exception as e:
            logger.error(f"Error loading gold labels: {str(e)}")
            raise
            
    def evaluate_predictions(self, predictions_path: Path) -> Dict[str, Any]:
        """
        Evaluate model predictions against gold labels.
        
        Args:
            predictions_path: Path to predictions CSV file
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Load predictions
            pred_df = pd.read_csv(predictions_path)
            
            # Merge with gold labels
            eval_df = pd.merge(
                pred_df,
                self.gold_df,
                on='talemaade_udtryk',
                how='inner'
            )
            
            if len(eval_df) == 0:
                raise ValueError("No matching idioms found between predictions and gold labels")
            
            # Calculate metrics
            accuracy = accuracy_score(
                eval_df['correct_label'],
                eval_df['predicted_label']
            )
            
            # Get detailed classification report
            report = classification_report(
                eval_df['correct_label'],
                eval_df['predicted_label'],
                output_dict=True
            )
            
            # Prepare results
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'total_samples': len(eval_df),
                'model': pred_df['model'].iloc[0],
                'timestamp': pred_df['timestamp'].iloc[0]
            }
            
            # Save detailed results
            self._save_detailed_results(eval_df, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {str(e)}")
            raise
            
    def _save_detailed_results(self, eval_df: pd.DataFrame, metrics: Dict[str, Any]):
        """Save detailed evaluation results."""
        try:
            output_dir = Path("results/evaluation")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save per-idiom results
            timestamp = metrics['timestamp']
            model_name = metrics['model']
            
            detailed_path = output_dir / f"{model_name}_detailed_eval_{timestamp}.csv"
            metrics_path = output_dir / f"{model_name}_metrics_{timestamp}.json"
            
            # Save detailed DataFrame
            eval_df.to_csv(detailed_path, index=False)
            
            # Save metrics as JSON
            pd.DataFrame([metrics]).to_json(metrics_path, orient='records')
            
            logger.info(f"Saved detailed evaluation to {detailed_path}")
            logger.info(f"Saved metrics to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise 