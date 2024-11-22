from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TalemaaderEvaluator:
    """Class for evaluating model performance on the talemåder dataset."""
    
    def __init__(self, model_name: str, save_dir: str = "results/metrics"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_metrics(self, 
                         true_labels: List[str], 
                         predicted_labels: List[str],
                         expressions: List[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            true_labels: List of correct labels (A, B, C, D)
            predicted_labels: List of model predictions
            expressions: Optional list of expressions for error analysis
            
        Returns:
            Dictionary containing various metrics
        """
        # Basic accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(
            true_labels, 
            predicted_labels,
            labels=['A', 'B', 'C', 'D']
        )
        
        # Per-class metrics
        classification_metrics = classification_report(
            true_labels,
            predicted_labels,
            labels=['A', 'B', 'C', 'D'],
            output_dict=True
        )
        
        # Calculate error patterns
        error_analysis = self._analyze_errors(
            true_labels, 
            predicted_labels,
            expressions if expressions else []
        )
        
        # Random baseline comparison
        random_baseline = 0.25  # 1/4 for 4 options
        improvement_over_random = ((accuracy - random_baseline) / random_baseline) * 100
        
        metrics = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "overall_accuracy": accuracy,
            "improvement_over_random": improvement_over_random,
            "confusion_matrix": conf_matrix.tolist(),
            "per_class_metrics": classification_metrics,
            "error_analysis": error_analysis,
            "sample_size": len(true_labels),
        }
        
        return metrics
    
    def _analyze_errors(self, 
                       true_labels: List[str], 
                       predicted_labels: List[str],
                       expressions: List[str]) -> Dict[str, Any]:
        """Analyze patterns in prediction errors."""
        error_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predicted_labels))
                        if true != pred]
        
        error_patterns = {
            "total_errors": len(error_indices),
            "error_rate": len(error_indices) / len(true_labels),
            "common_mistakes": {}
        }
        
        # Analyze common mistake patterns
        for i in error_indices:
            true = true_labels[i]
            pred = predicted_labels[i]
            pattern = f"{true}->{pred}"
            
            if pattern not in error_patterns["common_mistakes"]:
                error_patterns["common_mistakes"][pattern] = {
                    "count": 0,
                    "examples": []
                }
            
            error_patterns["common_mistakes"][pattern]["count"] += 1
            if expressions and len(expressions) > i:
                error_patterns["common_mistakes"][pattern]["examples"].append(expressions[i])
        
        return error_patterns
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray) -> None:
        """Create and save confusion matrix visualization."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['A', 'B', 'C', 'D'],
            yticklabels=['A', 'B', 'C', 'D']
        )
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save plot
        plt.savefig(self.save_dir / f"confusion_matrix_{self.model_name}.png")
        plt.close()
    
    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed metrics as JSON
        metrics_file = self.save_dir / f"metrics_{self.model_name}_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save summary to text file
        summary_file = self.save_dir / f"summary_{self.model_name}_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Summary for {self.model_name}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Improvement over Random: {metrics['improvement_over_random']:.2f}%\n\n")
            
            f.write("Per-Class Metrics:\n")
            for label in ['A', 'B', 'C', 'D']:
                class_metrics = metrics['per_class_metrics'][label]
                f.write(f"\nClass {label}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-score: {class_metrics['f1-score']:.4f}\n")
            
            f.write("\nTop Error Patterns:\n")
            for pattern, data in sorted(
                metrics['error_analysis']['common_mistakes'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:5]:
                f.write(f"  {pattern}: {data['count']} occurrences\n")
        
        logger.info(f"Saved metrics to {metrics_file} and {summary_file}")

# Example usage:
if __name__ == "__main__":
    # Sample data
    true_labels = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'] * 10
    predicted_labels = ['A', 'B', 'C', 'A', 'A', 'B', 'D', 'D'] * 10
    expressions = [f"Expression_{i}" for i in range(80)]
    
    # Create evaluator
    evaluator = TalemaaderEvaluator("claude")
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(true_labels, predicted_labels, expressions)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(np.array(metrics['confusion_matrix']))
    
    # Save results
    evaluator.save_metrics(metrics)