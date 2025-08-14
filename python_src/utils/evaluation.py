"""
Evaluation utilities for multi-label classification

This module provides comprehensive evaluation metrics for multi-label classification
tasks, matching the original MATLAB evaluation functions.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    hamming_loss, precision_score, recall_score, f1_score,
    accuracy_score, precision_recall_curve, average_precision_score
)
import warnings


class MultiLabelEvaluator:
    """
    Comprehensive evaluator for multi-label classification.
    
    Provides various evaluation metrics commonly used in multi-label learning,
    including both threshold-based and ranking-based metrics.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_score: np.ndarray = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray of shape (n_samples, n_labels)
            True binary labels
        y_pred : np.ndarray of shape (n_samples, n_labels)
            Predicted binary labels  
        y_score : np.ndarray of shape (n_samples, n_labels), optional
            Prediction scores/probabilities for ranking metrics
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing all evaluation metrics
        """
        self.metrics = {}
        
        # Validate inputs
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        if y_score is not None:
            y_score = np.asarray(y_score)
            if y_score.shape != y_true.shape:
                raise ValueError("y_score must have same shape as y_true")
        
        # Example-based metrics
        self.metrics.update(self._example_based_metrics(y_true, y_pred))
        
        # Label-based metrics
        self.metrics.update(self._label_based_metrics(y_true, y_pred))
        
        # Ranking-based metrics (require scores)
        if y_score is not None:
            self.metrics.update(self._ranking_based_metrics(y_true, y_score))
        
        # Additional metrics
        self.metrics.update(self._additional_metrics(y_true, y_pred))
        
        return self.metrics
    
    def _validate_inputs(self, y_true: np.ndarray, 
                        y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and convert inputs."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        
        if y_true.ndim != 2:
            raise ValueError("Inputs must be 2D arrays")
        
        return y_true, y_pred
    
    def _example_based_metrics(self, y_true: np.ndarray, 
                              y_pred: np.ndarray) -> Dict[str, float]:
        """Compute example-based evaluation metrics."""
        n_samples = y_true.shape[0]
        metrics = {}
        
        # Example-based precision
        intersection = np.sum(y_true * y_pred, axis=1)
        pred_sum = np.sum(y_pred, axis=1)
        precision_per_example = np.divide(intersection, pred_sum, 
                                        out=np.zeros_like(intersection, dtype=float),
                                        where=pred_sum!=0)
        metrics['example_based_precision'] = np.mean(precision_per_example)
        
        # Example-based recall
        true_sum = np.sum(y_true, axis=1)
        recall_per_example = np.divide(intersection, true_sum,
                                     out=np.zeros_like(intersection, dtype=float),
                                     where=true_sum!=0)
        metrics['example_based_recall'] = np.mean(recall_per_example)
        
        # Example-based F1
        f1_per_example = np.divide(2 * intersection, pred_sum + true_sum,
                                 out=np.zeros_like(intersection, dtype=float),
                                 where=(pred_sum + true_sum)!=0)
        metrics['example_based_f1'] = np.mean(f1_per_example)
        
        # Example-based accuracy (Jaccard index)
        union = np.sum((y_true + y_pred) > 0, axis=1)
        jaccard_per_example = np.divide(intersection, union,
                                      out=np.zeros_like(intersection, dtype=float),
                                      where=union!=0)
        metrics['example_based_accuracy'] = np.mean(jaccard_per_example)
        
        return metrics
    
    def _label_based_metrics(self, y_true: np.ndarray, 
                           y_pred: np.ndarray) -> Dict[str, float]:
        """Compute label-based evaluation metrics."""
        metrics = {}
        
        # Macro-averaged metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics['macro_precision'] = precision_score(y_true, y_pred, 
                                                       average='macro', 
                                                       zero_division=0)
            metrics['macro_recall'] = recall_score(y_true, y_pred, 
                                                 average='macro', 
                                                 zero_division=0)
            metrics['macro_f1'] = f1_score(y_true, y_pred, 
                                         average='macro', 
                                         zero_division=0)
        
        # Micro-averaged metrics
        metrics['micro_precision'] = precision_score(y_true, y_pred, 
                                                   average='micro', 
                                                   zero_division=0)
        metrics['micro_recall'] = recall_score(y_true, y_pred, 
                                             average='micro', 
                                             zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred, 
                                     average='micro', 
                                     zero_division=0)
        
        # Label-based accuracy
        label_accuracies = []
        for i in range(y_true.shape[1]):
            acc = accuracy_score(y_true[:, i], y_pred[:, i])
            label_accuracies.append(acc)
        metrics['label_based_accuracy'] = np.mean(label_accuracies)
        
        return metrics
    
    def _ranking_based_metrics(self, y_true: np.ndarray, 
                             y_score: np.ndarray) -> Dict[str, float]:
        """Compute ranking-based evaluation metrics."""
        metrics = {}
        
        # Hamming loss
        y_pred_binary = (y_score >= 0.5).astype(int)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred_binary)
        
        # Ranking loss
        metrics['ranking_loss'] = self._ranking_loss(y_true, y_score)
        
        # One error
        metrics['one_error'] = self._one_error(y_true, y_score)
        
        # Coverage
        metrics['coverage'] = self._coverage(y_true, y_score)
        
        # Average precision
        metrics['average_precision'] = self._average_precision(y_true, y_score)
        
        return metrics
    
    def _additional_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict[str, float]:
        """Compute additional evaluation metrics."""
        metrics = {}
        
        # Subset accuracy (exact match)
        exact_matches = np.all(y_true == y_pred, axis=1)
        metrics['subset_accuracy'] = np.mean(exact_matches)
        
        # Hamming loss for binary predictions
        metrics['hamming_loss_binary'] = hamming_loss(y_true, y_pred)
        
        return metrics
    
    def _ranking_loss(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute ranking loss."""
        n_samples, n_labels = y_true.shape
        loss = 0.0
        
        for i in range(n_samples):
            relevant = np.where(y_true[i] == 1)[0]
            irrelevant = np.where(y_true[i] == 0)[0]
            
            if len(relevant) == 0 or len(irrelevant) == 0:
                continue
            
            # Count inversions
            inversions = 0
            for rel in relevant:
                for irrel in irrelevant:
                    if y_score[i, rel] <= y_score[i, irrel]:
                        inversions += 1
            
            if len(relevant) * len(irrelevant) > 0:
                loss += inversions / (len(relevant) * len(irrelevant))
        
        return loss / n_samples
    
    def _one_error(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute one error."""
        n_samples = y_true.shape[0]
        errors = 0
        
        for i in range(n_samples):
            # Find the label with highest score
            max_label = np.argmax(y_score[i])
            # Check if it's not a relevant label
            if y_true[i, max_label] != 1:
                errors += 1
        
        return errors / n_samples
    
    def _coverage(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute coverage."""
        n_samples, n_labels = y_true.shape
        coverage_sum = 0.0
        
        for i in range(n_samples):
            if np.sum(y_true[i]) == 0:  # No relevant labels
                continue
            
            # Sort labels by score (descending)
            sorted_indices = np.argsort(-y_score[i])
            
            # Find position of last relevant label
            relevant_labels = np.where(y_true[i] == 1)[0]
            max_rank = 0
            
            for label in relevant_labels:
                rank = np.where(sorted_indices == label)[0][0]
                max_rank = max(max_rank, rank)
            
            coverage_sum += max_rank
        
        return coverage_sum / n_samples
    
    def _average_precision(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute average precision."""
        n_samples = y_true.shape[0]
        ap_sum = 0.0
        
        for i in range(n_samples):
            if np.sum(y_true[i]) == 0:  # No relevant labels
                continue
            
            # Sort by score (descending)
            sorted_indices = np.argsort(-y_score[i])
            sorted_labels = y_true[i, sorted_indices]
            
            # Compute precision at each position
            precisions = []
            relevant_count = 0
            
            for j, label in enumerate(sorted_labels):
                if label == 1:
                    relevant_count += 1
                    precision = relevant_count / (j + 1)
                    precisions.append(precision)
            
            if precisions:
                ap_sum += np.mean(precisions)
        
        return ap_sum / n_samples


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        y_score: np.ndarray = None) -> Dict[str, float]:
    """
    Convenience function for comprehensive evaluation.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray  
        Predicted binary labels
    y_score : np.ndarray, optional
        Prediction scores for ranking metrics
        
    Returns:
    --------
    Dict[str, float]
        Evaluation metrics
    """
    evaluator = MultiLabelEvaluator()
    return evaluator.evaluate(y_true, y_pred, y_score)


def print_evaluation_results(metrics: Dict[str, float], 
                           dataset_name: str = "Dataset") -> None:
    """
    Print evaluation results in a formatted way.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Evaluation metrics
    dataset_name : str
        Name of the dataset
    """
    print(f"\n=== Evaluation Results for {dataset_name} ===")
    print("-" * 50)
    
    # Group metrics by category
    example_based = {k: v for k, v in metrics.items() if 'example_based' in k}
    label_based = {k: v for k, v in metrics.items() if any(x in k for x in ['macro', 'micro', 'label_based'])}
    ranking_based = {k: v for k, v in metrics.items() if k in ['hamming_loss', 'ranking_loss', 'one_error', 'coverage', 'average_precision']}
    other_metrics = {k: v for k, v in metrics.items() if k not in {**example_based, **label_based, **ranking_based}}
    
    # Print each category
    if example_based:
        print("Example-based Metrics:")
        for name, value in example_based.items():
            print(f"  {name:<25}: {value:.4f}")
    
    if label_based:
        print("\nLabel-based Metrics:")
        for name, value in label_based.items():
            print(f"  {name:<25}: {value:.4f}")
    
    if ranking_based:
        print("\nRanking-based Metrics:")
        for name, value in ranking_based.items():
            print(f"  {name:<25}: {value:.4f}")
    
    if other_metrics:
        print("\nOther Metrics:")
        for name, value in other_metrics.items():
            print(f"  {name:<25}: {value:.4f}")
    
    print("-" * 50)


def compute_imbalance_ratio(Y: np.ndarray) -> np.ndarray:
    """
    Compute imbalance ratio for each label.
    
    Parameters:
    -----------
    Y : np.ndarray of shape (n_samples, n_labels)
        Binary label matrix
        
    Returns:
    --------
    np.ndarray of shape (n_labels,)
        Imbalance ratio for each label
    """
    n_samples, n_labels = Y.shape
    ratios = []
    
    for i in range(n_labels):
        positive_count = np.sum(Y[:, i])
        negative_count = n_samples - positive_count
        
        if positive_count == 0:
            ratio = float('inf')
        else:
            ratio = negative_count / positive_count
        
        ratios.append(ratio)
    
    return np.array(ratios)


def identify_tail_labels(Y: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """
    Identify tail labels based on imbalance ratio.
    
    Parameters:
    -----------
    Y : np.ndarray
        Binary label matrix
    threshold : float, default=2.0
        Imbalance ratio threshold for tail labels
        
    Returns:
    --------
    np.ndarray
        Indices of tail labels
    """
    ratios = compute_imbalance_ratio(Y)
    tail_indices = np.where(ratios >= threshold)[0]
    return tail_indices


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_samples, n_labels = 100, 5
    y_true = np.random.binomial(1, 0.3, (n_samples, n_labels))
    y_score = np.random.random((n_samples, n_labels))
    y_pred = (y_score >= 0.5).astype(int)
    
    # Evaluate
    metrics = evaluate_predictions(y_true, y_pred, y_score)
    print_evaluation_results(metrics, "Sample Dataset")
    
    # Compute imbalance ratios
    ratios = compute_imbalance_ratio(y_true)
    tail_labels = identify_tail_labels(y_true, threshold=2.0)
    
    print(f"\nImbalance ratios: {ratios}")
    print(f"Tail labels: {tail_labels}")
