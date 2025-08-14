"""
LLSF-DL MLSMOTE Hybrid Approach

This module implements the hybrid approach that combines LLSF-DL with MLSMOTE
for handling tail labels in multi-label classification.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from pathlib import Path

from algorithms.llsf_dl import LLSF_DL
from utils.evaluation import MultiLabelEvaluator, compute_imbalance_ratio, identify_tail_labels
from utils.data_utils import DataLoader, CrossValidator
from config import CONFIG

# Import MLSMOTE from the local algorithms directory
from algorithms.mlsmote import MLSMOTE


class LLSFDLMLSMOTEHybrid:
    """
    Hybrid approach combining LLSF-DL with MLSMOTE for tail label handling.
    
    This class implements the complete pipeline:
    1. Identify tail labels based on imbalance ratio
    2. Apply MLSMOTE to generate synthetic samples for tail labels
    3. Train LLSF-DL on the augmented dataset
    4. Evaluate performance
    """
    
    def __init__(self, 
                 llsf_params: Optional[Dict[str, Any]] = None,
                 mlsmote_params: Optional[Dict[str, Any]] = None,
                 tail_threshold: float = 2.0,
                 random_state: int = 42):
        """
        Initialize the hybrid approach.
        
        Parameters:
        -----------
        llsf_params : dict, optional
            Parameters for LLSF-DL algorithm
        mlsmote_params : dict, optional
            Parameters for MLSMOTE algorithm
        tail_threshold : float, default=2.0
            Imbalance ratio threshold for identifying tail labels
        random_state : int, default=42
            Random state for reproducibility
        """
        self.llsf_params = llsf_params or CONFIG.optm_parameter.copy()
        self.mlsmote_params = mlsmote_params or CONFIG.mlsmote_params.copy()
        self.tail_threshold = tail_threshold
        self.random_state = random_state
        
        # Model components
        self.llsf_model = None
        self.mlsmote = None
        self.tail_labels = None
        self.is_fitted = False
        
        # Evaluation results
        self.training_results = {}
        self.evaluation_results = {}
        
        np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, Y: np.ndarray, 
           method: str = 'minority') -> 'LLSFDLMLSMOTEHybrid':
        """
        Fit the hybrid model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        Y : np.ndarray of shape (n_samples, n_labels)
            Training labels (binary)
        method : str, default='minority'
            MLSMOTE application method:
            - 'minority': Apply to minority (tail) labels only
            - 'all': Apply to all labels
            
        Returns:
        --------
        self : LLSFDLMLSMOTEHybrid
            Fitted model
        """
        # Validate inputs
        X, Y = self._validate_inputs(X, Y)
        
        # Identify tail labels
        imbalance_ratios = compute_imbalance_ratio(Y)
        self.tail_labels = identify_tail_labels(Y, self.tail_threshold)
        
        print(f"Identified {len(self.tail_labels)} tail labels out of {Y.shape[1]} total labels")
        print(f"Tail label indices: {self.tail_labels}")
        print(f"Imbalance ratios for tail labels: {imbalance_ratios[self.tail_labels]}")
        
        # Apply MLSMOTE based on method
        if method == 'minority' and len(self.tail_labels) > 0:
            X_augmented, Y_augmented = self._apply_mlsmote_minority(X, Y)
        elif method == 'all':
            X_augmented, Y_augmented = self._apply_mlsmote_all(X, Y)
        else:
            print("No tail labels found or method='none', using original data")
            X_augmented, Y_augmented = X, Y
        
        print(f"Original dataset size: {X.shape}")
        print(f"Augmented dataset size: {X_augmented.shape}")
        
        # Train LLSF-DL on augmented data
        # Initialize LLSF-DL model
        llsf_init_params = self.llsf_params.copy()
        llsf_init_params['random_state'] = self.random_state
        
        # Remove parameters that LLSF_DL doesn't accept
        invalid_params = ['b_quiet']
        for param in invalid_params:
            llsf_init_params.pop(param, None)
            
        self.llsf_model = LLSF_DL(**llsf_init_params)
        self.llsf_model.fit(X_augmented, Y_augmented)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input features.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        np.ndarray of shape (n_samples, n_labels)
            Predicted label probabilities
        """
        if not self.is_fitted or self.llsf_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.llsf_model.predict(X)
    
    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
        threshold : float, default=0.5
            Decision threshold
            
        Returns:
        --------
        np.ndarray
            Binary predictions
        """
        if not self.is_fitted or self.llsf_model is None:
            raise ValueError("Model must be fitted before prediction")
            
        return self.llsf_model.predict_binary(X, threshold)
    
    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        Y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        Dict[str, float]
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        Y_score = self.predict(X_test)
        Y_pred = self.predict_binary(X_test)
        
        # Evaluate using comprehensive metrics
        evaluator = MultiLabelEvaluator()
        metrics = evaluator.evaluate(Y_test, Y_pred, Y_score)
        
        self.evaluation_results = metrics
        return metrics
    
    def _validate_inputs(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input arrays."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.int32)
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        
        return X, Y
    
    def _apply_mlsmote_minority(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply MLSMOTE to minority (tail) labels only."""
        if self.tail_labels is None or len(self.tail_labels) == 0:
            return X, Y
        
        # Initialize MLSMOTE
        self.mlsmote = MLSMOTE(k=self.mlsmote_params['k'], 
                              random_state=self.random_state)
        
        # Generate synthetic samples for tail labels
        synthetic_X_list = []
        synthetic_Y_list = []
        
        for label_idx in self.tail_labels:
            print(f"Applying MLSMOTE to tail label {label_idx}...")
            synth_X, synth_Y = self.mlsmote.fit_resample(
                X, Y, label_idx, self.mlsmote_params['a']
            )
            
            if len(synth_X) > 0:
                synthetic_X_list.append(synth_X)
                synthetic_Y_list.append(synth_Y)
                print(f"Generated {len(synth_X)} synthetic samples for label {label_idx}")
        
        # Combine original and synthetic data
        if synthetic_X_list:
            all_X = [X] + synthetic_X_list
            all_Y = [Y] + synthetic_Y_list
            
            X_augmented = np.vstack(all_X)
            Y_augmented = np.vstack(all_Y)
        else:
            X_augmented, Y_augmented = X, Y
        
        return X_augmented, Y_augmented
    
    def _apply_mlsmote_all(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply MLSMOTE to all labels."""
        # Initialize MLSMOTE
        self.mlsmote = MLSMOTE(k=self.mlsmote_params['k'], 
                              random_state=self.random_state)
        
        # Generate synthetic samples for all labels
        synthetic_X_list = []
        synthetic_Y_list = []
        
        for label_idx in range(Y.shape[1]):
            print(f"Applying MLSMOTE to label {label_idx}...")
            synth_X, synth_Y = self.mlsmote.fit_resample(
                X, Y, label_idx, self.mlsmote_params['a']
            )
            
            if len(synth_X) > 0:
                synthetic_X_list.append(synth_X)
                synthetic_Y_list.append(synth_Y)
                print(f"Generated {len(synth_X)} synthetic samples for label {label_idx}")
        
        # Combine original and synthetic data
        if synthetic_X_list:
            all_X = [X] + synthetic_X_list
            all_Y = [Y] + synthetic_Y_list
            
            X_augmented = np.vstack(all_X)
            Y_augmented = np.vstack(all_Y)
        else:
            X_augmented, Y_augmented = X, Y
        
        return X_augmented, Y_augmented


def run_experiment(dataset_name: str, 
                  method: str = 'minority',
                  n_folds: int = 5,
                  data_path: Optional[str] = None,
                  save_results: bool = True) -> Dict[str, Any]:
    """
    Run a complete experiment on a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    method : str, default='minority'
        MLSMOTE application method ('minority', 'all', or 'none')
    n_folds : int, default=5
        Number of cross-validation folds
    data_path : str, optional
        Path to datasets directory
    save_results : bool, default=True
        Whether to save results
        
    Returns:
    --------
    Dict[str, Any]
        Experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {dataset_name} with method '{method}'")
    print(f"{'='*60}")
    
    # Load dataset
    try:
        if dataset_name.lower() == 'demo':
            loader = DataLoader()
            X, Y = loader.create_synthetic_dataset()
            print("Using synthetic demo dataset")
        else:
            if data_path is not None:
                loader = DataLoader(data_path)
            else:
                loader = DataLoader()
            X, Y = loader.load_dataset(dataset_name)
            print(f"Loaded dataset: {X.shape}, {Y.shape}")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return {'error': str(e)}
    
    # Get dataset-specific parameters
    if dataset_name in CONFIG.dataset_params:
        llsf_params = CONFIG.get_dataset_params(dataset_name)
        print(f"Using dataset-specific parameters for {dataset_name}")
    else:
        llsf_params = CONFIG.optm_parameter.copy()
        print("Using default parameters")
    
    # Cross-validation
    cv = CrossValidator(n_splits=n_folds, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        print("-" * 30)
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # Initialize and train model
        model = LLSFDLMLSMOTEHybrid(
            llsf_params=llsf_params,
            mlsmote_params=CONFIG.mlsmote_params,
            tail_threshold=2.0,
            random_state=42
        )
        
        # Train model
        if method == 'none':
            # Baseline: LLSF-DL without MLSMOTE
            llsf_init_params = llsf_params.copy()
            llsf_init_params['random_state'] = 42
            
            # Remove parameters that LLSF_DL doesn't accept
            invalid_params = ['b_quiet']
            for param in invalid_params:
                llsf_init_params.pop(param, None)
                
            model.llsf_model = LLSF_DL(**llsf_init_params)
            model.llsf_model.fit(X_train, Y_train)
            model.is_fitted = True
        else:
            model.fit(X_train, Y_train, method=method)
        
        # Evaluate
        metrics = model.evaluate(X_test, Y_test)
        fold_results.append(metrics)
        
        # Print fold results
        print(f"Fold {fold_idx + 1} Results:")
        key_metrics = ['hamming_loss', 'ranking_loss', 'average_precision', 
                      'micro_f1', 'macro_f1', 'subset_accuracy']
        for metric in key_metrics:
            if metric in metrics:
                print(f"  {metric:<20}: {metrics[metric]:.4f}")
    
    # Aggregate results
    aggregated_results = _aggregate_fold_results(fold_results)
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"Final Results for {dataset_name} ({method})")
    print(f"{'='*60}")
    
    for metric, values in aggregated_results.items():
        if metric.endswith('_mean'):
            std_metric = metric.replace('_mean', '_std')
            std_val = aggregated_results.get(std_metric, 0.0)
            print(f"{metric[:-5]:<20}: {values:.4f} ± {std_val:.4f}")
    
    # Prepare final results
    results = {
        'dataset': dataset_name,
        'method': method,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'aggregated_results': aggregated_results,
        'dataset_info': {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_labels': Y.shape[1],
            'label_density': float(np.mean(Y))
        }
    }
    
    # Save results if requested
    if save_results:
        _save_results(results, dataset_name, method)
    
    return results


def _aggregate_fold_results(fold_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate results across folds."""
    if not fold_results:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for result in fold_results:
        all_metrics.update(result.keys())
    
    aggregated = {}
    
    for metric in all_metrics:
        values = [result.get(metric, np.nan) for result in fold_results]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_min"] = np.min(values)
            aggregated[f"{metric}_max"] = np.max(values)
    
    return aggregated


def _save_results(results: Dict[str, Any], dataset_name: str, method: str):
    """Save results to file."""
    import json
    from datetime import datetime
    
    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{dataset_name}_{method}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj
    
    # Recursively convert numpy objects
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy(obj)
    
    results_json = recursive_convert(results)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Testing LLSF-DL MLSMOTE Hybrid approach...")
    
    # Run experiment on demo dataset
    results = run_experiment(
        dataset_name='demo',
        method='minority',
        n_folds=3,
        save_results=True
    )
    
    print("\nExperiment completed successfully!")
