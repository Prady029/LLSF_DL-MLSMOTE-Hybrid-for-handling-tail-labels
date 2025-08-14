"""
Configuration settings for LLSF-DL MLSMOTE experiments

This module contains all the hyperparameters and settings used across different experiments.
Based on the original MATLAB configuration with adaptations for Python.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any


class Config:
    """Configuration class containing all experiment parameters."""
    
    def __init__(self):
        # Base paths
        self.base_path = Path(__file__).parent.parent
        self.datasets_path = self.base_path / "Datasets"
        self.results_path = self.base_path / "results"
        
        # Dataset configuration
        self.datasets = {
            1: "genbase",
            2: "emotions", 
            3: "rcv1-sample1",
            4: "recreation"
        }
        
        # Experiment parameters
        self.ttl_fold = 5  # Total number of folds
        self.max_self_iter = 3  # Maximum self-learning iterations
        self.ttl_eva = 15  # Total evaluation metrics
        self.max_iter = 100  # Maximum iterations for optimization
        self.a = 5  # Number of synthetic samples per minority instance
        
        # LLSF-DL optimization parameters
        self.optm_parameter = {
            'max_iter': self.max_iter,
            'minimum_loss_margin': 0.001,
            'b_quiet': True,
            'alpha': 4**(-3),  # label correlation
            'beta': 4**(-2),   # sparsity of label specific features
            'gamma': 4**(-1),  # sparsity of label specific dependent labels
            'rho': 0.1,
            'theta_x': 1.0,
            'theta_y': 1.0
        }
        
        # Dataset-specific parameter recommendations
        # Parameters: alpha, beta, gamma, rho
        self.dataset_params = {
            'emotions': {
                'alpha': 4**5,
                'beta': 4**3,
                'gamma': 4**3,
                'rho': 0.1
            },
            'rcv1-sample1': {
                'alpha': 4**5,
                'beta': 4**3, 
                'gamma': 4**3,
                'rho': 1.0
            },
            'genbase': {
                'alpha': 4**(-3),
                'beta': 4**(-2),
                'gamma': 4**(-1),
                'rho': 0.1
            },
            'recreation': {
                'alpha': 4**6,
                'beta': 4**4,
                'gamma': 4**5,
                'rho': 1.0
            }
        }
        
        # MLSMOTE parameters
        self.mlsmote_params = {
            'k': 5,  # Number of nearest neighbors
            'a': self.a  # Number of synthetic samples per minority instance
        }
        
        # Evaluation metrics
        self.evaluation_metrics = [
            'hamming_loss',
            'ranking_loss', 
            'one_error',
            'coverage',
            'average_precision',
            'macro_precision',
            'macro_recall',
            'macro_f1',
            'micro_precision',
            'micro_recall',
            'micro_f1',
            'subset_accuracy',
            'label_based_accuracy',
            'example_based_precision',
            'example_based_recall'
        ]
        
        # Output settings
        self.save_results = True
        self.generate_plots = True
        self.verbose = True
        self.compare_methods = True
        self.export_report = True
        
    def get_dataset_params(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset-specific parameters."""
        if dataset_name in self.dataset_params:
            params = self.optm_parameter.copy()
            params.update(self.dataset_params[dataset_name])
            return params
        return self.optm_parameter
    
    def create_directories(self):
        """Create necessary directories for results and data."""
        self.results_path.mkdir(exist_ok=True)
        self.datasets_path.mkdir(exist_ok=True)


def get_config() -> Config:
    """Get the configuration instance."""
    return Config()


# Global configuration instance
CONFIG = get_config()
