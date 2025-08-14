"""
Multi-Label Synthetic Minority Oversampling Technique (MLSMOTE)

Python implementation of MLSMOTE algorithm for handling imbalanced multi-label datasets.
This is a standalone version that doesn't require the original MATLAB-converted code.

Reference: Charte et al. "MLSMOTE: Approaching imbalanced multilabel 
learning through synthetic instance generation" (2015)
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Union, Optional
import warnings


class MLSMOTE:
    """
    Multi-Label Synthetic Minority Oversampling Technique
    
    This class implements the MLSMOTE algorithm for generating synthetic samples
    for minority class instances in multi-label classification tasks.
    """
    
    def __init__(self, k: int = 5, random_state: Optional[int] = None):
        """
        Initialize MLSMOTE.
        
        Parameters:
        -----------
        k : int, default=5
            Number of nearest neighbors to consider
        random_state : int, optional
            Random state for reproducibility
        """
        self.k = k
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X: np.ndarray, Y: np.ndarray, 
                    label_idx: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples for a specific label.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
        Y : np.ndarray of shape (n_samples, n_labels)
            Label matrix (binary)
        label_idx : int
            Index of the label to oversample (0-based)
        n_samples : int
            Number of synthetic samples to generate per minority instance
            
        Returns:
        --------
        train_data : np.ndarray
            Generated synthetic feature vectors
        train_target : np.ndarray
            Corresponding synthetic labels
        """
        # Validate inputs
        if label_idx < 0 or label_idx >= Y.shape[1]:
            raise ValueError(f"Label index {label_idx} is out of range [0, {Y.shape[1]-1}]")
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        
        # Get instances with the target label
        positive_mask = Y[:, label_idx] == 1
        positive_indices = np.where(positive_mask)[0]
        
        if len(positive_indices) == 0:
            warnings.warn(f"No positive instances found for label {label_idx}")
            return np.empty((0, X.shape[1])), np.empty((0, Y.shape[1]))
        
        # Combine features and labels for distance calculation
        data = np.hstack([X, Y])
        min_bag = data[positive_indices]
        
        # Generate synthetic samples
        synthetic_samples = []
        
        for j in range(len(min_bag)):
            # Calculate distances to all other instances in minBag
            current_instance = min_bag[j:j+1]  # Keep 2D shape
            distances = cdist(current_instance, min_bag)[0]
            
            # Sort by distance (ascending)
            sorted_indices = np.argsort(distances)
            
            # Generate n_samples synthetic instances for this minority instance
            for i in range(n_samples):
                synthetic_sample = self._generate_synthetic_sample(
                    min_bag, j, sorted_indices, X.shape[1], Y.shape[1]
                )
                synthetic_samples.append(synthetic_sample)
        
        if not synthetic_samples:
            return np.empty((0, X.shape[1])), np.empty((0, Y.shape[1]))
        
        # Convert to numpy arrays and separate features from labels
        synthetic_data = np.array(synthetic_samples)
        train_data = synthetic_data[:, :X.shape[1]]
        train_target = synthetic_data[:, X.shape[1]:]
        
        return train_data, train_target
    
    def _generate_synthetic_sample(self, min_bag: np.ndarray, current_idx: int,
                                 sorted_indices: np.ndarray, n_features: int,
                                 n_labels: int) -> np.ndarray:
        """
        Generate a single synthetic sample.
        
        Parameters:
        -----------
        min_bag : np.ndarray
            Instances with the target label
        current_idx : int
            Index of current instance in min_bag
        sorted_indices : np.ndarray
            Indices sorted by distance
        n_features : int
            Number of features
        n_labels : int
            Number of labels
            
        Returns:
        --------
        np.ndarray
            Synthetic sample (features + labels)
        """
        # Select neighbors (exclude self at index 0)
        if len(sorted_indices) >= self.k + 1:
            neighbor_indices = sorted_indices[1:self.k + 1]
        else:
            neighbor_indices = sorted_indices[1:] if len(sorted_indices) > 1 else sorted_indices
        
        if len(neighbor_indices) == 0:
            # If no neighbors available, return the original instance
            return min_bag[current_idx].copy()
        
        # Select a random neighbor
        neighbor_idx = np.random.choice(neighbor_indices)
        ref_neighbor = min_bag[neighbor_idx]
        current_instance = min_bag[current_idx]
        
        # Generate synthetic features
        diff = ref_neighbor[:n_features] - current_instance[:n_features]
        offset = diff * np.random.random()
        synthetic_features = current_instance[:n_features] + offset
        
        # Generate synthetic labels using majority voting
        neighbors = min_bag[neighbor_indices]
        
        # Count label occurrences (current instance + neighbors)
        label_counts = current_instance[n_features:].copy()
        label_counts += np.sum(neighbors[:, n_features:], axis=0)
        
        # Apply majority voting rule
        threshold = (len(neighbor_indices) + 1) / 2
        synthetic_labels = (label_counts > threshold).astype(int)
        
        return np.concatenate([synthetic_features, synthetic_labels])


def mlsmote_resample(X: np.ndarray, Y: np.ndarray, label_idx: int, 
                    n_samples: int, k: int = 5, 
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for MLSMOTE resampling.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    Y : np.ndarray  
        Label matrix
    label_idx : int
        Label index to oversample
    n_samples : int
        Number of synthetic samples per minority instance
    k : int, default=5
        Number of nearest neighbors
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        Synthetic features and labels
    """
    mlsmote = MLSMOTE(k=k, random_state=random_state)
    return mlsmote.fit_resample(X, Y, label_idx, n_samples)
