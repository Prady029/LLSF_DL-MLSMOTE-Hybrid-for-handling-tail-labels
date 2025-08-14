"""
Data utilities for loading and preprocessing multi-label datasets

This module provides utilities for loading MATLAB .mat files and preprocessing
multi-label datasets for the LLSF-DL MLSMOTE experiments.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


class DataLoader:
    """
    Data loader for multi-label datasets.
    
    Handles loading of .mat files and preprocessing of features and labels.
    """
    
    def __init__(self, data_path: Union[str, Path] = None):
        """
        Initialize DataLoader.
        
        Parameters:
        -----------
        data_path : str or Path, optional
            Path to the datasets directory
        """
        if data_path is None:
            self.data_path = Path(__file__).parent.parent.parent / "Datasets"
        else:
            self.data_path = Path(data_path)
    
    def load_dataset(self, dataset_name: str, 
                    normalize_features: bool = True,
                    feature_scaling: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a multi-label dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset (without .mat extension)
        normalize_features : bool, default=True
            Whether to normalize features
        feature_scaling : str, default='standard'
            Type of feature scaling ('standard', 'minmax', or 'none')
            
        Returns:
        --------
        tuple
            (X, Y) where X is features and Y is labels
        """
        # Construct file path
        file_path = self.data_path / f"{dataset_name}.mat"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load .mat file
        try:
            data = sio.loadmat(str(file_path))
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {e}")
        
        # Extract features and labels
        X, Y = self._extract_features_labels(data, dataset_name)
        
        # Validate data
        X, Y = self._validate_data(X, Y)
        
        # Normalize features if requested
        if normalize_features and feature_scaling != 'none':
            X = self._scale_features(X, feature_scaling)
        
        return X, Y
    
    def _extract_features_labels(self, data: Dict[str, Any], 
                               dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from loaded .mat data.
        
        Parameters:
        -----------
        data : dict
            Loaded .mat file data
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        tuple
            (features, labels)
        """
        # Common variable names in .mat files
        possible_feature_names = ['data', 'X', 'features', 'train_data']
        possible_label_names = ['target', 'Y', 'labels', 'train_target']
        
        # Try to find features
        X = None
        for name in possible_feature_names:
            if name in data:
                X = data[name]
                break
        
        if X is None:
            # Look for any numeric array that could be features
            for key, value in data.items():
                if (isinstance(value, np.ndarray) and 
                    value.ndim == 2 and 
                    not key.startswith('__')):
                    if X is None or value.shape[1] > X.shape[1]:  # Prefer larger feature dimension
                        X = value
        
        # Try to find labels
        Y = None
        for name in possible_label_names:
            if name in data:
                Y = data[name]
                break
        
        if Y is None:
            # Look for binary arrays that could be labels
            for key, value in data.items():
                if (isinstance(value, np.ndarray) and 
                    value.ndim == 2 and 
                    not key.startswith('__') and
                    np.all(np.isin(value, [0, 1]))):
                    Y = value
                    break
        
        if X is None or Y is None:
            available_keys = [k for k in data.keys() if not k.startswith('__')]
            raise ValueError(f"Could not find features and labels in {dataset_name}.mat. "
                           f"Available keys: {available_keys}")
        
        return X, Y
    
    def _validate_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and clean the loaded data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        Y : np.ndarray
            Label matrix
            
        Returns:
        --------
        tuple
            Validated (X, Y)
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.int32)
        
        # Ensure 2D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Check shapes
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Features and labels have different number of samples: "
                           f"{X.shape[0]} vs {Y.shape[0]}")
        
        # Check for invalid values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            warnings.warn("Features contain NaN or infinite values. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure binary labels
        if not np.all(np.isin(Y, [0, 1])):
            warnings.warn("Converting labels to binary (0/1)")
            Y = (Y > 0).astype(np.int32)
        
        return X, Y
    
    def _scale_features(self, X: np.ndarray, scaling_type: str) -> np.ndarray:
        """
        Scale features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        scaling_type : str
            Type of scaling ('standard' or 'minmax')
            
        Returns:
        --------
        np.ndarray
            Scaled features
        """
        if scaling_type == 'standard':
            scaler = StandardScaler()
        elif scaling_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")
        
        return scaler.fit_transform(X)
    
    def create_synthetic_dataset(self, n_samples: int = 200, 
                               n_features: int = 20, 
                               n_labels: int = 5,
                               label_density: float = 0.3,
                               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a synthetic multi-label dataset for testing.
        
        Parameters:
        -----------
        n_samples : int, default=200
            Number of samples
        n_features : int, default=20
            Number of features
        n_labels : int, default=5
            Number of labels
        label_density : float, default=0.3
            Average density of labels (probability of positive label)
        random_state : int, default=42
            Random state for reproducibility
            
        Returns:
        --------
        tuple
            (X, Y) synthetic dataset
        """
        np.random.seed(random_state)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate correlated labels
        W = np.random.randn(n_features, n_labels)
        logits = X @ W
        probabilities = 1 / (1 + np.exp(-logits))
        
        # Add label correlations
        for i in range(n_labels):
            for j in range(i + 1, n_labels):
                correlation = np.random.uniform(-0.3, 0.3)
                probabilities[:, j] += correlation * probabilities[:, i]
        
        # Ensure probabilities are in [0, 1]
        probabilities = np.clip(probabilities, 0, 1)
        
        # Adjust for desired label density
        current_density = np.mean(probabilities)
        probabilities = probabilities * (label_density / current_density)
        probabilities = np.clip(probabilities, 0, 1)
        
        # Generate binary labels
        Y = np.random.binomial(1, probabilities)
        
        return X, Y


class CrossValidator:
    """
    Cross-validation utilities for multi-label datasets.
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize CrossValidator.
        
        Parameters:
        -----------
        n_splits : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def split(self, X: np.ndarray, Y: np.ndarray):
        """
        Generate cross-validation splits.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        Y : np.ndarray
            Label matrix
            
        Yields:
        -------
        tuple
            (train_indices, test_indices) for each fold
        """
        for train_idx, test_idx in self.kfold.split(X):
            yield train_idx, test_idx
    
    def get_train_test_split(self, X: np.ndarray, Y: np.ndarray, 
                           fold_idx: int) -> Tuple[np.ndarray, np.ndarray, 
                                                  np.ndarray, np.ndarray]:
        """
        Get a specific train/test split.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        Y : np.ndarray
            Label matrix
        fold_idx : int
            Index of the fold (0-based)
            
        Returns:
        --------
        tuple
            (X_train, X_test, Y_train, Y_test)
        """
        splits = list(self.split(X, Y))
        if fold_idx >= len(splits):
            raise ValueError(f"Fold index {fold_idx} out of range [0, {len(splits)-1}]")
        
        train_idx, test_idx = splits[fold_idx]
        
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        return X_train, X_test, Y_train, Y_test


def load_dataset(dataset_name: str, data_path: Optional[str] = None,
                normalize_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    data_path : str, optional
        Path to datasets directory
    normalize_features : bool, default=True
        Whether to normalize features
        
    Returns:
    --------
    tuple
        (X, Y) loaded dataset
    """
    loader = DataLoader(data_path)
    return loader.load_dataset(dataset_name, normalize_features)


def create_demo_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a demo dataset for testing purposes.
    
    Returns:
    --------
    tuple
        (X, Y) demo dataset
    """
    loader = DataLoader()
    return loader.create_synthetic_dataset(
        n_samples=200, n_features=20, n_labels=5, 
        label_density=0.3, random_state=42
    )


if __name__ == "__main__":
    # Example usage
    print("Testing data utilities...")
    
    # Create synthetic dataset
    X, Y = create_demo_dataset()
    print(f"Synthetic dataset: {X.shape}, {Y.shape}")
    print(f"Label density: {np.mean(Y):.3f}")
    
    # Test cross-validation
    cv = CrossValidator(n_splits=3)
    for i, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
        print(f"Fold {i+1}: Train {len(train_idx)}, Test {len(test_idx)}")
    
    # Test specific split
    X_train, X_test, Y_train, Y_test = cv.get_train_test_split(X, Y, 0)
    print(f"Split 0: Train {X_train.shape}, Test {X_test.shape}")
    
    print("Data utilities test completed!")
