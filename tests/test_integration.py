"""
Integration tests for the complete LLSF-DL MLSMOTE implementation.

This test module validates the functionality that is currently tested
by the test_codebase() function in evaluate.py.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add python_src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python_src"))

from algorithms.llsf_dl import LLSF_DL
from algorithms.mlsmote import MLSMOTE
from utils.data_utils import create_demo_dataset, DataLoader, CrossValidator
from utils.evaluation import MultiLabelEvaluator
from hybrid_approach import LLSFDLMLSMOTEHybrid


class TestModuleImports:
    """Test that all modules can be imported successfully."""
    
    def test_algorithm_imports(self):
        """Test that algorithm modules import successfully."""
        assert LLSF_DL is not None
        assert MLSMOTE is not None
    
    def test_utility_imports(self):
        """Test that utility modules import successfully."""
        assert DataLoader is not None
        assert CrossValidator is not None
        assert MultiLabelEvaluator is not None
    
    def test_hybrid_imports(self):
        """Test that hybrid approach imports successfully."""
        assert LLSFDLMLSMOTEHybrid is not None


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_demo_dataset_creation(self):
        """Test that demo dataset can be created with expected properties."""
        X, Y = create_demo_dataset()
        
        assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
        assert X.ndim == 2, "X must be 2D array"
        assert Y.ndim == 2, "Y must be 2D array"
        assert X.shape[1] > 0, "X must have features"
        assert Y.shape[1] > 0, "Y must have labels"
        assert np.all((Y == 0) | (Y == 1)), "Y must be binary"


class TestLLSFDL:
    """Test LLSF-DL algorithm."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        X, Y = create_demo_dataset()
        train_size = int(0.8 * len(X))
        return {
            'X_train': X[:train_size],
            'Y_train': Y[:train_size], 
            'X_test': X[train_size:],
            'Y_test': Y[train_size:]
        }
    
    def test_llsf_dl_initialization(self):
        """Test LLSF-DL model initialization."""
        model = LLSF_DL(max_iter=10, random_state=42)
        
        assert model.max_iter == 10
        assert model.random_state == 42
        assert not model.is_fitted
    
    def test_llsf_dl_training(self, sample_data):
        """Test LLSF-DL model training."""
        model = LLSF_DL(max_iter=10, random_state=42)
        
        # Training should not raise exceptions
        model.fit(sample_data['X_train'], sample_data['Y_train'])
        
        assert model.is_fitted
        assert model.W_x is not None
        assert model.W_y is not None
        assert model.W_x.shape[0] == sample_data['X_train'].shape[1]
        assert model.W_x.shape[1] == sample_data['Y_train'].shape[1]
    
    def test_llsf_dl_prediction(self, sample_data):
        """Test LLSF-DL model prediction."""
        model = LLSF_DL(max_iter=10, random_state=42)
        model.fit(sample_data['X_train'], sample_data['Y_train'])
        
        predictions = model.predict(sample_data['X_test'])
        
        assert predictions.shape == sample_data['Y_test'].shape
        assert np.all(predictions >= 0) and np.all(predictions <= 1)


class TestMLSMOTE:
    """Test MLSMOTE algorithm."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        X, Y = create_demo_dataset()
        return X, Y
    
    def test_mlsmote_initialization(self):
        """Test MLSMOTE initialization."""
        mlsmote = MLSMOTE(k=5, random_state=42)
        
        assert mlsmote.k == 5
        assert mlsmote.random_state == 42
    
    def test_mlsmote_resampling(self, sample_data):
        """Test MLSMOTE synthetic sample generation."""
        X, Y = sample_data
        mlsmote = MLSMOTE(k=5, random_state=42)
        
        # Test resampling for first label
        X_synth, Y_synth = mlsmote.fit_resample(X, Y, label_idx=0, n_samples=10)
        
        assert X_synth.shape[0] == 10, "Should generate requested number of samples"
        assert X_synth.shape[1] == X.shape[1], "Features should match original"
        assert Y_synth.shape[0] == 10, "Labels should match samples"
        assert Y_synth.shape[1] == Y.shape[1], "Label dimensions should match"


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def test_evaluation_metrics(self):
        """Test that evaluation metrics can be computed."""
        # Create dummy predictions and true labels
        Y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        Y_pred = np.array([[0.8, 0.2, 0.7], [0.3, 0.9, 0.1], [0.6, 0.8, 0.3]])
        
        evaluator = MultiLabelEvaluator()
        metrics = evaluator.evaluate(Y_true, Y_pred)
        
        # Check that required metrics are present
        required_metrics = [
            'hamming_loss', 'ranking_loss', 'average_precision',
            'micro_f1', 'macro_f1', 'subset_accuracy'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float, np.number))


class TestHybridApproach:
    """Test the hybrid LLSF-DL + MLSMOTE approach."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        X, Y = create_demo_dataset()
        train_size = int(0.8 * len(X))
        return {
            'X_train': X[:train_size],
            'Y_train': Y[:train_size],
            'X_test': X[train_size:],
            'Y_test': Y[train_size:]
        }
    
    def test_hybrid_initialization(self):
        """Test hybrid approach initialization."""
        hybrid = LLSFDLMLSMOTEHybrid(
            llsf_params={'max_iter': 5},
            random_state=42
        )
        
        assert hybrid.random_state == 42
        assert not hybrid.is_fitted
    
    def test_hybrid_training_and_prediction(self, sample_data):
        """Test hybrid approach training and prediction."""
        hybrid = LLSFDLMLSMOTEHybrid(
            llsf_params={'max_iter': 5, 'random_state': 42},
            random_state=42
        )
        
        # Training should complete without errors
        hybrid.fit(sample_data['X_train'], sample_data['Y_train'], method='minority')
        
        assert hybrid.is_fitted
        
        # Prediction should work and return correct shape
        predictions = hybrid.predict(sample_data['X_test'])
        assert predictions.shape == sample_data['Y_test'].shape


def test_complete_pipeline():
    """Integration test for the complete pipeline."""
    # This replicates the test_codebase() functionality
    X, Y = create_demo_dataset()
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    # Test hybrid approach
    hybrid = LLSFDLMLSMOTEHybrid(
        llsf_params={'max_iter': 5, 'random_state': 42},
        random_state=42
    )
    
    hybrid.fit(X_train, Y_train, method='minority')
    predictions = hybrid.predict(X_test)
    
    assert predictions.shape == Y_test.shape
    assert np.all(predictions >= 0) and np.all(predictions <= 1)


if __name__ == "__main__":
    pytest.main([__file__])
