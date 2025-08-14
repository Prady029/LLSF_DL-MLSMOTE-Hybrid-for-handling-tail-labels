"""
Label-Specific Learning with Specific Features and Class-Dependent Labels (LLSF-DL)

Python implementation of the LLSF-DL algorithm for multi-label classification.
Based on the original MATLAB implementation from TKDE 2016.

This algorithm learns label-specific features and class-dependent labels 
for multi-label classification tasks.
"""

import numpy as np
from scipy.linalg import norm, solve
from scipy.spatial.distance import cdist
from typing import Dict, Any, Optional, Tuple
import warnings


class LLSF_DL:
    """
    Label-Specific Learning with Specific Features and Class-Dependent Labels
    
    This class implements the LLSF-DL algorithm which jointly learns:
    1. Label-specific feature transformations
    2. Class-dependent label correlations
    """
    
    def __init__(self, 
                 alpha: float = 4**(-3),
                 beta: float = 4**(-2), 
                 gamma: float = 4**(-1),
                 rho: float = 0.1,
                 theta_x: float = 1.0,
                 theta_y: float = 1.0,
                 max_iter: int = 100,
                 minimum_loss_margin: float = 0.001,
                 random_state: Optional[int] = None):
        """
        Initialize LLSF-DL.
        
        Parameters:
        -----------
        alpha : float, default=4**(-3)
            Label correlation regularization parameter
        beta : float, default=4**(-2)
            Sparsity regularization for label-specific features
        gamma : float, default=4**(-1)
            Sparsity regularization for label-dependent labels
        rho : float, default=0.1
            Ridge regularization parameter
        theta_x : float, default=1.0
            Feature learning weight
        theta_y : float, default=1.0
            Label learning weight
        max_iter : int, default=100
            Maximum number of iterations
        minimum_loss_margin : float, default=0.001
            Minimum loss improvement threshold
        random_state : int, optional
            Random state for reproducibility
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.max_iter = max_iter
        self.minimum_loss_margin = minimum_loss_margin
        self.random_state = random_state
        
        # Model parameters (learned during training)
        self.W_x: Optional[np.ndarray] = None  # Feature transformation matrix
        self.W_y: Optional[np.ndarray] = None  # Label dependency matrix
        self.n_features: int = 0
        self.n_labels: int = 0
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, Y: np.ndarray, 
            Y_pred: Optional[np.ndarray] = None) -> 'LLSF_DL':
        """
        Fit the LLSF-DL model.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix
        Y : np.ndarray of shape (n_samples, n_labels)
            Training label matrix (binary)
        Y_pred : np.ndarray of shape (n_samples, n_labels), optional
            Predicted labels from another classifier (e.g., LLSF)
            If provided, used for initialization
            
        Returns:
        --------
        self : LLSF_DL
            Fitted estimator
        """
        # Validate inputs
        X, Y = self._validate_inputs(X, Y)
        self.n_features = X.shape[1]
        self.n_labels = Y.shape[1]
        
        # Initialize model parameters
        self._initialize_parameters(X, Y, Y_pred)
        
        # Compute distance matrix for label correlations
        R = self._compute_label_correlation_matrix(Y)
        
        # Precompute matrices for efficiency
        XTX = X.T @ X
        XTY = X.T @ Y
        
        if Y_pred is not None:
            YTY = Y_pred.T @ Y_pred
            XTY_pred = X.T @ Y_pred
        else:
            YTY = Y.T @ Y
            XTY_pred = XTY
        
        # Compute Lipschitz constant for convergence
        lip_constant = self._compute_lipschitz_constant(XTX, YTY, XTY_pred, R)
        step_size = 1.0 / lip_constant
        
        # Optimization loop
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Store previous parameters
            W_x_prev = self.W_x.copy()
            W_y_prev = self.W_y.copy()
            
            # Update W_x (feature transformation matrix)
            grad_x = self._compute_gradient_x(X, Y, XTX, XTY, R)
            self.W_x = W_x_prev - step_size * grad_x
            
            # Update W_y (label dependency matrix)
            grad_y = self._compute_gradient_y(X, Y, Y_pred, YTY, XTY_pred, R)
            self.W_y = W_y_prev - step_size * grad_y
            
            # Apply shrinkage (soft thresholding) for sparsity
            self.W_x = self._soft_threshold(self.W_x, step_size * self.beta)
            self.W_y = self._soft_threshold(self.W_y, step_size * self.gamma)
            
            # Compute current loss
            current_loss = self._compute_loss(X, Y, Y_pred, R)
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.minimum_loss_margin:
                break
                
            prev_loss = current_loss
        
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
            Predicted labels (probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        assert self.W_x is not None, "Model weights must be initialized"
        
        # Simple prediction: X @ W_x gives us the direct label predictions
        predictions = X @ self.W_x
        
        # Apply sigmoid for probabilities
        predictions = self._sigmoid(predictions)
        
        return predictions
    
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
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def _validate_inputs(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input arrays."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        
        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array")
        
        return X, Y
    
    def _initialize_parameters(self, X: np.ndarray, Y: np.ndarray, 
                             Y_pred: Optional[np.ndarray] = None):
        """Initialize model parameters."""
        XTX = X.T @ X
        XTY = X.T @ Y
        
        # Initialize W_x using ridge regression
        try:
            self.W_x = solve(XTX + self.rho * np.eye(self.n_features), XTY)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            self.W_x = np.linalg.pinv(XTX + self.rho * np.eye(self.n_features)) @ XTY
        
        # Initialize W_y
        if Y_pred is not None:
            YTY = Y_pred.T @ Y_pred
            try:
                self.W_y = solve(YTY + self.rho * np.eye(self.n_labels), Y_pred.T @ Y)
            except np.linalg.LinAlgError:
                self.W_y = np.linalg.pinv(YTY + self.rho * np.eye(self.n_labels)) @ (Y_pred.T @ Y)
        else:
            YTY = Y.T @ Y
            try:
                self.W_y = solve(YTY + self.rho * np.eye(self.n_labels), YTY)
            except np.linalg.LinAlgError:
                self.W_y = np.linalg.pinv(YTY + self.rho * np.eye(self.n_labels)) @ YTY
    
    def _compute_label_correlation_matrix(self, Y: np.ndarray) -> np.ndarray:
        """Compute label correlation matrix using cosine distance."""
        # Add small epsilon to avoid division by zero
        Y_eps = Y.T + 1e-10
        # Compute cosine distance matrix
        R = cdist(Y_eps, Y_eps, metric='cosine')
        return R
    
    def _compute_lipschitz_constant(self, XTX: np.ndarray, YTY: np.ndarray,
                                  XTY: np.ndarray, R: np.ndarray) -> float:
        """Compute Lipschitz constant for gradient descent step size."""
        term1 = 3 * (self.theta_x**2 * norm(XTX))**2
        term2 = 3 * (self.alpha * norm(R))**2
        term3 = 3 * (self.theta_x * self.theta_y * norm(XTY.T))**2
        term4 = 2 * (self.theta_y**2 * norm(YTY))**2
        
        lip_constant = np.sqrt(term1 + term2 + term3 + term4)
        return max(lip_constant, 1e-10)  # Avoid zero
    
    def _compute_gradient_x(self, X: np.ndarray, Y: np.ndarray, 
                          XTX: np.ndarray, XTY: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Compute gradient with respect to W_x."""
        assert self.W_x is not None, "W_x must be initialized"
        
        # Feature reconstruction term
        grad_feat = self.theta_x**2 * (XTX @ self.W_x - XTY)
        
        # Label correlation term  
        grad_corr = self.alpha * (self.W_x @ R)
        
        assert self.W_x is not None and self.W_y is not None, "Model weights must be initialized"
        
        # Cross term
        grad_cross = self.theta_x * self.theta_y * (X.T @ (X @ self.W_x @ self.W_y.T - Y))
        
        return grad_feat + grad_corr + grad_cross
    
    def _compute_gradient_y(self, X: np.ndarray, Y: np.ndarray, 
                          Y_pred: Optional[np.ndarray], YTY: np.ndarray,
                          XTY_pred: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Compute gradient with respect to W_y."""
        assert self.W_x is not None and self.W_y is not None, "Model weights must be initialized"
        
        if Y_pred is not None:
            # Label prediction term
            grad_label = self.theta_y**2 * (YTY @ self.W_y - Y_pred.T @ Y)
        else:
            grad_label = self.theta_y**2 * (YTY @ self.W_y - Y.T @ Y)
        
        # Cross term
        transformed_X = X @ self.W_x
        grad_cross = self.theta_x * self.theta_y * (self.W_y @ transformed_X.T @ transformed_X - (transformed_X.T @ Y).T)
        
        return grad_label + grad_cross
    
    def _compute_loss(self, X: np.ndarray, Y: np.ndarray, 
                     Y_pred: Optional[np.ndarray], R: np.ndarray) -> float:
        """Compute the objective function value."""
        assert self.W_x is not None and self.W_y is not None, "Model weights must be initialized"
        
        # Feature reconstruction loss
        feature_loss = 0.5 * self.theta_x**2 * norm(X @ self.W_x - Y, 'fro')**2
        
        # Label prediction loss
        if Y_pred is not None:
            label_loss = 0.5 * self.theta_y**2 * norm(Y_pred @ self.W_y - Y, 'fro')**2
        else:
            label_loss = 0.5 * self.theta_y**2 * norm(Y @ self.W_y - Y, 'fro')**2
        
        # Label correlation loss
        # Correlation loss term
        correlation_loss = 0.5 * self.alpha * np.trace(R @ self.W_x.T @ self.W_x)
        
        # Sparsity regularization
        sparsity_x = self.beta * np.sum(np.abs(self.W_x))
        sparsity_y = self.gamma * np.sum(np.abs(self.W_y))
        
        # Ridge regularization
        ridge_x = 0.5 * self.rho * norm(self.W_x, 'fro')**2
        ridge_y = 0.5 * self.rho * norm(self.W_y, 'fro')**2
        
        total_loss = (feature_loss + label_loss + correlation_loss + 
                     sparsity_x + sparsity_y + ridge_x + ridge_y)
        
        return total_loss
    
    def _soft_threshold(self, matrix: np.ndarray, threshold: float) -> np.ndarray:
        """Apply soft thresholding for sparsity."""
        return np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Stable sigmoid function."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'rho': self.rho,
            'theta_x': self.theta_x,
            'theta_y': self.theta_y,
            'max_iter': self.max_iter,
            'minimum_loss_margin': self.minimum_loss_margin,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'LLSF_DL':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_samples, n_features, n_labels = 100, 20, 5
    X = np.random.randn(n_samples, n_features)
    Y = np.random.binomial(1, 0.3, (n_samples, n_labels))
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    # Train LLSF-DL
    model = LLSF_DL(alpha=0.1, beta=0.01, gamma=0.01, max_iter=50)
    model.fit(X_train, Y_train)
    
    # Make predictions
    Y_pred = model.predict(X_test)
    Y_pred_binary = model.predict_binary(X_test)
    
    print(f"Training data: {X_train.shape}, {Y_train.shape}")
    print(f"Test data: {X_test.shape}, {Y_test.shape}")
    print(f"Predictions shape: {Y_pred.shape}")
    print(f"Model fitted: {model.is_fitted}")
