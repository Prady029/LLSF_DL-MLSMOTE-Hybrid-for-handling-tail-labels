# Python Implementation

This directory contains the Python implementation of the LLSF-DL MLSMOTE hybrid approach.

For general information about the project, see the [main README](../README.md).

## � Development Setup

**Environment Setup:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy scipy scikit-learn pandas matplotlib seaborn tqdm
```

**Run Tests:**
```bash
python evaluate.py --test
```

## 📚 API Reference

### Quick Functions
```python
from python_src import quick_eval, test_codebase

# Test implementation
test_codebase()

# Quick evaluation
results = quick_eval(dataset_idx='demo', method='minority')
```

### Core Classes
```python
from python_src.hybrid_approach import LLSFDLMLSMOTEHybrid
from python_src.algorithms import LLSF_DL, MLSMOTE
from python_src.utils import MultiLabelEvaluator, DataLoader

# Hybrid approach
model = LLSFDLMLSMOTEHybrid()
model.fit(X_train, Y_train, method='minority')
predictions = model.predict(X_test)

# Individual algorithms
llsf = LLSF_DL(max_iter=100)
mlsmote = MLSMOTE(k=5)
```

## 🎯 Examples

**Complete experiment:**
```bash
# Demo with synthetic data
python evaluate.py --quick demo both

# Real datasets (if available)
python evaluate.py --comprehensive 1 minority all
```

**Custom experiment:**
```python
from hybrid_approach import run_experiment

results = run_experiment(
    dataset_name='demo',
    methods=['minority', 'all'],
    save_results=True
)
```

See [PYTHON_IMPLEMENTATION_SUMMARY.md](../PYTHON_IMPLEMENTATION_SUMMARY.md) for detailed implementation notes.
model = LLSFDLMLSMOTEHybrid()
model.fit(X, Y, method='minority')
predictions = model.predict(X)
```

## 📁 Directory Structure

```
python_src/
├── __init__.py                 # Main package initialization
├── config.py                  # Configuration settings
├── evaluate.py                # Main evaluation script (CLI)
├── hybrid_approach.py          # LLSF-DL + MLSMOTE hybrid implementation
├── algorithms/
│   ├── __init__.py
│   └── llsf_dl.py             # LLSF-DL algorithm implementation
├── utils/
│   ├── __init__.py
│   ├── evaluation.py          # Evaluation metrics and utilities
│   └── data_utils.py          # Data loading and preprocessing
└── results/                   # Generated results and reports
```

## 🔧 Key Components

### 1. LLSF-DL Algorithm (`algorithms/llsf_dl.py`)
- Complete Python implementation of Label-Specific Learning with Specific Features and Class-Dependent Labels
- Supports all original optimization parameters
- Uses efficient scipy/numpy operations

### 2. MLSMOTE Implementation (`../src/mlsmote.py`)
- Multi-Label Synthetic Minority Oversampling Technique
- Generates synthetic samples for tail labels
- Compatible with scikit-learn conventions

### 3. Hybrid Approach (`hybrid_approach.py`)
- Combines LLSF-DL with MLSMOTE
- Automatic tail label identification
- Supports multiple application strategies

### 4. Evaluation Framework (`utils/evaluation.py`)
- Comprehensive multi-label evaluation metrics
- Compatible with MATLAB results
- Includes ranking-based and threshold-based metrics

## 📊 Available Datasets

The implementation supports the same datasets as the original MATLAB version:

1. **genbase** - Text categorization (27 features, 32 labels)
2. **emotions** - Music emotion classification (72 features, 6 labels)
3. **rcv1-sample1** - Reuters text categorization (47,236 features, 101 labels)
4. **recreation** - Recreation domain (606 features, 20 labels)
5. **demo** - Synthetic dataset for testing

## 🎯 Methods Available

- **`minority`**: Apply MLSMOTE to tail (minority) labels only
- **`all`**: Apply MLSMOTE to all labels
- **`both`**: Compare both methods
- **`none`**: Baseline LLSF-DL without MLSMOTE

## 📈 Evaluation Metrics

The implementation provides the same metrics as the MATLAB version:

### Example-based Metrics
- Example-based precision, recall, F1
- Example-based accuracy (Jaccard index)

### Label-based Metrics  
- Macro/Micro precision, recall, F1
- Label-based accuracy

### Ranking-based Metrics
- Hamming loss
- Ranking loss
- One error
- Coverage  
- Average precision

### Additional Metrics
- Subset accuracy (exact match)

## 🔄 MATLAB Compatibility

The Python implementation provides equivalent functions to the MATLAB interface:

| MATLAB Function | Python Equivalent | Description |
|----------------|-------------------|-------------|
| `quick_eval()` | `quick_eval()` | Quick evaluation interface |
| `evaluate_llsf_mlsmote()` | `evaluate_llsf_mlsmote()` | Comprehensive evaluation |
| `test_codebase()` | `test_codebase()` | Validation tests |
| `LLSF_DL()` | `LLSF_DL()` | Core algorithm |
| `MLSMOTE()` | `MLSMOTE()` | Oversampling technique |

## ⚙️ Configuration

Modify parameters in `config.py`:

```python
# LLSF-DL parameters
CONFIG.optm_parameter = {
    'alpha': 4**(-3),     # Label correlation
    'beta': 4**(-2),      # Feature sparsity
    'gamma': 4**(-1),     # Label sparsity
    'rho': 0.1,           # Ridge regularization
    'max_iter': 100       # Maximum iterations
}

# MLSMOTE parameters
CONFIG.mlsmote_params = {
    'k': 5,               # Nearest neighbors
    'a': 5                # Synthetic samples per instance
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python evaluate.py --test
```

This validates:
- Module imports
- Synthetic data generation
- LLSF-DL training and prediction
- MLSMOTE synthetic sample generation
- Evaluation metrics computation
- Hybrid approach integration

## 📝 Example Usage Scenarios

### 1. Quick Evaluation
```python
# Test on demo data with both methods
results = quick_eval('demo', 'both')

# Test specific dataset
results = quick_eval(1, 'minority')  # genbase with minority method
```

### 2. Comprehensive Analysis
```python
# Full evaluation with all options
results = evaluate_llsf_mlsmote(
    dataset_idx=2,                    # emotions dataset
    experiment_types=['minority', 'all', 'none'],
    options={
        'save_results': True,
        'compare_methods': True,
        'verbose': True
    }
)
```

### 3. Custom Model Training
```python
from python_src.hybrid_approach import LLSFDLMLSMOTEHybrid
from python_src.utils.data_utils import create_demo_dataset

# Create data
X, Y = create_demo_dataset(n_samples=500, n_features=30, n_labels=8)

# Initialize model with custom parameters
model = LLSFDLMLSMOTEHybrid(
    llsf_params={'alpha': 0.1, 'beta': 0.01, 'max_iter': 50},
    mlsmote_params={'k': 3, 'a': 3},
    tail_threshold=1.5
)

# Train and evaluate
model.fit(X, Y, method='minority')
metrics = model.evaluate(X_test, Y_test)
```

## 🔍 Performance Considerations

The Python implementation is optimized for:
- **Memory efficiency**: Uses sparse matrices where appropriate
- **Computational speed**: Leverages NumPy/SciPy optimized operations
- **Scalability**: Handles large datasets efficiently
- **Numerical stability**: Robust to edge cases and numerical issues

## 🤝 Contributing

1. Follow the existing code structure and documentation style
2. Add comprehensive tests for new features
3. Ensure compatibility with the MATLAB interface
4. Update this README for significant changes

## 📊 Expected Results

The Python implementation should produce results very close to the MATLAB version. Small numerical differences may occur due to:
- Different random number generators
- Floating-point precision differences
- Library implementation variations

These differences are typically within acceptable tolerances for machine learning applications.

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Dataset not found**: Use demo dataset or check file paths
3. **Memory issues**: Reduce dataset size or adjust parameters
4. **Numerical instability**: Check for degenerate cases in data

### Getting Help

1. Run `python evaluate.py --test` to validate installation
2. Check the generated log files for detailed error information
3. Review the configuration parameters for your specific use case

## 📄 License

This Python implementation follows the same license as the original MATLAB codebase.
