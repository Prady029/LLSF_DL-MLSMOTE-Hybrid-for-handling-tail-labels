# Contributing to LLSF-DL MLSMOTE

We welcome contributions to the LLSF-DL MLSMOTE project! This document provides guidelines for contributing.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/LLSF_DL-MLSMOTE-Hybrid-for-handling-tail-labels.git
   cd LLSF_DL-MLSMOTE-Hybrid-for-handling-tail-labels
   ```

3. **Set up development environment**:
   ```bash
   make setup-dev
   source .venv/bin/activate
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes** and test them:
   ```bash
   make test
   make all-checks
   ```

6. **Submit a pull request**

## 🔧 Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Environment Setup
```bash
# Use the provided Makefile
make setup-dev

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run the main test suite
make test

# Run with pytest (when available)
make test-pytest

# Run specific tests
python -m pytest tests/test_integration.py::TestLLSFDL -v
```

## 📝 Code Style

We follow Python best practices and use automated tools for code quality:

### Formatting
- **Black** for code formatting
- **isort** for import sorting
- Line length: 88 characters

```bash
make format
```

### Linting
- **flake8** for style checking
- **mypy** for type checking

```bash
make lint
make type-check
```

### Type Hints
All public functions should include type hints:

```python
def example_function(data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Example function with proper type hints."""
    pass
```

## 🏗️ Project Structure

```
├── python_src/           # Main Python package
│   ├── algorithms/       # Core algorithms (LLSF-DL, MLSMOTE)
│   ├── utils/           # Utility functions
│   ├── config.py        # Configuration management
│   ├── evaluate.py      # Main evaluation interface
│   └── hybrid_approach.py  # Hybrid implementation
├── tests/               # Test suite
├── src/                # Original MATLAB implementation
└── docs/               # Documentation (if added)
```

## 🧪 Testing Guidelines

### Test Categories
1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Functional Tests**: Test complete workflows

### Writing Tests
- Use descriptive test names
- Include docstrings explaining test purpose
- Test both success and failure cases
- Use fixtures for common test data

Example:
```python
def test_llsf_dl_handles_invalid_input():
    """Test that LLSF-DL raises appropriate errors for invalid input."""
    model = LLSF_DL()
    
    with pytest.raises(ValueError, match="X and Y must have the same number of samples"):
        model.fit(np.array([[1, 2]]), np.array([[1], [0]]))
```

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code follows style guidelines (`make all-checks` passes)
- [ ] Tests pass (`make test` passes)
- [ ] New functionality includes tests
- [ ] Documentation is updated if needed
- [ ] CHANGELOG.md is updated for significant changes

### Pull Request Description
Include:
- **Purpose**: What does this PR accomplish?
- **Changes**: What specific changes were made?
- **Testing**: How was this tested?
- **Breaking Changes**: Any backwards incompatible changes?

### Example PR Description
```
## Purpose
Add support for custom distance metrics in MLSMOTE algorithm

## Changes
- Added `metric` parameter to MLSMOTE constructor
- Updated distance calculation to use specified metric
- Added validation for supported metrics

## Testing
- Added unit tests for new functionality
- Verified existing tests still pass
- Tested with euclidean, manhattan, and cosine metrics

## Breaking Changes
None - new parameter is optional with sensible default
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**: Python version, OS, package versions
2. **Reproduction**: Minimal code to reproduce the issue
3. **Expected vs Actual**: What you expected vs what happened
4. **Error Messages**: Full error traceback if applicable

Use the issue template:
```
**Environment:**
- Python: 3.9.7
- OS: Ubuntu 20.04
- Package version: 1.0.0

**Bug Description:**
Brief description of the issue

**Reproduction:**
```python
# Minimal code to reproduce
```

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Error Message:**
```
Full traceback here
```
```

## 💡 Feature Requests

For new features:
1. **Check existing issues** to avoid duplicates
2. **Describe the use case** - why is this needed?
3. **Propose implementation** if you have ideas
4. **Consider backwards compatibility**

## 🏆 Code Review Process

1. **Automated checks** must pass (CI/CD when available)
2. **Manual review** by maintainers
3. **Discussion** and iteration if needed
4. **Approval** and merge

## 📚 Documentation

### Code Documentation
- Use clear, descriptive docstrings
- Follow NumPy docstring style
- Include examples for complex functions

Example:
```python
def compute_imbalance_ratio(Y: np.ndarray, label_idx: int) -> float:
    """
    Compute imbalance ratio for a specific label.
    
    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_labels)
        Binary label matrix
    label_idx : int
        Index of the label to compute ratio for
        
    Returns
    -------
    float
        Imbalance ratio (positive_samples / negative_samples)
        
    Examples
    --------
    >>> Y = np.array([[1, 0], [0, 1], [1, 1]])
    >>> compute_imbalance_ratio(Y, 0)
    2.0
    """
```

### Adding Documentation
- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new functionality

## 🤝 Community Guidelines

- **Be respectful** and inclusive
- **Help others** learn and contribute
- **Ask questions** if something is unclear
- **Provide constructive feedback**

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Comments**: For implementation questions

Thank you for contributing to LLSF-DL MLSMOTE! 🎉
