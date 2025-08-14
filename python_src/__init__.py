"""
LLSF-DL MLSMOTE Python Implementation

A Python implementation of the hybrid approach combining LLSF-DL with MLSMOTE
for handling tail labels in multi-label classification.
"""

__version__ = "1.0.0"
__author__ = "Python Implementation Team"

# Main exports
from .config import CONFIG, get_config
from .hybrid_approach import LLSFDLMLSMOTEHybrid, run_experiment
from .evaluate import quick_eval, evaluate_llsf_mlsmote, test_codebase

__all__ = [
    'CONFIG',
    'get_config', 
    'LLSFDLMLSMOTEHybrid',
    'run_experiment',
    'quick_eval',
    'evaluate_llsf_mlsmote',
    'test_codebase'
]
