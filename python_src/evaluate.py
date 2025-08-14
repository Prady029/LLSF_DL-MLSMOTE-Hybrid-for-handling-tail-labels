"""
Main evaluation script for LLSF-DL MLSMOTE experiments

This script provides the Python equivalent of the MATLAB evaluation functions,
offering both simple and comprehensive evaluation interfaces.
"""

import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime

from config import CONFIG
from hybrid_approach import run_experiment, LLSFDLMLSMOTEHybrid
from utils.data_utils import DataLoader, create_demo_dataset
from utils.evaluation import print_evaluation_results


def quick_eval(dataset_idx: Union[int, str] = 1, 
              method: str = 'both',
              mode: str = 'normal',
              verbose: bool = True) -> Dict[str, Any]:
    """
    Quick evaluation function - Python equivalent of MATLAB quick_eval().
    
    Parameters:
    -----------
    dataset_idx : int, default=1
        Dataset index (1-4) or special values:
        1: genbase, 2: emotions, 3: rcv1-sample1, 4: recreation
        'demo': synthetic demo dataset
        'all': run on all datasets
    method : str, default='both'
        Method to run:
        'minority': MLSMOTE on tail labels only
        'all': MLSMOTE on all labels  
        'both': compare both methods
        'none': baseline LLSF-DL without MLSMOTE
    mode : str, default='normal'
        Execution mode:
        'normal': standard execution
        'silent': minimal output
        'demo': use synthetic data
        'test': quick validation
        'compare': comprehensive comparison
    verbose : bool, default=True
        Whether to print detailed output
        
    Returns:
    --------
    Dict[str, Any]
        Evaluation results
    """
    if mode == 'silent':
        verbose = False
    
    # Handle special dataset cases
    if dataset_idx == 'demo' or mode == 'demo':
        dataset_name = 'demo'
    elif dataset_idx == 'test':
        return _run_test_mode(verbose)
    elif dataset_idx == 'compare':
        return _run_compare_mode(verbose)
    elif dataset_idx == 'all':
        return _run_all_datasets(method, verbose)
    else:
        # Map dataset index to name
        dataset_map = {1: 'genbase', 2: 'emotions', 3: 'rcv1-sample1', 4: 'recreation'}
        if dataset_idx not in dataset_map:
            raise ValueError(f"Invalid dataset index: {dataset_idx}. Use 1-4 or 'demo'/'all'")
        dataset_name = dataset_map[dataset_idx]
    
    if verbose:
        print(f"Running quick evaluation...")
        print(f"Dataset: {dataset_name}")
        print(f"Method: {method}")
        print(f"Mode: {mode}")
    
    # Run experiments based on method
    if method == 'both':
        results = {}
        for m in ['minority', 'all']:
            if verbose:
                print(f"\nRunning method: {m}")
            results[m] = run_experiment(
                dataset_name=dataset_name,
                method=m,
                n_folds=5 if mode != 'test' else 3,
                save_results=True
            )
        
        # Compare results
        if verbose:
            _compare_methods(results, dataset_name)
        
        return results
    
    else:
        return run_experiment(
            dataset_name=dataset_name,
            method=method,
            n_folds=5 if mode != 'test' else 3,
            save_results=True
        )


def evaluate_llsf_mlsmote(dataset_idx: Union[int, str] = 1,
                         experiment_types: Optional[List[str]] = None,
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation function - Python equivalent of MATLAB evaluate_llsf_mlsmote().
    
    Parameters:
    -----------
    dataset_idx : int, default=1
        Dataset index or 'all' for all datasets
    experiment_types : list, default=['minority', 'all']
        List of experiment types to run
    options : dict, optional
        Evaluation options:
        - save_results: bool (default True)
        - generate_plots: bool (default True)  
        - verbose: bool (default True)
        - compare_methods: bool (default True)
        - export_report: bool (default True)
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive evaluation results
    """
    # Set defaults
    if experiment_types is None:
        experiment_types = ['minority', 'all']
    
    if options is None:
        options = {}
    
    # Default options
    default_options = {
        'save_results': True,
        'generate_plots': True,
        'verbose': True,
        'compare_methods': True,
        'export_report': True
    }
    default_options.update(options)
    options = default_options
    
    if options['verbose']:
        print("=" * 80)
        print("LLSF-DL MLSMOTE Comprehensive Evaluation")
        print("=" * 80)
    
    # Handle dataset selection
    if dataset_idx == 'all':
        datasets = ['genbase', 'emotions', 'rcv1-sample1', 'recreation']
    else:
        dataset_map = {1: 'genbase', 2: 'emotions', 3: 'rcv1-sample1', 4: 'recreation'}
        if dataset_idx not in dataset_map:
            raise ValueError(f"Invalid dataset index: {dataset_idx}")
        datasets = [dataset_map[dataset_idx]]
    
    all_results = {}
    
    # Run experiments for each dataset
    for dataset_name in datasets:
        if options['verbose']:
            print(f"\nProcessing dataset: {dataset_name}")
            print("-" * 50)
        
        dataset_results = {}
        
        # Run each experiment type
        for exp_type in experiment_types:
            if options['verbose']:
                print(f"\nRunning experiment: {exp_type}")
            
            try:
                result = run_experiment(
                    dataset_name=dataset_name,
                    method=exp_type,
                    n_folds=5,
                    save_results=options['save_results']
                )
                dataset_results[exp_type] = result
                
            except Exception as e:
                print(f"Error in experiment {exp_type} on {dataset_name}: {e}")
                dataset_results[exp_type] = {'error': str(e)}
        
        all_results[dataset_name] = dataset_results
        
        # Compare methods for this dataset
        if options['compare_methods'] and len(experiment_types) > 1:
            _compare_methods(dataset_results, dataset_name)
    
    # Generate comprehensive report
    if options['export_report']:
        _generate_comprehensive_report(all_results, experiment_types)
    
    # Generate plots (placeholder - would implement visualization)
    if options['generate_plots']:
        if options['verbose']:
            print("\nNote: Plot generation not yet implemented")
    
    return all_results


def test_codebase() -> bool:
    """
    Test the codebase - Python equivalent of MATLAB test_codebase().
    
    Returns:
    --------
    bool
        True if all tests pass
    """
    print("Testing LLSF-DL MLSMOTE Python Implementation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 6
    
    try:
        # Test 1: Import all modules
        print("Test 1: Module imports... ", end="")
        from config import CONFIG
        from algorithms.llsf_dl import LLSF_DL
        from utils.evaluation import MultiLabelEvaluator
        from utils.data_utils import DataLoader
        from hybrid_approach import LLSFDLMLSMOTEHybrid
        print("PASSED")
        tests_passed += 1
        
        # Test 2: Create synthetic dataset
        print("Test 2: Synthetic data generation... ", end="")
        X, Y = create_demo_dataset()
        assert X.shape[0] == Y.shape[0], "Feature and label sample counts mismatch"
        assert X.shape == (200, 20), f"Expected (200, 20), got {X.shape}"
        assert Y.shape == (200, 5), f"Expected (200, 5), got {Y.shape}"
        print("PASSED")
        tests_passed += 1
        
        # Test 3: LLSF-DL model
        print("Test 3: LLSF-DL model training... ", end="")
        llsf = LLSF_DL(max_iter=10, random_state=42)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]
        
        llsf.fit(X_train, Y_train)
        Y_pred = llsf.predict(X_test)
        assert Y_pred.shape == Y_test.shape, "Prediction shape mismatch"
        print("PASSED")
        tests_passed += 1
        
        # Test 4: MLSMOTE
        print("Test 4: MLSMOTE synthetic sample generation... ", end="")
        from algorithms.mlsmote import MLSMOTE
        mlsmote = MLSMOTE(k=3, random_state=42)
        synth_X, synth_Y = mlsmote.fit_resample(X_train, Y_train, label_idx=0, n_samples=2)
        assert synth_X.shape[1] == X_train.shape[1], "Feature dimension mismatch"
        assert synth_Y.shape[1] == Y_train.shape[1], "Label dimension mismatch"
        print("PASSED")
        tests_passed += 1
        
        # Test 5: Evaluation metrics
        print("Test 5: Evaluation metrics... ", end="")
        evaluator = MultiLabelEvaluator()
        Y_pred_binary = (Y_pred >= 0.5).astype(int)
        metrics = evaluator.evaluate(Y_test, Y_pred_binary, Y_pred)
        required_metrics = ['hamming_loss', 'micro_f1', 'macro_f1', 'subset_accuracy']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        print("PASSED")
        tests_passed += 1
        
        # Test 6: Hybrid approach
        print("Test 6: Hybrid approach integration... ", end="")
        hybrid = LLSFDLMLSMOTEHybrid(
            llsf_params={'max_iter': 5, 'random_state': 42},
            random_state=42
        )
        hybrid.fit(X_train, Y_train, method='minority')
        Y_hybrid_pred = hybrid.predict(X_test)
        assert Y_hybrid_pred.shape == Y_test.shape, "Hybrid prediction shape mismatch"
        print("PASSED")
        tests_passed += 1
        
    except Exception as e:
        print(f"FAILED - {str(e)}")
    
    print(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! The codebase is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


def _run_test_mode(verbose: bool) -> Dict[str, Any]:
    """Run quick test mode with minimal data."""
    if verbose:
        print("Running in test mode with synthetic data...")
    
    return run_experiment(
        dataset_name='demo',
        method='minority',
        n_folds=2,
        save_results=False
    )


def _run_compare_mode(verbose: bool) -> Dict[str, Any]:
    """Run comprehensive comparison mode."""
    if verbose:
        print("Running comprehensive comparison mode...")
    
    results = {}
    methods = ['none', 'minority', 'all']
    
    for method in methods:
        if verbose:
            print(f"\nRunning baseline with method: {method}")
        results[method] = run_experiment(
            dataset_name='demo',
            method=method,
            n_folds=3,
            save_results=True
        )
    
    if verbose:
        _compare_methods(results, 'demo')
    
    return results


def _run_all_datasets(method: str, verbose: bool) -> Dict[str, Any]:
    """Run on all available datasets."""
    datasets = ['genbase', 'emotions', 'rcv1-sample1', 'recreation']
    results = {}
    
    for dataset in datasets:
        if verbose:
            print(f"\nProcessing dataset: {dataset}")
        
        try:
            results[dataset] = run_experiment(
                dataset_name=dataset,
                method=method,
                n_folds=5,
                save_results=True
            )
        except FileNotFoundError:
            if verbose:
                print(f"Dataset {dataset} not found, using demo data instead")
            results[dataset] = run_experiment(
                dataset_name='demo',
                method=method,
                n_folds=3,
                save_results=True
            )
        except Exception as e:
            if verbose:
                print(f"Error processing {dataset}: {e}")
            results[dataset] = {'error': str(e)}
    
    return results


def _compare_methods(results: Dict[str, Any], dataset_name: str):
    """Compare results between different methods."""
    print(f"\n{'='*60}")
    print(f"Method Comparison for {dataset_name}")
    print(f"{'='*60}")
    
    # Key metrics for comparison
    key_metrics = ['hamming_loss', 'ranking_loss', 'average_precision', 
                  'micro_f1', 'macro_f1', 'subset_accuracy']
    
    print(f"{'Method':<12}", end="")
    for metric in key_metrics:
        print(f"{metric:<15}", end="")
    print()
    print("-" * (12 + 15 * len(key_metrics)))
    
    for method, result in results.items():
        if 'error' in result:
            print(f"{method:<12} ERROR: {result['error']}")
            continue
        
        if 'aggregated_results' not in result:
            print(f"{method:<12} No aggregated results")
            continue
        
        print(f"{method:<12}", end="")
        for metric in key_metrics:
            mean_key = f"{metric}_mean"
            if mean_key in result['aggregated_results']:
                value = result['aggregated_results'][mean_key]
                print(f"{value:<15.4f}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()


def _generate_comprehensive_report(results: Dict[str, Any], 
                                 experiment_types: List[str]):
    """Generate a comprehensive evaluation report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(__file__).parent / "results" / f"comprehensive_report_{timestamp}.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("LLSF-DL MLSMOTE Comprehensive Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment types: {', '.join(experiment_types)}\n\n")
        
        for dataset_name, dataset_results in results.items():
            f.write(f"\nDataset: {dataset_name}\n")
            f.write("-" * 40 + "\n")
            
            for exp_type, result in dataset_results.items():
                f.write(f"\nMethod: {exp_type}\n")
                
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                    continue
                
                if 'aggregated_results' in result:
                    f.write("Key Metrics (mean ± std):\n")
                    key_metrics = ['hamming_loss', 'ranking_loss', 'average_precision',
                                  'micro_f1', 'macro_f1', 'subset_accuracy']
                    
                    for metric in key_metrics:
                        mean_key = f"{metric}_mean"
                        std_key = f"{metric}_std"
                        
                        if mean_key in result['aggregated_results']:
                            mean_val = result['aggregated_results'][mean_key]
                            std_val = result['aggregated_results'].get(std_key, 0.0)
                            f.write(f"  {metric:<20}: {mean_val:.4f} ± {std_val:.4f}\n")
                
                if 'dataset_info' in result:
                    info = result['dataset_info']
                    f.write(f"\nDataset Info:\n")
                    f.write(f"  Samples: {info['n_samples']}\n")
                    f.write(f"  Features: {info['n_features']}\n")
                    f.write(f"  Labels: {info['n_labels']}\n")
                    f.write(f"  Label density: {info['label_density']:.3f}\n")
    
    print(f"Comprehensive report saved to: {report_path}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="LLSF-DL MLSMOTE Python Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --quick 1 minority          # Quick eval on genbase with minority method
  python evaluate.py --quick demo both           # Demo dataset with both methods
  python evaluate.py --test                      # Run test suite
  python evaluate.py --comprehensive 2 minority all  # Full evaluation on emotions
        """
    )
    
    # Add arguments
    parser.add_argument('--quick', nargs=2, metavar=('DATASET', 'METHOD'),
                       help='Quick evaluation: dataset_idx method')
    parser.add_argument('--comprehensive', nargs='+', metavar='ARGS',
                       help='Comprehensive evaluation: dataset_idx method1 [method2 ...]')
    parser.add_argument('--test', action='store_true',
                       help='Run test suite')
    parser.add_argument('--silent', action='store_true',
                       help='Silent mode (minimal output)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    if args.test:
        success = test_codebase()
        sys.exit(0 if success else 1)
    
    elif args.quick:
        dataset_arg, method = args.quick
        
        # Parse dataset argument
        try:
            dataset_idx = int(dataset_arg)
        except ValueError:
            dataset_idx = dataset_arg
        
        mode = 'silent' if args.silent else 'normal'
        
        results = quick_eval(
            dataset_idx=dataset_idx,
            method=method,
            mode=mode,
            verbose=not args.silent
        )
        
        if not args.silent:
            print("\nQuick evaluation completed!")
    
    elif args.comprehensive:
        dataset_arg = args.comprehensive[0]
        methods = args.comprehensive[1:] if len(args.comprehensive) > 1 else ['minority', 'all']
        
        # Parse dataset argument
        try:
            dataset_idx = int(dataset_arg)
        except ValueError:
            dataset_idx = dataset_arg
        
        options = {
            'verbose': not args.silent,
            'save_results': not args.no_save,
            'generate_plots': True,
            'compare_methods': True,
            'export_report': True
        }
        
        results = evaluate_llsf_mlsmote(
            dataset_idx=dataset_idx,
            experiment_types=methods,
            options=options
        )
        
        if not args.silent:
            print("\nComprehensive evaluation completed!")
    
    else:
        # Default: run quick evaluation on demo data
        print("Running default quick evaluation on demo data...")
        results = quick_eval(dataset_idx='demo', method='both')


if __name__ == "__main__":
    main()
