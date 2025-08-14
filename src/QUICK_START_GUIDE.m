% QUICK_START_GUIDE - Getting started with LLSF-DL + MLSMOTE
%
% This guide shows the most common usage patterns for the streamlined codebase.

%% Basic Usage - Start here!

% 1. Run with default settings (genbase dataset, both methods)
quick_eval()

% 2. Test with specific dataset
quick_eval(2)                   % emotions dataset
quick_eval(3)                   % rcv1-sample1 dataset

% 3. Test specific method
quick_eval(1, 'all')           % all labels method
quick_eval(1, 'minority')      % minority labels only

%% Advanced Usage

% 4. Compare all methods on specific dataset
quick_eval(1, 'both')          % comprehensive comparison

% 5. Run on all datasets
quick_eval('all', 'minority')  % minority method on all datasets

% 6. Silent mode (minimal output)
quick_eval(1, 'both', 'silent')

%% Testing and Validation

% 7. Demo mode (uses synthetic data)
quick_eval('demo')

% 8. Test mode (quick validation)
quick_eval('test')

% 9. Full comparison mode
quick_eval('compare')

%% Advanced Evaluation

% 10. For maximum control, use the full evaluation function
options.verbose = true;
options.generate_plots = true;
options.save_results = true;
evaluate_llsf_mlsmote(1, {'all', 'minority'}, options);

%% Configuration

% 11. Modify parameters in config.m, then run:
quick_eval()

% 12. Check available datasets
config = get_config();
fprintf('Available datasets: %s\n', strjoin(config.datasets, ', '));

%% Results

% After running, check for:
% - performance_results.png (plots)
% - evaluation_report_*.txt (detailed reports)  
% - evaluation_results_*.mat (numerical results)

fprintf('Quick start guide completed!\n');
fprintf('For more help, see README.md or run: help quick_eval\n');
