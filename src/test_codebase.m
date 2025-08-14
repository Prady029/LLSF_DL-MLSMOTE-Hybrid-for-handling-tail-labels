function test_codebase()
% TEST_CODEBASE - Validate that all core functions work correctly
%
% This function performs a quick validation of the streamlined codebase
% to ensure all dependencies are properly configured.

fprintf('🧪 TESTING STREAMLINED CODEBASE\n');
fprintf('================================\n\n');

%% Test 1: Core function availability
fprintf('Test 1: Core Function Availability\n');
required_functions = {
    'get_config',
    'Imbalance_ratio', 
    'kfoldpartation',
    'MLSMOTE',
    'plot_results',
    'quick_eval',
    'evaluate_llsf_mlsmote'
};

all_functions_ok = true;
for i = 1:length(required_functions)
    func_name = required_functions{i};
    if exist(func_name, 'file') == 2
        fprintf('  ✓ %s\n', func_name);
    else
        fprintf('  ✗ %s - MISSING!\n', func_name);
        all_functions_ok = false;
    end
end

if all_functions_ok
    fprintf('  → All core functions available ✓\n\n');
else
    fprintf('  → Some functions missing ✗\n\n');
    return;
end

%% Test 2: Configuration loading
fprintf('Test 2: Configuration Loading\n');
try
    config = get_config();
    fprintf('  ✓ Configuration loaded successfully\n');
    fprintf('  ✓ %d datasets configured\n', length(config.datasets));
    fprintf('  ✓ Parameters: α=%.3f, β=%.3f, γ=%.3f\n', ...
        config.optmParameter.alpha, config.optmParameter.beta, config.optmParameter.gamma);
    fprintf('  → Configuration test passed ✓\n\n');
catch ME
    fprintf('  ✗ Configuration test failed: %s\n\n', ME.message);
    return;
end

%% Test 3: Dataset availability
fprintf('Test 3: Dataset Availability\n');
dataset_names = {'genbase', 'emotions', 'rcv1-sample1', 'recreation'};
available_count = 0;

for i = 1:length(dataset_names)
    dataset_path = sprintf('../Datasets/%s.mat', dataset_names{i});
    if exist(dataset_path, 'file')
        fprintf('  ✓ %s.mat\n', dataset_names{i});
        available_count = available_count + 1;
    else
        fprintf('  ⚠ %s.mat - not found\n', dataset_names{i});
    end
end

fprintf('  → %d/%d datasets available\n\n', available_count, length(dataset_names));

%% Test 4: Basic algorithm functions
fprintf('Test 4: Algorithm Functions\n');
try
    % Create small test data
    rng(42);
    X_test = randn(50, 10);
    Y_test = rand(50, 5) > 0.7;
    
    % Test imbalance ratio
    [IR, mean_ir] = Imbalance_ratio(Y_test);
    fprintf('  ✓ Imbalance ratio calculation\n');
    
    % Test k-fold partitioning
    rand_idx = randperm(50);
    partitions = kfoldpartation(X_test, Y_test, 3, rand_idx);
    fprintf('  ✓ K-fold partitioning (%d folds)\n', length(partitions));
    
    % Test MLSMOTE (if there are positive examples)
    label_idx = find(sum(Y_test) > 1, 1);
    if ~isempty(label_idx)
        [synth_X, synth_Y] = MLSMOTE(X_test, label_idx, Y_test, 2);
        fprintf('  ✓ MLSMOTE synthesis (%d samples)\n', size(synth_X, 1));
    else
        fprintf('  ⚠ MLSMOTE test skipped (no suitable labels)\n');
    end
    
    fprintf('  → Algorithm functions test passed ✓\n\n');
    
catch ME
    fprintf('  ✗ Algorithm functions test failed: %s\n\n', ME.message);
    return;
end

%% Test 5: Demo mode
fprintf('Test 5: Demo Mode\n');
try
    fprintf('  Running quick_eval demo...\n');
    quick_eval('demo');
    fprintf('  → Demo mode test passed ✓\n\n');
catch ME
    fprintf('  ✗ Demo mode test failed: %s\n\n', ME.message);
    return;
end

%% Summary
fprintf('🎉 CODEBASE VALIDATION SUMMARY\n');
fprintf('==============================\n');
fprintf('✓ All core functions available\n');
fprintf('✓ Configuration system working\n');
fprintf('✓ %d/%d datasets found\n', available_count, length(dataset_names));
fprintf('✓ Algorithm functions operational\n');
fprintf('✓ Demo mode functional\n\n');

if available_count > 0
    fprintf('🚀 Ready to run experiments!\n');
    fprintf('   Try: quick_eval() to get started\n\n');
else
    fprintf('⚠️  No datasets found. Place .mat files in ../Datasets/\n');
    fprintf('   Or use: quick_eval(''demo'') to test with synthetic data\n\n');
end

fprintf('Codebase validation completed successfully! 🎯\n');

end
