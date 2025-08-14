function quick_eval(varargin)
% QUICK_EVAL - Simple interface for LLSF-DL + MLSMOTE evaluation
%
% This function provides the simplest way to run evaluations with
% sensible defaults and minimal configuration.
%
% Usage Examples:
%   quick_eval()                           % Run on genbase, both methods
%   quick_eval(2)                          % Run on emotions, both methods  
%   quick_eval('all')                      % Run on all datasets, both methods
%   quick_eval(1, 'minority')              % Run on genbase, minority only
%   quick_eval([1,2], 'all')               % Run on genbase & emotions, all labels
%   quick_eval('all', 'both', 'silent')    % Run all, minimal output
%
% Shortcuts:
%   quick_eval('demo')                     % Demo mode with synthetic data
%   quick_eval('test')                     % Test mode with first dataset
%   quick_eval('compare')                  % Compare all methods on genbase

%% Parse inputs
if nargin == 0
    % Default: genbase, both methods
    dataset_idx = 1;
    experiment_types = 'both';
    mode = 'normal';
elseif nargin == 1
    arg1 = varargin{1};
    if ischar(arg1)
        switch lower(arg1)
            case 'demo'
                run_demo_mode();
                return;
            case 'test'
                dataset_idx = 1;
                experiment_types = 'all';
                mode = 'test';
            case 'compare'
                dataset_idx = 1;
                experiment_types = 'both';
                mode = 'compare';
            case 'all'
                dataset_idx = 'all';
                experiment_types = 'both';
                mode = 'normal';
            otherwise
                experiment_types = arg1;
                dataset_idx = 1;
                mode = 'normal';
        end
    else
        dataset_idx = arg1;
        experiment_types = 'both';
        mode = 'normal';
    end
elseif nargin == 2
    dataset_idx = varargin{1};
    experiment_types = varargin{2};
    mode = 'normal';
elseif nargin == 3
    dataset_idx = varargin{1};
    experiment_types = varargin{2};
    mode = varargin{3};
else
    error('Too many arguments. See help quick_eval for usage.');
end

%% Set options based on mode
switch lower(mode)
    case 'silent'
        options.verbose = false;
        options.generate_plots = false;
        options.export_report = false;
        options.save_results = true;
        options.compare_methods = false;
        
    case 'test'
        options.verbose = true;
        options.generate_plots = false;
        options.export_report = false;
        options.save_results = false;
        options.compare_methods = false;
        fprintf('🧪 TEST MODE: Quick validation run\n');
        
    case 'compare'
        options.verbose = true;
        options.generate_plots = true;
        options.export_report = true;
        options.save_results = true;
        options.compare_methods = true;
        fprintf('⚖️  COMPARE MODE: Comprehensive comparison\n');
        
    otherwise % 'normal'
        options.verbose = true;
        options.generate_plots = true;
        options.export_report = true;
        options.save_results = true;
        options.compare_methods = true;
end

%% Display startup information
fprintf('\n🚀 QUICK EVALUATION STARTING\n');

if isnumeric(dataset_idx)
    dataset_names = {'genbase', 'emotions', 'rcv1-sample1', 'recreation'};
    if length(dataset_idx) == 1
        fprintf('Dataset: %s\n', dataset_names{dataset_idx});
    else
        selected_names = dataset_names(dataset_idx);
        fprintf('Datasets: %s\n', strjoin(selected_names, ', '));
    end
else
    fprintf('Datasets: %s\n', dataset_idx);
end

if ischar(experiment_types)
    fprintf('Method: %s\n', experiment_types);
else
    fprintf('Methods: %s\n', strjoin(experiment_types, ', '));
end

fprintf('Mode: %s\n', mode);

%% Run evaluation
try
    evaluate_llsf_mlsmote(dataset_idx, experiment_types, options);
    
    fprintf('\n✅ EVALUATION COMPLETED SUCCESSFULLY!\n');
    
    % Provide next steps
    fprintf('\n📋 NEXT STEPS:\n');
    fprintf('• Check the generated plots and reports\n');
    fprintf('• Review saved .mat files for detailed results\n');
    fprintf('• Use plot_results() function for custom visualizations\n');
    fprintf('• Try evaluate_llsf_mlsmote() for advanced options\n');
    
    if strcmp(mode, 'test')
        fprintf('• Run quick_eval(''compare'') for full comparison\n');
    end
    
catch ME
    fprintf('\n❌ EVALUATION FAILED!\n');
    fprintf('Error: %s\n', ME.message);
    
    % Provide troubleshooting tips
    fprintf('\n🔧 TROUBLESHOOTING:\n');
    fprintf('• Ensure datasets are in ../Datasets/ folder\n');
    fprintf('• Check that all required functions are available\n');
    fprintf('• Try quick_eval(''demo'') to test with synthetic data\n');
    fprintf('• Use evaluate_llsf_mlsmote() for more detailed error info\n');
    
    rethrow(ME);
end

end

%% Demo mode with synthetic data
function run_demo_mode()

fprintf('\n🎮 DEMO MODE: Testing with synthetic data\n');

try
    % Create synthetic multi-label dataset
    rng(42); % For reproducibility
    n_samples = 200;
    n_features = 50;
    n_labels = 8;
    
    % Generate synthetic features
    X = randn(n_samples, n_features);
    
    % Generate correlated synthetic labels with imbalance
    label_probs = [0.3, 0.25, 0.4, 0.15, 0.1, 0.35, 0.2, 0.05]; % Some rare labels
    Y = zeros(n_samples, n_labels);
    
    for i = 1:n_samples
        for j = 1:n_labels
            % Create some label correlations
            base_prob = label_probs(j);
            if j > 1 && Y(i, j-1) == 1
                base_prob = base_prob * 1.5; % Increase prob if previous label is 1
            end
            Y(i, j) = rand() < base_prob;
        end
    end
    
    % Split into train/test
    train_idx = 1:150;
    test_idx = 151:200;
    
    X_train = X(train_idx, :);
    Y_train = Y(train_idx, :);
    X_test = X(test_idx, :);
    Y_test = Y(test_idx, :);
    
    % Save synthetic dataset
    save('synthetic_demo_dataset.mat', 'X', 'Y', 'Xt', 'Yt', ...
         'X_train', 'Y_train', 'X_test', 'Y_test');
    
    % Display dataset info
    fprintf('📊 Synthetic Dataset Created:\n');
    fprintf('  Samples: %d | Features: %d | Labels: %d\n', n_samples, n_features, n_labels);
    fprintf('  Label frequencies: %s\n', mat2str(sum(Y), 3));
    
    % Calculate imbalance ratios
    [IR_label, meanir] = Imbalance_ratio(Y);
    minority_labels = sum(IR_label > meanir);
    fprintf('  Minority labels: %d/%d\n', minority_labels, n_labels);
    fprintf('  Imbalance ratios: %s\n', mat2str(IR_label, 2));
    
    % Test basic functions
    fprintf('\n🧪 Testing core functions:\n');
    
    % Test k-fold partitioning
    rand_idx = randperm(n_samples);
    partitions = kfoldpartation(X, Y, 3, rand_idx);
    fprintf('  ✓ K-fold partitioning: %d folds created\n', length(partitions));
    
    % Test MLSMOTE on a minority label
    minority_idx = find(IR_label > meanir, 1);
    if ~isempty(minority_idx)
        try
            [synth_X, synth_Y] = MLSMOTE(X_train, minority_idx, Y_train, 2);
            fprintf('  ✓ MLSMOTE: Generated %d synthetic samples\n', size(synth_X, 1));
        catch ME
            fprintf('  ⚠️  MLSMOTE test failed: %s\n', ME.message);
        end
    end
    
    fprintf('\n✅ Demo completed successfully!\n');
    fprintf('📁 Synthetic dataset saved as: synthetic_demo_dataset.mat\n');
    fprintf('💡 You can now test the full pipeline with real datasets\n');
    
catch ME
    fprintf('❌ Demo failed: %s\n', ME.message);
    rethrow(ME);
end

end
