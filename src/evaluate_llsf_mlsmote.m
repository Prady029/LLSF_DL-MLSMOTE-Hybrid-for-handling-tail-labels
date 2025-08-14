function evaluate_llsf_mlsmote(dataset_idx, experiment_types, options)
% EVALUATE_LLSF_MLSMOTE - Comprehensive evaluation script for LLSF-DL + MLSMOTE
%
% This script provides a unified evaluation framework that can run multiple
% experiment types, compare results, and generate comprehensive reports.
%
% Inputs:
%   dataset_idx - Dataset index (1-4) or 'all' for all datasets
%                 1: genbase, 2: emotions, 3: rcv1-sample1, 4: recreation
%   experiment_types - Cell array of experiment types or single string:
%                      'all', 'minority', 'both' (default), or {'all', 'minority'}
%   options - Optional structure with evaluation options:
%             .save_results: boolean (default: true)
%             .generate_plots: boolean (default: true)
%             .verbose: boolean (default: true)
%             .compare_methods: boolean (default: true)
%             .export_report: boolean (default: true)
%
% Usage Examples:
%   evaluate_llsf_mlsmote(1, 'both');                    % genbase, both methods
%   evaluate_llsf_mlsmote('all', 'minority');            % all datasets, minority only
%   evaluate_llsf_mlsmote(2, {'all', 'minority'});       % emotions, compare methods
%
% Outputs:
%   - Detailed console output with performance metrics
%   - Saved .mat files with results
%   - Performance plots and comparison charts
%   - Comprehensive evaluation report

%% Input validation and default settings
if nargin < 1
    dataset_idx = 1; % Default to genbase
    fprintf('No dataset specified. Using genbase (dataset 1).\n');
end

if nargin < 2
    experiment_types = 'both'; % Default to both methods
end

if nargin < 3
    options = struct();
end

% Set default options
if ~isfield(options, 'save_results'), options.save_results = true; end
if ~isfield(options, 'generate_plots'), options.generate_plots = true; end
if ~isfield(options, 'verbose'), options.verbose = true; end
if ~isfield(options, 'compare_methods'), options.compare_methods = true; end
if ~isfield(options, 'export_report'), options.export_report = true; end

% Handle experiment types
if ischar(experiment_types)
    if strcmp(experiment_types, 'both')
        experiment_types = {'all', 'minority'};
    else
        experiment_types = {experiment_types};
    end
end

% Validate experiment types
valid_types = {'all', 'minority'};
for i = 1:length(experiment_types)
    if ~ismember(experiment_types{i}, valid_types)
        error('Invalid experiment type: %s. Valid types are: all, minority', experiment_types{i});
    end
end

%% Initialize evaluation
addpath(genpath('.'));
config = get_config();
dataset_names = {'genbase', 'emotions', 'rcv1-sample1', 'recreation'};

% Handle dataset selection
if ischar(dataset_idx) && strcmp(dataset_idx, 'all')
    dataset_indices = 1:4;
else
    if isnumeric(dataset_idx)
        dataset_indices = dataset_idx;
    else
        error('Dataset index must be numeric (1-4) or ''all''');
    end
end

% Validate dataset indices
for idx = dataset_indices
    if idx < 1 || idx > 4
        error('Dataset index must be between 1 and 4');
    end
end

fprintf('=== LLSF-DL + MLSMOTE Comprehensive Evaluation ===\n');
fprintf('Datasets: %s\n', mat2str(dataset_indices));
fprintf('Experiments: %s\n', strjoin(experiment_types, ', '));
fprintf('Started at: %s\n\n', datestr(now));

%% Main evaluation loop
results_summary = struct();
evaluation_start_time = tic;

for d_idx = dataset_indices
    dataset_name = dataset_names{d_idx};
    fprintf('\n>>> EVALUATING DATASET: %s (Index %d) <<<\n', upper(dataset_name), d_idx);
    
    % Check dataset availability
    dataset_file = sprintf('../Datasets/%s.mat', dataset_name);
    if ~exist(dataset_file, 'file')
        fprintf('⚠️  Dataset file not found: %s\n', dataset_file);
        fprintf('   Skipping this dataset.\n');
        continue;
    end
    
    % Load dataset for basic info
    fprintf('Loading dataset for analysis...\n');
    try
        data = load(dataset_file);
        dataX = [data.X; data.Xt];
        dataY = [data.Y; data.Yt];
        
        [n_samples, n_features] = size(dataX);
        [~, n_labels] = size(dataY);
        label_cardinality = mean(sum(dataY, 2));
        label_density = label_cardinality / n_labels;
        
        fprintf('  📊 Dataset Statistics:\n');
        fprintf('     Samples: %d | Features: %d | Labels: %d\n', n_samples, n_features, n_labels);
        fprintf('     Label Cardinality: %.2f | Label Density: %.3f\n', label_cardinality, label_density);
        
        % Calculate imbalance information
        [IR_label, meanir] = Imbalance_ratio(dataY);
        minority_labels = sum(IR_label > meanir);
        fprintf('     Imbalanced Labels: %d/%d (%.1f%%)\n', minority_labels, n_labels, (minority_labels/n_labels)*100);
        fprintf('     Mean Imbalance Ratio: %.2f\n', meanir);
        
    catch ME
        fprintf('❌ Error loading dataset: %s\n', ME.message);
        continue;
    end
    
    % Store dataset info
    results_summary.(sprintf('dataset_%d', d_idx)).name = dataset_name;
    results_summary.(sprintf('dataset_%d', d_idx)).stats = struct(...
        'n_samples', n_samples, 'n_features', n_features, 'n_labels', n_labels, ...
        'label_cardinality', label_cardinality, 'label_density', label_density, ...
        'minority_labels', minority_labels, 'mean_ir', meanir);
    
    % Run experiments for this dataset
    for exp_idx = 1:length(experiment_types)
        exp_type = experiment_types{exp_idx};
        fprintf('\n  🚀 Running %s labels experiment...\n', upper(exp_type));
        
        exp_start_time = tic;
        
        try
            % Run the appropriate experiment
            [results, config_used] = run_single_experiment(d_idx, exp_type, config, options);
            
            exp_duration = toc(exp_start_time);
            fprintf('  ✅ %s experiment completed in %.1f seconds\n', upper(exp_type), exp_duration);
            
            % Store results
            results_summary.(sprintf('dataset_%d', d_idx)).(exp_type) = results;
            results_summary.(sprintf('dataset_%d', d_idx)).(exp_type).duration = exp_duration;
            results_summary.(sprintf('dataset_%d', d_idx)).(exp_type).config = config_used;
            
            % Display key metrics
            display_key_metrics(results, exp_type, options.verbose);
            
        catch ME
            fprintf('  ❌ %s experiment failed: %s\n', upper(exp_type), ME.message);
            if options.verbose
                fprintf('     Stack trace: %s\n', ME.getReport);
            end
        end
    end
end

total_duration = toc(evaluation_start_time);
fprintf('\n=== EVALUATION COMPLETED ===\n');
fprintf('Total evaluation time: %.1f seconds\n', total_duration);

%% Post-processing and analysis
if length(experiment_types) > 1 && options.compare_methods
    fprintf('\n🔍 COMPARATIVE ANALYSIS\n');
    generate_comparison_analysis(results_summary, dataset_indices, experiment_types, options);
end

if options.export_report
    fprintf('\n📄 GENERATING EVALUATION REPORT\n');
    generate_evaluation_report(results_summary, dataset_indices, experiment_types, options);
end

% Save complete results
if options.save_results
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    results_filename = sprintf('evaluation_results_%s.mat', timestamp);
    save(results_filename, 'results_summary', 'dataset_indices', 'experiment_types', 'options', 'total_duration');
    fprintf('📁 Complete results saved to: %s\n', results_filename);
end

fprintf('\n🎉 Evaluation completed successfully!\n');

end

%% Helper function to run a single experiment
function [results, config_used] = run_single_experiment(dataset_idx, exp_type, config, options)

% Set the dataset in config
config.current_dataset = dataset_idx;
config_used = config;

switch exp_type
    case 'all'
        results = run_all_labels_experiment_eval(dataset_idx, config, options);
    case 'minority'
        results = run_minority_labels_experiment_eval(dataset_idx, config, options);
    otherwise
        error('Unknown experiment type: %s', exp_type);
end

end

%% Helper function for all labels experiment
function results = run_all_labels_experiment_eval(dataset_idx, config, options)

if options.verbose
    fprintf('    Initializing all labels experiment...\n');
end

% Load dataset
filename = config.datasets{dataset_idx};
load(filename);

% Initialize parameters
ttlFold = config.ttlFold;
maxSelfIter = config.maxSelfIter;
ttlEva = config.ttlEva;
optmParameter = config.optmParameter;
a = config.a;

% Initialize results storage
Avg_LLSF_Self_Smote = zeros(ttlEva, maxSelfIter);
Result_LLSF_SMOTE = cell(ttlFold, 1);
for i = 1:ttlFold
    Result_LLSF_SMOTE{i} = zeros(ttlEva, maxSelfIter);
end

% Prepare data
dataX = [X; Xt];
dataY = [Y; Yt];
N = size(dataY, 1);
rand_idx = randperm(N);
partationData = kfoldpartation(dataX, dataY, ttlFold, rand_idx);

if options.verbose
    fprintf('    Running %d-fold cross-validation...\n', ttlFold);
end

% Main experiment loop
for runNo = 1:ttlFold
    if options.verbose
        fprintf('      Fold %d/%d', runNo, ttlFold);
    end
    
    X_train = full(partationData{runNo}.X);
    Y_train = full(partationData{runNo}.Y);
    X_test = full(partationData{runNo}.Xt);
    Y_test = full(partationData{runNo}.Yt);
    
    [IR_label, meanir] = Imbalance_ratio(Y_train);
    
    for selfIterNo = 1:maxSelfIter
        % Train model
        W = LLSF_DL(X_train, Y_train, optmParameter);
        [Outputs_llsf, predict_Label] = LLSF_TrainAndPredict(X_train, Y_train, X_test, optmParameter);
        [Pre_Labels, Outputs_DL] = LLSF_DL_Predict(W, X_test, predict_Label, 3);
        
        % Evaluate
        Result_LLSF_SMOTE{runNo}(:, selfIterNo) = EvaluationAll(predict_Label, Y_test', Outputs_llsf);
        
        % Generate synthetic samples for ALL labels
        Xnew = [];
        Ynew = [];
        for L = 1:size(Y_train, 2)
            [train_data, train_label] = MLSMOTE(X_train, L, Y_train, a);
            Xnew = [Xnew; train_data];
            Ynew = [Ynew; train_label];
        end
        
        % Filter and add new samples
        newIDXToRetain = sum(Ynew .* repmat(ones(1, size(Ynew, 2)), size(Xnew, 1), 1), 2) > 0;
        X_train = [X_train; Xnew(newIDXToRetain, :)];
        Y_train = [Y_train; Ynew(newIDXToRetain, :)];
    end
    
    if options.verbose
        fprintf(' ✓\n');
    end
end

% Calculate averages
for runNo = 1:ttlFold
    Avg_LLSF_Self_Smote = Avg_LLSF_Self_Smote + Result_LLSF_SMOTE{runNo};
end
Avg_LLSF_Self_Smote = Avg_LLSF_Self_Smote ./ ttlFold;

% Package results
results = struct();
results.avg_results = Avg_LLSF_Self_Smote;
results.fold_results = Result_LLSF_SMOTE;
results.method = 'all_labels';
results.dataset_idx = dataset_idx;

end

%% Helper function for minority labels experiment
function results = run_minority_labels_experiment_eval(dataset_idx, config, options)

if options.verbose
    fprintf('    Initializing minority labels experiment...\n');
end

% Load dataset
filename = config.datasets{dataset_idx};
load(filename);

% Initialize parameters (same as all labels)
ttlFold = config.ttlFold;
maxSelfIter = config.maxSelfIter;
ttlEva = config.ttlEva;
optmParameter = config.optmParameter;
a = config.a;

% Initialize results storage
Avg_LLSF_Self_Smote = zeros(ttlEva, maxSelfIter);
Result_LLSF_SMOTE = cell(ttlFold, 1);
for i = 1:ttlFold
    Result_LLSF_SMOTE{i} = zeros(ttlEva, maxSelfIter);
end

% Prepare data
dataX = [X; Xt];
dataY = [Y; Yt];
N = size(dataY, 1);
rand_idx = randperm(N);
partationData = kfoldpartation(dataX, dataY, ttlFold, rand_idx);

if options.verbose
    fprintf('    Running %d-fold cross-validation...\n', ttlFold);
end

% Main experiment loop
for runNo = 1:ttlFold
    if options.verbose
        fprintf('      Fold %d/%d', runNo, ttlFold);
    end
    
    X_full = full(partationData{runNo}.X);
    Y_full = full(partationData{runNo}.Y);
    X_test = full(partationData{runNo}.Xt);
    Y_test_full = full(partationData{runNo}.Yt);
    
    % Identify minority labels
    [IR_label, meanir] = Imbalance_ratio(Y_full);
    minorityL = IR_label > meanir;
    
    % Extract only minority labels
    Y_train = Y_full(:, minorityL);
    Y_test = Y_test_full(:, minorityL);
    X_train = X_full;
    
    for selfIterNo = 1:maxSelfIter
        % Train model on minority labels only
        W = LLSF_DL(X_train, Y_train, optmParameter);
        [Outputs_llsf, predict_Label] = LLSF_TrainAndPredict(X_train, Y_train, X_test, optmParameter);
        [Pre_Labels, Outputs_DL] = LLSF_DL_Predict(W, X_test, predict_Label, 3);
        
        % Evaluate
        Result_LLSF_SMOTE{runNo}(:, selfIterNo) = EvaluationAll(predict_Label, Y_test', Outputs_llsf);
        
        % Generate synthetic samples for minority labels only
        Xnew = [];
        Ynew = [];
        for L = 1:size(Y_train, 2)
            [train_data, train_label] = MLSMOTE(X_train, L, Y_train, a);
            Xnew = [Xnew; train_data];
            Ynew = [Ynew; train_label];
        end
        
        % Filter and add new samples
        newIDXToRetain = sum(Ynew .* repmat(ones(1, size(Ynew, 2)), size(Xnew, 1), 1), 2) > 0;
        X_train = [X_train; Xnew(newIDXToRetain, :)];
        Y_train = [Y_train; Ynew(newIDXToRetain, :)];
    end
    
    if options.verbose
        fprintf(' ✓\n');
    end
end

% Calculate averages
for runNo = 1:ttlFold
    Avg_LLSF_Self_Smote = Avg_LLSF_Self_Smote + Result_LLSF_SMOTE{runNo};
end
Avg_LLSF_Self_Smote = Avg_LLSF_Self_Smote ./ ttlFold;

% Package results
results = struct();
results.avg_results = Avg_LLSF_Self_Smote;
results.fold_results = Result_LLSF_SMOTE;
results.method = 'minority_labels';
results.dataset_idx = dataset_idx;
results.minority_count = sum(minorityL);
results.total_labels = length(minorityL);

end

%% Helper function to display key metrics
function display_key_metrics(results, exp_type, verbose)

if ~verbose
    return;
end

% Metric names (standard order from evaluation functions)
metric_names = {
    'Hamming Loss', 'Ranking Loss', 'One Error', 'Coverage', 'Average Precision',
    'Macro Precision', 'Macro Recall', 'Macro F-measure', 
    'Micro Precision', 'Micro Recall', 'Micro F-measure',
    'Subset Accuracy', 'Label-Based Accuracy', 'Example-Based Precision', 'Example-Based Recall'
};

avg_results = results.avg_results;
[num_metrics, num_iterations] = size(avg_results);

% Show key metrics from last iteration
fprintf('    📈 Key Performance Metrics (Final Iteration):\n');

key_metrics = [1, 2, 5, 8, 11, 12]; % Hamming Loss, Ranking Loss, Avg Precision, Macro F1, Micro F1, Subset Accuracy
for i = 1:min(length(key_metrics), num_metrics)
    metric_idx = key_metrics(i);
    if metric_idx <= length(metric_names)
        final_value = avg_results(metric_idx, end);
        fprintf('       %-20s: %.4f\n', metric_names{metric_idx}, final_value);
    end
end

% Show improvement from first to last iteration
if num_iterations > 1
    fprintf('    📊 Improvement (First → Last Iteration):\n');
    for i = 1:min(length(key_metrics), num_metrics)
        metric_idx = key_metrics(i);
        if metric_idx <= length(metric_names)
            first_val = avg_results(metric_idx, 1);
            last_val = avg_results(metric_idx, end);
            if first_val > 0
                improvement = ((last_val - first_val) / abs(first_val)) * 100;
                fprintf('       %-20s: %+.2f%%\n', metric_names{metric_idx}, improvement);
            end
        end
    end
end

end

%% Helper function for comparison analysis
function generate_comparison_analysis(results_summary, dataset_indices, experiment_types, options)

if length(experiment_types) < 2
    return;
end

fprintf('  Comparing %s vs %s methods...\n', upper(experiment_types{1}), upper(experiment_types{2}));

for d_idx = dataset_indices
    dataset_field = sprintf('dataset_%d', d_idx);
    if ~isfield(results_summary, dataset_field)
        continue;
    end
    
    dataset_name = results_summary.(dataset_field).name;
    fprintf('    📊 %s:\n', upper(dataset_name));
    
    % Compare final iteration results
    method1 = experiment_types{1};
    method2 = experiment_types{2};
    
    if isfield(results_summary.(dataset_field), method1) && ...
       isfield(results_summary.(dataset_field), method2)
        
        results1 = results_summary.(dataset_field).(method1).avg_results;
        results2 = results_summary.(dataset_field).(method2).avg_results;
        
        % Key metrics for comparison
        key_metrics = [1, 2, 5, 8, 11, 12];
        metric_names = {'Hamming Loss', 'Ranking Loss', 'Average Precision', 'Macro F1', 'Micro F1', 'Subset Accuracy'};
        
        for i = 1:min(length(key_metrics), size(results1, 1))
            metric_idx = key_metrics(i);
            if metric_idx <= size(results1, 1) && metric_idx <= size(results2, 1)
                val1 = results1(metric_idx, end);
                val2 = results2(metric_idx, end);
                
                % Determine which is better (lower for loss metrics, higher for others)
                if metric_idx <= 2 % Loss metrics
                    better = val1 < val2;
                    diff = val2 - val1;
                else % Performance metrics
                    better = val1 > val2;
                    diff = val1 - val2;
                end
                
                winner = better ? method1 : method2;
                fprintf('       %-18s: %.4f vs %.4f (%.4f) → %s\n', ...
                    metric_names{i}, val1, val2, abs(diff), upper(winner));
            end
        end
    end
end

end

%% Helper function to generate evaluation report
function generate_evaluation_report(results_summary, dataset_indices, experiment_types, options)

timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
report_filename = sprintf('evaluation_report_%s.txt', datestr(now, 'yyyymmdd_HHMMSS'));

fid = fopen(report_filename, 'w');
if fid == -1
    fprintf('Warning: Could not create report file\n');
    return;
end

fprintf(fid, '=== LLSF-DL + MLSMOTE Evaluation Report ===\n');
fprintf(fid, 'Generated: %s\n', timestamp);
fprintf(fid, 'Datasets: %s\n', mat2str(dataset_indices));
fprintf(fid, 'Methods: %s\n\n', strjoin(experiment_types, ', '));

% Write detailed results for each dataset
for d_idx = dataset_indices
    dataset_field = sprintf('dataset_%d', d_idx);
    if ~isfield(results_summary, dataset_field)
        continue;
    end
    
    dataset_info = results_summary.(dataset_field);
    fprintf(fid, '--- DATASET: %s ---\n', upper(dataset_info.name));
    fprintf(fid, 'Samples: %d | Features: %d | Labels: %d\n', ...
        dataset_info.stats.n_samples, dataset_info.stats.n_features, dataset_info.stats.n_labels);
    fprintf(fid, 'Label Cardinality: %.2f | Density: %.3f\n', ...
        dataset_info.stats.label_cardinality, dataset_info.stats.label_density);
    fprintf(fid, 'Minority Labels: %d/%d (%.1f%%)\n\n', ...
        dataset_info.stats.minority_labels, dataset_info.stats.n_labels, ...
        (dataset_info.stats.minority_labels/dataset_info.stats.n_labels)*100);
    
    % Write results for each method
    for exp_idx = 1:length(experiment_types)
        exp_type = experiment_types{exp_idx};
        if isfield(dataset_info, exp_type)
            fprintf(fid, '  %s Method Results:\n', upper(exp_type));
            results = dataset_info.(exp_type).avg_results;
            
            % Write key metrics
            metric_names = {'Hamming Loss', 'Ranking Loss', 'One Error', 'Coverage', 'Average Precision'};
            for i = 1:min(5, size(results, 1))
                fprintf(fid, '    %s: %.4f\n', metric_names{i}, results(i, end));
            end
            fprintf(fid, '    Duration: %.1f seconds\n\n', dataset_info.(exp_type).duration);
        end
    end
end

fprintf(fid, '=== END REPORT ===\n');
fclose(fid);

fprintf('  📄 Detailed report saved to: %s\n', report_filename);

end
