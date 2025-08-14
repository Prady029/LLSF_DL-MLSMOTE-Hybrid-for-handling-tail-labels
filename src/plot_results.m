function plot_results(Avg_LLSF_Self_Smote, config)
% PLOT_RESULTS - Visualize experimental results
%
% This function creates plots to visualize the performance metrics
% across different self-learning iterations.
%
% Inputs:
%   Avg_LLSF_Self_Smote - Average results matrix (metrics x iterations)
%   config - Configuration structure containing experiment parameters
%
% Usage:
%   plot_results(Avg_LLSF_Self_Smote, config);

if nargin < 2
    error('plot_results:InvalidInput', 'Both results and config are required');
end

% Metric names (standard multi-label evaluation metrics)
metric_names = {
    'Hamming Loss', 'Ranking Loss', 'One Error', 'Coverage', 'Average Precision',
    'Macro Precision', 'Macro Recall', 'Macro F-measure', 
    'Micro Precision', 'Micro Recall', 'Micro F-measure',
    'Subset Accuracy', 'Label-Based Accuracy', 'Example-Based Precision', 'Example-Based Recall'
};

% Ensure we don't exceed available metrics
num_metrics = min(length(metric_names), size(Avg_LLSF_Self_Smote, 1));
iterations = 1:size(Avg_LLSF_Self_Smote, 2);

% Create figure
figure('Position', [100, 100, 1200, 800]);

% Plot key metrics
key_metrics_idx = [1, 2, 3, 4, 5, 8, 11]; % Select important metrics
num_key_metrics = min(length(key_metrics_idx), num_metrics);

for i = 1:num_key_metrics
    subplot(3, 3, i);
    metric_idx = key_metrics_idx(i);
    
    if metric_idx <= num_metrics
        plot(iterations, Avg_LLSF_Self_Smote(metric_idx, :), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
        title(metric_names{metric_idx}, 'FontSize', 12, 'FontWeight', 'bold');
        xlabel('Self-Learning Iteration');
        ylabel('Metric Value');
        grid on;
        
        % Set appropriate y-axis limits based on metric type
        if ismember(metric_idx, [1, 2, 3]) % Loss metrics (lower is better)
            ylim([0, max(Avg_LLSF_Self_Smote(metric_idx, :)) * 1.1]);
        else % Performance metrics (higher is better)
            ylim([0, 1]);
        end
    end
end

% Add overall title
suptitle(sprintf('LLSF-DL + MLSMOTE Performance Across %d Iterations', config.maxSelfIter));

% Save the plot
saveas(gcf, 'performance_results.png');
fprintf('Results plot saved as performance_results.png\n');

% Display summary statistics
fprintf('\n=== PERFORMANCE SUMMARY ===\n');
fprintf('Dataset: %s\n', config.datasets{1}); % Assuming first dataset for display
fprintf('Self-learning iterations: %d\n', config.maxSelfIter);
fprintf('Cross-validation folds: %d\n', config.ttlFold);

% Show improvement from first to last iteration for key metrics
fprintf('\nKey Metrics (First → Last iteration):\n');
for i = 1:num_key_metrics
    metric_idx = key_metrics_idx(i);
    if metric_idx <= num_metrics
        first_val = Avg_LLSF_Self_Smote(metric_idx, 1);
        last_val = Avg_LLSF_Self_Smote(metric_idx, end);
        improvement = ((last_val - first_val) / first_val) * 100;
        
        fprintf('%-20s: %.4f → %.4f (%.2f%% change)\n', ...
            metric_names{metric_idx}, first_val, last_val, improvement);
    end
end

end
