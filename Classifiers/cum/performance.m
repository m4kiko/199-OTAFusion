function performance(filename)
    % PERFORMANCE Academic AMC results analyzer for Cooperative SDR study.
    % Focused on Accuracy, SNR Robustness, Distance Emulation, and Confusion.
    
    if ~exist(filename, 'file')
        error('File %s not found.', filename);
    end

    % Read the CSV file
    data = readtable(filename);
    
    % Display basic information
    fprintf('==================================================\n');
    fprintf('                PERFORMANCE REPORT      \n');
    fprintf('==================================================\n');
    fprintf('Filename: %s\n', filename);
    fprintf('Total samples: %d\n', height(data));
    
    %% 1. GLOBAL CLASSIFICATION METRICS
    if all(ismember({'TrueLabel', 'Prediction', 'Correct'}, data.Properties.VariableNames))
        fprintf('\n--- OVERALL PERFORMANCE ---\n');
        
        accuracy = mean(data.Correct) * 100;
        fprintf('Overall Classification Accuracy: %.2f%%\n', accuracy);
        fprintf('Correct predictions: %d | Incorrect: %d\n', sum(data.Correct), sum(~data.Correct));
        
        % Confusion matrix analysis
        unique_labels = unique([data.TrueLabel; data.Prediction]);
        n_classes = length(unique_labels);
        confusion_mat = zeros(n_classes, n_classes);
        
        for i = 1:height(data)
            true_idx = find(strcmp(unique_labels, data.TrueLabel{i}));
            pred_idx = find(strcmp(unique_labels, data.Prediction{i}));
            confusion_mat(true_idx, pred_idx) = confusion_mat(true_idx, pred_idx) + 1;
        end
        
        % Print Text-based Confusion Matrix
        fprintf('\nConfusion Matrix (Rows: True, Cols: Predicted):\n');
        fprintf('%12s', '');
        for i = 1:n_classes
            fprintf('%10s', unique_labels{i});
        end
        fprintf('\n');
        for i = 1:n_classes
            fprintf('%12s', unique_labels{i});
            for j = 1:n_classes
                fprintf('%10d', confusion_mat(i, j));
            end
            fprintf('\n');
        end
        
        % Per-class statistics
        fprintf('\n--- PER-CLASS METRICS ---\n');
        fprintf('%-12s | %-10s | %-10s | %-10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
        fprintf('------------------------------------------------------------\n');
        for i = 1:n_classes
            tp = confusion_mat(i, i);
            fn = sum(confusion_mat(i, :)) - tp;
            fp = sum(confusion_mat(:, i)) - tp;
            
            precision = tp / (tp + fp);
            recall = tp / (tp + fn);
            f1_score = 2 * (precision * recall) / (precision + recall);
            
            fprintf('%-12s | %-10.4f | %-10.4f | %-10.4f\n', ...
                unique_labels{i}, precision, recall, f1_score);
        end
    end
    
    %% 2. SNR & DISTANCE ANALYSIS (Study Core)
    if ismember('SNR', data.Properties.VariableNames)
        fprintf('\n--- SNR & DISTANCE ROBUSTNESS ---\n');
        snr_bins = unique(data.SNR);
        snr_max = max(snr_bins);
        
        fprintf('%-10s | %-15s | %-10s | %-10s\n', 'SNR (dB)', 'Rel. Distance', 'Samples', 'Accuracy');
        fprintf('------------------------------------------------------------\n');
        
        accuracy_per_snr = zeros(size(snr_bins));
        dist_per_snr = zeros(size(snr_bins));
        
        for i = 1:length(snr_bins)
            mask = data.SNR == snr_bins(i);
            n_samples = sum(mask);
            accuracy_per_snr(i) = mean(data.Correct(mask)) * 100;
            
            % Friis Equation: 6dB loss = 2x distance
            dist_per_snr(i) = 10^((snr_max - snr_bins(i))/20);
            
            fprintf('%-10.1f | %-15.2f | %-10d | %-10.2f%%\n', ...
                snr_bins(i), dist_per_snr(i), n_samples, accuracy_per_snr(i));
        end
    end
    
    %% 3. GENERATING STUDY PLOTS
    fprintf('\nGenerating visualizations...\n');
    figure('Color', 'w', 'Position', [100, 100, 1100, 450]);
    
    % Subplot 1: Accuracy vs SNR & Distance
    if exist('accuracy_per_snr', 'var')
        subplot(1, 2, 1);
        yyaxis left
        plot(snr_bins, accuracy_per_snr, '-bo', 'LineWidth', 2, 'MarkerFaceColor', 'b');
        ylabel('Classification Accuracy (%)');
        xlabel('Signal-to-Noise Ratio (dB)');
        ylim([0 105]);
        grid on;
        
        yyaxis right
        plot(snr_bins, dist_per_snr, '--rs', 'LineWidth', 1.5);
        ylabel('Emulated Relative Distance (D/D0)');
        set(gca, 'XDir', 'reverse'); % Lower SNR = Further Distance
        
        title('Robustness across Channel Conditions');
        legend('Accuracy', 'Rel. Distance', 'Location', 'southwest');
    end
    
    % Subplot 2: Confusion Matrix Heatmap
    if exist('confusion_mat', 'var')
        subplot(1, 2, 2);
        h = heatmap(unique_labels, unique_labels, confusion_mat);
        h.Title = 'Modulation Confusion Matrix';
        h.XLabel = 'Predicted Class';
        h.YLabel = 'True Class';
        h.Colormap = sky;
    end
    
    fprintf('\nAnalysis complete. Reports and plots generated.\n');
end