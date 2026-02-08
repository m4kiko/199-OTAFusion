function fusion_performance(filename)
    % FUSION_PERFORMANCE Comparative analyzer for Cooperative AMC Fusion results.
    % Handles CSVs with individual (ViT, HOC, Cyclo) and Fused (DS) predictions.
    
    if ~exist(filename, 'file')
        error('File %s not found.', filename);
    end

    % Read the CSV file
    data = readtable(filename);
    
    % Ensure column names are standardized (case-insensitive check)
    cols = data.Properties.VariableNames;
    
    % Display basic information
    fprintf('==================================================\n');
    fprintf('           COOPERATIVE FUSION REPORT      \n');
    fprintf('==================================================\n');
    fprintf('Filename: %s\n', filename);
    fprintf('Total samples: %d\n', height(data));
    
    %% 1. COMPARATIVE ACCURACY ANALYSIS
    % Calculate correctness for each source if not already in CSV
    data.ViT_Match = strcmp(data.ViT_Pred, data.TrueLabel);
    data.HOC_Match = strcmp(data.HOC_Pred, data.TrueLabel);
    data.Cyclo_Match = strcmp(data.Cyclo_Pred, data.TrueLabel);
    data.DS_Match = strcmp(data.DS_Fusion_Pred, data.TrueLabel);
    
    acc_vit = mean(data.ViT_Match) * 100;
    acc_hoc = mean(data.HOC_Match) * 100;
    acc_cyc = mean(data.Cyclo_Match) * 100;
    acc_ds  = mean(data.DS_Match) * 100;
    
    fprintf('\n--- COMPARATIVE ACCURACY ---\n');
    fprintf('ViT Accuracy:       %.2f%%\n', acc_vit);
    fprintf('HOC Accuracy:       %.2f%%\n', acc_hoc);
    fprintf('Cyclo Accuracy:     %.2f%%\n', acc_cyc);
    fprintf('----------------------------\n');
    fprintf('DS FUSION ACCURACY: %.2f%% (Gain: %+.2f%%)\n', acc_ds, acc_ds - max([acc_vit, acc_hoc, acc_cyc]));

    %% 2. SNR ROBUSTNESS ANALYSIS
    if ismember('SNR', cols)
        snr_bins = unique(data.SNR);
        n_snr = length(snr_bins);
        
        % Preallocate for SNR curves
        snr_acc_vit = zeros(n_snr, 1);
        snr_acc_hoc = zeros(n_snr, 1);
        snr_acc_cyc = zeros(n_snr, 1);
        snr_acc_ds  = zeros(n_snr, 1);
        snr_avg_conflict = zeros(n_snr, 1);
        
        fprintf('\n--- PERFORMANCE BY SNR (FUSION FOCUS) ---\n');
        fprintf('%-8s | %-10s | %-10s | %-10s\n', 'SNR(dB)', 'Fusion Acc', 'Avg Conf.', 'Avg Conflict');
        fprintf('----------------------------------------------------------\n');
        
        for i = 1:n_snr
            mask = data.SNR == snr_bins(i);
            snr_acc_vit(i) = mean(data.ViT_Match(mask)) * 100;
            snr_acc_hoc(i) = mean(data.HOC_Match(mask)) * 100;
            snr_acc_cyc(i) = mean(data.Cyclo_Match(mask)) * 100;
            snr_acc_ds(i)  = mean(data.DS_Match(mask)) * 100;
            
            avg_conf = mean(data.Confidence(mask));
            avg_k = mean(data.Conflict(mask));
            snr_avg_conflict(i) = avg_k;
            
            fprintf('%-8.1f | %-10.2f%% | %-10.4f | %-10.4f\n', ...
                snr_bins(i), snr_acc_ds(i), avg_conf, avg_k);
        end
    end

    %% 3. GENERATING VISUALIZATIONS
    fprintf('\nGenerating Study Visualizations...\n');
    figure('Color', 'w', 'Position', [100, 100, 1500, 500]);
    
    % Plot 1: Cooperative Performance Gain (Superimposed SNR Curves)
    subplot(1, 3, 1);
    hold on;
    plot(snr_bins, snr_acc_vit, '--o', 'DisplayName', 'ViT (DL)', 'Color', [0.4 0.4 0.4]);
    plot(snr_bins, snr_acc_hoc, '--s', 'DisplayName', 'HOC (Stats)', 'Color', [0.6 0.6 0.6]);
    plot(snr_bins, snr_acc_cyc, '--^', 'DisplayName', 'Cyclo (Spectral)', 'Color', [0.7 0.7 0.7]);
    plot(snr_bins, snr_acc_ds, '-ko', 'LineWidth', 2.5, 'MarkerFaceColor', 'k', 'DisplayName', 'DS FUSION');
    hold off;
    grid on;
    xlabel('SNR (dB)');
    ylabel('Accuracy (%)');
    title('Cooperative Performance Gain');
    legend('Location', 'southeast');
    ylim([0 105]);
    
    % Plot 2: Evidence Conflict Analysis
    subplot(1, 3, 2);
    [yy, h1, h2] = plotyy(snr_bins, snr_acc_ds, snr_bins, snr_avg_conflict);
    grid on;
    title('Conflict (K) vs. Fusion Success');
    
    ylabel(yy(1), 'Fusion Accuracy (%)');
    ylabel(yy(2), 'Dempster Conflict (K)');
    xlabel('SNR (dB)');
    
    h1.LineStyle = '-'; h1.LineWidth = 2; h1.Marker = 'o';
    h2.LineStyle = '--'; h2.LineWidth = 1.5; h2.Marker = 'x'; h2.Color = 'r';
    yy(2).YColor = 'r';
    
    % Plot 3: Fusion Confusion Matrix
    subplot(1, 3, 3);
    unique_labels = unique(data.TrueLabel);
    confMat = confusionmat(data.TrueLabel, data.DS_Fusion_Pred, 'Order', unique_labels);
    h = heatmap(unique_labels, unique_labels, confMat);
    h.Title = 'Fused System Confusion Matrix';
    h.XLabel = 'Fused Prediction';
    h.YLabel = 'True Modulation';
    h.Colormap = summer;

    fprintf('\nAnalysis complete. Results stored in current figure.\n');
end