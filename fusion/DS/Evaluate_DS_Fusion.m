%% ========================================================================
%  DEMPSTER-SHAFER FUSION EVALUATION (PRODUCTION VERSION)
%  Uses consolidated classifier outputs from Python consolidation script
%  Handles zero probabilities for unsupported classes (HOC-BPSK)
%% ========================================================================

clear; close all; clc;

fprintf('========================================\n');
fprintf('DEMPSTER-SHAFER FUSION EVALUATION\n');
fprintf('(Using Consolidated Classifier Results)\n');
fprintf('========================================\n\n');

%% CONFIGURATION
fprintf('[1/6] Loading configuration...\n');

% Add helper functions to path
addpath('ds_helpers');

% Fusion classes
FUSION_CLASSES = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'};

% DS Configuration
ds_config = struct();
ds_config.fusion_classes = FUSION_CLASSES;
ds_config.uncertainty_threshold = 0.05;
ds_config.conflict_threshold = 0.95;
ds_config.use_reliability_weighting = true;

% Output directory
outputDir = 'Phase1_Wired_Dataset/DS_Fusion_Results';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('  ✓ Configuration loaded\n\n');

%% LOAD CONSOLIDATED RESULTS
fprintf('[2/6] Loading consolidated classifier results...\n');

% Load consolidated MAT file
consolidatedFile = 'D:\w\Documents\199\data\Phase1_Wired_Dataset\Consolidated\Consolidated_AMC_Results.mat';

if ~exist(consolidatedFile, 'file')
    error(['Consolidated results not found at: %s\n' ...
           'Please run the Python consolidation script first:\n' ...
           '  python csv_to_meta_npy.py'], consolidatedFile);
end

% Load data
data = load(consolidatedFile);

% Extract results structure
if isfield(data, 'results')
    results_data = data.results;
else
    error('MAT file does not contain "results" structure');
end

% Get number of frames
numFrames = length(results_data.FrameIndex);

fprintf('  ✓ Consolidated data loaded\n');
fprintf('    Total frames: %d\n', numFrames);
fprintf('    SNR range: %.1f to %.1f dB\n\n', ...
    min(results_data.SNR), max(results_data.SNR));

%% VERIFY DATA STRUCTURE
fprintf('[3/6] Verifying data integrity...\n');

% Check probability matrix dimensions
assert(size(results_data.ViT.probs, 1) == numFrames, ...
    'ViT prob matrix size mismatch: expected %d rows, got %d', ...
    numFrames, size(results_data.ViT.probs, 1));
assert(size(results_data.ViT.probs, 2) == 5, ...
    'ViT prob matrix should have 5 columns, got %d', size(results_data.ViT.probs, 2));
assert(size(results_data.HOC.probs, 1) == numFrames, 'HOC prob matrix size mismatch');
assert(size(results_data.HOC.probs, 2) == 5, 'HOC should have 5 columns');
assert(size(results_data.Cyclo.probs, 1) == numFrames, 'Cyclo prob matrix size mismatch');
assert(size(results_data.Cyclo.probs, 2) == 5, 'Cyclo should have 5 columns');

% Statistics on zero probabilities (expected for HOC BPSK column)
vit_zeros = sum(all(results_data.ViT.probs == 0, 2));
hoc_zeros = sum(all(results_data.HOC.probs == 0, 2));
cyclo_zeros = sum(all(results_data.Cyclo.probs == 0, 2));

fprintf('  Zero probability frames:\n');
fprintf('    ViT:   %d (%.1f%%)\n', vit_zeros, vit_zeros/numFrames*100);
fprintf('    HOC:   %d (%.1f%%) - Expected for BPSK frames\n', hoc_zeros, hoc_zeros/numFrames*100);
fprintf('    Cyclo: %d (%.1f%%)\n\n', cyclo_zeros, cyclo_zeros/numFrames*100);

% Convert TrueLabel and predictions to column vectors (MAT may load as 1xN row)
trueLabels = results_data.TrueLabel(:);
vit_preds = results_data.ViT.pred(:);
hoc_preds = results_data.HOC.pred(:);
cyclo_preds = results_data.Cyclo.pred(:);

fprintf('  ✓ Data structure verified\n\n');

%% INITIALIZE FUSION RESULTS
fprintf('[4/6] Running Dempster-Shafer fusion...\n');

% Initialize fusion results
fusion_results = struct();
fusion_results.predictions = cell(numFrames, 1);
fusion_results.confidences = zeros(numFrames, 1);
fusion_results.conflicts = zeros(numFrames, 1);
fusion_results.bpa_combined = zeros(numFrames, 6);  % 5 classes + uncertainty
fusion_results.trueLabels = trueLabels;
fusion_results.snrValues = results_data.SNR;
fusion_results.frameIndices = results_data.FrameIndex;

% Track fusion statistics
stats = struct();
stats.zero_prob_frames = 0;
stats.high_conflict_frames = 0;
stats.fusion_failures = 0;

% Progress tracking
progressBar = waitbar(0, 'Fusing predictions...', 'Name', 'DS Fusion');
startTime = tic;

%% RUN DS FUSION FOR EACH FRAME
for i = 1:numFrames
    
    % Update progress every 100 frames
    if mod(i, 100) == 0
        elapsed = toc(startTime);
        remaining = elapsed * (numFrames - i) / i;
        waitbar(i / numFrames, progressBar, ...
            sprintf('Fusing: %d/%d (%.1f%%) | ETA: %.1f min', ...
            i, numFrames, 100*i/numFrames, remaining/60));
    end
    
    % Get SNR for this frame
    snr = results_data.SNR(i);
    
    % Extract probability vectors (order: BPSK, QPSK, 8PSK, 16QAM, 64QAM)
    classifier_outputs = struct();
    classifier_outputs.vit   = results_data.ViT.probs(i, :);
    classifier_outputs.hoc   = results_data.HOC.probs(i, :);
    classifier_outputs.cyclo = results_data.Cyclo.probs(i, :);
    
    % Check if any classifier has all zeros
    if sum(classifier_outputs.vit) < 1e-6 || ...
       sum(classifier_outputs.cyclo) < 1e-6
        % ViT or Cyclo failed - this shouldn't happen in normal operation
        stats.zero_prob_frames = stats.zero_prob_frames + 1;
    end
    
    % HOC having all zeros is expected for BPSK frames
    % The calculate_bpa function will handle this by assigning all mass to uncertainty
    
    % Run DS fusion
    try
        output = Dempster_Shafer_Fusion(classifier_outputs, snr, ds_config);
        
        fusion_results.predictions{i} = output.prediction;
        fusion_results.confidences(i) = output.confidence;
        fusion_results.conflicts(i) = output.conflict;
        fusion_results.bpa_combined(i, :) = output.bpa_combined;
        
        % Track high conflict
        if output.conflict > 0.95
            stats.high_conflict_frames = stats.high_conflict_frames + 1;
        end
        
    catch ME
        warning('DS fusion failed for frame %d: %s', i, ME.message);
        stats.fusion_failures = stats.fusion_failures + 1;
        
        % Fallback: Use majority voting
        preds = {vit_preds{i}, hoc_preds{i}, cyclo_preds{i}};
        [~, ~, idx] = unique(preds);
        counts = histcounts(idx, 'BinMethod', 'integers');
        [~, maxIdx] = max(counts);
        fusion_results.predictions{i} = preds{maxIdx};
        fusion_results.confidences(i) = 0.33;  % Low confidence for fallback
        fusion_results.conflicts(i) = 1.0;
        fusion_results.bpa_combined(i, :) = [0.2, 0.2, 0.2, 0.2, 0.2, 0];
    end
end

close(progressBar);
totalTime = toc(startTime);

fprintf('  ✓ Fusion complete!\n');
fprintf('    Total time: %.2f minutes\n', totalTime/60);
fprintf('    Average time per frame: %.2f ms\n', totalTime*1000/numFrames);
fprintf('    Average conflict: %.2f%%\n', mean(fusion_results.conflicts) * 100);
fprintf('    High conflict frames: %d (%.1f%%)\n', ...
    stats.high_conflict_frames, stats.high_conflict_frames/numFrames*100);
fprintf('    Fusion failures: %d\n\n', stats.fusion_failures);

%% CALCULATE PERFORMANCE METRICS
fprintf('[5/6] Calculating performance metrics...\n');

% Overall accuracy
correctPredictions = strcmp(fusion_results.predictions, fusion_results.trueLabels);
overallAccuracy = sum(correctPredictions) / numFrames * 100;

fprintf('  Fusion Accuracy: %.2f%% (%d/%d)\n', ...
    overallAccuracy, sum(correctPredictions), numFrames);

% Calculate individual accuracies
vit_correct = strcmp(vit_preds, trueLabels);
hoc_correct = strcmp(hoc_preds, trueLabels);
cyclo_correct = strcmp(cyclo_preds, trueLabels);

vit_accuracy = sum(vit_correct) / numFrames * 100;
hoc_accuracy = sum(hoc_correct) / numFrames * 100;
cyclo_accuracy = sum(cyclo_correct) / numFrames * 100;

fprintf('\n  Individual Classifier Accuracies:\n');
fprintf('    ViT:   %.2f%%\n', vit_accuracy);
fprintf('    HOC:   %.2f%%\n', hoc_accuracy);
fprintf('    Cyclo: %.2f%%\n', cyclo_accuracy);
fprintf('    ─────────────────\n');
fprintf('    DS Fusion: %.2f%% ', overallAccuracy);

% Calculate fusion gain
best_individual = max([vit_accuracy, hoc_accuracy, cyclo_accuracy]);
fusion_gain = overallAccuracy - best_individual;

if fusion_gain > 0
    fprintf('(+%.2f%% gain) ✓\n\n', fusion_gain);
elseif fusion_gain == 0
    fprintf('(no change) ─\n\n');
else
    fprintf('(%.2f%% loss) ⚠\n\n', fusion_gain);
end

% Per-modulation accuracy
fprintf('  Per-Modulation Accuracy:\n');
fprintf('  %-8s | Fusion | ViT   | HOC   | Cyclo\n', 'Mod');
fprintf('  %s\n', repmat('-', 1, 45));
perModAccuracy = struct();

for i = 1:length(FUSION_CLASSES)
    modName = FUSION_CLASSES{i};
    modField = matlab.lang.makeValidName(modName);  % e.g. 8PSK -> x8PSK (invalid to start with digit)
    
    isThisMod = strcmp(trueLabels, modName);
    numThisMod = sum(isThisMod);
    
    if numThisMod > 0
        % Fusion accuracy
        correctThisMod = sum(correctPredictions & isThisMod);
        fus_acc = correctThisMod / numThisMod * 100;
        perModAccuracy.(modField) = fus_acc;
        
        % Individual accuracies
        vit_acc = sum(vit_correct & isThisMod) / numThisMod * 100;
        hoc_acc = sum(hoc_correct & isThisMod) / numThisMod * 100;
        cyc_acc = sum(cyclo_correct & isThisMod) / numThisMod * 100;
        
        fprintf('  %-8s | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%%\n', ...
            modName, fus_acc, vit_acc, hoc_acc, cyc_acc);
    else
        perModAccuracy.(modField) = NaN;
        fprintf('  %-8s | No samples\n', modName);
    end
end

% SNR-based accuracy
fprintf('\n  Accuracy by SNR:\n');
fprintf('  SNR (dB) | Fusion | ViT   | HOC   | Cyclo\n');
fprintf('  %s\n', repmat('-', 1, 48));

uniqueSNRs = unique(fusion_results.snrValues);
snrAccuracy = zeros(length(uniqueSNRs), 1);
snr_vit = zeros(length(uniqueSNRs), 1);
snr_hoc = zeros(length(uniqueSNRs), 1);
snr_cyclo = zeros(length(uniqueSNRs), 1);

for i = 1:length(uniqueSNRs)
    snrVal = uniqueSNRs(i);
    isSNR = (fusion_results.snrValues(:) == snrVal);
    
    totalAtSNR = sum(isSNR);
    
    % Fusion accuracy at this SNR (use (:) so & yields vector, sum yields scalar)
    correctAtSNR = sum(correctPredictions(:) & isSNR);
    snrAccuracy(i) = correctAtSNR / totalAtSNR * 100;
    
    % Individual classifier accuracies at this SNR
    snr_vit(i) = sum(vit_correct(:) & isSNR) / totalAtSNR * 100;
    snr_hoc(i) = sum(hoc_correct(:) & isSNR) / totalAtSNR * 100;
    snr_cyclo(i) = sum(cyclo_correct(:) & isSNR) / totalAtSNR * 100;
    
    fprintf('  %7.1f  | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%%\n', ...
        snrVal, snrAccuracy(i), snr_vit(i), snr_hoc(i), snr_cyclo(i));
end

% Confidence and conflict statistics
avgConfidence = mean(fusion_results.confidences) * 100;
avgConflict = mean(fusion_results.conflicts) * 100;

fprintf('\n  Confidence & Conflict Statistics:\n');
fprintf('    Average Confidence: %.2f%%\n', avgConfidence);
fprintf('    Min Confidence: %.2f%%\n', min(fusion_results.confidences) * 100);
fprintf('    Max Confidence: %.2f%%\n', max(fusion_results.confidences) * 100);
fprintf('    Average Conflict: %.2f%%\n', avgConflict);
fprintf('    Max Conflict: %.2f%%\n', max(fusion_results.conflicts) * 100);
fprintf('    High Conflict (>95%%): %d frames (%.1f%%)\n\n', ...
    stats.high_conflict_frames, stats.high_conflict_frames/numFrames*100);

%% CONFUSION MATRIX
confMat = confusionmat(fusion_results.trueLabels, fusion_results.predictions, ...
    'Order', FUSION_CLASSES);

confMatTable = array2table(confMat, ...
    'RowNames', FUSION_CLASSES, ...
    'VariableNames', FUSION_CLASSES);

fprintf('Confusion Matrix (Rows: True, Columns: Predicted):\n');
disp(confMatTable);
fprintf('\n');

%% SAVE RESULTS
fprintf('[6/6] Saving results...\n');

% Save complete results
resultsFile = fullfile(outputDir, 'DS_Fusion_Results.mat');
individualClassifierResults = struct();
individualClassifierResults.vit_predictions = vit_preds;
individualClassifierResults.hoc_predictions = hoc_preds;
individualClassifierResults.cyclo_predictions = cyclo_preds;
individualClassifierResults.vit_accuracy = vit_accuracy;
individualClassifierResults.hoc_accuracy = hoc_accuracy;
individualClassifierResults.cyclo_accuracy = cyclo_accuracy;

save(resultsFile, 'fusion_results', 'perModAccuracy', 'snrAccuracy', ...
    'confMat', 'overallAccuracy', 'ds_config', 'individualClassifierResults', ...
    'fusion_gain', 'uniqueSNRs', 'snr_vit', 'snr_hoc', 'snr_cyclo', 'stats', '-v7.3');
fprintf('  ✓ Results saved: %s\n', resultsFile);

% Export to CSV (force all to column vectors so table has consistent row count)
csvFile = fullfile(outputDir, 'DS_Fusion_Results.csv');
resultsTable = table(...
    fusion_results.frameIndices(:), ...
    fusion_results.trueLabels(:), ...
    vit_preds(:), ...
    hoc_preds(:), ...
    cyclo_preds(:), ...
    fusion_results.predictions(:), ...
    correctPredictions(:), ...
    fusion_results.confidences(:), ...
    fusion_results.conflicts(:), ...
    fusion_results.snrValues(:), ...
    'VariableNames', {'FrameIndex', 'TrueLabel', 'ViT_Pred', 'HOC_Pred', ...
                      'Cyclo_Pred', 'DS_Fusion_Pred', 'Correct', ...
                      'Confidence', 'Conflict', 'SNR'});
writetable(resultsTable, csvFile);
fprintf('  ✓ CSV exported: %s\n', csvFile);

% Save confusion matrix
confMatCSV = fullfile(outputDir, 'Confusion_Matrix.csv');
writetable(confMatTable, confMatCSV, 'WriteRowNames', true);
fprintf('  ✓ Confusion matrix saved: %s\n', confMatCSV);

% Summary report
summaryFile = fullfile(outputDir, 'DS_Summary.txt');
fid = fopen(summaryFile, 'w');
fprintf(fid, 'DEMPSTER-SHAFER FUSION EVALUATION REPORT\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Date: %s\n\n', datestr(now));
fprintf(fid, 'Data Source:\n');
fprintf(fid, '  File: %s\n', consolidatedFile);
fprintf(fid, '  Total frames: %d\n', numFrames);
fprintf(fid, '  SNR range: %.1f to %.1f dB\n\n', min(uniqueSNRs), max(uniqueSNRs));
fprintf(fid, 'DS Configuration:\n');
fprintf(fid, '  Uncertainty threshold: %.3f\n', ds_config.uncertainty_threshold);
fprintf(fid, '  Conflict threshold: %.3f\n', ds_config.conflict_threshold);
fprintf(fid, '  Reliability weighting: %s\n\n', ...
    string(ds_config.use_reliability_weighting));
fprintf(fid, 'Individual Classifier Performance:\n');
fprintf(fid, '  ViT:              %.2f%%\n', vit_accuracy);
fprintf(fid, '  HOC:              %.2f%%\n', hoc_accuracy);
fprintf(fid, '  Cyclostationary:  %.2f%%\n', cyclo_accuracy);
fprintf(fid, '  Best Individual:  %.2f%%\n\n', best_individual);
fprintf(fid, 'Fusion Performance:\n');
fprintf(fid, '  DS Fusion Accuracy:   %.2f%%\n', overallAccuracy);
fprintf(fid, '  Fusion Gain:          %+.2f%%\n', fusion_gain);
fprintf(fid, '  Average Confidence:   %.2f%%\n', avgConfidence);
fprintf(fid, '  Average Conflict:     %.2f%%\n', avgConflict);
fprintf(fid, '  High Conflict (>95%%): %d frames (%.1f%%)\n', ...
    stats.high_conflict_frames, stats.high_conflict_frames/numFrames*100);
fprintf(fid, '  Fusion Failures:      %d\n\n', stats.fusion_failures);
fprintf(fid, 'Per-Modulation Accuracy:\n');
fprintf(fid, '  Modulation | Fusion | ViT   | HOC   | Cyclo\n');
fprintf(fid, '  %s\n', repmat('-', 1, 48));
for i = 1:length(FUSION_CLASSES)
    modName = FUSION_CLASSES{i};
    modField = matlab.lang.makeValidName(modName);
    isThisMod = strcmp(trueLabels, modName);
    numThisMod = sum(isThisMod);
    if numThisMod > 0
        fus_acc = perModAccuracy.(modField);
        vit_acc = sum(vit_correct & isThisMod) / numThisMod * 100;
        hoc_acc = sum(hoc_correct & isThisMod) / numThisMod * 100;
        cyc_acc = sum(cyclo_correct & isThisMod) / numThisMod * 100;
        fprintf(fid, '  %-10s | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%%\n', ...
            modName, fus_acc, vit_acc, hoc_acc, cyc_acc);
    end
end
fprintf(fid, '\nSNR Performance:\n');
fprintf(fid, '  SNR (dB) | Fusion | ViT   | HOC   | Cyclo\n');
fprintf(fid, '  %s\n', repmat('-', 1, 48));
for i = 1:length(uniqueSNRs)
    fprintf(fid, '  %7.1f  | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%%\n', ...
        uniqueSNRs(i), snrAccuracy(i), snr_vit(i), snr_hoc(i), snr_cyclo(i));
end
fclose(fid);
fprintf('  ✓ Summary saved: %s\n\n', summaryFile);

fprintf('========================================\n');
fprintf('DS FUSION EVALUATION COMPLETE\n');
fprintf('========================================\n\n');
fprintf('Summary:\n');
fprintf('  DS Fusion Accuracy:  %.2f%%\n', overallAccuracy);
fprintf('  Best Individual:     %.2f%%\n', best_individual);
fprintf('  Fusion Gain:         %+.2f%%\n', fusion_gain);
fprintf('  Average Conflict:    %.2f%%\n', avgConflict);
fprintf('  Processing Time:     %.2f minutes\n\n', totalTime/60);
fprintf('Output directory: %s\n', outputDir);
fprintf('========================================\n\n');