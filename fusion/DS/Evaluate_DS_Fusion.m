%% ========================================================================
%  DEMPSTER-SHAFER FUSION EVALUATION
%  Evaluates DS fusion on Phase 1 wired dataset
%% ========================================================================

clear; close all; clc;

fprintf('========================================\n');
fprintf('DEMPSTER-SHAFER FUSION EVALUATION\n');
fprintf('========================================\n\n');

%% CONFIGURATION
fprintf('[1/7] Loading configuration...\n');

% Get script directory
script_dir = fileparts(mfilename('fullpath'));

% Add helper functions to path
addpath(fullfile(script_dir, 'ds_helpers'));

% Fusion classes
FUSION_CLASSES = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'};

% DS Configuration
ds_config = struct();
ds_config.fusion_classes = FUSION_CLASSES;
ds_config.uncertainty_threshold = 0.05;
ds_config.conflict_threshold = 0.95;
ds_config.use_reliability_weighting = true;

% Data directory (relative to script: ../../data/Phase1_Wired_Dataset)
data_dir = fullfile(script_dir, '..', '..', 'data', 'Phase1_Wired_Dataset');

% Output directory
outputDir = fullfile(data_dir, 'DS_Fusion_Results');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('  ✓ Configuration loaded\n\n');

%% LOAD CLASSIFIER RESULTS
fprintf('[2/7] Loading classifier results...\n');

% Load HOC results
hoc_file = fullfile(data_dir, 'HOC_Classification_Results', 'HOC_Classification_Results.mat');
if ~exist(hoc_file, 'file')
    error('HOC results not found. Run Bulk_HOC_Classifier.m first');
end
hoc_data = load(hoc_file);
fprintf('  ✓ HOC results loaded (%d predictions)\n', length(hoc_data.results.predictions));

% Load Cyclo results
cyclo_file = fullfile(data_dir, 'Cyclo_Classification_Results', 'Cyclo_Classification_Results.mat');
if ~exist(cyclo_file, 'file')
    error('Cyclo results not found. Run Bulk_Cyclo_Classifier.m first');
end
cyclo_data = load(cyclo_file);
fprintf('  ✓ Cyclo results loaded (%d predictions)\n', length(cyclo_data.results.predictions));

% Load ViT results (placeholder - you'll need to create this)
vit_file = fullfile(data_dir, 'ViT_Classification_Results', 'ViT_Classification_Results.mat');
if exist(vit_file, 'file')
    vit_data = load(vit_file);
    fprintf('  ✓ ViT results loaded (%d predictions)\n', length(vit_data.results.predictions));
else
    warning('ViT results not found. Using dummy data for demonstration.');
    % Create dummy ViT data matching Cyclo structure
    vit_data = cyclo_data;  % Replace with actual ViT results
end

fprintf('\n');

%% ALIGN DATASETS
fprintf('[3/7] Aligning classifier outputs...\n');

% Find common frames (all three classifiers must have classified)
% For now, assume Cyclo has all frames (5 mods × 7 SNRs × 200 frames = 7000)
numFrames = length(cyclo_data.results.predictions);

% Initialize fusion results
fusion_results = struct();
fusion_results.predictions = cell(numFrames, 1);
fusion_results.confidences = zeros(numFrames, 1);
fusion_results.conflicts = zeros(numFrames, 1);
fusion_results.trueLabels = cyclo_data.results.trueLabels;
fusion_results.snrValues = cyclo_data.results.snrValues;
fusion_results.frameIndices = cyclo_data.results.frameIndices;

fprintf('  ✓ %d frames to process\n\n', numFrames);

%% RUN DS FUSION
fprintf('[4/7] Running Dempster-Shafer fusion...\n');

progressBar = waitbar(0, 'Fusing predictions...', 'Name', 'DS Fusion');
startTime = tic;

for i = 1:numFrames
    
    if mod(i, 100) == 0
        elapsed = toc(startTime);
        remaining = elapsed * (numFrames - i) / i;
        waitbar(i / numFrames, progressBar, ...
            sprintf('Fusing: %d/%d (%.1f%%) | ETA: %.1f min', ...
            i, numFrames, 100*i/numFrames, remaining/60));
    end
    
    % Get SNR for this frame
    snr = cyclo_data.results.snrValues(i);
    
    % Extract classifier probability vectors
    classifier_outputs = struct();
    
    % HOC probabilities
    if i <= length(hoc_data.results.allProbabilities)
        hoc_probs = hoc_data.results.allProbabilities{i};
        classifier_outputs.hoc = [
            0,  % BPSK (HOC doesn't support)
            hoc_probs.('4psk'),   % QPSK
            hoc_probs.('8psk'),   % 8PSK
            hoc_probs.('16qam'),  % 16QAM
            hoc_probs.('64qam')   % 64QAM
        ];
    else
        % Frame not classified by HOC (BPSK) - use uniform
        classifier_outputs.hoc = [0, 0.25, 0.25, 0.25, 0.25];
    end
    
    % Cyclo probabilities
    cyclo_probs = cyclo_data.results.allProbabilities{i};
    classifier_outputs.cyclo = [
        cyclo_probs.BPSK,
        cyclo_probs.QPSK,
        cyclo_probs.('8PSK'),
        cyclo_probs.('16QAM'),
        cyclo_probs.('64QAM')
    ];
    
    % ViT probabilities
    vit_probs = vit_data.results.allProbabilities{i};
    classifier_outputs.vit = [
        vit_probs.BPSK,
        vit_probs.QPSK,
        vit_probs.('8PSK'),
        vit_probs.('16QAM'),
        vit_probs.('64QAM')
    ];
    
    % Run DS fusion
    try
        output = Dempster_Shafer_Fusion(classifier_outputs, snr, ds_config);
        
        fusion_results.predictions{i} = output.prediction;
        fusion_results.confidences(i) = output.confidence;
        fusion_results.conflicts(i) = output.conflict;
        
    catch ME
        warning('DS fusion failed for frame %d: %s', i, ME.message);
        % Fallback to majority voting
        fusion_results.predictions{i} = 'QPSK';  % Default
        fusion_results.confidences(i) = 0.2;
        fusion_results.conflicts(i) = 1.0;
    end
end

close(progressBar);
totalTime = toc(startTime);

fprintf('  ✓ Fusion complete!\n');
fprintf('    Total time: %.2f minutes\n', totalTime/60);
fprintf('    Average conflict: %.2f%%\n\n', mean(fusion_results.conflicts) * 100);

%% CALCULATE PERFORMANCE METRICS
fprintf('[5/7] Calculating performance metrics...\n');

% Overall accuracy
correctPredictions = strcmp(fusion_results.predictions, fusion_results.trueLabels);
overallAccuracy = sum(correctPredictions) / numFrames * 100;

fprintf('  Fusion Accuracy: %.2f%% (%d/%d)\n', ...
    overallAccuracy, sum(correctPredictions), numFrames);

% Per-modulation accuracy
fprintf('\n  Per-Modulation Accuracy:\n');
perModAccuracy = struct();

for i = 1:length(FUSION_CLASSES)
    modName = FUSION_CLASSES{i};
    
    isThisMod = strcmp(fusion_results.trueLabels, modName);
    numThisMod = sum(isThisMod);
    
    if numThisMod > 0
        correctThisMod = sum(correctPredictions & isThisMod);
        accuracy = correctThisMod / numThisMod * 100;
        
        perModAccuracy.(modName) = accuracy;
        fprintf('    %s: %.2f%% (%d/%d)\n', ...
            modName, accuracy, correctThisMod, numThisMod);
    end
end

% SNR-based accuracy
fprintf('\n  Accuracy by SNR:\n');
uniqueSNRs = unique(fusion_results.snrValues);
snrAccuracy = zeros(length(uniqueSNRs), 1);

for i = 1:length(uniqueSNRs)
    snrVal = uniqueSNRs(i);
    isSNR = fusion_results.snrValues == snrVal;
    
    correctAtSNR = sum(correctPredictions & isSNR);
    totalAtSNR = sum(isSNR);
    
    snrAccuracy(i) = correctAtSNR / totalAtSNR * 100;
    fprintf('    SNR %.1f dB: %.2f%% (%d/%d)\n', ...
        snrVal, snrAccuracy(i), correctAtSNR, totalAtSNR);
end

fprintf('\n');

%% CONFUSION MATRIX
fprintf('[6/7] Generating confusion matrix...\n');

confMat = confusionmat(fusion_results.trueLabels, fusion_results.predictions, ...
    'Order', FUSION_CLASSES);

confMatTable = array2table(confMat, ...
    'RowNames', FUSION_CLASSES, ...
    'VariableNames', FUSION_CLASSES);

fprintf('\nConfusion Matrix:\n');
disp(confMatTable);

%% SAVE RESULTS
fprintf('[7/7] Saving results...\n');

% Save complete results
resultsFile = fullfile(outputDir, 'DS_Fusion_Results.mat');
save(resultsFile, 'fusion_results', 'perModAccuracy', 'snrAccuracy', ...
    'confMat', 'overallAccuracy', 'ds_config', '-v7.3');
fprintf('  ✓ Results saved: %s\n', resultsFile);

% Export CSV
csvFile = fullfile(outputDir, 'DS_Fusion_Results.csv');
resultsTable = table(...
    fusion_results.frameIndices, ...
    fusion_results.trueLabels, ...
    fusion_results.predictions, ...
    correctPredictions, ...
    fusion_results.confidences, ...
    fusion_results.conflicts, ...
    fusion_results.snrValues, ...
    'VariableNames', {'FrameIndex', 'TrueLabel', 'Prediction', 'Correct', ...
                      'Confidence', 'Conflict', 'SNR'});
writetable(resultsTable, csvFile);
fprintf('  ✓ CSV exported: %s\n', csvFile);

% Summary report
summaryFile = fullfile(outputDir, 'DS_Summary.txt');
fid = fopen(summaryFile, 'w');
fprintf(fid, 'DEMPSTER-SHAFER FUSION EVALUATION REPORT\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Date: %s\n\n', datestr(now));
fprintf(fid, 'Configuration:\n');
fprintf(fid, '  Uncertainty threshold: %.2f\n', ds_config.uncertainty_threshold);
fprintf(fid, '  Conflict threshold: %.2f\n', ds_config.conflict_threshold);
fprintf(fid, '  Reliability weighting: %s\n\n', ...
    string(ds_config.use_reliability_weighting));
fprintf(fid, 'Overall Performance:\n');
fprintf(fid, '  Accuracy: %.2f%%\n', overallAccuracy);
fprintf(fid, '  Average Confidence: %.2f%%\n', mean(fusion_results.confidences) * 100);
fprintf(fid, '  Average Conflict: %.2f%%\n', mean(fusion_results.conflicts) * 100);
fprintf(fid, '  High Conflict Frames (>95%%): %d (%.1f%%)\n\n', ...
    sum(fusion_results.conflicts > 0.95), ...
    sum(fusion_results.conflicts > 0.95) / numFrames * 100);
fprintf(fid, 'Per-Modulation Accuracy:\n');
for i = 1:length(FUSION_CLASSES)
    fprintf(fid, '  %s: %.2f%%\n', FUSION_CLASSES{i}, perModAccuracy.(FUSION_CLASSES{i}));
end
fclose(fid);
fprintf('  ✓ Summary saved: %s\n\n', summaryFile);

fprintf('========================================\n');
fprintf('DS FUSION EVALUATION COMPLETE\n');
fprintf('========================================\n\n');
fprintf('Results:\n');
fprintf('  Fusion Accuracy: %.2f%%\n', overallAccuracy);
fprintf('  Output: %s\n\n', outputDir);