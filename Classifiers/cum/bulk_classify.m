%% ========================================================================
%  BULK HOC CLASSIFIER - PHASE 1 DATASET EVALUATION (FIXED)
%  Fix: Matched Filtering, Downsampling, and Renormalization
%  Modified: Classifies ALL 7,000 files (BPSK probability forced to 0)
%% ========================================================================
clear; close all; clc;

fprintf('========================================\n');
fprintf('BULK HOC CLASSIFICATION - SYNC VERSION\n');
fprintf('========================================\n\n');

%% ========================================================================
%  CONFIGURATION
%% ========================================================================
fprintf('[1/6] Loading dataset...\n');
datasetPath = 'D:\w\Documents\199\data\Phase1_Wired_Dataset\Separated_Components\Payload_Only_Dataset.mat';
if ~exist(datasetPath, 'file')
    error('Dataset not found at: %s', datasetPath);
end
load(datasetPath, 'payloadOnlyDataset', 'config');

outputDir = fullfile('Phase1_Wired_Dataset', 'HOC_Classification_Results');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Target subsets for the project
fullExperimentalSet = {'bpsk', '4psk', '8psk', '16qam', '64qam'};
hocSupportedMods = {'4psk', '8psk', '16qam', '64qam'};

% Mapping dataset names to internal labels
modNameMap = containers.Map(...
    {'QPSK', '8PSK', '16QAM', '64QAM', 'BPSK'}, ...  
    {'4psk', '8psk', '16qam', '64qam', 'bpsk'});

%% ========================================================================
%  FILTER DATASET (CLASSIFY ALL)
%% ========================================================================
% Process all indices in the dataset (1 to 7,000)
numCompatible = length(payloadOnlyDataset);
compatibleIndices = 1:numCompatible;

fprintf('  → Classifying all %d frames in the dataset...\n\n', numCompatible);

%% ========================================================================
%  BULK CLASSIFICATION (WITH SYNC & RENORMALIZATION)
%% ========================================================================
fprintf('[3/6] Running bulk classification...\n');

results = struct();
results.predictions = cell(numCompatible, 1);
results.confidences = zeros(numCompatible, 1);
results.trueLabels = cell(numCompatible, 1);
results.snrValues = zeros(numCompatible, 1);
results.frameIndices = zeros(numCompatible, 1);
results.allProbabilities = cell(numCompatible, 1);

originalDiary = get(0, 'Diary');
diary off;

progressBar = waitbar(0, 'Classifying signals...', 'Name', 'HOC Bulk Classification');
startTime = tic;

% RRC Parameters for Matched Filter
rolloff = 0.25;
span = 8;

for i = 1:numCompatible
    if mod(i, 50) == 0
        elapsed = toc(startTime);
        waitbar(i / numCompatible, progressBar, sprintf('Classifying: %d/%d', i, numCompatible));
    end
    
    datasetIdx = compatibleIndices(i);
    
    % --- 1. SIGNAL PRE-PROCESSING (Sync & Normalize) ---
    raw_signal = payloadOnlyDataset(datasetIdx).signal;
    
    if isfield(payloadOnlyDataset(datasetIdx), 'sps')
        sps = payloadOnlyDataset(datasetIdx).sps;
    elseif isfield(payloadOnlyDataset(datasetIdx), 'symbolRate') && isfield(config, 'fs')
        sr = payloadOnlyDataset(datasetIdx).symbolRate;
        sps = round(config.fs / max(sr, 1)); 
    else
        sps = 4;
    end
    
    % Matched Filter (RRC)
    rxFilter = rcosdesign(rolloff, span, sps, 'sqrt');
    filteredSig = filter(rxFilter, 1, [raw_signal; zeros(span*sps, 1)]);
    
    % Downsample to symbol peaks
    delay = span * sps / 2;
    symbolSamples = filteredSig(delay + 1 : sps : end);
    symbolSamples = symbolSamples(1:floor(1024/sps)); 
    symbolSamples = symbolSamples / sqrt(mean(abs(symbolSamples).^2));
    
    % Format for 2xN Real/Imag blackbox input
    signal_2xN = [real(symbolSamples)'; imag(symbolSamples)'];
    
    % --- 2. CALL FROZEN BLACKBOX ---
    evalc('output = blackbox(signal_2xN);');
    
    % --- 3. SUBSET RENORMALIZATION ---
    rawProbs = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for k = 1:size(output.topGuesses, 1)
        rawProbs(lower(output.topGuesses{k, 1})) = output.topGuesses{k, 2};
    end
    
    finalSubset = containers.Map('KeyType', 'char', 'ValueType', 'double');
    subsetSum = 0;
    for k = 1:length(fullExperimentalSet)
        mName = fullExperimentalSet{k};
        if ismember(mName, hocSupportedMods) && isKey(rawProbs, mName)
            val = rawProbs(mName);
            finalSubset(mName) = val;
            subsetSum = subsetSum + val;
        else
            finalSubset(mName) = 0.0; % FORCE 0 on BPSK
        end
    end
    
    % Renormalize and Determine Best Guess
    probStruct = struct();
    bestVal = -1;
    bestName = '';
    for k = 1:length(fullExperimentalSet)
        mName = fullExperimentalSet{k};
        if subsetSum > 0
            finalSubset(mName) = finalSubset(mName) / subsetSum;
        end
        if finalSubset(mName) > bestVal
            bestVal = finalSubset(mName);
            bestName = mName;
        end
        validField = matlab.lang.makeValidName(mName);
        probStruct.(validField) = finalSubset(mName);
    end
    
    % Store Results
    results.predictions{i} = bestName;
    results.confidences(i) = bestVal;
    results.trueLabels{i} = modNameMap(payloadOnlyDataset(datasetIdx).modulation);
    results.snrValues(i) = payloadOnlyDataset(datasetIdx).snr;
    results.frameIndices(i) = payloadOnlyDataset(datasetIdx).frameIndex;
    results.allProbabilities{i} = probStruct;
end

close(progressBar);
if strcmp(originalDiary, 'on'), diary on; end

%% ========================================================================
%  SAVE RESULTS (EXPORT FULL CSV)
%% ========================================================================
fprintf('[4/6] Exporting results...\n');

% Create the full table
finalTable = table(...
    results.frameIndices, ...
    results.trueLabels, ...
    results.predictions, ...
    strcmp(results.predictions, results.trueLabels), ...
    results.confidences, ...
    results.snrValues, ...
    'VariableNames', {'FrameIndex', 'TrueLabel', 'Prediction', 'Correct', 'Confidence', 'SNR'});

% Add probability columns: prob_bpsk, prob_4psk, prob_8psk, prob_16qam, prob_64qam
for k = 1:length(fullExperimentalSet)
    mName = fullExperimentalSet{k};
    colName = ['prob_', mName];
    field = matlab.lang.makeValidName(mName);
    colData = zeros(numCompatible, 1);
    for j = 1:numCompatible
        colData(j) = results.allProbabilities{j}.(field);
    end
    finalTable.(colName) = colData;
end

% Calculate overall accuracy
overallAccuracy = mean(finalTable.Correct) * 100;

csvFile = fullfile(outputDir, 'HOC_Results.csv');
writetable(finalTable, csvFile);
fprintf(' ✓ Done. Accuracy: %.2f%%\n', overallAccuracy);