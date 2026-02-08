%% ========================================================================
%  BULK CYCLOSTATIONARY CLASSIFIER - PHASE 1 
%  Modes: 1. Full Dataset Evaluation | 2. Specific Modulation File
%% ========================================================================
clear; close all; clc;

%% ========================================================================
%  1. INPUT SELECTION
%% ========================================================================
% Select the file to process
[fileName, filePath] = uigetfile('*.mat', 'Select Dataset (Payload_Only_Dataset.mat OR [MOD]_PayloadOnly.mat)');
if isequal(fileName,0), return; end
inputPath = fullfile(filePath, fileName);

% Determine Mode
isFullDataset = contains(fileName, 'Payload_Only_Dataset');
[~, namePart] = fileparts(fileName);

% Define Fusion Classes (Internal labels)
FUSION_CLASSES = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'};
% CSV Header mapping (Internal -> requested CSV column names)
csvHeaderMap = containers.Map(...
    {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'}, ...
    {'prob_bpsk', 'prob_4psk', 'prob_8psk', 'prob_16qam', 'prob_64qam'});

fprintf('Processing: %s\n', fileName);

%% ========================================================================
%  2. DATA LOADING & PRE-PROCESSING
%% ========================================================================
if isFullDataset
    load(inputPath, 'payloadOnlyDataset', 'config');
    numFrames = length(payloadOnlyDataset);
    modeName = 'Full_Dataset';
else
    % Mode 2: Specific Modulation (e.g., 8PSK_PayloadOnly.mat)
    data = load(inputPath);
    if isfield(data, 'payloadWaveform')
        rawWaveform = data.payloadWaveform;
    else
        % Fallback for other variable names
        f = fields(data);
        rawWaveform = data.(f{1});
    end
    
    % Extract true label from filename (e.g., "8PSK" from "8PSK_PayloadOnly")
    trueMod = strrep(namePart, '_PayloadOnly', '');
    modeName = trueMod;
    
    % Reshape continuous waveform into 1024-sample frames
    numFrames = floor(length(rawWaveform) / 1024);
    frames = reshape(rawWaveform(1:numFrames*1024), 1024, numFrames).';
    
    % Convert to struct-like for loop compatibility
    payloadOnlyDataset = struct();
    for i = 1:numFrames
        payloadOnlyDataset(i).signal = frames(i, :).';
        payloadOnlyDataset(i).modulation = trueMod;
        payloadOnlyDataset(i).snr = 15.0; % Default for these files
        payloadOnlyDataset(i).frameIndex = i;
        payloadOnlyDataset(i).sps = 2; % Adjust if generation SPS differs
    end
end

%% ========================================================================
%  3. BULK CLASSIFICATION
%% ========================================================================
% Initialize results for CSV
res_FrameIndex = (1:numFrames)';
res_TrueLabel = cell(numFrames, 1);
res_Prediction = cell(numFrames, 1);
res_Correct = zeros(numFrames, 1);
res_Confidence = zeros(numFrames, 1);
res_SNR = zeros(numFrames, 1);
res_Probs = zeros(numFrames, 5); % One for each FUSION_CLASS

originalDiary = get(0, 'Diary'); diary off;
progressBar = waitbar(0, 'Classifying signals...', 'Name', ['Cyclo: ' modeName]);

for i = 1:numFrames
    if mod(i, 50) == 0, waitbar(i/numFrames, progressBar); end
    
    % Signal Extraction
    sig = payloadOnlyDataset(i).signal;
    % Ensure Unit Power Normalization (Study Consistency)
    sig = sig / sqrt(mean(abs(sig).^2));
    
    % Prepare input for frozen blackbox
    inputStruct.signal = [real(sig)'; imag(sig)'];
    inputStruct.sps = 2; 
    
    % Call Classifier
    try
        evalc('output = classify_with_top5(inputStruct);');
    catch
        output.bestGuess = 'Unknown'; output.confidence = 0; output.topGuesses = {};
    end
    
    % Map probabilities to the 5-class vector
    prob_vec = zeros(1, 5);
    if isfield(output, 'topGuesses') && ~isempty(output.topGuesses)
        for r = 1:size(output.topGuesses, 1)
            name = upper(output.topGuesses{r, 1});
            prob = output.topGuesses{r, 2};
            idx = find(strcmp(FUSION_CLASSES, name), 1);
            if ~isempty(idx), prob_vec(idx) = prob; end
        end
    end
    
    % Renormalize (ensure Total Prob = 1)
    if sum(prob_vec) > 0, prob_vec = prob_vec / sum(prob_vec); else, prob_vec = ones(1,5)/5; end
    
    % Identify best within our 5 classes
    [maxVal, maxIdx] = max(prob_vec);
    
    % Store for CSV
    res_TrueLabel{i} = payloadOnlyDataset(i).modulation;
    res_Prediction{i} = FUSION_CLASSES{maxIdx};
    res_Correct(i) = strcmp(res_Prediction{i}, res_TrueLabel{i});
    res_Confidence(i) = maxVal;
    res_SNR(i) = payloadOnlyDataset(i).snr;
    res_Probs(i, :) = prob_vec;
end

close(progressBar);
if strcmp(originalDiary, 'on'), diary on; end

%% ========================================================================
%  4. EXPORT TO SPECIFIC CSV FORMAT
%% ========================================================================
% Construct the table with requested column names
resultsTable = table(res_FrameIndex, res_TrueLabel, res_Prediction, res_Correct, res_Confidence, res_SNR, ...
    'VariableNames', {'FrameIndex', 'TrueLabel', 'Prediction', 'Correct', 'Confidence', 'SNR'});

% Add probability columns with requested names
for k = 1:length(FUSION_CLASSES)
    colName = csvHeaderMap(FUSION_CLASSES{k});
    resultsTable.(colName) = res_Probs(:, k);
end

% Save path is the input path
outputCSV = fullfile(filePath, [namePart '_Cyclo_Results.csv']);
writetable(resultsTable, outputCSV);

fprintf('✓ Processed %d frames.\n', numFrames);
fprintf('✓ CSV Saved to: %s\n', outputCSV);

% Calculate and display basic accuracy
acc = mean(res_Correct) * 100;
fprintf('✓ Accuracy: %.2f%%\n', acc);