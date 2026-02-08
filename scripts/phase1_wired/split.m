%% ========================================================================
%  ZC PREAMBLE AND PAYLOAD SEPARATOR
%  Phase 1: Wired Calibration Dataset Processing
%  Separates Zadoff-Chu preamble from payload for analysis/transmission
%% ========================================================================

clear; close all; clc;

%% ========================================================================
%  LOAD GENERATED DATASET
%% ========================================================================

fprintf('========================================\n');
fprintf('ZC/PAYLOAD SEPARATOR\n');
fprintf('========================================\n\n');

% Load the complete dataset
fprintf('[1/4] Loading dataset...\n');
datasetPath = 'Phase1_Wired_Dataset/Phase1_Complete_Dataset.mat';

if ~exist(datasetPath, 'file')
    error('Dataset not found! Please run the signal generator first.');
end

load(datasetPath, 'dataset', 'config', 'zcSeq');
fprintf('  ✓ Loaded %d frames\n', length(dataset));
fprintf('  ✓ ZC Length: %d samples\n', config.zcLength);
fprintf('  ✓ Payload Length: %d samples\n\n', config.numPayloadSamples);

%% ========================================================================
%  CREATE SEPARATED DATASET STRUCTURE
%% ========================================================================

fprintf('[2/4] Separating ZC preambles from payloads...\n');

% Create new structure with separated components
separatedDataset = struct();

for i = 1:length(dataset)
    % Extract components
    fullFrame = dataset(i).signal;
    
    separatedDataset(i).zcPreamble = fullFrame(1:config.zcLength);
    separatedDataset(i).payload = fullFrame(config.zcLength+1:end);
    separatedDataset(i).fullFrame = fullFrame; % Keep original for reference
    
    % Copy all metadata
    separatedDataset(i).modulation = dataset(i).modulation;
    separatedDataset(i).modulationIdx = dataset(i).modulationIdx;
    separatedDataset(i).snr = dataset(i).snr;
    separatedDataset(i).symbolRate = dataset(i).symbolRate;
    separatedDataset(i).frameIndex = dataset(i).frameIndex;
    separatedDataset(i).timestamp = dataset(i).timestamp;
    separatedDataset(i).sps = dataset(i).sps;
    
    % Add size information
    separatedDataset(i).zcLength = config.zcLength;
    separatedDataset(i).payloadLength = config.numPayloadSamples;
    separatedDataset(i).frameLength = config.frameLength;
end

fprintf('  ✓ All frames separated successfully\n\n');

%% ========================================================================
%  SAVE SEPARATED DATASET
%% ========================================================================

fprintf('[3/4] Saving separated dataset...\n');

% Create output directory
separatedDir = fullfile('Phase1_Wired_Dataset', 'Separated_Components');
if ~exist(separatedDir, 'dir')
    mkdir(separatedDir);
end

% Save complete separated dataset
separatedFile = fullfile(separatedDir, 'Separated_Dataset.mat');
save(separatedFile, 'separatedDataset', 'config', 'zcSeq', '-v7.3');
fprintf('  ✓ Saved: %s\n', separatedFile);

% Save ONLY payloads (for classifier testing without ZC)
payloadOnlyDataset = struct();
for i = 1:length(separatedDataset)
    payloadOnlyDataset(i).signal = separatedDataset(i).payload;
    payloadOnlyDataset(i).modulation = separatedDataset(i).modulation;
    payloadOnlyDataset(i).modulationIdx = separatedDataset(i).modulationIdx;
    payloadOnlyDataset(i).snr = separatedDataset(i).snr;
    payloadOnlyDataset(i).symbolRate = separatedDataset(i).symbolRate;
    payloadOnlyDataset(i).frameIndex = separatedDataset(i).frameIndex;
end

payloadFile = fullfile(separatedDir, 'Payload_Only_Dataset.mat');
save(payloadFile, 'payloadOnlyDataset', 'config');
fprintf('  ✓ Saved payload-only dataset: %s\n', payloadFile);

% Save ONLY ZC preambles (for synchronization testing)
zcOnlyDataset = struct();
for i = 1:length(separatedDataset)
    zcOnlyDataset(i).signal = separatedDataset(i).zcPreamble;
    zcOnlyDataset(i).snr = separatedDataset(i).snr;
    zcOnlyDataset(i).frameIndex = separatedDataset(i).frameIndex;
end

zcFile = fullfile(separatedDir, 'ZC_Only_Dataset.mat');
save(zcFile, 'zcOnlyDataset', 'config', 'zcSeq');
fprintf('  ✓ Saved ZC-only dataset: %s\n\n', zcFile);

%% ========================================================================
%  CREATE TRANSMISSION FILES (ZC AND PAYLOAD SEPARATE)
%% ========================================================================

fprintf('[4/4] Creating separate transmission files for PlutoSDR...\n');

txSeparatedDir = fullfile(separatedDir, 'TX_Separated');
if ~exist(txSeparatedDir, 'dir')
    mkdir(txSeparatedDir);
end

for modIdx = 1:config.numModulations
    modType = config.modTypes{modIdx};
    
    % Get frames for this modulation at high SNR (15 dB)
    modFrames = separatedDataset([separatedDataset.modulationIdx] == modIdx & ...
                                  [separatedDataset.snr] == 15);
    
    numTxFrames = min(1000, length(modFrames));
    
    % ==========================================
    % Option 1: ZC + Payload concatenated
    % ==========================================
    fullWaveform = [];
    for i = 1:numTxFrames
        fullWaveform = [fullWaveform; modFrames(i).fullFrame];
    end
    
    % Save full frames
    fullFile = fullfile(txSeparatedDir, sprintf('%s_Full.mat', modType));
    save(fullFile, 'fullWaveform');
    
    fullBinFile = fullfile(txSeparatedDir, sprintf('%s_Full.dat', modType));
    fid = fopen(fullBinFile, 'wb');
    fwrite(fid, [real(fullWaveform)'; imag(fullWaveform)'], 'float32');
    fclose(fid);
    
    % ==========================================
    % Option 2: Payload ONLY (no ZC)
    % ==========================================
    payloadWaveform = [];
    for i = 1:numTxFrames
        payloadWaveform = [payloadWaveform; modFrames(i).payload];
    end
    
    % Save payload only
    payloadFile = fullfile(txSeparatedDir, sprintf('%s_PayloadOnly.mat', modType));
    save(payloadFile, 'payloadWaveform');
    
    payloadBinFile = fullfile(txSeparatedDir, sprintf('%s_PayloadOnly.dat', modType));
    fid = fopen(payloadBinFile, 'wb');
    fwrite(fid, [real(payloadWaveform)'; imag(payloadWaveform)'], 'float32');
    fclose(fid);
    
    % ==========================================
    % Option 3: ZC preambles ONLY (for sync testing)
    % ==========================================
    zcWaveform = [];
    for i = 1:numTxFrames
        zcWaveform = [zcWaveform; modFrames(i).zcPreamble];
    end
    
    % Save ZC only
    zcFile = fullfile(txSeparatedDir, sprintf('%s_ZC_Only.mat', modType));
    save(zcFile, 'zcWaveform');
    
    zcBinFile = fullfile(txSeparatedDir, sprintf('%s_ZC_Only.dat', modType));
    fid = fopen(zcBinFile, 'wb');
    fwrite(fid, [real(zcWaveform)'; imag(zcWaveform)'], 'float32');
    fclose(fid);
    
    % ==========================================
    % Option 4: Interleaved (ZC, Payload, ZC, Payload, ...)
    % ==========================================
    interleavedWaveform = [];
    for i = 1:numTxFrames
        interleavedWaveform = [interleavedWaveform; 
                               modFrames(i).zcPreamble; 
                               modFrames(i).payload];
    end
    
    % Save interleaved (same as full, but explicitly constructed)
    interleavedFile = fullfile(txSeparatedDir, sprintf('%s_Interleaved.mat', modType));
    save(interleavedFile, 'interleavedWaveform');
    
    fprintf('  ✓ %s: Full, Payload-Only, ZC-Only, Interleaved variants created\n', modType);
end

fprintf('\n');

%% ========================================================================
%  GENERATE VALIDATION PLOTS
%% ========================================================================

fprintf('Generating validation plots...\n');

plotDir = fullfile(separatedDir, 'Validation_Plots');
if ~exist(plotDir, 'dir')
    mkdir(plotDir);
end

% Get a sample frame
sampleIdx = find([separatedDataset.modulationIdx] == 1 & [separatedDataset.snr] == 15, 1);
sampleFrame = separatedDataset(sampleIdx);

% Plot 1: ZC vs Payload Comparison
fig1 = figure('Position', [100 100 1200 800]);

% ZC Preamble - Real
subplot(3,2,1);
plot(real(sampleFrame.zcPreamble), 'b-', 'LineWidth', 1.5);
grid on;
title('ZC Preamble - In-Phase');
xlabel('Sample'); ylabel('Amplitude');
xlim([1 config.zcLength]);

% ZC Preamble - Imag
subplot(3,2,2);
plot(imag(sampleFrame.zcPreamble), 'r-', 'LineWidth', 1.5);
grid on;
title('ZC Preamble - Quadrature');
xlabel('Sample'); ylabel('Amplitude');
xlim([1 config.zcLength]);

% Payload - Real
subplot(3,2,3);
plot(real(sampleFrame.payload), 'b-', 'LineWidth', 1);
grid on;
title('Payload - In-Phase');
xlabel('Sample'); ylabel('Amplitude');
xlim([1 config.numPayloadSamples]);

% Payload - Imag
subplot(3,2,4);
plot(imag(sampleFrame.payload), 'r-', 'LineWidth', 1);
grid on;
title('Payload - Quadrature');
xlabel('Sample'); ylabel('Amplitude');
xlim([1 config.numPayloadSamples]);

% Full Frame - Real
subplot(3,2,5);
plot(real(sampleFrame.fullFrame), 'b-', 'LineWidth', 1);
hold on;
xline(config.zcLength, 'k--', 'LineWidth', 2);
grid on;
title('Full Frame - In-Phase (ZC | Payload)');
xlabel('Sample'); ylabel('Amplitude');
legend('Signal', 'ZC/Payload Boundary');

% Full Frame - Imag
subplot(3,2,6);
plot(imag(sampleFrame.fullFrame), 'r-', 'LineWidth', 1);
hold on;
xline(config.zcLength, 'k--', 'LineWidth', 2);
grid on;
title('Full Frame - Quadrature (ZC | Payload)');
xlabel('Sample'); ylabel('Amplitude');
legend('Signal', 'ZC/Payload Boundary');

sgtitle(sprintf('Frame Components: %s @ SNR=15dB', sampleFrame.modulation));
saveas(fig1, fullfile(plotDir, 'ZC_Payload_Comparison.png'));
fprintf('  ✓ Component comparison plot saved\n');

% Plot 2: Power Spectral Density Comparison
fig2 = figure('Position', [100 100 1200 400]);

% ZC PSD
subplot(1,3,1);
[psd_zc, f_zc] = pwelch(sampleFrame.zcPreamble, [], [], [], config.fs, 'centered');
plot(f_zc/1e6, 10*log10(psd_zc), 'LineWidth', 1.5);
grid on;
title('ZC Preamble PSD');
xlabel('Frequency (MHz)'); ylabel('Power (dB)');

% Payload PSD
subplot(1,3,2);
[psd_payload, f_payload] = pwelch(sampleFrame.payload, [], [], [], config.fs, 'centered');
plot(f_payload/1e6, 10*log10(psd_payload), 'LineWidth', 1.5);
grid on;
title('Payload PSD');
xlabel('Frequency (MHz)'); ylabel('Power (dB)');

% Full Frame PSD
subplot(1,3,3);
[psd_full, f_full] = pwelch(sampleFrame.fullFrame, [], [], [], config.fs, 'centered');
plot(f_full/1e6, 10*log10(psd_full), 'LineWidth', 1.5);
grid on;
title('Full Frame PSD');
xlabel('Frequency (MHz)'); ylabel('Power (dB)');

saveas(fig2, fullfile(plotDir, 'PSD_Comparison.png'));
fprintf('  ✓ PSD comparison plot saved\n');

% Plot 3: ZC Autocorrelation Properties
fig3 = figure('Position', [100 100 1000 600]);

subplot(2,1,1);
[acf, lags] = xcorr(sampleFrame.zcPreamble, 'normalized');
plot(lags, abs(acf), 'LineWidth', 2);
grid on;
title('ZC Preamble Autocorrelation (CAZAC Property)');
xlabel('Lag (samples)'); ylabel('|Autocorrelation|');
ylim([0 1.1]);

subplot(2,1,2);
plot(lags, 20*log10(abs(acf)), 'LineWidth', 2);
grid on;
title('ZC Preamble Autocorrelation (dB scale)');
xlabel('Lag (samples)'); ylabel('Autocorrelation (dB)');
yline(-3, 'r--', 'LineWidth', 1.5);
legend('Autocorrelation', '3dB Threshold');

saveas(fig3, fullfile(plotDir, 'ZC_Autocorrelation_Properties.png'));
fprintf('  ✓ ZC autocorrelation plot saved\n');

% Plot 4: Constellation (Payload Only)
fig4 = figure('Position', [100 100 1200 800]);
for modIdx = 1:config.numModulations
    subplot(2, 3, modIdx);
    
    % Get sample at high SNR
    sampleIdx = find([separatedDataset.modulationIdx] == modIdx & ...
                     [separatedDataset.snr] == 15, 1);
    sampleData = separatedDataset(sampleIdx);
    
    % Downsample payload to symbol rate
    sps = sampleData.sps;
    symbols = downsample(sampleData.payload, round(sps));
    
    plot(real(symbols), imag(symbols), 'b.', 'MarkerSize', 3);
    grid on; axis equal;
    title(sprintf('%s (Payload Only)', config.modTypes{modIdx}));
    xlabel('In-Phase'); ylabel('Quadrature');
end
saveas(fig4, fullfile(plotDir, 'Payload_Constellations.png'));
fprintf('  ✓ Payload constellation plot saved\n\n');

%% ========================================================================
%  GENERATE HELPER FUNCTIONS FILE
%% ========================================================================

fprintf('Creating helper functions file...\n');

helperFile = fullfile(separatedDir, 'load_helpers.m');
fid = fopen(helperFile, 'w');

fprintf(fid, '%% Helper functions to load separated dataset components\n\n');

fprintf(fid, 'function [zcPreambles, payloads, metadata] = loadSeparatedData(modType, snrRange)\n');
fprintf(fid, '    %% Load separated ZC and payload data\n');
fprintf(fid, '    %% Inputs:\n');
fprintf(fid, '    %%   modType: ''BPSK'', ''QPSK'', ''8PSK'', ''16QAM'', or ''64QAM''\n');
fprintf(fid, '    %%   snrRange: [min max] or specific SNR value (optional)\n');
fprintf(fid, '    \n');
fprintf(fid, '    load(''Separated_Dataset.mat'', ''separatedDataset'');\n');
fprintf(fid, '    \n');
fprintf(fid, '    if nargin < 2\n');
fprintf(fid, '        snrRange = [-inf inf];\n');
fprintf(fid, '    elseif length(snrRange) == 1\n');
fprintf(fid, '        snrRange = [snrRange snrRange];\n');
fprintf(fid, '    end\n');
fprintf(fid, '    \n');
fprintf(fid, '    % Filter by modulation and SNR\n');
fprintf(fid, '    mask = strcmp({separatedDataset.modulation}, modType) & ...\n');
fprintf(fid, '           [separatedDataset.snr] >= snrRange(1) & ...\n');
fprintf(fid, '           [separatedDataset.snr] <= snrRange(2);\n');
fprintf(fid, '    \n');
fprintf(fid, '    filtered = separatedDataset(mask);\n');
fprintf(fid, '    \n');
fprintf(fid, '    zcPreambles = {filtered.zcPreamble};\n');
fprintf(fid, '    payloads = {filtered.payload};\n');
fprintf(fid, '    metadata = rmfield(filtered, {''zcPreamble'', ''payload'', ''fullFrame''});\n');
fprintf(fid, 'end\n\n');

fprintf(fid, 'function payload = extractPayload(frame, zcLength)\n');
fprintf(fid, '    %% Extract payload from a full frame\n');
fprintf(fid, '    if nargin < 2\n');
fprintf(fid, '        zcLength = 32; %% Default ZC length\n');
fprintf(fid, '    end\n');
fprintf(fid, '    payload = frame(zcLength+1:end);\n');
fprintf(fid, 'end\n\n');

fprintf(fid, 'function zcPreamble = extractZC(frame, zcLength)\n');
fprintf(fid, '    %% Extract ZC preamble from a full frame\n');
fprintf(fid, '    if nargin < 2\n');
fprintf(fid, '        zcLength = 32; %% Default ZC length\n');
fprintf(fid, '    end\n');
fprintf(fid, '    zcPreamble = frame(1:zcLength);\n');
fprintf(fid, 'end\n\n');

fprintf(fid, 'function fullFrame = combineZCandPayload(zcPreamble, payload)\n');
fprintf(fid, '    %% Combine ZC preamble and payload into full frame\n');
fprintf(fid, '    fullFrame = [zcPreamble; payload];\n');
fprintf(fid, 'end\n');

fclose(fid);
fprintf('  ✓ Helper functions saved: %s\n\n', helperFile);

%% ========================================================================
%  SUMMARY REPORT
%% ========================================================================

fprintf('========================================\n');
fprintf('SEPARATION COMPLETE\n');
fprintf('========================================\n\n');

fprintf('Dataset Structure:\n');
fprintf('  - Full frames: %d samples each\n', config.frameLength);
fprintf('  - ZC preamble: %d samples each\n', config.zcLength);
fprintf('  - Payload: %d samples each\n', config.numPayloadSamples);
fprintf('\nOutput Files:\n');
fprintf('  1. Separated_Dataset.mat - Full dataset with components\n');
fprintf('  2. Payload_Only_Dataset.mat - Just payloads (for classifiers)\n');
fprintf('  3. ZC_Only_Dataset.mat - Just ZC preambles (for sync testing)\n');
fprintf('  4. TX_Separated/ - Transmission files (Full, Payload, ZC, Interleaved)\n');
fprintf('  5. load_helpers.m - Utility functions\n');
fprintf('\nSaved to: %s\n', separatedDir);
fprintf('========================================\n\n');

% Save summary
summaryFile = fullfile(separatedDir, 'Separation_Summary.txt');
fid = fopen(summaryFile, 'w');
fprintf(fid, 'ZC/PAYLOAD SEPARATION SUMMARY\n');
fprintf(fid, '============================\n\n');
fprintf(fid, 'Separation Date: %s\n', datestr(now));
fprintf(fid, 'Total Frames Processed: %d\n\n', length(separatedDataset));
fprintf(fid, 'Component Sizes:\n');
fprintf(fid, '  ZC Preamble: %d samples (%.2f µs @ 1 Msps)\n', config.zcLength, config.zcLength);
fprintf(fid, '  Payload: %d samples (%.2f ms @ 1 Msps)\n', config.numPayloadSamples, config.numPayloadSamples/1000);
fprintf(fid, '  Full Frame: %d samples (%.2f ms @ 1 Msps)\n\n', config.frameLength, config.frameLength/1000);
fprintf(fid, 'Usage Examples:\n');
fprintf(fid, '  1. Load full dataset: load(''Separated_Dataset.mat'')\n');
fprintf(fid, '  2. Load payloads only: load(''Payload_Only_Dataset.mat'')\n');
fprintf(fid, '  3. Use helpers: [zc, payload, meta] = loadSeparatedData(''QPSK'', [10 15])\n');
fclose(fid);

fprintf('✓ Summary saved: %s\n\n', summaryFile);
fprintf('Ready for Phase 1 wired testing!\n\n');