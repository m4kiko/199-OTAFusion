%% ========================================================================
%  PHASE 1: WIRED CALIBRATION DATASET GENERATOR
%  Cooperative AMC - Signal Generation for PlutoSDR
%  Author: Based on thesis specifications
%  Date: 2025
%% ========================================================================

clear; close all; clc;

%% ========================================================================
%  CONFIGURATION PARAMETERS
%% ========================================================================

% --- Path Configuration ---
config.outputDir = 'Phase1_Wired_Dataset';
if ~exist(config.outputDir, 'dir')
    mkdir(config.outputDir);
end

% --- Signal Parameters ---
config.fs = 1e6;                    % Sample rate: 1 Msps
config.fc = 918e6;                  % Carrier frequency: 918 MHz (NTC approved)
config.numPayloadSamples = 1024;    % Payload length per frame
config.zcLength = 32;               % Zadoff-Chu preamble length
config.frameLength = config.zcLength + config.numPayloadSamples; % 1056 total

% --- Modulation Configuration ---
config.modTypes = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'};
config.numModulations = length(config.modTypes);

% Symbol rates optimized for each modulation to fit 1024 samples cleanly
config.symbolRates = [250e3, 200e3, 150e3, 125e3, 100e3]; % ksym/s per modulation

% --- Pulse Shaping ---
config.rrcRolloff = 0.25;           % RRC roll-off factor
config.rrcSpan = 8;                 % Filter span in symbols

% --- SNR Configuration ---
config.snrRange = 0:2.5:15;         % SNR from 0 to 15 dB in 2.5 dB steps
config.numSNRlevels = length(config.snrRange);

% --- Dataset Size ---
config.framesPerCondition = 200;    % Frames per (modulation, SNR) pair
config.totalFrames = config.numModulations * config.numSNRlevels * config.framesPerCondition;

% --- Zadoff-Chu Parameters ---
config.zcRoot = 25;                 % Root index (coprime with 32)

% --- Normalization (to match RadioML 2018.01A) ---
config.targetRMS = 0.993;           % Target RMS energy

fprintf('========================================\n');
fprintf('PHASE 1: SIGNAL GENERATION INITIALIZED\n');
fprintf('========================================\n');
fprintf('Output Directory: %s\n', config.outputDir);
fprintf('Total Frames to Generate: %d\n', config.totalFrames);
fprintf('Modulations: %s\n', strjoin(config.modTypes, ', '));
fprintf('SNR Range: %.1f to %.1f dB (%.1f dB steps)\n', ...
    min(config.snrRange), max(config.snrRange), config.snrRange(2)-config.snrRange(1));
fprintf('========================================\n\n');

%% ========================================================================
%  GENERATE ZADOFF-CHU PREAMBLE
%% ========================================================================

fprintf('[1/5] Generating Zadoff-Chu Preamble...\n');
zcSeq = generateZadoffChu(config.zcLength, config.zcRoot);
fprintf('      ✓ ZC sequence generated (Length=%d, Root=%d)\n', config.zcLength, config.zcRoot);
fprintf('      ✓ Peak autocorrelation gain: %.2f dB\n\n', 10*log10(config.zcLength));

%% ========================================================================
%  GENERATE SIGNALS FOR ALL CONDITIONS
%% ========================================================================

fprintf('[2/5] Generating Modulated Signals...\n');

% Initialize storage structure
dataset = struct();
frameCounter = 1;

% Progress tracking
progressBar = waitbar(0, 'Generating dataset...', 'Name', 'Phase 1 Signal Generation');

for modIdx = 1:config.numModulations
    modType = config.modTypes{modIdx};
    symbolRate = config.symbolRates(modIdx);
    % Ensure samples-per-symbol is integer-valued for rcosdesign/upsample
    sps = round(config.fs / symbolRate); % Integer samples per symbol
    
    fprintf('  [Mod %d/%d] %s (Symbol Rate: %.0f ksps, SPS: %d)\n', ...
        modIdx, config.numModulations, modType, symbolRate/1e3, sps);
    
    for snrIdx = 1:config.numSNRlevels
        targetSNR = config.snrRange(snrIdx);
        
        for frameIdx = 1:config.framesPerCondition
            
            % Update progress
            progress = frameCounter / config.totalFrames;
            waitbar(progress, progressBar, ...
                sprintf('Generating: %s @ SNR=%.1f dB | Frame %d/%d', ...
                modType, targetSNR, frameCounter, config.totalFrames));
            
            % ==========================================
            % STEP 1: Generate Random Symbols
            % ==========================================
            numSymbols = ceil(config.numPayloadSamples / sps) + config.rrcSpan;
            symbols = generateSymbols(modType, numSymbols);
            
            % ==========================================
            % STEP 2: Pulse Shaping (RRC Filter)
            % ==========================================
            txFilter = rcosdesign(config.rrcRolloff, config.rrcSpan, sps, 'sqrt');
            upsampled = upsample(symbols, sps);
            shapedSignal = filter(txFilter, 1, upsampled);
            
            % Remove transient and extract exactly 1024 samples
            delay = config.rrcSpan * sps / 2;
            shapedSignal = shapedSignal(delay+1:delay+config.numPayloadSamples);
            
            % ==========================================
            % STEP 3: Normalize to Unit Power
            % ==========================================
            signalPower = mean(abs(shapedSignal).^2);
            shapedSignal = shapedSignal / sqrt(signalPower);
            
            % ==========================================
            % STEP 4: Add AWGN to Target SNR
            % ==========================================
            noisySignal = awgn(shapedSignal, targetSNR, 'measured');
            
            % ==========================================
            % STEP 5: Prepend Zadoff-Chu Preamble
            % ==========================================
            framedSignal = [zcSeq; noisySignal];
            
            % ==========================================
            % STEP 6: Final Normalization (RadioML Style)
            % ==========================================
            framedSignal = normalizeRadioML(framedSignal, config.targetRMS);
            
            % ==========================================
            % STEP 7: Store Frame and Metadata
            % ==========================================
            dataset(frameCounter).signal = framedSignal;
            dataset(frameCounter).modulation = modType;
            dataset(frameCounter).modulationIdx = modIdx;
            dataset(frameCounter).snr = targetSNR;
            dataset(frameCounter).symbolRate = symbolRate;
            dataset(frameCounter).frameIndex = frameIdx;
            dataset(frameCounter).timestamp = datetime('now');
            dataset(frameCounter).sps = sps;
            
            frameCounter = frameCounter + 1;
        end
    end
end

close(progressBar);
fprintf('  ✓ All signals generated successfully!\n\n');

%% ========================================================================
%  SAVE DATASET
%% ========================================================================

fprintf('[3/5] Saving Dataset...\n');

% Save complete dataset (use default MAT version for SciPy compatibility)
datasetFile = fullfile(config.outputDir, 'Phase1_Complete_Dataset.mat');
save(datasetFile, 'dataset', 'config', 'zcSeq');
fprintf('  ✓ Complete dataset saved: %s\n', datasetFile);

% Save organized by modulation type (for easier loading)
for modIdx = 1:config.numModulations
    modType = config.modTypes{modIdx};
    modData = dataset([dataset.modulationIdx] == modIdx);
    modFile = fullfile(config.outputDir, sprintf('Dataset_%s.mat', modType));
    save(modFile, 'modData', 'config', 'zcSeq');
    fprintf('  ✓ %s dataset saved: %s\n', modType, modFile);
end

% Save ZC sequence separately for receiver synchronization
zcFile = fullfile(config.outputDir, 'ZC_Preamble.mat');
save(zcFile, 'zcSeq', 'config');
fprintf('  ✓ ZC preamble saved: %s\n\n', zcFile);

%% ========================================================================
%  GENERATE TRANSMISSION FILES FOR PLUTOSDR
%% ========================================================================

fprintf('[4/5] Generating PlutoSDR Transmission Files...\n');

txDir = fullfile(config.outputDir, 'PlutoSDR_TX_Files');
if ~exist(txDir, 'dir')
    mkdir(txDir);
end

% Create continuous transmission files (1000 frames per file for testing)
for modIdx = 1:config.numModulations
    modType = config.modTypes{modIdx};
    
    % Get all frames for this modulation at high SNR (15 dB)
    modFrames = dataset([dataset.modulationIdx] == modIdx & [dataset.snr] == 15);
    
    % Concatenate first 1000 frames (or all available)
    numTxFrames = min(1000, length(modFrames));
    txWaveform = [];
    for i = 1:numTxFrames
        txWaveform = [txWaveform; modFrames(i).signal];
    end
    
    % Save as complex single precision (PlutoSDR format)
    txFile = fullfile(txDir, sprintf('TX_%s_Continuous.mat', modType));
    save(txFile, 'txWaveform');
    
    % Also save as binary for GNU Radio compatibility
    binFile = fullfile(txDir, sprintf('TX_%s_Continuous.dat', modType));
    fid = fopen(binFile, 'wb');
    fwrite(fid, [real(txWaveform)'; imag(txWaveform)'], 'float32');
    fclose(fid);
    
    fprintf('  ✓ %s TX waveform: %d frames, %.2f seconds\n', ...
        modType, numTxFrames, length(txWaveform)/config.fs);
end

fprintf('\n');

%% ========================================================================
%  GENERATE VALIDATION PLOTS
%% ========================================================================

fprintf('[5/5] Generating Validation Plots...\n');

figDir = fullfile(config.outputDir, 'Validation_Plots');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

% Plot 1: Constellation Diagrams
fig1 = figure('Position', [100 100 1200 800]);
for modIdx = 1:config.numModulations
    subplot(2, 3, modIdx);
    
    % Get sample at high SNR
    sampleFrame = dataset(find([dataset.modulationIdx] == modIdx & [dataset.snr] == 15, 1));
    payload = sampleFrame.signal(config.zcLength+1:end);
    
    % Downsample to symbol rate for constellation
    sps = sampleFrame.sps;
    symbols = downsample(payload, round(sps));
    
    plot(real(symbols), imag(symbols), 'b.', 'MarkerSize', 3);
    grid on; axis equal;
    title(sprintf('%s Constellation (SNR=15dB)', config.modTypes{modIdx}));
    xlabel('In-Phase'); ylabel('Quadrature');
end
saveas(fig1, fullfile(figDir, 'Constellations.png'));
fprintf('  ✓ Constellation plots saved\n');

% Plot 2: ZC Autocorrelation
fig2 = figure('Position', [100 100 800 400]);
[acf, lags] = xcorr(zcSeq, 'normalized');
plot(lags, abs(acf), 'LineWidth', 2);
grid on;
title('Zadoff-Chu Autocorrelation (CAZAC Property)');
xlabel('Lag (samples)'); ylabel('Normalized Correlation');
saveas(fig2, fullfile(figDir, 'ZC_Autocorrelation.png'));
fprintf('  ✓ ZC autocorrelation plot saved\n');

% Plot 3: Sample Frame Structure
fig3 = figure('Position', [100 100 1000 600]);
sampleFrame = dataset(1).signal;
subplot(2,1,1);
plot(real(sampleFrame), 'LineWidth', 1);
hold on;
xline(config.zcLength, 'r--', 'LineWidth', 2);
grid on;
title('Frame Structure: In-Phase Component');
xlabel('Sample Index'); ylabel('Amplitude');
legend('I-Channel', 'ZC/Payload Boundary');

subplot(2,1,2);
plot(imag(sampleFrame), 'LineWidth', 1);
hold on;
xline(config.zcLength, 'r--', 'LineWidth', 2);
grid on;
title('Frame Structure: Quadrature Component');
xlabel('Sample Index'); ylabel('Amplitude');
legend('Q-Channel', 'ZC/Payload Boundary');
saveas(fig3, fullfile(figDir, 'Frame_Structure.png'));
fprintf('  ✓ Frame structure plot saved\n');

% Plot 4: SNR Distribution Validation
fig4 = figure('Position', [100 100 800 600]);
measuredSNR = zeros(length(dataset), 1);
for i = 1:length(dataset)
    payload = dataset(i).signal(config.zcLength+1:end);
    % snr in this MATLAB version expects a real-valued vector
    measuredSNR(i) = snr(abs(payload)); % use magnitude of complex payload
end

scatter([dataset.snr], measuredSNR, 20, 'filled');
hold on;
plot([0 15], [0 15], 'r--', 'LineWidth', 2);
grid on;
xlabel('Target SNR (dB)'); ylabel('Measured SNR (dB)');
title('SNR Calibration Validation');
legend('Measured', 'Ideal', 'Location', 'northwest');
saveas(fig4, fullfile(figDir, 'SNR_Validation.png'));
fprintf('  ✓ SNR validation plot saved\n\n');

%% ========================================================================
%  GENERATE DATASET SUMMARY REPORT
%% ========================================================================

fprintf('========================================\n');
fprintf('DATASET GENERATION COMPLETE\n');
fprintf('========================================\n');
fprintf('Total Frames Generated: %d\n', length(dataset));
fprintf('Total File Size: %.2f MB\n', dir(datasetFile).bytes / 1e6);
fprintf('Frame Length: %d samples (%.2f ms)\n', config.frameLength, config.frameLength/config.fs*1000);
fprintf('Dataset organized by:\n');
fprintf('  - Modulation Types: %d\n', config.numModulations);
fprintf('  - SNR Levels: %d (%.1f to %.1f dB)\n', config.numSNRlevels, min(config.snrRange), max(config.snrRange));
fprintf('  - Frames per condition: %d\n', config.framesPerCondition);
fprintf('\nFiles saved to: %s\n', config.outputDir);
fprintf('========================================\n');

% Save summary report
summaryFile = fullfile(config.outputDir, 'Dataset_Summary.txt');
fid = fopen(summaryFile, 'w');
fprintf(fid, 'PHASE 1 DATASET SUMMARY\n');
fprintf(fid, '======================\n\n');
fprintf(fid, 'Generation Date: %s\n', datestr(now));
fprintf(fid, 'Total Frames: %d\n', length(dataset));
fprintf(fid, 'Modulations: %s\n', strjoin(config.modTypes, ', '));
fprintf(fid, 'SNR Range: %.1f to %.1f dB (%.1f dB steps)\n', min(config.snrRange), max(config.snrRange), config.snrRange(2)-config.snrRange(1));
fprintf(fid, 'Sample Rate: %.2f Msps\n', config.fs/1e6);
fprintf(fid, 'Carrier Frequency: %.2f MHz\n', config.fc/1e6);
fprintf(fid, 'Frame Structure: %d ZC + %d Payload = %d Total\n', config.zcLength, config.numPayloadSamples, config.frameLength);
fclose(fid);

fprintf('\n✓ Summary report saved: %s\n\n', summaryFile);

%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function zcSeq = generateZadoffChu(N, u)
    % Generate Zadoff-Chu sequence with CAZAC property
    % N: Sequence length (should be prime for perfect autocorrelation)
    % u: Root index (coprime with N)
    
    n = (0:N-1)';
    zcSeq = exp(-1j * pi * u * n .* (n + 1) / N);
end

function symbols = generateSymbols(modType, numSymbols)
    % Generate random symbols for specified modulation type
    
    switch modType
        case 'BPSK'
            M = 2;
            data = randi([0 M-1], numSymbols, 1);
            symbols = pskmod(data, M, 0, 'gray');
            
        case 'QPSK'
            M = 4;
            data = randi([0 M-1], numSymbols, 1);
            symbols = pskmod(data, M, 0, 'gray');
            
        case '8PSK'
            M = 8;
            data = randi([0 M-1], numSymbols, 1);
            symbols = pskmod(data, M, 0, 'gray');
            
        case '16QAM'
            M = 16;
            data = randi([0 M-1], numSymbols, 1);
            symbols = qammod(data, M, 'gray', 'UnitAveragePower', true);
            
        case '64QAM'
            M = 64;
            data = randi([0 M-1], numSymbols, 1);
            symbols = qammod(data, M, 'gray', 'UnitAveragePower', true);
            
        otherwise
            error('Unsupported modulation type: %s', modType);
    end
end

function normalizedSignal = normalizeRadioML(signal, targetRMS)
    % Two-stage normalization to match RadioML 2018.01A dataset
    
    % Stage 1: Unit power normalization
    signalPower = sqrt(mean(abs(signal).^2));
    signal = signal / signalPower;
    
    % Stage 2: RMS energy scaling
    currentRMS = sqrt(mean(abs(signal).^2));
    normalizedSignal = (targetRMS / currentRMS) * signal;
end