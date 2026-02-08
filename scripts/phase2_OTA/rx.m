%% ========================================================================
%  OTA BATCH CAPTURE & VERIFY (CHAMBER MODE)
%  Strictly following Phase 1 Dataset Generator logic for extraction
%% ========================================================================
clear; clc; close all;

%% 1. CONFIGURATION
config.fs = 1e6;
config.fc = 918e6;
config.numPayloadSamples = 1024;
config.zcLength = 32;
config.zcRoot = 25;
config.frameLength = config.zcLength + config.numPayloadSamples;

%% 2. USER INTERACTION
fprintf('--- OTA Chamber: Batch Capture ---\n');
numFrames = input('Enter number of frames to capture (e.g., 500): ');
rxGain = input('Enter RX Gain (Manual, e.g., 30): ');
tag = input('Enter filename tag (e.g., 16QAM_Dist2m): ', 's');

%% 3. HARDWARE SETUP
try
    rx = sdrrx('Pluto', 'RadioID', 'usb:0', 'CenterFrequency', config.fc, 'BasebandSampleRate', config.fs);
    rx.GainSource = 'Manual';
    rx.Gain = rxGain;
    % Buffer size: 5x frame length to ensure ZC is captured
    rx.SamplesPerFrame = config.frameLength * 5; 
catch ME
    error('Pluto SDR RX Initialization failed: %s', ME.message);
end

%% 4. PREPARE REFERENCE ZC
n_zc = (0:config.zcLength-1)';
zcRef = exp(-1i * pi * config.zcRoot * n_zc .* (n_zc + 1) / config.zcLength);

%% 5. CAPTURE LOOP
batchData = zeros(numFrames, config.frameLength, 'like', 1j);
syncPeaks = zeros(numFrames, 1);

fprintf('Capturing %d frames...\n', numFrames);
progressBar = waitbar(0, 'Capturing OTA data...');

for f = 1:numFrames
    rx_raw = rx();
    
    % Cross-correlation for Timing Sync (Appendix C logic)
    [xc, lags] = xcorr(rx_raw, zcRef);
    [val, maxIdx] = max(abs(xc));
    
    % Store Normalized Correlation Peak (Reliability Metric)
    syncPeaks(f) = val / (config.zcLength * mean(abs(rx_raw)));
    
    % Sync and Extract
    if (maxIdx + config.frameLength - 1) <= length(rx_raw)
        batchData(f, :) = rx_raw(maxIdx : maxIdx + config.frameLength - 1).';
    else
        % Logic fallback: Skip failed syncs in batch
        batchData(f, :) = 0;
    end
    
    if mod(f, 50) == 0, waitbar(f/numFrames, progressBar); end
end
close(progressBar);
release(rx);

%% 6. QUALITY VERIFICATION (Visual)
all_payload = batchData(:, config.zcLength+1:end);
all_payload = all_payload(all_payload ~= 0); % Filter failed syncs

figure('Color', 'w', 'Name', ['OTA Verification: ' tag]);
subplot(2,1,1);
plot(all_payload(:), '.', 'Color', [0 0.4 0.7 0.03], 'MarkerSize', 1);
axis square; grid on; xlim([-2 2]); ylim([-2 2]);
title(['Density Constellation: ', tag]);

subplot(2,1,2);
plot(syncPeaks, 'Color', [0.8 0.2 0], 'LineWidth', 1);
title('Link Stability (ZC Peaks)');
ylabel('Peak Mag'); grid on;

%% 7. SAVE RESULTS
timestamp = datestr(now, 'HHMMSS');
fileName = sprintf('OTA_Batch_%s_%s.mat', tag, timestamp);
save(fileName, 'batchData', 'syncPeaks', 'config', 'tag');
fprintf('\n✓ Saved %d frames to: %s\n', numFrames, fileName);