%% Adalm-Pluto Hardware Stress Test (10m Cable Verification)
clear; clc;

% --- Configuration ---
sampleRate = 2e6;        % 2 MHz (Standard for AMC)
centerFreq = 918e6;      % Your study frequency
duration = 30;           % Test duration in seconds
numSamples = 1024*4;     % Buffer size

% --- Initialization ---
try
    rx = sdrrx('Pluto', ...
        'CenterFrequency', centerFreq, ...
        'BasebandSampleRate', sampleRate, ...
        'SamplesPerFrame', numSamples, ...
        'OutputDataType', 'double');
    fprintf('Device found! Starting 30-second stress test...\n');
catch ME
    error('Pluto not found. Check DC power and USB connection.');
end

% --- Stress Test Loop ---
totalLostSamples = 0;
numIterations = ceil((duration * sampleRate) / numSamples);
tic;

for i = 1:numIterations
    [data, datavalid, overrange] = rx();
    
    % Check for overflows (Indicates the 10m cable is lagging)
    if ~datavalid
        totalLostSamples = totalLostSamples + numSamples;
    end
    
    % Visual feedback every 100 frames
    if mod(i, 100) == 0
        fprintf('Progress: %.1f%% | Lost Samples: %d\n', ...
            (i/numIterations)*100, totalLostSamples);
    end
end

% --- Final Report ---
testTime = toc;
release(rx);

fprintf('\n--- Test Results ---\n');
fprintf('Actual Duration: %.2f seconds\n', testTime);
if totalLostSamples == 0
    fprintf('STATUS: PASSED. The 10m Jasoz cable is stable.\n');
else
    fprintf('STATUS: FAILED. %d samples lost. Data integrity compromised.\n', totalLostSamples);
    fprintf('Suggestion: Reduce Sample Rate or check Ferrite placement.\n');
end

% Plot a snippet to ensure signal isn't "flatlined"
plot(real(data(1:200)));
title('Real Component of Received Signal');
xlabel('Sample Index'); ylabel('Amplitude');