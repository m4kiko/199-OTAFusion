%% ========================================================================
%  OTA PERPETUAL TRANSMITTER (CHAMBER MODE)
%  Strictly following Phase 1 Dataset Generator logic
%% ========================================================================
clear; clc; close all;

%% 1. CONFIGURATION (Strictly from provided Phase 1 logic)
config.fs = 1e6;                 % 1 Msps
config.fc = 918e6;                % 918 MHz
config.numPayloadSamples = 1024;
config.zcLength = 32;
config.zcRoot = 25;
config.targetRMS = 0.993;
config.rrcRolloff = 0.25;
config.rrcSpan = 8;

% Parameters for each modulation
config.modTypes = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'};
config.symbolRates = [250e3, 200e3, 150e3, 125e3, 100e3]; % ksps

%% 2. USER INTERACTION
fprintf('--- OTA Chamber: Perpetual TX ---\n');
for i = 1:length(config.modTypes)
    fprintf('%d: %s\n', i, config.modTypes{i});
end
choice = input('Select Modulation to Transmit (1-5): ');
selectedMod = config.modTypes{choice};
symbolRate = config.symbolRates(choice);
sps = round(config.fs / symbolRate);

txGain = input('Enter TX Gain (e.g., -15): ');

%% 3. SIGNAL GENERATION (Religious adherence to provided logic)
% 3.1 Generate ZC Preamble
zcSeq = generateZadoffChu(config.zcLength, config.zcRoot);

% 3.2 Generate Payload Symbols
numSymbols = ceil(config.numPayloadSamples / sps) + config.rrcSpan;
symbols = generateSymbols(selectedMod, numSymbols);

% 3.3 Pulse Shaping
txFilter = rcosdesign(config.rrcRolloff, config.rrcSpan, sps, 'sqrt');
upsampled = upsample(symbols, sps);
shapedSignal = filter(txFilter, 1, upsampled);

% 3.4 Transient Removal (extract 1024)
delay = config.rrcSpan * sps / 2;
shapedSignal = shapedSignal(delay+1 : delay+config.numPayloadSamples);

% 3.5 Unit Power Normalization
shapedSignal = shapedSignal / sqrt(mean(abs(shapedSignal).^2));

% 3.6 Assembly and Final RadioML Normalization (0.993 RMS)
framedSignal = [zcSeq; shapedSignal];
tx_frame = normalizeRadioML(framedSignal, config.targetRMS);

%% 4. HARDWARE TRANSMISSION
try
    tx = sdrtx('Pluto', 'RadioID', 'usb:0', 'CenterFrequency', config.fc, 'BasebandSampleRate', config.fs);
    tx.Gain = txGain;
    
    fprintf('\nTransmitting %s perpetually at %.2f MHz...\n', selectedMod, config.fc/1e6);
    fprintf('Press Ctrl+C or clear tx to stop.\n');
    
    transmitRepeat(tx, tx_frame);
catch ME
    error('Pluto SDR TX Initialization failed: %s', ME.message);
end

%% ========================================================================
%  HELPER FUNCTIONS (Strictly from provided source)
%% ========================================================================
function zcSeq = generateZadoffChu(N, u)
    n = (0:N-1)';
    zcSeq = exp(-1i * pi * u * n .* (n + 1) / N);
end

function symbols = generateSymbols(modType, numSymbols)
    switch modType
        case 'BPSK', M = 2; symbols = pskmod(randi([0 M-1], numSymbols, 1), M, 0, 'gray');
        case 'QPSK', M = 4; symbols = pskmod(randi([0 M-1], numSymbols, 1), M, 0, 'gray');
        case '8PSK', M = 8; symbols = pskmod(randi([0 M-1], numSymbols, 1), M, 0, 'gray');
        case '16QAM', M = 16; symbols = qammod(randi([0 M-1], numSymbols, 1), M, 'gray', 'UnitAveragePower', true);
        case '64QAM', M = 64; symbols = qammod(randi([0 M-1], numSymbols, 1), M, 'gray', 'UnitAveragePower', true);
    end
end

function normalizedSignal = normalizeRadioML(signal, targetRMS)
    signal = signal / sqrt(mean(abs(signal).^2)); % Stage 1
    normalizedSignal = (targetRMS / sqrt(mean(abs(signal).^2))) * signal; % Stage 2
end