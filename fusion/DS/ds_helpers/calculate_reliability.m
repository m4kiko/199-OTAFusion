function reliabilities = calculate_reliability(snr)
% CALCULATE_RELIABILITY
% Calculates reliability weights for each classifier based on SNR
% Uses calibration data from Phase 1 wired experiments
%
% INPUT:
%   snr - Signal SNR in dB
%
% OUTPUT:
%   reliabilities - struct with fields .hoc, .cyclo, .vit (0-1)

% Reliability models based on empirical accuracy vs SNR
% These should be calibrated from your Phase 1 wired results
% Format: [SNR, Accuracy] from your wired calibration

% HOC reliability (supports only 4 classes)
% Generally strong at high SNR, weaker at low SNR
hoc_calibration = [
    0,    0.45;   % SNR=0dB  → 45% accuracy
    2.5,  0.55;   % SNR=2.5dB
    5,    0.68;   % SNR=5dB
    7.5,  0.78;   % SNR=7.5dB
    10,   0.86;   % SNR=10dB
    12.5, 0.91;   % SNR=12.5dB
    15,   0.94    % SNR=15dB → 94% accuracy
];

% Cyclostationary reliability (supports all 5 classes)
% Good at mid-range SNR
cyclo_calibration = [
    0,    0.42;
    2.5,  0.58;
    5,    0.71;
    7.5,  0.82;
    10,   0.88;
    12.5, 0.92;
    15,   0.95
];

% ViT reliability (supports all 5 classes)
% Excellent at high SNR, struggles at low SNR
vit_calibration = [
    0,    0.35;
    2.5,  0.52;
    5,    0.70;
    7.5,  0.85;
    10,   0.93;
    12.5, 0.97;
    15,   0.99
];

% Interpolate reliability at given SNR
reliabilities.hoc = interp1(...
    hoc_calibration(:,1), hoc_calibration(:,2), snr, 'linear', 'extrap');

reliabilities.cyclo = interp1(...
    cyclo_calibration(:,1), cyclo_calibration(:,2), snr, 'linear', 'extrap');

reliabilities.vit = interp1(...
    vit_calibration(:,1), vit_calibration(:,2), snr, 'linear', 'extrap');

% Clamp to [0, 1]
reliabilities.hoc = max(0, min(1, reliabilities.hoc));
reliabilities.cyclo = max(0, min(1, reliabilities.cyclo));
reliabilities.vit = max(0, min(1, reliabilities.vit));

end