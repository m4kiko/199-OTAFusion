function output = Dempster_Shafer_Fusion(classifier_outputs, snr, config)
% DEMPSTER_SHAFER_FUSION
% Fuses predictions from multiple classifiers using Dempster-Shafer theory
%
% INPUTS:
%   classifier_outputs - struct with fields:
%       .hoc   - [1×5] probability vector (may have zeros for unsupported classes)
%       .cyclo - [1×5] probability vector
%       .vit   - [1×5] probability vector
%   snr - Signal SNR in dB (used for reliability weighting)
%   config - Configuration struct (optional)
%
% OUTPUT:
%   output - struct with fields:
%       .prediction - Predicted class name (string)
%       .confidence - Confidence in prediction (0-1)
%       .bpa_combined - Final combined BPA [1×5]
%       .conflict - Conflict measure (0-1)
%       .individual_bpas - Individual BPAs before combination
%       .reliabilities - Reliability weights used

% Default configuration
if nargin < 3
    config = struct();
end

% Set defaults
if ~isfield(config, 'fusion_classes')
    config.fusion_classes = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'};
end
if ~isfield(config, 'uncertainty_threshold')
    config.uncertainty_threshold = 0.05;  % Minimum mass for uncertainty
end
if ~isfield(config, 'conflict_threshold')
    config.conflict_threshold = 0.95;  % High conflict warning
end
if ~isfield(config, 'use_reliability_weighting')
    config.use_reliability_weighting = true;
end

num_classes = length(config.fusion_classes);

%% STEP 1: CALCULATE RELIABILITY WEIGHTS
% Based on SNR and historical performance
if config.use_reliability_weighting
    reliabilities = calculate_reliability(snr);
else
    reliabilities = struct('hoc', 1, 'cyclo', 1, 'vit', 1);
end

%% STEP 2: CONVERT PROBABILITIES TO BASIC PROBABILITY ASSIGNMENTS (BPAs)
% BPA represents belief in each hypothesis

% HOC BPA
bpa_hoc = calculate_bpa(...
    classifier_outputs.hoc, ...
    reliabilities.hoc, ...
    config.uncertainty_threshold);

% Cyclo BPA  
bpa_cyclo = calculate_bpa(...
    classifier_outputs.cyclo, ...
    reliabilities.cyclo, ...
    config.uncertainty_threshold);

% ViT BPA
bpa_vit = calculate_bpa(...
    classifier_outputs.vit, ...
    reliabilities.vit, ...
    config.uncertainty_threshold);

% Store individual BPAs
output.individual_bpas = struct();
output.individual_bpas.hoc = bpa_hoc;
output.individual_bpas.cyclo = bpa_cyclo;
output.individual_bpas.vit = bpa_vit;

%% STEP 3: COMBINE EVIDENCE USING DEMPSTER'S RULE
% Combine pairwise: (HOC ⊕ Cyclo) ⊕ ViT

% First combination: HOC ⊕ Cyclo
[bpa_temp, conflict_12] = combine_evidence(bpa_hoc, bpa_cyclo);

% Second combination: (HOC ⊕ Cyclo) ⊕ ViT
[bpa_combined, conflict_123] = combine_evidence(bpa_temp, bpa_vit);

% Total conflict
output.conflict = 1 - (1 - conflict_12) * (1 - conflict_123);

%% STEP 4: HANDLE HIGH CONFLICT
if output.conflict > config.conflict_threshold
    % High conflict detected - use alternative strategy
    warning('DS:HighConflict', ...
        'High conflict detected (%.2f%%). Using fallback strategy.', ...
        output.conflict * 100);
    
    % Fallback: Use weighted average instead
    bpa_combined = resolve_conflict(...
        bpa_hoc, bpa_cyclo, bpa_vit, ...
        reliabilities);
end

%% STEP 5: MAKE FINAL DECISION
% Extract belief for each class (excluding uncertainty)
beliefs = bpa_combined(1:num_classes);

% Normalize to get final probabilities
if sum(beliefs) > 0
    final_probs = beliefs / sum(beliefs);
else
    % All mass on uncertainty - use uniform distribution
    final_probs = ones(1, num_classes) / num_classes;
end

% Get prediction
[max_prob, max_idx] = max(final_probs);

output.prediction = config.fusion_classes{max_idx};
output.confidence = max_prob;
output.bpa_combined = bpa_combined;
output.final_probabilities = final_probs;
output.reliabilities = reliabilities;

end