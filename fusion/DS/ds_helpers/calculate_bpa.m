function bpa = calculate_bpa(probabilities, reliability, uncertainty_threshold)
% CALCULATE_BPA
% Converts classifier probability vector to Basic Probability Assignment
%
% INPUTS:
%   probabilities - [1×5] probability vector from classifier
%   reliability - Reliability weight (0-1) for this classifier
%   uncertainty_threshold - Minimum mass to assign to uncertainty
%
% OUTPUT:
%   bpa - [1×6] BPA vector [class1, class2, ..., class5, uncertainty]

num_classes = length(probabilities);

% Initialize BPA (classes + uncertainty)
bpa = zeros(1, num_classes + 1);

% Discount probabilities by reliability
% m(A) = r * P(A)
% m(Θ) = 1 - r + r * threshold
discounted_probs = reliability * probabilities;

% Assign mass to each class
bpa(1:num_classes) = discounted_probs;

% Calculate remaining mass for uncertainty (frame of discernment Θ)
% This represents "don't know"
total_belief = sum(bpa(1:num_classes));
remaining_mass = 1 - total_belief;

% Ensure minimum uncertainty threshold
uncertainty = max(remaining_mass, uncertainty_threshold);

% Renormalize if needed
if total_belief + uncertainty > 1
    scale_factor = (1 - uncertainty) / total_belief;
    bpa(1:num_classes) = bpa(1:num_classes) * scale_factor;
end

% Assign uncertainty mass
bpa(end) = uncertainty;

% Verify BPA sums to 1
assert(abs(sum(bpa) - 1) < 1e-6, 'BPA must sum to 1');

end