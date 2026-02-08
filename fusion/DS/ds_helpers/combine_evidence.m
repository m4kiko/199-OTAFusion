function [bpa_combined, conflict] = combine_evidence(bpa1, bpa2)
% COMBINE_EVIDENCE
% Combines two BPAs using Dempster's combination rule
%
% Dempster's Rule:
%   m₁₂(A) = [Σ m₁(B)m₂(C)] / (1 - K)
%            B∩C=A
%   where K = Σ m₁(B)m₂(C) is the conflict mass
%           B∩C=∅
%
% INPUTS:
%   bpa1 - [1×6] First BPA
%   bpa2 - [1×6] Second BPA
%
% OUTPUTS:
%   bpa_combined - [1×6] Combined BPA
%   conflict - Conflict measure K (0-1)

num_elements = length(bpa1);
num_classes = num_elements - 1;  % Last element is uncertainty

% Initialize combined BPA
bpa_combined = zeros(1, num_elements);

% Calculate conflict mass (K)
conflict = 0;

% Iterate over all pairs of focal elements
for i = 1:num_elements
    for j = 1:num_elements
        mass_product = bpa1(i) * bpa2(j);
        
        if i == j && i <= num_classes
            % Same singleton class: B ∩ C = A
            bpa_combined(i) = bpa_combined(i) + mass_product;
            
        elseif i == num_elements || j == num_elements
            % One is uncertainty (Θ): B ∩ Θ = B or Θ ∩ C = C
            if i == num_elements
                % bpa1 is uncertainty, take bpa2's element
                bpa_combined(j) = bpa_combined(j) + mass_product;
            else
                % bpa2 is uncertainty, take bpa1's element
                bpa_combined(i) = bpa_combined(i) + mass_product;
            end
            
        else
            % Different singleton classes: B ∩ C = ∅ (conflict!)
            conflict = conflict + mass_product;
        end
    end
end

% Normalize by (1 - K) - Dempster's normalization
if conflict < 1
    normalization_factor = 1 / (1 - conflict);
    bpa_combined = bpa_combined * normalization_factor;
else
    % Total conflict - return uniform uncertainty
    warning('DS:TotalConflict', 'Total conflict detected (K=%.4f)', conflict);
    bpa_combined = zeros(1, num_elements);
    bpa_combined(end) = 1;  % All mass on uncertainty
end

% Verify BPA sums to 1
assert(abs(sum(bpa_combined) - 1) < 1e-6, 'Combined BPA must sum to 1');

end