function bpa_resolved = resolve_conflict(bpa_hoc, bpa_cyclo, bpa_vit, reliabilities)
% RESOLVE_CONFLICT
% Fallback strategy when Dempster's rule encounters high conflict
% Uses weighted average based on classifier reliabilities
%
% INPUTS:
%   bpa_hoc, bpa_cyclo, bpa_vit - Individual BPAs
%   reliabilities - Reliability weights
%
% OUTPUT:
%   bpa_resolved - Resolved BPA using weighted average

num_elements = length(bpa_hoc);

% Calculate total reliability (for normalization)
total_reliability = reliabilities.hoc + reliabilities.cyclo + reliabilities.vit;

% Weighted average of BPAs
if total_reliability > 0
    bpa_resolved = (...
        reliabilities.hoc * bpa_hoc + ...
        reliabilities.cyclo * bpa_cyclo + ...
        reliabilities.vit * bpa_vit ...
    ) / total_reliability;
else
    % All unreliable - uniform uncertainty
    bpa_resolved = zeros(1, num_elements);
    bpa_resolved(end) = 1;
end

% Ensure normalization
bpa_resolved = bpa_resolved / sum(bpa_resolved);

end