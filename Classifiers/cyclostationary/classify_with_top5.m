function output = classify_with_top5(inputStruct)
% CLASSIFY_WITH_TOP5 (Normalized Edition)
% Runs the Logic Classifier for accuracy, but blends probabilities
% so the Top 5 list sums to 100% mathematically.

    % 1. RUN LOGIC CLASSIFIER (The Accuracy King)
    main_result = classify_dobre_bruteforce(inputStruct);
    best_guess = main_result.bestGuess;
    
    % 2. RUN DISTANCE CHECK (The Runner-Up Finder)
    % Get raw probabilities for all classes
    [all_classes, all_probs] = internal_distance_check(inputStruct);
    
    % 3. MERGE AND NORMALIZE
    % If Logic picked a winner, we boost its probability to 95%
    % and squash the others into the remaining 5%.
    
    FORCED_CONFIDENCE = 0.95;
    
    % Find where the logic winner is in the list
    winner_idx = find(strcmp(all_classes, best_guess));
    
    % Scale everyone else down
    % If original sum was 1.0, and we want them to sum to (1 - 0.95)
    % we multiply by (0.05 / sum_of_others)
    
    % Remove winner from "others" calculation to avoid double counting
    others_mask = true(size(all_probs));
    others_mask(winner_idx) = false;
    
    sum_others = sum(all_probs(others_mask));
    
    % Avoid divide by zero if sum_others is tiny
    if sum_others < 1e-9
        scale_factor = 0;
    else
        scale_factor = (1 - FORCED_CONFIDENCE) / sum_others;
    end
    
    % Apply scaling
    final_probs = all_probs * scale_factor;
    final_probs(winner_idx) = FORCED_CONFIDENCE; % Force winner
    
    % 4. SORT AND FORMAT
    [sorted_probs, idx] = sort(final_probs, 'descend');
    
    output.bestGuess = best_guess;
    output.confidence = sorted_probs(1);
    
    output.topGuesses = cell(5, 2);
    for i = 1:5
        output.topGuesses{i, 1} = all_classes{idx(i)};
        output.topGuesses{i, 2} = sorted_probs(i);
    end
end

% --- INTERNAL HELPER ---
function [classes, probs] = internal_distance_check(inputStruct)
    persistent LibraryMap FeatureStats;
    if isempty(LibraryMap)
        if isfile('Trained_Enhanced_Clean.mat')
            tmp = load('Trained_Enhanced_Clean.mat');
            LibraryMap = tmp.LibraryMap;
            FeatureStats = tmp.FeatureStats;
        else
             % Fallback if training missing
             classes = {'unknown'}; probs = 0; return;
        end
    end
    
    raw = inputStruct.signal;
    if ismatrix(raw) && size(raw, 1) == 2
        sig = double(raw(1,:) + 1j*raw(2,:)).';
    else
        sig = double(raw(:));
    end
    
    F = extract_enhanced_features(sig);
    
    classes = LibraryMap.keys;
    scores = zeros(1, length(classes));
    W = [8 8 6 6 6 4 5 3 4];
    
    for k = 1:length(classes)
        ref = LibraryMap(classes{k});
        diff = (F - ref) ./ FeatureStats.range;
        d = norm(diff .* W);
        scores(k) = -d;
    end
    
    probs = softmax(scores);
end

% --- EXTRACTOR ---
function F = extract_enhanced_features(r)
    r = r - mean(r);
    pwr = mean(abs(r).^2);
    if pwr > 0, r = r / sqrt(pwr); end
    N = length(r);
    amp = abs(r);
    phase = angle(r);
    f1 = std(amp) / (mean(amp) + 1e-6);
    m20 = mean(r.^2); m42 = mean(abs(r).^4);
    f2 = abs(m42 - abs(m20)^2 - 2);
    x2 = r.^2; X2 = abs(fft(x2, N)); f3 = max(X2) / N;
    x4 = r.^4; X4 = abs(fft(x4, N)); f4 = max(X4) / N;
    x8 = r.^8; X8 = abs(fft(x8, N)); f5 = max(X8) / N;
    phase_diff = diff(unwrap(phase)); f6 = std(phase_diff);
    f7 = kurtosis(amp);
    phase_fft = abs(fft(phase_diff)); f8 = max(phase_fft) / length(phase_fft);
    spec = abs(fft(r)).^2; spec = spec / sum(spec);
    spec_sorted = sort(spec, 'descend');
    idx_90 = find(cumsum(spec_sorted) >= 0.9, 1);
    if isempty(idx_90), idx_90 = N; end
    f9 = idx_90 / N;
    F = [f1, f2, f3, f4, f5, f6, f7, f8, f9];
    F(isnan(F) | isinf(F)) = 0;
end

function p = softmax(x)
    x = x - max(x); ex = exp(x); p = ex / sum(ex);
end