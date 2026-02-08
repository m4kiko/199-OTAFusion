% classify_dobre_clean.m
% FIXED: Normalized weighted distance classifier
function output = classify_dobre_bruteforce(inputStruct)

persistent LibraryMap FeatureStats;

if isempty(LibraryMap)
    if isfile('Trained_Enhanced_Clean.mat')
        tmp = load('Trained_Enhanced_Clean.mat');
        LibraryMap = tmp.LibraryMap;
        FeatureStats = tmp.FeatureStats;
    else
        error('Training file missing! Run train_dobre_clean.m.');
    end
end

raw = inputStruct.signal;

if ismatrix(raw) && size(raw,1) == 2
    sig = double(raw(1,:) + 1j*raw(2,:)).';
else
    sig = double(raw(:));
end

test_features = extract_enhanced_features(sig);

classes = LibraryMap.keys;
num_classes = length(classes);
scores = zeros(1, num_classes);

% Weighted distance
W = [8 8 6 6 6 4 5 3 4];

for k = 1:num_classes
    ref = LibraryMap(classes{k});

    diff = (test_features - ref) ./ FeatureStats.range;
    distance = norm(diff .* W);

    scores(k) = -distance;
end

[sorted_scores, idx] = sort(scores, 'descend');

output.bestGuess = classes{idx(1)};

if num_classes > 1
    margin = sorted_scores(1) - sorted_scores(2);
    output.confidence = 1/(1+exp(-margin));
else
    output.confidence = 1.0;
end

temp_probs = softmax(sorted_scores);
output.topGuesses = cell(min(5,num_classes), 2);

for i = 1:min(5, num_classes)
    output.topGuesses{i,1} = classes{idx(i)};
    output.topGuesses{i,2} = temp_probs(i);
end

end

% --------------------------------------------
% Feature extraction (same as training)
% --------------------------------------------
function F = extract_enhanced_features(r)

r = r - mean(r);
pwr = mean(abs(r).^2);
if pwr > 0, r = r / sqrt(pwr); end

N = length(r);
amp = abs(r);
phase = angle(r);

f1 = std(amp) / (mean(amp) + 1e-6);

m20 = mean(r.^2);
m42 = mean(abs(r).^4);
c42 = m42 - abs(m20)^2 - 2;
f2 = abs(c42);

x2 = r.^2; X2 = abs(fft(x2, N));
x4 = r.^4; X4 = abs(fft(x4, N));
x8 = r.^8; X8 = abs(fft(x8, N));
f3 = max(X2) / N;
f4 = max(X4) / N;
f5 = max(X8) / N;

phase_diff = diff(unwrap(phase));
f6 = std(phase_diff);

f7 = kurtosis(amp);

phase_fft = abs(fft(phase_diff));
f8 = max(phase_fft) / length(phase_fft);

spec = abs(fft(r)).^2;
spec = spec / sum(spec);
spec_sorted = sort(spec, 'descend');
cumsum_spec = cumsum(spec_sorted);
idx_90 = find(cumsum_spec >= 0.9, 1);
if isempty(idx_90), idx_90 = N; end
f9 = idx_90 / N;

F = [f1 f2 f3 f4 f5 f6 f7 f8 f9];
F(isnan(F) | isinf(F)) = 0;
end

function p = softmax(x)
x = x - max(x);
ex = exp(x);
p = ex / sum(ex);
end
