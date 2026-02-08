% train_dobre_clean.m
% FIXED: Only train on high-SNR files to avoid corrupted features

clear all; clc;

dataset_root = 'C:\Users\Mio\Documents\196\amc_dataset_mat_converted';
MIN_TRAINING_SNR = 18;      % Only use clean signals
FILES_PER_CLASS = 500;      % Use 500 high-SNR files per class

supported_mods = {'4ask', '8ask', 'bpsk', 'qpsk', '8psk', ...
                  '16psk', '16qam', '32qam', '64qam'};

fprintf('========================================\n');
fprintf('TRAINING WITH CLEAN DATA (SNR >= %d dB)\n', MIN_TRAINING_SNR);
fprintf('========================================\n');

LibraryMap = containers.Map();
FeatureStats = struct();
all_features_collection = {};

for i = 1:length(supported_mods)
    mod_name = supported_mods{i};

    % Locate folder ignoring case
    folder_name = find_folder_name(dataset_root, mod_name);
    if isempty(folder_name)
        warning('Folder not found for %s', mod_name);
        continue;
    end

    folder_path = fullfile(dataset_root, folder_name);
    files = dir(fullfile(folder_path, '*.mat'));

    fprintf('Training %-8s ... ', mod_name);

    feature_matrix = [];
    count = 0;

    for f = 1:length(files)
        if count >= FILES_PER_CLASS
            break;
        end

        filename = files(f).name;

        % Filter high-SNR
        snr_tokens = regexp(filename, 'snr_?(-?\d+)', 'tokens');
        if isempty(snr_tokens), continue; end

        file_snr = str2double(snr_tokens{1}{1});
        if file_snr < MIN_TRAINING_SNR, continue; end

        % Load signal
        try
            file_path = fullfile(folder_path, filename);
            data = load(file_path);
            vars = fieldnames(data);
            raw_sig = data.(vars{1});

            if size(raw_sig, 1) == 2
                sig = double(raw_sig(1,:) + 1j*raw_sig(2,:)).';
            else
                sig = double(raw_sig(:));
            end

            features = extract_enhanced_features(sig);
            feature_matrix = [feature_matrix; features];
            count = count + 1;

        catch
            continue;
        end
    end

    if count == 0
        warning('No valid files for %s', mod_name);
        continue;
    end

    avg_features = mean(feature_matrix, 1);
    LibraryMap(mod_name) = avg_features;

    all_features_collection{end+1} = feature_matrix;

    fprintf('Done (%d files)\n', count);
    fprintf('  Features: [%.2f %.2f %.1f %.1f %.1f %.2f %.2f %.2f %.2f]\n', ...
            avg_features);
end

% Compute global normalization stats
all_features = vertcat(all_features_collection{:});
FeatureStats.min = min(all_features);
FeatureStats.max = max(all_features);
FeatureStats.range = FeatureStats.max - FeatureStats.min;
FeatureStats.range(FeatureStats.range < 1e-6) = 1;

fprintf('\n=== Feature Ranges ===\n');
fprintf('AV: %.2f | C42: %.2f | P2: %.2f | P4: %.2f | P8: %.2f\n', ...
        FeatureStats.range(1:5));
fprintf('PhS: %.2f | EK: %.2f | PP: %.2f | SI: %.2f\n', ...
        FeatureStats.range(6:9));

save('Trained_Enhanced_Clean.mat', 'LibraryMap', 'FeatureStats');
fprintf('\n=== Saved: Trained_Enhanced_Clean.mat ===\n');

% --------------------------------------------
% Helper: Find folder ignoring casing
% --------------------------------------------
function folder_name = find_folder_name(root, mod_name)
items = dir(root);
folders = items([items.isdir]);

for i = 1:length(folders)
    if strcmpi(folders(i).name, mod_name)
        folder_name = folders(i).name;
        return;
    end
end
folder_name = '';
end

% --------------------------------------------
% Feature extraction
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

x2 = r.^2;  X2 = abs(fft(x2, N));
x4 = r.^4;  X4 = abs(fft(x4, N));
x8 = r.^8;  X8 = abs(fft(x8, N));
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
