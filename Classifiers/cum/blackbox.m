function output = blackbox(inputData)
% BLACKBOX_HOC (Cumulant Based Classifier)
% Input:  inputData (struct with .signal field OR raw 2xN matrix)
% Output: Structured output compatible with test harnesses.

    % --- 1. INPUT SANITIZATION ---
    % Allows this to work with your .mat files immediately
    if isstruct(inputData)
        if isfield(inputData, 'signal')
            raw_sig = inputData.signal;
        else
            error('Input struct must have a .signal field');
        end
    else
        raw_sig = inputData;
    end

    % Convert 2xN (Real/Imag) to 1xN Complex Vector
    if ismatrix(raw_sig) && size(raw_sig, 1) == 2
        I = double(raw_sig(1, :));
        Q = double(raw_sig(2, :));
        Sig = I + 1j*Q;
    else
        Sig = double(raw_sig);
        if size(Sig, 1) > 1, Sig = Sig.'; end % Ensure row vector
    end

    % --- 2. DEFINE POOL & REFERENCE ---
    % Note: HOC works best on this specific subset. 
    % BPSK and ASK are hard to distinguish from QPSK/PAM using only these features 
    % without checking C40 conjugate properties, so we stick to your provided pool.
    modulationPool = {'4psk', '8psk', '16qam', '64qam'};
    
    % Reference Table (C40n, C42n, C63n)
    cumu_ref = [
        1,       -1,       4;         % 4psk (QPSK)
        0,       -1,       4;         % 8psk
        -0.68,   -0.68,    2.08;      % 16qam
        -0.6191, -0.6191,  1.797      % 64qam
    ];

    % --- 3. CALCULATE FEATURES (Your Code) ---
    C21 = mean(abs(Sig).^2); % Signal Power
    
    % Safety check for empty/silent signals
    if C21 < 1e-10
        output.bestGuess = 'Noise';
        output.confidence = 0.0;
        return;
    end
    
    % Normalize signal power to 1 for calculation stability
    % (Ref table assumes unit power)
    Sig = Sig / sqrt(C21); 
    C21 = 1; 

    C20 = mean(Sig.^2);
    M40 = mean(Sig.^4);
    
    C40 = M40 - 3*C20^2;
    C42 = mean(abs(Sig).^4) - abs(C20)^2 - 2*C21^2;
    C63 = mean(abs(Sig).^6) - 9*mean(abs(Sig).^4)*C21 + 12*(abs(C20)^2)*C21 + 12*C21^3;

    % Feature Vector
    cumu = [
        C40/(C21^2), ...
        C42/(C21^2), ...
        C63/(C21^3) 
    ];

    % --- 4. CALCULATE DISTANCE & PROBABILITIES ---
    cumu_vec = repmat(cumu, size(cumu_ref, 1), 1);
    err = abs(cumu_vec - cumu_ref);
    err_totals = sum(err, 2); 
    
    % Convert error to "probability" (Heuristic: inverse distance)
    scores = 1 ./ (err_totals + 1e-6); % 1e-6 prevents div by zero
    probabilities = scores / sum(scores);

    % --- 5. SORT & FORMAT OUTPUT ---
    [sortedProbs, sortIndices] = sort(probabilities, 'descend');
    
    output.bestGuess = modulationPool{sortIndices(1)};
    output.confidence = sortedProbs(1);
    
    % Prepare Top Guesses Table
    numToShow = min(5, length(modulationPool));
    output.topGuesses = cell(numToShow, 2);
    
    fprintf('\n---------------------------------\n');
    fprintf('   BLACKBOX HOC CLASSIFIER       \n');
    fprintf('---------------------------------\n');
    
    for k = 1:numToShow
        idx = sortIndices(k);
        modName = modulationPool{idx};
        probVal = sortedProbs(k);
        
        output.topGuesses{k, 1} = modName;
        output.topGuesses{k, 2} = probVal;
        
        % Print formatted line
        fprintf('%d. %-10s : %6.2f%%\n', k, modName, probVal * 100);
    end
    fprintf('---------------------------------\n');
    
    % Debug: Show calculated features vs Best Match Reference
    % This helps debug why a signal might be misclassified
    fprintf('Debug Features: [%.2f, %.2f, %.2f]\n', cumu(1), cumu(2), cumu(3));
    fprintf('\n');
end