function [augmentedFeatures, augmentedLabels] = applySMOTEAdvanced(features, labels, options)
%APPLYSMOTEADVANCED Advanced SMOTE with modern enhancements
%   [AUG_FEAT, AUG_LAB] = applySMOTEAdvanced(FEAT, LAB, OPTIONS) applies
%   SMOTE with additional modern augmentation techniques.
%
%   Inputs:
%       features - Feature matrix (N x D)
%       labels - Categorical or cell array of labels
%       options - Structure with options:
%           .k - Number of nearest neighbors (default: 5)
%           .ratio - Oversampling ratio (default: 1.0)
%           .method - 'smote', 'borderline', 'adasyn' (default: 'smote')
%           .applyPCA - Apply PCA before SMOTE (default: false)
%           .pcaComponents - Number of PCA components (default: 0.95 variance)
%
%   Outputs:
%       augmentedFeatures - Augmented feature matrix
%       augmentedLabels - Augmented labels

if nargin < 3
    options = struct();
end

% Helper function to check GPU
function gpuAvailable = canUseGPU()
    try
        gpuAvailable = canUseGPU && parallel.gpu.GPUDevice.isAvailable;
    catch
        gpuAvailable = false;
    end
end

% Default options
if ~isfield(options, 'k'), options.k = 5; end
if ~isfield(options, 'ratio'), options.ratio = 1.0; end
if ~isfield(options, 'method'), options.method = 'smote'; end
if ~isfield(options, 'applyPCA'), options.applyPCA = false; end
if ~isfield(options, 'pcaComponents'), options.pcaComponents = 0.95; end

% Convert labels to categorical if needed
if ~iscategorical(labels)
    labels = categorical(labels);
end

uniqueLabels = categories(labels);
numClasses = length(uniqueLabels);

fprintf('Applying %s augmentation...\n', upper(options.method));
fprintf('Number of classes: %d\n', numClasses);

% Normalize features (z-score normalization)
% Use GPU if available for large feature matrices
if size(features, 1) > 1000 && canUseGPU
    featuresGPU = gpuArray(features);
    featMean = gather(mean(featuresGPU, 1));
    featStd = gather(std(featuresGPU, [], 1));
    featuresNorm = gather((featuresGPU - featMean) ./ (featStd + eps));
    clear featuresGPU;
else
    featuresNorm = (features - mean(features, 1)) ./ (std(features, [], 1) + eps);
end

% Apply PCA if requested
pcaTransform = [];
if options.applyPCA
    fprintf('Applying PCA dimensionality reduction...\n');
    if options.pcaComponents < 1
        [coeff, score, ~, ~, explained] = pca(featuresNorm);
        cumsumExplained = cumsum(explained);
        numComponents = find(cumsumExplained >= options.pcaComponents * 100, 1);
    else
        [coeff, score] = pca(featuresNorm, 'NumComponents', options.pcaComponents);
        numComponents = options.pcaComponents;
    end
    featuresNorm = score(:, 1:numComponents);
    pcaTransform = coeff(:, 1:numComponents);
    fprintf('Reduced to %d components (%.1f%% variance)\n', numComponents, sum(explained(1:numComponents)));
end

% Process each class
augmentedFeatures = featuresNorm;
augmentedLabels = labels;

for classIdx = 1:numClasses
    currentLabel = uniqueLabels{classIdx};
    classMask = labels == currentLabel;
    classFeatures = featuresNorm(classMask, :);
    numSamples = size(classFeatures, 1);
    
    fprintf('Processing class %s: %d samples\n', char(currentLabel), numSamples);
    
    % Determine number of synthetic samples to generate
    numSynthetic = round(numSamples * options.ratio);
    
    if numSynthetic == 0
        continue;
    end
    
    % Generate synthetic samples based on method
    switch lower(options.method)
        case 'smote'
            synthetic = generateSMOTE(classFeatures, numSynthetic, options.k);
            
        case 'borderline'
            synthetic = generateBorderlineSMOTE(classFeatures, featuresNorm, ...
                                                classMask, numSynthetic, options.k);
            
        case 'adasyn'
            synthetic = generateADASYN(classFeatures, featuresNorm, ...
                                       classMask, numSynthetic, options.k);
            
        otherwise
            error('Unknown SMOTE method: %s', options.method);
    end
    
    % Append synthetic samples
    % Create synthetic labels as categorical to match augmentedLabels
    if iscategorical(augmentedLabels)
        % Create categorical array with same categories as augmentedLabels
        % Use categorical constructor to ensure proper type
        syntheticLabels = categorical(repmat({char(currentLabel)}, size(synthetic, 1), 1), uniqueLabels);
    else
        syntheticLabels = repmat(currentLabel, size(synthetic, 1), 1);
    end
    augmentedFeatures = [augmentedFeatures; synthetic];
    augmentedLabels = [augmentedLabels; syntheticLabels];
end

% Transform back from PCA space if needed
if options.applyPCA && ~isempty(pcaTransform)
    augmentedFeatures = augmentedFeatures * pcaTransform';
end

fprintf('Augmentation completed. Original: %d samples, Augmented: %d samples\n', ...
        length(labels), length(augmentedLabels));
end

function synthetic = generateSMOTE(minoritySamples, numSynthetic, k)
% Standard SMOTE generation
numSamples = size(minoritySamples, 1);
synthetic = zeros(numSynthetic, size(minoritySamples, 2));

% Find k-nearest neighbors for each sample
if numSamples > k
    idx = knnsearch(minoritySamples, minoritySamples, 'K', k+1);
    idx = idx(:, 2:end); % Remove self
else
    idx = repmat(1:numSamples, numSamples, 1);
    idx = idx - eye(numSamples) * numSamples;
    idx(idx <= 0) = 1;
end

% Generate synthetic samples
for i = 1:numSynthetic
    % Randomly select a sample
    sampleIdx = randi(numSamples);
    
    % Randomly select a neighbor
    neighborIdx = idx(sampleIdx, randi(k));
    
    % Generate synthetic sample
    lambda = rand();
    synthetic(i, :) = minoritySamples(sampleIdx, :) + ...
                     lambda * (minoritySamples(neighborIdx, :) - minoritySamples(sampleIdx, :));
end
end

function synthetic = generateBorderlineSMOTE(minoritySamples, allSamples, ...
                                             minorityMask, numSynthetic, k)
% Borderline SMOTE: only oversample samples near the decision boundary
numSamples = size(minoritySamples, 1);
numAll = size(allSamples, 1);

% Find k-nearest neighbors in all samples
idx = knnsearch(allSamples, minoritySamples, 'K', k+1);

% Count how many neighbors are from minority class
minorityCount = sum(~minorityMask(idx(:, 2:end)), 2);

% Borderline samples: more than half neighbors are from majority
borderlineMask = minorityCount >= k/2 & minorityCount < k;
borderlineSamples = minoritySamples(borderlineMask, :);

if isempty(borderlineSamples)
    % Fallback to standard SMOTE
    synthetic = generateSMOTE(minoritySamples, numSynthetic, k);
    return;
end

% Generate synthetic samples from borderline samples
synthetic = generateSMOTE(borderlineSamples, numSynthetic, k);
end

function synthetic = generateADASYN(minoritySamples, allSamples, ...
                                    minorityMask, numSynthetic, k)
% ADASYN: Adaptive Synthetic Sampling
numSamples = size(minoritySamples, 1);
numAll = size(allSamples, 1);

% Find k-nearest neighbors
idx = knnsearch(allSamples, minoritySamples, 'K', k+1);

% Calculate density distribution
density = zeros(numSamples, 1);
for i = 1:numSamples
    neighbors = idx(i, 2:end);
    density(i) = sum(~minorityMask(neighbors)) / k;
end

% Normalize density to get sampling weights
density = density / (sum(density) + eps);

% Calculate number of synthetic samples per minority sample
numPerSample = round(numSynthetic * density);
numPerSample = min(numPerSample, numSynthetic); % Cap at total

% Generate synthetic samples
synthetic = [];
for i = 1:numSamples
    if numPerSample(i) > 0
        % Find neighbors from minority class
        neighbors = idx(i, 2:end);
        minorityNeighbors = neighbors(minorityMask(neighbors));
        
        if ~isempty(minorityNeighbors)
            for j = 1:numPerSample(i)
                neighborIdx = minorityNeighbors(randi(length(minorityNeighbors)));
                lambda = rand();
                newSample = minoritySamples(i, :) + ...
                           lambda * (allSamples(neighborIdx, :) - minoritySamples(i, :));
                synthetic = [synthetic; newSample];
            end
        end
    end
end

% If we didn't generate enough, use standard SMOTE for remainder
if size(synthetic, 1) < numSynthetic
    remaining = numSynthetic - size(synthetic, 1);
    additional = generateSMOTE(minoritySamples, remaining, k);
    synthetic = [synthetic; additional];
end
end

