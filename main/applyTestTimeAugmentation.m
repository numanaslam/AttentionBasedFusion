function predictions = applyTestTimeAugmentation(classifier, features, testLabels, options)
%APPLYTESTTIMEAUGMENTATION Apply test-time augmentation for robust predictions
%   PREDICTIONS = applyTestTimeAugmentation(CLASSIFIER, FEATURES, LABELS, OPTIONS)
%   applies multiple augmentations to test features and averages predictions.
%
%   Inputs:
%       classifier - Trained classifier
%       features - Test feature matrix (N x D)
%       testLabels - Test labels (for evaluation, optional)
%       options - Structure with options:
%           .numAugmentations - Number of augmentations per sample (default: 5)
%           .augmentationTypes - Cell array of augmentation types:
%               'gaussian_noise', 'dropout', 'scale', 'shift' (default: all)
%           .noiseStd - Standard deviation for Gaussian noise (default: 0.01)
%           .dropoutRate - Feature dropout rate (default: 0.1)
%           .scaleRange - Scale range for feature scaling (default: [0.95, 1.05])
%           .shiftRange - Shift range for feature shifting (default: [-0.05, 0.05])
%           .votingMethod - 'average' or 'majority' (default: 'average')
%
%   Outputs:
%       predictions - Averaged predictions (N x 1) or (N x C) for probabilities

if nargin < 4
    options = struct();
end

% Default options
if ~isfield(options, 'numAugmentations'), options.numAugmentations = 5; end
if ~isfield(options, 'augmentationTypes')
    options.augmentationTypes = {'gaussian_noise', 'dropout', 'scale', 'shift'};
end
if ~isfield(options, 'noiseStd'), options.noiseStd = 0.01; end
if ~isfield(options, 'dropoutRate'), options.dropoutRate = 0.1; end
if ~isfield(options, 'scaleRange'), options.scaleRange = [0.95, 1.05]; end
if ~isfield(options, 'shiftRange'), options.shiftRange = [-0.05, 0.05]; end
if ~isfield(options, 'votingMethod'), options.votingMethod = 'average'; end

numSamples = size(features, 1);
numAug = options.numAugmentations;

fprintf('Applying test-time augmentation (%d augmentations per sample)...\n', numAug);

% Get all predictions
allPredictions = cell(numAug + 1, 1);  % +1 for original

% Original predictions (no augmentation)
fprintf('  Original predictions...\n');
try
    [~, scores] = predict(classifier, features);
    allPredictions{1} = scores;
catch
    % If classifier doesn't support scores, use class predictions
    pred = predict(classifier, features);
    allPredictions{1} = pred;
end

% Augmented predictions
for augIdx = 1:numAug
    fprintf('  Augmentation %d/%d...\n', augIdx, numAug);
    
    % Select random augmentation type
    augType = options.augmentationTypes{randi(length(options.augmentationTypes))};
    
    % Apply augmentation
    switch augType
        case 'gaussian_noise'
            % Add Gaussian noise
            noise = randn(size(features)) * options.noiseStd;
            augFeatures = features + noise;
            
        case 'dropout'
            % Random feature dropout
            mask = rand(size(features)) > options.dropoutRate;
            augFeatures = features .* mask;
            
        case 'scale'
            % Random feature scaling
            scale = options.scaleRange(1) + ...
                rand(1, size(features, 2)) * diff(options.scaleRange);
            augFeatures = features .* scale;
            
        case 'shift'
            % Random feature shifting
            shift = options.shiftRange(1) + ...
                rand(1, size(features, 2)) * diff(options.shiftRange);
            augFeatures = features + shift;
            
        otherwise
            % Default: Gaussian noise
            noise = randn(size(features)) * options.noiseStd;
            augFeatures = features + noise;
    end
    
    % Get predictions
    try
        [~, scores] = predict(classifier, augFeatures);
        allPredictions{augIdx + 1} = scores;
    catch
        pred = predict(classifier, augFeatures);
        allPredictions{augIdx + 1} = pred;
    end
end

% Combine predictions
fprintf('Combining predictions using %s voting...\n', options.votingMethod);

if iscell(allPredictions{1}) || iscategorical(allPredictions{1})
    % Categorical predictions - use majority voting
    % Convert to numeric for voting
    if iscategorical(allPredictions{1})
        uniqueLabels = categories(allPredictions{1});
    else
        uniqueLabels = unique(allPredictions{1});
    end
    
    numClasses = length(uniqueLabels);
    voteMatrix = zeros(numSamples, numClasses);
    
    for i = 1:length(allPredictions)
        pred = allPredictions{i};
        if iscategorical(pred)
            pred = double(pred);
        end
        for j = 1:numSamples
            classIdx = find(strcmp(uniqueLabels, char(pred(j))));
            if ~isempty(classIdx)
                voteMatrix(j, classIdx) = voteMatrix(j, classIdx) + 1;
            end
        end
    end
    
    [~, maxIdx] = max(voteMatrix, [], 2);
    predictions = categorical(uniqueLabels(maxIdx), uniqueLabels);
    
else
    % Numeric scores/probabilities - use averaging
    if isvector(allPredictions{1})
        % Single column (class indices or scores)
        allPredArray = zeros(numSamples, length(allPredictions));
        for i = 1:length(allPredictions)
            allPredArray(:, i) = double(allPredictions{i});
        end
        predictions = mean(allPredArray, 2);
    else
        % Matrix (probabilities per class)
        allPredArray = zeros(numSamples, size(allPredictions{1}, 2), length(allPredictions));
        for i = 1:length(allPredictions)
            allPredArray(:, :, i) = double(allPredictions{i});
        end
        predictions = mean(allPredArray, 3);
    end
end

fprintf('Test-time augmentation completed.\n');

end

