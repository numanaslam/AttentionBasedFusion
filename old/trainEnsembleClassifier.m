function [ensembleClassifier, results] = trainEnsembleClassifier(features, labels, options)
%TRAINENSEMBLECLASSIFIER Train ensemble of multiple classifiers
%   [ENSEMBLE, RESULTS] = trainEnsembleClassifier(FEAT, LAB, OPTIONS) trains
%   an ensemble of multiple classifiers and combines their predictions.
%
%   Inputs:
%       features - Feature matrix (N x D) - should be normalized
%       labels - Categorical labels
%       options - Structure with options:
%           .classifiers - Cell array of classifier types (default: {'svm', 'ensemble', 'neural'})
%           .votingMethod - 'majority', 'weighted', 'stacking' (default: 'weighted')
%           .cvFolds - Cross-validation folds (default: 5)
%           .optimizeHyperparams - Optimize hyperparameters (default: true)
%           .featuresAlreadyNormalized - Skip normalization (default: false)
%
%   Outputs:
%       ensembleClassifier - Structure containing trained classifiers and voting weights
%       results - Structure with training results

if nargin < 3
    options = struct();
end

% Default options
if ~isfield(options, 'classifiers')
    options.classifiers = {'svm', 'ensemble', 'neural'};
end
if ~isfield(options, 'votingMethod'), options.votingMethod = 'weighted'; end
if ~isfield(options, 'cvFolds'), options.cvFolds = 5; end
if ~isfield(options, 'optimizeHyperparams'), options.optimizeHyperparams = true; end
if ~isfield(options, 'featuresAlreadyNormalized'), options.featuresAlreadyNormalized = false; end

fprintf('========================================\n');
fprintf('Training Ensemble Classifier\n');
fprintf('========================================\n');
fprintf('Classifiers: %s\n', strjoin(options.classifiers, ', '));
fprintf('Voting method: %s\n', options.votingMethod);
fprintf('\n');

% Initialize results
results = struct();
results.classifiers = options.classifiers;
results.votingMethod = options.votingMethod;
results.individualResults = cell(length(options.classifiers), 1);
results.weights = zeros(length(options.classifiers), 1);

% Train individual classifiers
trainedClassifiers = cell(length(options.classifiers), 1);
cvAccuracies = zeros(length(options.classifiers), 1);

for i = 1:length(options.classifiers)
    fprintf('Training classifier %d/%d: %s\n', i, length(options.classifiers), ...
            upper(options.classifiers{i}));
    
    % Train classifier
    classifierOptions = struct();
    classifierOptions.cvFolds = options.cvFolds;
    classifierOptions.optimizeHyperparams = options.optimizeHyperparams;
    classifierOptions.featuresAlreadyNormalized = options.featuresAlreadyNormalized;
    classifierOptions.useGPU = false;  % Ensemble training on CPU for compatibility
    
    [trainedClassifiers{i}, classifierResults] = trainModernClassifier(...
        features, labels, options.classifiers{i}, classifierOptions);
    
    % Store results
    results.individualResults{i} = classifierResults;
    cvAccuracies(i) = classifierResults.cvAccuracy;
    
    fprintf('  CV Accuracy: %.2f%%\n', cvAccuracies(i) * 100);
    fprintf('\n');
end

% Compute voting weights
switch lower(options.votingMethod)
    case 'majority'
        % Equal weights
        results.weights = ones(length(options.classifiers), 1) / length(options.classifiers);
        
    case 'weighted'
        % Weight by CV accuracy
        % Normalize to sum to 1
        results.weights = cvAccuracies / sum(cvAccuracies);
        
    case 'stacking'
        % Use meta-learner (simplified: use average of top performers)
        [~, sortedIdx] = sort(cvAccuracies, 'descend');
        topN = min(3, length(options.classifiers));
        results.weights = zeros(length(options.classifiers), 1);
        results.weights(sortedIdx(1:topN)) = 1 / topN;
        
    otherwise
        % Default: equal weights
        results.weights = ones(length(options.classifiers), 1) / length(options.classifiers);
end

% Create ensemble structure
ensembleClassifier = struct();
ensembleClassifier.classifiers = trainedClassifiers;
ensembleClassifier.weights = results.weights;
ensembleClassifier.classifierTypes = options.classifiers;
ensembleClassifier.votingMethod = options.votingMethod;
ensembleClassifier.classNames = categories(labels);

% Evaluate ensemble on training data (for reporting)
fprintf('Evaluating ensemble...\n');
ensemblePredictions = predictEnsemble(ensembleClassifier, features);
ensembleAccuracy = sum(ensemblePredictions == labels) / length(labels) * 100;
fprintf('Ensemble training accuracy: %.2f%%\n', ensembleAccuracy);

% Cross-validation for ensemble
fprintf('Performing ensemble cross-validation...\n');
cvPartition = cvpartition(labels, 'KFold', options.cvFolds);
cvAccuraciesEnsemble = zeros(options.cvFolds, 1);

for fold = 1:options.cvFolds
    trainIdx = training(cvPartition, fold);
    testIdx = test(cvPartition, fold);
    
    % Train ensemble on fold
    foldEnsemble = struct();
    foldEnsemble.classifiers = cell(length(options.classifiers), 1);
    
    for i = 1:length(options.classifiers)
        foldOptions = struct();
        foldOptions.cvFolds = 3;  % Reduced for speed
        foldOptions.optimizeHyperparams = false;  % Skip optimization in CV
        foldOptions.featuresAlreadyNormalized = options.featuresAlreadyNormalized;
        foldOptions.useGPU = false;
        
        [foldEnsemble.classifiers{i}, ~] = trainModernClassifier(...
            features(trainIdx, :), labels(trainIdx), ...
            options.classifiers{i}, foldOptions);
    end
    
    foldEnsemble.weights = results.weights;
    foldEnsemble.classifierTypes = options.classifiers;
    foldEnsemble.votingMethod = options.votingMethod;
    foldEnsemble.classNames = categories(labels);
    
    % Predict on test fold
    foldPredictions = predictEnsemble(foldEnsemble, features(testIdx, :));
    cvAccuraciesEnsemble(fold) = sum(foldPredictions == labels(testIdx)) / ...
                                  length(labels(testIdx));
end

results.cvAccuracy = mean(cvAccuraciesEnsemble);
results.cvStd = std(cvAccuraciesEnsemble);
results.individualCVAccuracies = cvAccuracies;
results.ensembleTrainingAccuracy = ensembleAccuracy / 100;

fprintf('\n========================================\n');
fprintf('Ensemble Training Summary\n');
fprintf('========================================\n');
for i = 1:length(options.classifiers)
    fprintf('%s: CV=%.2f%%, Weight=%.3f\n', ...
            upper(options.classifiers{i}), ...
            cvAccuracies(i) * 100, results.weights(i));
end
fprintf('Ensemble CV Accuracy: %.2f%% (std: %.2f%%)\n', ...
        results.cvAccuracy * 100, results.cvStd * 100);
fprintf('========================================\n\n');

end

function predictions = predictEnsemble(ensembleClassifier, features)
%PREDICTENSEMBLE Predict using ensemble of classifiers

numSamples = size(features, 1);
numClassifiers = length(ensembleClassifier.classifiers);
classNames = ensembleClassifier.classNames;
numClasses = length(classNames);

% Get predictions from all classifiers
allPredictions = cell(numClassifiers, 1);
allScores = zeros(numSamples, numClasses, numClassifiers);

for i = 1:numClassifiers
    try
        % Try to get probability scores
        [pred, scores] = predict(ensembleClassifier.classifiers{i}, features);
        allPredictions{i} = pred;
        
        % Convert scores to probability matrix if needed
        if size(scores, 2) == numClasses
            allScores(:, :, i) = scores;
        else
            % Convert class predictions to one-hot
            for j = 1:numSamples
                classIdx = find(strcmp(classNames, char(pred(j))));
                if ~isempty(classIdx)
                    allScores(j, classIdx, i) = 1.0;
                end
            end
        end
    catch
        % Fallback: use class predictions only
        pred = predict(ensembleClassifier.classifiers{i}, features);
        allPredictions{i} = pred;
        
        % Convert to one-hot
        for j = 1:numSamples
            classIdx = find(strcmp(classNames, char(pred(j))));
            if ~isempty(classIdx)
                allScores(j, classIdx, i) = 1.0;
            end
        end
    end
end

% Combine predictions based on voting method
switch lower(ensembleClassifier.votingMethod)
    case 'majority'
        % Majority voting
        voteMatrix = zeros(numSamples, numClasses);
        for i = 1:numClassifiers
            pred = allPredictions{i};
            for j = 1:numSamples
                classIdx = find(strcmp(classNames, char(pred(j))));
                if ~isempty(classIdx)
                    voteMatrix(j, classIdx) = voteMatrix(j, classIdx) + 1;
                end
            end
        end
        [~, maxIdx] = max(voteMatrix, [], 2);
        predictions = categorical(classNames(maxIdx), classNames);
        
    case {'weighted', 'stacking'}
        % Weighted voting using scores
        weightedScores = zeros(numSamples, numClasses);
        for i = 1:numClassifiers
            weightedScores = weightedScores + ...
                allScores(:, :, i) * ensembleClassifier.weights(i);
        end
        [~, maxIdx] = max(weightedScores, [], 2);
        predictions = categorical(classNames(maxIdx), classNames);
        
    otherwise
        % Default: majority voting
        voteMatrix = zeros(numSamples, numClasses);
        for i = 1:numClassifiers
            pred = allPredictions{i};
            for j = 1:numSamples
                classIdx = find(strcmp(classNames, char(pred(j))));
                if ~isempty(classIdx)
                    voteMatrix(j, classIdx) = voteMatrix(j, classIdx) + 1;
                end
            end
        end
        [~, maxIdx] = max(voteMatrix, [], 2);
        predictions = categorical(classNames(maxIdx), classNames);
end

end

