function [classifier, results] = trainModernClassifier(features, labels, classifierType, options)
%TRAINMODERNCLASSIFIER Train modern classifier with hyperparameter tuning
%   [CLASSIFIER, RESULTS] = trainModernClassifier(FEAT, LAB, TYPE, OPTIONS)
%   trains a classifier with modern techniques and hyperparameter optimization.
%
%   Inputs:
%       features - Feature matrix (N x D) - should be already normalized
%       labels - Categorical labels
%       classifierType - 'svm', 'ensemble', 'xgboost', 'neural'
%       options - Structure with training options
%           .featuresAlreadyNormalized - If true, skip normalization (default: false)
%
%   Outputs:
%       classifier - Trained classifier
%       results - Structure with training results

if nargin < 4
    options = struct();
end

% Default options
if ~isfield(options, 'cvFolds'), options.cvFolds = 5; end
if ~isfield(options, 'optimizeHyperparams'), options.optimizeHyperparams = true; end
if ~isfield(options, 'usePCA'), options.usePCA = false; end
if ~isfield(options, 'pcaVariance'), options.pcaVariance = 0.95; end
if ~isfield(options, 'pcaVariance'), options.pcaVariance = 0.95; end
if ~isfield(options, 'featuresAlreadyNormalized'), options.featuresAlreadyNormalized = false; end
if ~isfield(options, 'augmentation'), options.augmentation = struct('enabled', false); end

fprintf('Training %s classifier...\n', upper(classifierType));

% Initialize results structure
results = struct();

% Normalize features (z-score normalization) if not already normalized
if options.featuresAlreadyNormalized
    fprintf('Features are already normalized, skipping normalization step.\n');
    featuresNorm = features;
    % Use identity normalization (mean=0, std=1) since features are already normalized
    results.normalizationMean = zeros(1, size(features, 2));
    results.normalizationStd = ones(1, size(features, 2));
else
    % Compute normalization statistics from training data
    trainMean = mean(features, 1);
    trainStd = std(features, [], 1);
    featuresNorm = (features - trainMean) ./ (trainStd + eps);
    results.normalizationMean = trainMean;
    results.normalizationStd = trainStd;
end

% Apply PCA if requested
if options.usePCA
    fprintf('Applying PCA (%.0f%% variance)...\n', options.pcaVariance * 100);
    [coeff, score, ~, ~, explained] = pca(featuresNorm);
    cumsumExplained = cumsum(explained);
    numComponents = find(cumsumExplained >= options.pcaVariance * 100, 1);
    featuresNorm = score(:, 1:numComponents);
    fprintf('Reduced to %d components\n', numComponents);
end

% Convert labels to categorical if needed
if ~iscategorical(labels)
    labels = categorical(labels);
end

% Train classifier based on type
switch lower(classifierType)
    case 'svm'
        [classifier, classifierResults] = trainSVM(featuresNorm, labels, options);
        
    case 'ensemble'
        [classifier, classifierResults] = trainEnsemble(featuresNorm, labels, options);
        
    case 'xgboost'
        [classifier, classifierResults] = trainXGBoost(featuresNorm, labels, options);
        
    case 'neural'
        [classifier, classifierResults] = trainNeuralNetwork(featuresNorm, labels, options);
        
    otherwise
        error('Unknown classifier type: %s', classifierType);
end

% Perform custom Cross-Validation if augmentation is enabled
if options.augmentation.enabled && options.optimizeHyperparams
    fprintf('Performing custom Cross-Validation with internal augmentation...\n');
    cvAccuracies = zeros(options.cvFolds, 1);
    
    % Create folds
    cvPart = cvpartition(labels, 'KFold', options.cvFolds);
    
    for k = 1:options.cvFolds
        % Get fold indices
        trainIdx = cvPart.training(k);
        valIdx = cvPart.test(k);
        
        % Split data
        X_train = featuresNorm(trainIdx, :);
        y_train = labels(trainIdx);
        X_val = featuresNorm(valIdx, :);
        y_val = labels(valIdx);
        
        % Apply Augmentation to Training Fold ONLY
        augOpts = options.augmentation;
        if isfield(augOpts, 'method') && strcmpi(augOpts.method, 'smote')
             % Use applySMOTEAdvanced (assuming it's in path)
             [X_train_aug, y_train_aug] = applySMOTEAdvanced(X_train, y_train, augOpts);
        else
             X_train_aug = X_train;
             y_train_aug = y_train;
        end
        
        % Train on augmented fold (disable internal CV/optimization for speed in loop)
        foldOptions = options;
        foldOptions.optimizeHyperparams = false; % Don't nest optimization
        foldOptions.augmentation.enabled = false; % Don't recurse
        
        switch lower(classifierType)
            case 'svm'
                [foldClassifier, ~] = trainSVM(X_train_aug, y_train_aug, foldOptions);
            case 'ensemble'
                [foldClassifier, ~] = trainEnsemble(X_train_aug, y_train_aug, foldOptions);
            case 'xgboost'
                [foldClassifier, ~] = trainXGBoost(X_train_aug, y_train_aug, foldOptions);
            case 'neural'
                [foldClassifier, ~] = trainNeuralNetwork(X_train_aug, y_train_aug, foldOptions);
        end
        
        % Evaluate on validation fold
        valPred = predict(foldClassifier.classifier, X_val); % Note: accessing internal classifier object
        cvAccuracies(k) = sum(valPred == y_val) / length(y_val);
    end
    
    classifierResults.cvAccuracy = mean(cvAccuracies);
    fprintf('Custom CV Accuracy with Augmentation: %.2f%%\n', classifierResults.cvAccuracy * 100);
    
    % Re-train final model on FULL dataset (augmented)
    fprintf('Retraining final model on full augmented dataset...\n');
    if isfield(options.augmentation, 'method') && strcmpi(options.augmentation.method, 'smote')
        [featuresAug, labelsAug] = applySMOTEAdvanced(featuresNorm, labels, options.augmentation);
    else
        featuresAug = featuresNorm;
        labelsAug = labels;
    end
    
    % Train final model
    finalOptions = options;
    finalOptions.optimizeHyperparams = false; % Use default/fixed for final to save time or implement separate search
    finalOptions.augmentation.enabled = false;
    
    switch lower(classifierType)
        case 'svm'
            [classifier, ~] = trainSVM(featuresAug, labelsAug, finalOptions);
        case 'ensemble'
            [classifier, ~] = trainEnsemble(featuresAug, labelsAug, finalOptions);
        case 'xgboost'
            [classifier, ~] = trainXGBoost(featuresAug, labelsAug, finalOptions);
        case 'neural'
            [classifier, ~] = trainNeuralNetwork(featuresAug, labelsAug, finalOptions);
    end
    
    % Update results with custom CV accuracy
    classifierResults.cvAccuracy = mean(cvAccuracies);
end

% Preserve normalization stats and merge with classifier results
normMean = results.normalizationMean;
normStd = results.normalizationStd;
results = classifierResults;
results.normalizationMean = normMean;
results.normalizationStd = normStd;
results.classifierType = classifierType;
results.featureDim = size(featuresNorm, 2);
end

function [classifier, results] = trainSVM(features, labels, options)
% Train SVM with hyperparameter optimization

fprintf('Training SVM classifier...\n');

% Hyperparameter search space
if options.optimizeHyperparams
    fprintf('Optimizing hyperparameters...\n');
    
    % Try different kernel types
    kernels = {'linear', 'polynomial', 'rbf', 'gaussian'};
    bestAccuracy = 0;
    bestClassifier = [];
    bestKernel = '';
    
    for kernelIdx = 1:length(kernels)
        kernel = kernels{kernelIdx};
        fprintf('  Testing %s kernel...\n', kernel);
        
        try
            switch kernel
                case 'linear'
                    t = templateSVM('KernelFunction', 'linear', 'Standardize', false);
                    
                case 'polynomial'
                    % Try different polynomial degrees
                    for degree = 2:3
                        t = templateSVM('KernelFunction', 'polynomial', ...
                                       'PolynomialOrder', degree, 'Standardize', false);
                        cvPartition = cvpartition(labels, 'KFold', options.cvFolds);
                        cvModel = fitcecoc(features, labels, 'Learners', t, ...
                                          'CVPartition', cvPartition);
                        cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'classiferror');
                        
                        if cvAccuracy > bestAccuracy
                            bestAccuracy = cvAccuracy;
                            bestKernel = sprintf('polynomial_%d', degree);
                            bestClassifier = fitcecoc(features, labels, 'Learners', t);
                        end
                    end
                    continue;
                    
                case 'rbf'
                    % Try different scale parameters
                    scales = [0.1, 1, 10, 100];
                    for scale = scales
                        t = templateSVM('KernelFunction', 'rbf', ...
                                       'KernelScale', scale, 'Standardize', false);
                        cvPartition = cvpartition(labels, 'KFold', options.cvFolds);
                        cvModel = fitcecoc(features, labels, 'Learners', t, ...
                                          'CVPartition', cvPartition);
                        cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'classiferror');
                        
                        if cvAccuracy > bestAccuracy
                            bestAccuracy = cvAccuracy;
                            bestKernel = sprintf('rbf_scale_%.1f', scale);
                            bestClassifier = fitcecoc(features, labels, 'Learners', t);
                        end
                    end
                    continue;
                    
                case 'gaussian'
                    t = templateSVM('KernelFunction', 'gaussian', 'Standardize', false);
            end
            
            % Cross-validation
            cvPartition = cvpartition(labels, 'KFold', options.cvFolds);
            cvModel = fitcecoc(features, labels, 'Learners', t, ...
                              'CVPartition', cvPartition);
            cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'classiferror');
            
            if cvAccuracy > bestAccuracy
                bestAccuracy = cvAccuracy;
                bestKernel = kernel;
                bestClassifier = fitcecoc(features, labels, 'Learners', t);
            end
            
        catch ME
            warning('Error with %s kernel: %s', kernel, ME.message);
        end
    end
    
    classifier = bestClassifier;
    results.cvAccuracy = bestAccuracy;
    results.bestKernel = bestKernel;
    results.classifier = classifier;
    fprintf('Best kernel: %s (CV Accuracy: %.2f%%)\n', bestKernel, bestAccuracy * 100);
    
else
    % Use cubic SVM (best from original work)
    t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 3, ...
                   'Standardize', false);
    classifier = fitcecoc(features, labels, 'Learners', t);
    
    % Cross-validation
    cvPartition = cvpartition(labels, 'KFold', options.cvFolds);
    cvModel = fitcecoc(features, labels, 'Learners', t, ...
                      'CVPartition', cvPartition);
    results.cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'classiferror');
    results.bestKernel = 'polynomial_3';
end

results.classifier = classifier;
end

% Helper to make predict work uniformly
function pred = predict(model, X)
    if isa(model, 'ClassificationECOC') || isa(model, 'classreg.learning.classif.CompactClassificationECOC')
        pred = predict(model, X);
    elseif isa(model, 'classreg.learning.classif.CompactClassificationEnsemble') || isa(model, 'classreg.learning.classif.ClassificationEnsemble')
        pred = predict(model, X);
    elseif isstruct(model) && isfield(model, 'net')
        % Neural net
        outputs = model.net(X');
        [~, idx] = max(outputs);
        pred = model.uniqueLabels(idx);
    else
        try
             pred = predict(model, X);
        catch
             error('Unknown model type for prediction');
        end
    end
end

function [classifier, results] = trainEnsemble(features, labels, options)
% Train ensemble classifier

fprintf('Training ensemble classifier...\n');

% Use bagged trees with different learners
t1 = templateTree('MaxNumSplits', 100, 'Surrogate', 'on');
t2 = templateDiscriminant('DiscrimType', 'pseudolinear');

% Note: 'Bag' method doesn't support LearnRate parameter
% LearnRate is only for boosting methods (AdaBoost, GentleBoost, etc.)
classifier = fitcensemble(features, labels, 'Method', 'Bag', ...
                         'Learners', {t1, t2}, ...
                         'NumLearningCycles', 100);

% Cross-validation
% Note: 'Verbose' is not a valid parameter for fitcensemble with KFold
cvPartition = cvpartition(labels, 'KFold', options.cvFolds);
cvModel = fitcensemble(features, labels, 'Method', 'Bag', ...
                      'Learners', {t1, t2}, ...
                      'NumLearningCycles', 100, ...
                      'CVPartition', cvPartition);
results.cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'classiferror');
results.classifier = classifier;
end

function [classifier, results] = trainXGBoost(features, labels, options)
% Train XGBoost (if available) or gradient boosting

fprintf('Training gradient boosting classifier...\n');

% Use MATLAB's gradient boosting
t = templateTree('MaxNumSplits', 50);
classifier = fitcensemble(features, labels, 'Method', 'GentleBoost', ...
                         'Learners', t, 'NumLearningCycles', 200, ...
                         'LearnRate', 0.1);

cvPartition = cvpartition(labels, 'KFold', options.cvFolds);
cvModel = fitcensemble(features, labels, 'Method', 'GentleBoost', ...
                      'Learners', t, 'NumLearningCycles', 200, ...
                      'LearnRate', 0.1, ...
                      'CVPartition', cvPartition);
results.cvAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'classiferror');
results.classifier = classifier;
end

function [classifier, results] = trainNeuralNetwork(features, labels, options)
% Train shallow neural network

fprintf('Training neural network classifier...\n');

% Convert labels to numeric
uniqueLabels = categories(labels);
numClasses = length(uniqueLabels);
labelNumeric = zeros(length(labels), 1);
for i = 1:numClasses
    labelNumeric(labels == uniqueLabels{i}) = i;
end

% Create neural network
hiddenLayerSize = min(100, round(size(features, 2) / 2));
net = patternnet(hiddenLayerSize);
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = true;
net.trainParam.epochs = 100;
net.trainParam.max_fail = 10;

% Train network
[net, tr] = train(net, features', dummyvar(labelNumeric)');

classifier.net = net;
classifier.uniqueLabels = uniqueLabels;

% Cross-validation accuracy (simplified)
results.cvAccuracy = 1 - tr.best_perf;
results.classifier = classifier;
end

