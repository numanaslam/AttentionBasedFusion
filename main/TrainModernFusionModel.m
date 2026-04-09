% TrainModernFusionModel.m
% Modernized training pipeline for GI tract image classification
% Combines state-of-the-art CNNs, handcrafted features, and modern ML techniques
%
% Author: Updated for 2024-2025
% Date: 2024

clc; clear; close all;

%% Configuration
config = struct();
config.datasetPath = 'kvasir-dataset';  % Update with your path
config.inputSize = [224 224 3];
config.trainRatio = 0.8;
config.testRatio = 0.2;
% GPU configuration
config.useGPU = canUseGPU;
if config.useGPU
    try
        gpuInfo = gpuDevice;
        fprintf('GPU detected: %s (%.2f GB memory)\n', gpuInfo.Name, gpuInfo.AvailableMemory / 1e9);
    catch
        fprintf('GPU check failed, using CPU\n');
        config.useGPU = false;
    end
else
    fprintf('GPU not available, using CPU\n');
end

% CNN models to use
config.cnnModels = {'resnet50', 'densenet201'};  % Can add more: 'resnet101', 'mobilenetv2'

% Feature fusion method
% 'attention': Learnable attention weights based on feature importance and 
%              cross-modal correlation (justifies "Attention-Based" in title)
% 'multimodal': Cross-attention mechanism with softmax (also attention-based)
% 'concat': Simple concatenation (baseline)
% 'weighted': Variance-based weighted fusion
% 'bilinear': Bilinear pooling (richer representations)
config.fusionMethod = 'attention';  % Recommended: 'attention' or 'multimodal' for attention-based fusion

% Image-level augmentation (applied before feature extraction)
config.imageAugmentation = struct();
config.imageAugmentation.enabled = false;  % Set to true to enable (recommended for accuracy)
config.imageAugmentation.rotation = [-15, 15];  % Rotation range in degrees
config.imageAugmentation.translation = [-10, 10];  % Translation range in pixels
config.imageAugmentation.scale = [0.9, 1.1];  % Scale range
config.imageAugmentation.flip = 'horizontal';  % 'horizontal', 'vertical', 'both', 'none'
config.imageAugmentation.brightness = [0.8, 1.2];  % Brightness range
config.imageAugmentation.contrast = [0.8, 1.2];  % Contrast range

% Feature-level augmentation (SMOTE)
config.augmentation = struct();
config.augmentation.enabled = true;
config.augmentation.method = 'smote';  % 'smote', 'borderline', 'adasyn'
config.augmentation.ratio = 1.0;  % Oversampling ratio
config.augmentation.k = 5;  % Number of neighbors for SMOTE

% Classification options
config.classifier = struct();
config.classifier.type = 'svm';  % 'svm', 'ensemble', 'xgboost', 'neural', 'ensemble_multi'
config.classifier.optimizeHyperparams = true;
config.classifier.cvFolds = 5;
config.classifier.usePCA = false;  % Set to true for dimensionality reduction (recommended)
config.classifier.pcaVariance = 0.98;  % Keep 98% variance (increased from 0.95)

% Ensemble classifier options (when type = 'ensemble_multi')
config.classifier.ensembleClassifiers = {'svm', 'ensemble', 'neural'};  % Classifiers to combine
config.classifier.ensembleVoting = 'weighted';  % 'majority', 'weighted', 'stacking'

% Test-time augmentation options
config.testTimeAugmentation = struct();
config.testTimeAugmentation.enabled = false;  % Set to true to enable (recommended for accuracy)
config.testTimeAugmentation.numAugmentations = 5;  % Number of augmentations per test sample
config.testTimeAugmentation.augmentationTypes = {'gaussian_noise', 'dropout', 'scale', 'shift'};

% Save options
config.saveFeatures = true;  % Save extracted features to avoid recalculation
config.saveModel = true;
config.outputDir = 'results';
config.forceRecalculate = true;  % Set to true to force recalculation (use if normalization issues occur)
% IMPORTANT: Set to true if you see normalization warnings or low test accuracy
% This will regenerate all features from scratch with proper normalization
% Note: Features are automatically loaded if they exist, saving significant time
% on subsequent runs. Set forceRecalculate=true to recompute features.
% IMPORTANT: If you see warnings about pre-normalized features AND test accuracy is low, set this to true!

% Create output directory if it doesn't exist
if ~isfolder(config.outputDir)
    mkdir(config.outputDir);
end

fprintf('========================================\n');
fprintf('Modern GI Tract Classification Pipeline\n');
fprintf('========================================\n\n');

%% Step 1: Load and Prepare Dataset
fprintf('Step 1: Loading dataset...\n');
if ~isfolder(config.datasetPath)
    error('Dataset path not found: %s', config.datasetPath);
end

imds = imageDatastore(config.datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('Total images: %d\n', numel(imds.Files));
fprintf('Classes: %s\n', strjoin(categories(imds.Labels), ', '));

% Shuffle dataset
imds = shuffle(imds);

% Split into train and test
[imdsTrain, imdsTest] = splitEachLabel(imds, config.trainRatio, 'randomized');
fprintf('Training images: %d\n', numel(imdsTrain.Files));
fprintf('Test images: %d\n', numel(imdsTest.Files));

%% Step 2: Extract Handcrafted Features
fprintf('\nStep 2: Extracting handcrafted features...\n');
handcraftedFile = fullfile(config.outputDir, 'handcrafted_features.mat');

if exist(handcraftedFile, 'file') && config.saveFeatures && ~config.forceRecalculate
    fprintf('Loading existing handcrafted features from %s...\n', handcraftedFile);
    load(handcraftedFile, 'hcFeaturesTrain', 'hcFeaturesTest');
    fprintf('Handcrafted features loaded successfully.\n');
    handcraftedTime = 0;  % No time spent if loaded
else
    fprintf('Extracting handcrafted features (this may take a while)...\n');
    fprintf('Note: Handcrafted features use CPU (GPU not applicable for texture analysis)\n');
    tic;
    hcFeaturesTrain = extractHandcraftedFeaturesModern(imdsTrain);
    hcFeaturesTest = extractHandcraftedFeaturesModern(imdsTest);
    handcraftedTime = toc;
    fprintf('Handcrafted features extracted in %.2f seconds\n', handcraftedTime);
    
    if config.saveFeatures
        if ~isfolder(config.outputDir)
            mkdir(config.outputDir);
        end
        save(handcraftedFile, 'hcFeaturesTrain', 'hcFeaturesTest', '-v7.3');
        fprintf('Handcrafted features saved to %s\n', handcraftedFile);
    end
end
fprintf('Feature dimension: %d\n', size(hcFeaturesTrain, 2));

%% Step 3: Extract CNN Features
fprintf('\nStep 3: Extracting CNN features...\n');
cnnFeaturesTrain = cell(length(config.cnnModels), 1);
cnnFeaturesTest = cell(length(config.cnnModels), 1);
modelInfo = cell(length(config.cnnModels), 1);

for modelIdx = 1:length(config.cnnModels)
    modelType = config.cnnModels{modelIdx};
    cnnFeatureFile = fullfile(config.outputDir, sprintf('%s_features.mat', modelType));
    
    if exist(cnnFeatureFile, 'file') && config.saveFeatures && ~config.forceRecalculate
        fprintf('\nLoading existing %s features from %s...\n', upper(modelType), cnnFeatureFile);
        load(cnnFeatureFile, 'featTrain', 'featTest', 'infoTrain');
        
        % Check if features appear to be normalized (sanity check)
        trainStd = std(featTrain(:));
        testStd = std(featTest(:));
        if abs(trainStd - 1.0) < 0.1 || abs(testStd - 1.0) < 0.1 || abs(testStd - 0.5) < 0.1
            fprintf('  WARNING: %s features may be pre-normalized (train std: %.4f, test std: %.4f)\n', ...
                    upper(modelType), trainStd, testStd);
            fprintf('  Consider setting config.forceRecalculate = true to regenerate.\n');
        end
        
        % Check for distribution mismatch
        trainMean = mean(featTrain(:));
        testMean = mean(featTest(:));
        if abs(trainMean - testMean) > 0.5 || abs(trainStd - testStd) > 0.5
            fprintf('  WARNING: Train and test %s features have different distributions!\n', upper(modelType));
            fprintf('  Train: mean=%.4f, std=%.4f; Test: mean=%.4f, std=%.4f\n', ...
                    trainMean, trainStd, testMean, testStd);
        end
        
        cnnFeaturesTrain{modelIdx} = featTrain;
        cnnFeaturesTest{modelIdx} = featTest;
        modelInfo{modelIdx} = infoTrain;
        fprintf('%s features loaded successfully. Dimension: %d\n', upper(modelType), size(featTrain, 2));
    else
        fprintf('\nExtracting features using %s (this may take a while)...\n', upper(modelType));
        
        % Apply image augmentation if enabled
        if config.imageAugmentation.enabled
            fprintf('Applying image-level augmentation...\n');
            augOptions = struct();
            augOptions.enabled = true;
            augOptions.rotation = config.imageAugmentation.rotation;
            augOptions.translation = config.imageAugmentation.translation;
            augOptions.scale = config.imageAugmentation.scale;
            augOptions.flip = config.imageAugmentation.flip;
            augOptions.brightness = config.imageAugmentation.brightness;
            augOptions.contrast = config.imageAugmentation.contrast;
            augOptions.inputSize = config.inputSize;
            
            imdsTrainAug = applyImageAugmentation(imdsTrain, augOptions);
            % Test set: no augmentation during feature extraction (TTA applied later if enabled)
            augOptionsTest = augOptions;
            augOptionsTest.enabled = false;
            imdsTestAug = applyImageAugmentation(imdsTest, augOptionsTest);
        else
            imdsTrainAug = imdsTrain;
            imdsTestAug = imdsTest;
        end
        
        [featTrain, infoTrain] = extractModernCNNFeatures(imdsTrainAug, modelType);
        [featTest, infoTest] = extractModernCNNFeatures(imdsTestAug, modelType);
        
        cnnFeaturesTrain{modelIdx} = featTrain;
        cnnFeaturesTest{modelIdx} = featTest;
        modelInfo{modelIdx} = infoTrain;
        
        fprintf('%s features: %d dimensions\n', upper(modelType), size(featTrain, 2));
        
        if config.saveFeatures
            save(cnnFeatureFile, 'featTrain', 'featTest', 'infoTrain', '-v7.3');
            fprintf('%s features saved to %s\n', upper(modelType), cnnFeatureFile);
        end
    end
end

%% Step 4: Fuse Features
fprintf('\nStep 4: Fusing features using %s method...\n', config.fusionMethod);
fusedFeatureFile = fullfile(config.outputDir, sprintf('fused_features_%s.mat', config.fusionMethod));

if exist(fusedFeatureFile, 'file') && config.saveFeatures && ~config.forceRecalculate
    fprintf('Loading existing fused features from %s...\n', fusedFeatureFile);
    % Load all available variables
    fusionData = load(fusedFeatureFile);
    fusedTrain = fusionData.fusedTrain;
    fusedTest = fusionData.fusedTest;
    
    % Verify features are not already normalized (check if they look normalized)
    % If mean is near 0 and std is near 1, they might be pre-normalized
    trainCheckMean = mean(fusedTrain(:));
    trainCheckStd = std(fusedTrain(:));
    testCheckMean = mean(fusedTest(:));
    testCheckStd = std(fusedTest(:));
    
    if abs(trainCheckMean) < 0.1 && abs(trainCheckStd - 1.0) < 0.1
        fprintf('Warning: Loaded training features appear to be pre-normalized.\n');
        fprintf('  Consider setting config.forceRecalculate = true to regenerate features.\n');
    end
    
    if abs(testCheckStd - 0.5) < 0.1 || (abs(testCheckMean) < 0.1 && abs(testCheckStd - 1.0) < 0.1)
        fprintf('Warning: Loaded test features appear to be pre-normalized (std=%.4f).\n', testCheckStd);
        fprintf('  This will cause normalization issues. Set config.forceRecalculate = true.\n');
    end
    
    % Check for distribution mismatch
    if abs(trainCheckMean - testCheckMean) > 0.5 || abs(trainCheckStd - testCheckStd) > 0.5
        fprintf('Warning: Train and test fused features have different distributions!\n');
        fprintf('  Train: mean=%.4f, std=%.4f; Test: mean=%.4f, std=%.4f\n', ...
                trainCheckMean, trainCheckStd, testCheckMean, testCheckStd);
        fprintf('  This may cause poor test performance. Consider forcing recalculation.\n');
    end
    
    % Try to load fusion info if available
    if isfield(fusionData, 'fusionInfoTrain')
        fusionInfoTrain = fusionData.fusionInfoTrain;
    else
        fusionInfoTrain = [];
    end
    if isfield(fusionData, 'fusionInfoTest')
        fusionInfoTest = fusionData.fusionInfoTest;
    else
        fusionInfoTest = [];
    end
    fprintf('Fused features loaded successfully.\n');
else
    fprintf('Fusing features (this may take a while)...\n');
    if config.useGPU
        fprintf('Using GPU for feature fusion...\n');
    end
    [fusedTrain, fusionInfoTrain] = fuseFeaturesModern(cnnFeaturesTrain, hcFeaturesTrain, config.fusionMethod, config.useGPU);
    [fusedTest, fusionInfoTest] = fuseFeaturesModern(cnnFeaturesTest, hcFeaturesTest, config.fusionMethod, config.useGPU);
    
    if config.saveFeatures
        save(fusedFeatureFile, 'fusedTrain', 'fusedTest', 'fusionInfoTrain', 'fusionInfoTest', '-v7.3');
        fprintf('Fused features saved to %s\n', fusedFeatureFile);
    end
end
fprintf('Fused feature dimension: %d\n', size(fusedTrain, 2));

%% Step 5: Normalize Features BEFORE Augmentation
% Normalize original training features first (before augmentation)
% This ensures SMOTE works on properly scaled features
fprintf('\nStep 5: Normalizing features before augmentation...\n');
if config.useGPU
    fprintf('Using GPU for normalization operations...\n');
    % Move to GPU for faster computation
    fusedTrainGPU = gpuArray(fusedTrain);
    fusedTestGPU = gpuArray(fusedTest);
    
    % Compute statistics on GPU
    trainMean = gather(mean(fusedTrainGPU, 1));
    trainStd = gather(std(fusedTrainGPU, [], 1));
    
    % Handle near-zero variance features (set minimum std threshold)
    minStd = 1e-6;  % Minimum standard deviation threshold
    trainStd(trainStd < minStd) = 1.0;  % Set to 1.0 for constant features
    
    % Normalize on GPU
    fusedTrainNorm = gather((fusedTrainGPU - trainMean) ./ (trainStd + eps));
    fusedTestNorm = gather((fusedTestGPU - trainMean) ./ (trainStd + eps));
    
    % Clip extreme values (outliers beyond 5 standard deviations)
    fusedTrainNorm = gather(max(min(fusedTrainNorm, 5), -5));
    fusedTestNorm = gather(max(min(fusedTestNorm, 5), -5));
    
    % Clear GPU arrays
    clear fusedTrainGPU fusedTestGPU;
else
    % CPU version
    trainMean = mean(fusedTrain, 1);
    trainStd = std(fusedTrain, [], 1);
    
    % Handle near-zero variance features (set minimum std threshold)
    minStd = 1e-6;  % Minimum standard deviation threshold
    trainStd(trainStd < minStd) = 1.0;  % Set to 1.0 for constant features
    
    % Normalize
    fusedTrainNorm = (fusedTrain - trainMean) ./ (trainStd + eps);
    fusedTestNorm = (fusedTest - trainMean) ./ (trainStd + eps);
    
    % Clip extreme values (outliers beyond 5 standard deviations)
    fusedTrainNorm = max(min(fusedTrainNorm, 5), -5);
    fusedTestNorm = max(min(fusedTestNorm, 5), -5);
end

% Diagnostic: Check for problematic features
numLowVarFeatures = sum(trainStd < 1e-3);
if numLowVarFeatures > 0
    fprintf('Warning: %d features have very low variance (< 0.001). These may cause issues.\n', numLowVarFeatures);
end

%% Step 6: Data Augmentation
if config.augmentation.enabled
    fprintf('\nStep 6: Applying %s augmentation...\n', upper(config.augmentation.method));
    tic;
    
    augOptions = struct();
    augOptions.k = config.augmentation.k;
    augOptions.ratio = config.augmentation.ratio;
    augOptions.method = config.augmentation.method;
    augOptions.applyPCA = false;
    
    % Apply SMOTE on normalized features
    [fusedTrainAug, labelsTrainAug] = applySMOTEAdvanced(...
        fusedTrainNorm, imdsTrain.Labels, augOptions);
    
    augmentationTime = toc;
    fprintf('Augmentation completed in %.2f seconds\n', augmentationTime);
    fprintf('Original samples: %d, Augmented samples: %d\n', ...
            size(fusedTrainNorm, 1), size(fusedTrainAug, 1));
    
    fusedTrain = fusedTrainAug;
    labelsTrain = labelsTrainAug;
else
    fusedTrain = fusedTrainNorm;
    labelsTrain = imdsTrain.Labels;
end

%% Step 7: Train Classifier
% Note: Features are already normalized, so trainModernClassifier should not normalize again
fprintf('\nStep 7: Training %s classifier...\n', upper(config.classifier.type));
fprintf('Note: Features are already normalized, classifier will use them as-is.\n');
tic;

classifierOptions = config.classifier;
% Pass already-normalized features to classifier
% Tell classifier that features are already normalized to avoid double normalization
classifierOptions.featuresAlreadyNormalized = true;

% Check if ensemble classifier is requested
if strcmpi(config.classifier.type, 'ensemble_multi')
    % Use ensemble of multiple classifiers
    classifierOptions.classifiers = config.classifier.ensembleClassifiers;
    classifierOptions.votingMethod = config.classifier.ensembleVoting;
    [classifier, trainResults] = trainEnsembleClassifier(...
        fusedTrain, labelsTrain, classifierOptions);
else
    % Use single classifier
    [classifier, trainResults] = trainModernClassifier(...
        fusedTrain, labelsTrain, config.classifier.type, classifierOptions);
end

trainingTime = toc;
fprintf('Training completed in %.2f seconds\n', trainingTime);
fprintf('Cross-validation accuracy: %.2f%%\n', trainResults.cvAccuracy * 100);

if config.saveModel
    save(fullfile(config.outputDir, 'trained_classifier.mat'), ...
         'classifier', 'trainResults', 'config', '-v7.3');
end

%% Step 8: Evaluate on Test Set
fprintf('\nStep 8: Evaluating on test set...\n');
% Test features are already normalized in Step 5 using training statistics

% Diagnostic checks
fprintf('Feature statistics check:\n');
fprintf('  Training features (original, before norm) - Mean: %.4f, Std: %.4f\n', ...
    mean(fusedTrain(:)), std(fusedTrain(:)));
fprintf('  Training features (original, before norm) - Min: %.4f, Max: %.4f\n', ...
    min(fusedTrain(:)), max(fusedTrain(:)));
fprintf('  Test features (before norm) - Mean: %.4f, Std: %.4f\n', ...
    mean(fusedTest(:)), std(fusedTest(:)));
fprintf('  Test features (before norm) - Min: %.4f, Max: %.4f\n', ...
    min(fusedTest(:)), max(fusedTest(:)));

% Check if test features appear pre-normalized
% Only warn if there's a significant mismatch AND test accuracy is poor
testStdBeforeNorm = std(fusedTest(:));
trainStdBeforeNorm = std(fusedTrain(:));
stdMismatch = abs(trainStdBeforeNorm - testStdBeforeNorm);

% More lenient check: only warn if std is very close to 0.5 AND there's a mismatch
if (abs(testStdBeforeNorm - 0.5) < 0.005) && (stdMismatch > 0.3)
    fprintf('\n  ⚠️  WARNING: Test features may have normalization mismatch (test std: %.4f, train std: %.4f)\n', ...
            testStdBeforeNorm, trainStdBeforeNorm);
    fprintf('  ⚠️  If test accuracy is low (< 50%%), consider setting config.forceRecalculate = true.\n');
    fprintf('  ⚠️  Current setting: config.forceRecalculate = %d\n', config.forceRecalculate);
elseif abs(testStdBeforeNorm - 0.5) < 0.01
    % Just an informational note if std is close to 0.5 but no mismatch
    fprintf('\n  ℹ️  Note: Test features have std ≈ %.4f (close to 0.5, but this may be normal)\n', testStdBeforeNorm);
end

fprintf('  Test features (after norm) - Mean: %.4f, Std: %.4f\n', ...
    mean(fusedTestNorm(:)), std(fusedTestNorm(:)));
fprintf('  Test features (after norm) - Min: %.4f, Max: %.4f\n', ...
    min(fusedTestNorm(:)), max(fusedTestNorm(:)));
fprintf('  Test features - NaN count: %d, Inf count: %d\n', ...
    sum(isnan(fusedTestNorm(:))), sum(isinf(fusedTestNorm(:))));

% Additional check: Verify normalization consistency
% Use original normalized training features (before augmentation) for comparison
trainNormMean = mean(fusedTrainNorm(:));
trainNormStd = std(fusedTrainNorm(:));
testNormMean = mean(fusedTestNorm(:));
testNormStd = std(fusedTestNorm(:));

fprintf('  Normalization consistency check:\n');
fprintf('    Train (norm, before aug) - Mean: %.4f, Std: %.4f\n', trainNormMean, trainNormStd);
fprintf('    Test (norm) - Mean: %.4f, Std: %.4f\n', testNormMean, testNormStd);

if abs(testNormMean) > 0.1 || abs(testNormStd - 1.0) > 0.1
    fprintf('  WARNING: Test features may not be properly normalized!\n');
    fprintf('  Expected: Mean ≈ 0, Std ≈ 1\n');
end

% Verify label alignment
fprintf('\nLabel alignment check:\n');
fprintf('  Training labels: %s\n', strjoin(categories(labelsTrain), ', '));
fprintf('  Test labels: %s\n', strjoin(categories(imdsTest.Labels), ', '));

% Check if classifier classes match test labels
if isstruct(classifier) && isfield(classifier, 'classifiers')
    % Ensemble - check first classifier
    classifierClasses = classifier.classNames;
else
    % Single classifier
    classifierClasses = classifier.ClassNames;
end

fprintf('  Classifier classes: %s\n', strjoin(cellstr(classifierClasses), ', '));

% Check for class mismatch
trainClasses = categories(labelsTrain);
testClasses = categories(imdsTest.Labels);
if ~isequal(sort(trainClasses), sort(testClasses))
    fprintf('  WARNING: Training and test classes do not match!\n');
end
if ~isequal(sort(cellstr(classifierClasses)), sort(testClasses))
    fprintf('  WARNING: Classifier classes do not match test classes!\n');
end

% Make predictions and extract scores for ROC curves
fprintf('\nMaking predictions...\n');
if config.testTimeAugmentation.enabled
    fprintf('Applying test-time augmentation...\n');
    ttaOptions = struct();
    ttaOptions.numAugmentations = config.testTimeAugmentation.numAugmentations;
    ttaOptions.augmentationTypes = config.testTimeAugmentation.augmentationTypes;
    predictions = applyTestTimeAugmentation(classifier, fusedTestNorm, [], ttaOptions);
    % Note: TTA doesn't return scores, so we'll extract them separately
    testScores = [];
else
    % Standard prediction
    if isstruct(classifier) && isfield(classifier, 'classifiers')
        % Ensemble classifier
        predictions = predictEnsemble(classifier, fusedTestNorm);
        % Extract scores using the same method as predictions
        if exist('getEnsembleScores', 'file')
            testScores = getEnsembleScores(classifier, fusedTestNorm);
        else
            % Fallback: extract scores manually
            [~, testScores] = predict(classifier.classifiers{1}, fusedTestNorm);
        end
    else
        % Single classifier
        [predictions, testScores] = predict(classifier, fusedTestNorm);
    end
end
testLabels = imdsTest.Labels;

% Check prediction distribution
fprintf('\nPrediction distribution:\n');
uniquePreds = categories(predictions);
for i = 1:length(uniquePreds)
    count = sum(predictions == uniquePreds{i});
    fprintf('  %s: %d (%.1f%%)\n', char(uniquePreds{i}), count, count/length(predictions)*100);
end

% Calculate metrics
testAccuracy = mean(predictions == testLabels);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);

% Confusion matrix
cm = confusionmat(testLabels, predictions);
fprintf('\nConfusion Matrix:\n');
disp(cm);

% Per-class accuracy
uniqueLabels = categories(testLabels);
numClasses = length(uniqueLabels);
perClassAcc = zeros(numClasses, 1);
for i = 1:numClasses
    currentClass = uniqueLabels{i};
    classMask = testLabels == currentClass;
    perClassAcc(i) = mean(predictions(classMask) == testLabels(classMask));
    fprintf('Class %s: %.2f%%\n', char(currentClass), perClassAcc(i) * 100);
end

% Calculate sensitivity and specificity (for binary, extend for multiclass)
if numClasses == 2
    TP = cm(2, 2);
    TN = cm(1, 1);
    FP = cm(1, 2);
    FN = cm(2, 1);
    
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    
    fprintf('\nSensitivity: %.2f%%\n', sensitivity * 100);
    fprintf('Specificity: %.2f%%\n', specificity * 100);
end

%% Step 8: Save Results
results = struct();
results.testAccuracy = testAccuracy;
results.cvAccuracy = trainResults.cvAccuracy;
results.confusionMatrix = cm;
results.perClassAccuracy = perClassAcc;
results.trainingTime = trainingTime;
results.handcraftedTime = handcraftedTime;
results.featureDim = size(fusedTrain, 2);  % Fused feature dimension
if config.augmentation.enabled
    results.augmentationTime = augmentationTime;
end
results.modelInfo = modelInfo;
results.config = config;

% Save test scores for ROC curve generation
if exist('testScores', 'var') && ~isempty(testScores)
    results.testScores = testScores;
    results.testLabels = testLabels;
    fprintf('Test scores saved for ROC curve generation.\n');
end

if config.saveModel
    save(fullfile(config.outputDir, 'results.mat'), 'results', '-v7.3');
    % Also save scores separately for easy access
    if exist('testScores', 'var') && ~isempty(testScores)
        save(fullfile(config.outputDir, 'test_scores.mat'), 'testScores', 'testLabels', '-v7.3');
    end
end

%% Summary
fprintf('\n========================================\n');
fprintf('Training Summary\n');
fprintf('========================================\n');
fprintf('CNN Models: %s\n', strjoin(config.cnnModels, ', '));
fprintf('Fusion Method: %s\n', config.fusionMethod);
fprintf('Augmentation: %s (ratio: %.1f)\n', config.augmentation.method, config.augmentation.ratio);
fprintf('Classifier: %s\n', config.classifier.type);
fprintf('Cross-validation Accuracy: %.2f%%\n', trainResults.cvAccuracy * 100);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);
fprintf('========================================\n');

fprintf('\nResults saved to: %s\n', config.outputDir);

