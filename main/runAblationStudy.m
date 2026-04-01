% runAblationStudy.m
% Comprehensive ablation study for publication
% Tests different components and configurations

clc; clear; close all;

%% Configuration
config = struct();
config.datasetPath = 'kvasir-dataset';
config.outputDir = 'ablation_results';
config.saveResults = true;

% Create output directory
if ~isfolder(config.outputDir)
    mkdir(config.outputDir);
end

fprintf('========================================\n');
fprintf('Ablation Study\n');
fprintf('========================================\n\n');

%% Load Dataset (once)
fprintf('Loading dataset...\n');
imds = imageDatastore(config.datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
imds = shuffle(imds);
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('Dataset loaded: %d training, %d test images\n', ...
        numel(imdsTrain.Files), numel(imdsTest.Files));

%% Load Pre-extracted Features (if available)
fprintf('\nLoading pre-extracted features...\n');
resultsDir = 'results';

% Handcrafted features
hcFile = fullfile(resultsDir, 'handcrafted_features.mat');
if exist(hcFile, 'file')
    load(hcFile, 'hcFeaturesTrain', 'hcFeaturesTest');
    
    % Check for normalization issues
    hcTrainStd = std(hcFeaturesTrain(:));
    hcTestStd = std(hcFeaturesTest(:));
    if abs(hcTrainStd - 1.0) < 0.1 || abs(hcTestStd - 0.5) < 0.1 || abs(hcTestStd - 1.0) < 0.1
        fprintf('Warning: Handcrafted features may be pre-normalized (train std: %.4f, test std: %.4f)\n', ...
                hcTrainStd, hcTestStd);
    end
    fprintf('Handcrafted features loaded.\n');
else
    fprintf('Extracting handcrafted features...\n');
    hcFeaturesTrain = extractHandcraftedFeaturesModern(imdsTrain);
    hcFeaturesTest = extractHandcraftedFeaturesModern(imdsTest);
end

% CNN features
resnetFile = fullfile(resultsDir, 'resnet50_features.mat');
densenetFile = fullfile(resultsDir, 'densenet201_features.mat');

if exist(resnetFile, 'file')
    load(resnetFile, 'featTrain', 'featTest');
    resnetTrain = featTrain;
    resnetTest = featTest;
    
    % Check for normalization issues
    resnetTrainStd = std(resnetTrain(:));
    resnetTestStd = std(resnetTest(:));
    if abs(resnetTrainStd - 1.0) < 0.1 || abs(resnetTestStd - 0.5) < 0.1 || abs(resnetTestStd - 1.0) < 0.1
        fprintf('Warning: ResNet-50 features may be pre-normalized (train std: %.4f, test std: %.4f)\n', ...
                resnetTrainStd, resnetTestStd);
    end
    fprintf('ResNet-50 features loaded.\n');
else
    fprintf('Extracting ResNet-50 features...\n');
    [resnetTrain, ~] = extractModernCNNFeatures(imdsTrain, 'resnet50');
    [resnetTest, ~] = extractModernCNNFeatures(imdsTest, 'resnet50');
end

if exist(densenetFile, 'file')
    load(densenetFile, 'featTrain', 'featTest');
    densenetTrain = featTrain;
    densenetTest = featTest;
    
    % Check for normalization issues
    densenetTrainStd = std(densenetTrain(:));
    densenetTestStd = std(densenetTest(:));
    if abs(densenetTrainStd - 1.0) < 0.1 || abs(densenetTestStd - 0.5) < 0.1 || abs(densenetTestStd - 1.0) < 0.1
        fprintf('Warning: DenseNet-201 features may be pre-normalized (train std: %.4f, test std: %.4f)\n', ...
                densenetTrainStd, densenetTestStd);
    end
    fprintf('DenseNet-201 features loaded.\n');
else
    fprintf('Extracting DenseNet-201 features...\n');
    [densenetTrain, ~] = extractModernCNNFeatures(imdsTrain, 'densenet201');
    [densenetTest, ~] = extractModernCNNFeatures(imdsTest, 'densenet201');
end

%% Initialize Results Structure
ablationResults = struct();
ablationResults.experiments = {};
ablationResults.accuracies = [];
ablationResults.cvAccuracies = [];
ablationResults.configs = {};

expIdx = 0;

%% Experiment 1: Baseline - CNN Only (ResNet-50)
fprintf('\n========================================\n');
fprintf('Experiment 1: CNN Only (ResNet-50)\n');
fprintf('========================================\n');
expIdx = expIdx + 1;
tic;
[classifier, trainResults] = trainSingleExperiment(...
    resnetTrain, imdsTrain.Labels, resnetTest, imdsTest.Labels, 'svm', false);
testAcc = evaluateClassifier(classifier, resnetTest, imdsTest.Labels, ...
                             trainResults.normalizationMean, trainResults.normalizationStd);
time = toc;

ablationResults.experiments{expIdx} = 'CNN Only (ResNet-50)';
ablationResults.accuracies(expIdx) = testAcc;
ablationResults.cvAccuracies(expIdx) = trainResults.cvAccuracy;
ablationResults.configs{expIdx} = struct('cnn', 'resnet50', 'fusion', 'none', 'augmentation', false);
fprintf('Test Accuracy: %.2f%%, CV Accuracy: %.2f%%, Time: %.2f sec\n', ...
        testAcc * 100, trainResults.cvAccuracy * 100, time);

%% Experiment 2: CNN Only (DenseNet-201)
fprintf('\n========================================\n');
fprintf('Experiment 2: CNN Only (DenseNet-201)\n');
fprintf('========================================\n');
expIdx = expIdx + 1;
tic;
[classifier, trainResults] = trainSingleExperiment(...
    densenetTrain, imdsTrain.Labels, densenetTest, imdsTest.Labels, 'svm', false);
testAcc = evaluateClassifier(classifier, densenetTest, imdsTest.Labels, ...
                             trainResults.normalizationMean, trainResults.normalizationStd);
time = toc;

ablationResults.experiments{expIdx} = 'CNN Only (DenseNet-201)';
ablationResults.accuracies(expIdx) = testAcc;
ablationResults.cvAccuracies(expIdx) = trainResults.cvAccuracy;
ablationResults.configs{expIdx} = struct('cnn', 'densenet201', 'fusion', 'none', 'augmentation', false);
fprintf('Test Accuracy: %.2f%%, CV Accuracy: %.2f%%, Time: %.2f sec\n', ...
        testAcc * 100, trainResults.cvAccuracy * 100, time);

%% Experiment 3: Handcrafted Features Only
fprintf('\n========================================\n');
fprintf('Experiment 3: Handcrafted Features Only\n');
fprintf('========================================\n');
expIdx = expIdx + 1;
tic;
[classifier, trainResults] = trainSingleExperiment(...
    hcFeaturesTrain, imdsTrain.Labels, hcFeaturesTest, imdsTest.Labels, 'svm', false);
testAcc = evaluateClassifier(classifier, hcFeaturesTest, imdsTest.Labels, ...
                             trainResults.normalizationMean, trainResults.normalizationStd);
time = toc;

ablationResults.experiments{expIdx} = 'Handcrafted Only';
ablationResults.accuracies(expIdx) = testAcc;
ablationResults.cvAccuracies(expIdx) = trainResults.cvAccuracy;
ablationResults.configs{expIdx} = struct('cnn', 'none', 'fusion', 'none', 'augmentation', false);
fprintf('Test Accuracy: %.2f%%, CV Accuracy: %.2f%%, Time: %.2f sec\n', ...
        testAcc * 100, trainResults.cvAccuracy * 100, time);

%% Experiment 4: Simple Concatenation
fprintf('\n========================================\n');
fprintf('Experiment 4: Simple Concatenation\n');
fprintf('========================================\n');
expIdx = expIdx + 1;
fusedTrain = fuseFeaturesModern({resnetTrain, densenetTrain}, hcFeaturesTrain, 'concat', false);
fusedTest = fuseFeaturesModern({resnetTest, densenetTest}, hcFeaturesTest, 'concat', false);
tic;
[classifier, trainResults] = trainSingleExperiment(...
    fusedTrain, imdsTrain.Labels, fusedTest, imdsTest.Labels, 'svm', false);
testAcc = evaluateClassifier(classifier, fusedTest, imdsTest.Labels, ...
                             trainResults.normalizationMean, trainResults.normalizationStd);
time = toc;

ablationResults.experiments{expIdx} = 'Concatenation';
ablationResults.accuracies(expIdx) = testAcc;
ablationResults.cvAccuracies(expIdx) = trainResults.cvAccuracy;
ablationResults.configs{expIdx} = struct('cnn', 'both', 'fusion', 'concat', 'augmentation', false);
fprintf('Test Accuracy: %.2f%%, CV Accuracy: %.2f%%, Time: %.2f sec\n', ...
        testAcc * 100, trainResults.cvAccuracy * 100, time);

%% Experiment 5: Weighted Fusion
fprintf('\n========================================\n');
fprintf('Experiment 5: Weighted Fusion\n');
fprintf('========================================\n');
expIdx = expIdx + 1;
fusedTrain = fuseFeaturesModern({resnetTrain, densenetTrain}, hcFeaturesTrain, 'weighted', false);
fusedTest = fuseFeaturesModern({resnetTest, densenetTest}, hcFeaturesTest, 'weighted', false);
tic;
[classifier, trainResults] = trainSingleExperiment(...
    fusedTrain, imdsTrain.Labels, fusedTest, imdsTest.Labels, 'svm', false);
testAcc = evaluateClassifier(classifier, fusedTest, imdsTest.Labels, ...
                             trainResults.normalizationMean, trainResults.normalizationStd);
time = toc;

ablationResults.experiments{expIdx} = 'Weighted Fusion';
ablationResults.accuracies(expIdx) = testAcc;
ablationResults.cvAccuracies(expIdx) = trainResults.cvAccuracy;
ablationResults.configs{expIdx} = struct('cnn', 'both', 'fusion', 'weighted', 'augmentation', false);
fprintf('Test Accuracy: %.2f%%, CV Accuracy: %.2f%%, Time: %.2f sec\n', ...
        testAcc * 100, trainResults.cvAccuracy * 100, time);

%% Experiment 6: Attention-Based Fusion (Proposed)
fprintf('\n========================================\n');
fprintf('Experiment 6: Attention-Based Fusion (Proposed)\n');
fprintf('========================================\n');
expIdx = expIdx + 1;
fusedTrain = fuseFeaturesModern({resnetTrain, densenetTrain}, hcFeaturesTrain, 'attention', false);
fusedTest = fuseFeaturesModern({resnetTest, densenetTest}, hcFeaturesTest, 'attention', false);
tic;
[classifier, trainResults] = trainSingleExperiment(...
    fusedTrain, imdsTrain.Labels, fusedTest, imdsTest.Labels, 'svm', false);
testAcc = evaluateClassifier(classifier, fusedTest, imdsTest.Labels, ...
                             trainResults.normalizationMean, trainResults.normalizationStd);
time = toc;

ablationResults.experiments{expIdx} = 'Attention-Based (Proposed)';
ablationResults.accuracies(expIdx) = testAcc;
ablationResults.cvAccuracies(expIdx) = trainResults.cvAccuracy;
ablationResults.configs{expIdx} = struct('cnn', 'both', 'fusion', 'attention', 'augmentation', false);
fprintf('Test Accuracy: %.2f%%, CV Accuracy: %.2f%%, Time: %.2f sec\n', ...
        testAcc * 100, trainResults.cvAccuracy * 100, time);

%% Experiment 7: Attention + SMOTE
fprintf('\n========================================\n');
fprintf('Experiment 7: Attention + SMOTE\n');
fprintf('========================================\n');
expIdx = expIdx + 1;
fusedTrain = fuseFeaturesModern({resnetTrain, densenetTrain}, hcFeaturesTrain, 'attention', false);
fusedTest = fuseFeaturesModern({resnetTest, densenetTest}, hcFeaturesTest, 'attention', false);

% Define SMOTE options
augOptions = struct();
augOptions.enabled = true;
augOptions.k = 5;
augOptions.ratio = 1.0;
augOptions.method = 'smote';
augOptions.applyPCA = false;

tic;
% Pass augmentation options to trainSingleExperiment
% Note: featuresAlreadyNorm is false because we want the function to handle normalization
[classifier, trainResults] = trainSingleExperiment(...
    fusedTrain, imdsTrain.Labels, fusedTest, imdsTest.Labels, 'svm', false, augOptions);
testAcc = evaluateClassifier(classifier, fusedTest, imdsTest.Labels, ...
                             trainResults.normalizationMean, trainResults.normalizationStd);
time = toc;

ablationResults.experiments{expIdx} = 'Attention + SMOTE (Full Proposed)';
ablationResults.accuracies(expIdx) = testAcc;
ablationResults.cvAccuracies(expIdx) = trainResults.cvAccuracy;
ablationResults.configs{expIdx} = struct('cnn', 'both', 'fusion', 'attention', 'augmentation', true);
fprintf('Test Accuracy: %.2f%%, CV Accuracy: %.2f%%, Time: %.2f sec\n', ...
        testAcc * 100, trainResults.cvAccuracy * 100, time);

%% Experiment 8: Different Classifiers
fprintf('\n========================================\n');
fprintf('Experiment 8: Different Classifiers\n');
fprintf('========================================\n');
fusedTrain = fuseFeaturesModern({resnetTrain, densenetTrain}, hcFeaturesTrain, 'attention', false);
fusedTest = fuseFeaturesModern({resnetTest, densenetTest}, hcFeaturesTest, 'attention', false);

classifiers = {'svm', 'ensemble'};
for c = 1:length(classifiers)
    expIdx = expIdx + 1;
    fprintf('\nTesting %s classifier...\n', upper(classifiers{c}));
    tic;
    [classifier, trainResults] = trainSingleExperiment(...
        fusedTrain, imdsTrain.Labels, fusedTest, imdsTest.Labels, classifiers{c}, false);
    testAcc = evaluateClassifier(classifier, fusedTest, imdsTest.Labels, ...
                                 trainResults.normalizationMean, trainResults.normalizationStd);
    time = toc;
    
    ablationResults.experiments{expIdx} = sprintf('Attention + %s', upper(classifiers{c}));
    ablationResults.accuracies(expIdx) = testAcc;
    ablationResults.cvAccuracies(expIdx) = trainResults.cvAccuracy;
    ablationResults.configs{expIdx} = struct('cnn', 'both', 'fusion', 'attention', ...
                                            'augmentation', false, 'classifier', classifiers{c});
    fprintf('Test Accuracy: %.2f%%, CV Accuracy: %.2f%%, Time: %.2f sec\n', ...
            testAcc * 100, trainResults.cvAccuracy * 100, time);
end

%% Save Results
if config.saveResults
    save(fullfile(config.outputDir, 'ablation_results.mat'), 'ablationResults', '-v7.3');
    fprintf('\nAblation results saved to: %s\n', fullfile(config.outputDir, 'ablation_results.mat'));
end

%% Generate Ablation Study Figure
fprintf('\nGenerating ablation study visualization...\n');
generateAblationFigure(ablationResults, config.outputDir);

%% Display Summary
fprintf('\n========================================\n');
fprintf('Ablation Study Summary\n');
fprintf('========================================\n');
fprintf('%-40s | Test Acc | CV Acc\n', 'Experiment');
fprintf('%-40s-|----------|--------\n', repmat('-', 1, 40));
for i = 1:length(ablationResults.experiments)
    fprintf('%-40s | %7.2f%% | %6.2f%%\n', ...
            ablationResults.experiments{i}, ...
            ablationResults.accuracies(i) * 100, ...
            ablationResults.cvAccuracies(i) * 100);
end
fprintf('========================================\n');

% Helper functions are in separate files:
% - trainSingleExperiment.m
% - evaluateClassifier.m
% - generateAblationFigure.m

