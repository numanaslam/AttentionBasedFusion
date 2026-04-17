clc; clear; close all;
rng(42);  % Fixed seed for reproducibility

fprintf('========================================\n');
fprintf('Ablation Study - Adaptive Variance-Correlation Weighted Fusion\n');
fprintf('========================================\n\n');

%% Configuration
config.datasetPath = 'kvasir-dataset';
config.outputDir   = 'ablation_results';
config.saveResults = true;

if ~isfolder(config.outputDir)
    mkdir(config.outputDir);
end

%% Load Dataset
fprintf('Loading dataset...\n');
imds = imageDatastore(config.datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds = shuffle(imds);
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');
fprintf('Dataset loaded: %d training, %d test images\n', numel(imdsTrain.Files), numel(imdsTest.Files));

%% Load / Extract Features
resultsDir = 'results';

% Handcrafted
hcFile = fullfile(resultsDir, 'handcrafted_features.mat');
if exist(hcFile, 'file')
    load(hcFile, 'hcFeaturesTrain', 'hcFeaturesTest');
else
    fprintf('Extracting handcrafted features...\n');
    hcFeaturesTrain = extractHandcraftedFeaturesModern(imdsTrain);
    hcFeaturesTest  = extractHandcraftedFeaturesModern(imdsTest);
end

% ResNet-50
resnetFile = fullfile(resultsDir, 'resnet50_features.mat');
if exist(resnetFile, 'file')
    load(resnetFile, 'featTrain', 'featTest');
    resnetTrain = featTrain; resnetTest = featTest;
else
    fprintf('Extracting ResNet-50 features...\n');
    [resnetTrain, ~] = extractModernCNNFeatures(imdsTrain, 'resnet50');
    [resnetTest,  ~] = extractModernCNNFeatures(imdsTest,  'resnet50');
end

% DenseNet-201
densenetFile = fullfile(resultsDir, 'densenet201_features.mat');
if exist(densenetFile, 'file')
    load(densenetFile, 'featTrain', 'featTest');
    densenetTrain = featTrain; densenetTest = featTest;
else
    fprintf('Extracting DenseNet-201 features...\n');
    [densenetTrain, ~] = extractModernCNNFeatures(imdsTrain, 'densenet201');
    [densenetTest,  ~] = extractModernCNNFeatures(imdsTest,  'densenet201');
end

%% Define Experiments (fixed cell array)
experiments = {
    'CNN Only (ResNet-50)',                  'resnet',      'svm';
    'CNN Only (DenseNet-201)',               'densenet',    'svm';
    'Handcrafted Features Only',             'handcrafted', 'svm';
    'Simple Concatenation',                  'concat',      'svm';
    'Weighted Fusion',                       'weighted',    'svm';
    'Variance-Correlation Weighted Fusion',  'varianceCorrelationWeighted', 'svm';
    'Variance-Correlation Weighted + SMOTE', 'varianceCorrelationWeighted', 'svm';
    'Variance-Correlation Weighted + SVM',   'varianceCorrelationWeighted', 'svm';
    'Proposed: Variance-Correlation Weighted + Ensemble', 'varianceCorrelationWeighted', 'ensemble';
};

fprintf('Defined %d experiments.\n', size(experiments,1));

%% Run Ablation
ablationTable = table('Size', [size(experiments,1) 3], ...
    'VariableTypes', {'string','double','double'}, ...
    'VariableNames', {'Experiment','Test_Accuracy','CV_Accuracy'});

for i = 1:size(experiments,1)
    expName        = experiments{i,1};
    fusionType     = experiments{i,2};
    classifierType = experiments{i,3};
    
    fprintf('\n========================================\n');
    fprintf('Experiment %d: %s\n', i, expName);
    fprintf('========================================\n');
    
    % Select features
    if strcmp(fusionType, 'resnet')
        fTrain = resnetTrain; fTest = resnetTest;
    elseif strcmp(fusionType, 'densenet')
        fTrain = densenetTrain; fTest = densenetTest;
    elseif strcmp(fusionType, 'handcrafted')
        fTrain = hcFeaturesTrain; fTest = hcFeaturesTest;
    else
        % Fusion
        fTrain = fuseFeaturesModern({resnetTrain, densenetTrain}, hcFeaturesTrain, fusionType, true);
        fTest  = fuseFeaturesModern({resnetTest,  densenetTest},  hcFeaturesTest,  fusionType, true);
    end
    
    useSMOTE = contains(expName, 'SMOTE');
    
    [classifier, trainResults] = trainSingleExperiment(fTrain, imdsTrain.Labels, ...
        fTest, imdsTest.Labels, classifierType, useSMOTE);
    
    testAcc = evaluateClassifier(classifier, fTest, imdsTest.Labels, ...
        trainResults.normalizationMean, trainResults.normalizationStd);
    
    ablationTable.Experiment(i)    = expName;
    ablationTable.Test_Accuracy(i) = testAcc * 100;
    ablationTable.CV_Accuracy(i)   = trainResults.cvAccuracy * 100;
    
    fprintf('Test Accuracy: %.2f%% | CV Accuracy: %.2f%%\n', testAcc*100, trainResults.cvAccuracy*100);
end

%% Save and Show Summary
if config.saveResults
    save(fullfile(config.outputDir, 'ablation_results.mat'), 'ablationTable');
    writetable(ablationTable, fullfile(config.outputDir, 'ablation_results.csv'));
    fprintf('\nResults saved to %s\n', config.outputDir);
end

disp(ablationTable);
fprintf('\nAblation study completed successfully.\n');
