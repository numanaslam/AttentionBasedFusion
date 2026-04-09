% generateSampleImages.m
% Generate sample images with predictions for publication

clc; clear; close all;

%% Configuration
config = struct();
config.datasetPath = 'kvasir-dataset';
config.resultsDir = 'results';
config.outputDir = 'figures/samples';
config.numSamplesPerClass = 4;  % Number of sample images per class

% Create output directory
if ~isfolder(config.outputDir)
    mkdir(config.outputDir);
end

fprintf('Generating sample images with predictions...\n');

%% Load Classifier and Test Data
load(fullfile(config.resultsDir, 'trained_classifier.mat'), 'classifier');
load(fullfile(config.resultsDir, 'fused_features_attention.mat'), 'fusedTest');
load(fullfile(config.resultsDir, 'results.mat'), 'results');

% Get test labels
imds = imageDatastore(config.datasetPath, 'IncludeSubfolders', true, ...
                     'LabelSource', 'foldernames');
[~, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Normalize test features
trainMean = results.config.trainMean;
trainStd = results.config.trainStd;
fusedTestNorm = (fusedTest - trainMean) ./ (trainStd + eps);

% Get predictions
predictions = predict(classifier, fusedTestNorm);
testLabels = imdsTest.Labels;

%% Get Class Names
classNames = categories(testLabels);
numClasses = length(classNames);

%% Generate Sample Images
fprintf('Creating sample image grid...\n');

% Create figure for each class
for classIdx = 1:numClasses
    className = classNames{classIdx};
    classMask = testLabels == className;
    classIndices = find(classMask);
    
    % Get correct and incorrect predictions
    correctMask = predictions(classMask) == testLabels(classMask);
    incorrectMask = ~correctMask;
    
    correctIndices = classIndices(correctMask);
    incorrectIndices = classIndices(incorrectMask);
    
    % Select samples
    numCorrect = min(config.numSamplesPerClass, length(correctIndices));
    numIncorrect = min(2, length(incorrectIndices));  % Show some errors
    
    selectedCorrect = correctIndices(randperm(length(correctIndices), numCorrect));
    selectedIncorrect = [];
    if ~isempty(incorrectIndices)
        selectedIncorrect = incorrectIndices(randperm(length(incorrectIndices), numIncorrect));
    end
    
    selectedIndices = [selectedCorrect; selectedIncorrect];
    
    % Create figure
    fig = figure('Position', [100, 100, 1200, 300 * ceil(length(selectedIndices) / 4)]);
    
    for i = 1:length(selectedIndices)
        idx = selectedIndices(i);
        img = readimage(imdsTest, idx);
        trueLabel = char(testLabels(idx));
        predLabel = char(predictions(idx));
        isCorrect = strcmp(trueLabel, predLabel);
        
        subplot(ceil(length(selectedIndices) / 4), 4, i);
        imshow(img);
        
        % Title with prediction result
        if isCorrect
            title(sprintf('True: %s\nPred: %s ✓', trueLabel, predLabel), ...
                  'Color', 'green', 'FontSize', 10, 'FontWeight', 'bold');
        else
            title(sprintf('True: %s\nPred: %s ✗', trueLabel, predLabel), ...
                  'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
        end
    end
    
    % Save figure
    filename = sprintf('SampleImages_%s.png', strrep(className, ' ', '_'));
    saveas(fig, fullfile(config.outputDir, filename), 'png');
    close(fig);
end

fprintf('Sample images saved to: %s\n', config.outputDir);

