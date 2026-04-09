function [classifier, trainResults] = trainSingleExperiment(featuresTrain, labelsTrain, ...
                                                           featuresTest, labelsTest, ...
                                                           classifierType, featuresAlreadyNorm, augmentationOptions)
%TRAINSINGLEEXPERIMENT Helper function for ablation study
%   Trains a classifier for a single ablation experiment

    if nargin < 7
        augmentationOptions = struct('enabled', false);
    end

    % Always normalize features to ensure consistency
    % Even if features appear pre-normalized, we need to compute stats from training data
    % to properly normalize test features
    if ~featuresAlreadyNorm
        % Compute normalization statistics from training data
        trainMean = mean(featuresTrain, 1);
        trainStd = std(featuresTrain, [], 1);
        
        % Handle near-zero variance features
        minStd = 1e-6;
        trainStd(trainStd < minStd) = 1.0;
        
        % Normalize training features
        featuresTrain = (featuresTrain - trainMean) ./ (trainStd + eps);
        
        % Clip extreme values
        featuresTrain = max(min(featuresTrain, 5), -5);
        
        % Check if features were already normalized (for diagnostic)
        trainStdCheck = std(featuresTrain(:));
        trainMeanCheck = mean(featuresTrain(:));
        if abs(trainMeanCheck) < 0.1 && abs(trainStdCheck - 1.0) < 0.1
            fprintf('  Note: Features were already normalized, but re-normalized for consistency.\n');
        end
    else
        % Features already normalized, use identity normalization
        trainMean = zeros(1, size(featuresTrain, 2));
        trainStd = ones(1, size(featuresTrain, 2));
    end
    
    % Train classifier
    options = struct();
    options.optimizeHyperparams = false;  % Faster for ablation
    options.cvFolds = 5;
    options.featuresAlreadyNormalized = featuresAlreadyNorm;
    options.augmentation = augmentationOptions;
    
    [classifier, trainResults] = trainModernClassifier(...
        featuresTrain, labelsTrain, classifierType, options);
    
    % Store normalization stats
    trainResults.normalizationMean = trainMean;
    trainResults.normalizationStd = trainStd;
end

