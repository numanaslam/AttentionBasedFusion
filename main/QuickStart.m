% QuickStart.m
% Quick start guide for the modernized GI tract classification pipeline
% This script demonstrates how to use the new codebase

clc; clear; close all;

fprintf('========================================\n');
fprintf('Quick Start Guide\n');
fprintf('========================================\n\n');

%% Example 1: Basic Usage (Recommended for first-time users)
fprintf('Example 1: Basic Usage\n');
fprintf('----------------------\n');
fprintf('This uses ResNet-50 + attention fusion + SMOTE + SVM\n\n');

% Uncomment to run:
% config.datasetPath = 'kvasir-dataset';
% config.cnnModels = {'resnet50'};  % Single model for faster training
% config.fusionMethod = 'attention';
% config.augmentation.enabled = true;
% config.augmentation.method = 'smote';
% config.classifier.type = 'svm';
% config.classifier.optimizeHyperparams = true;
% 
% % Run training (modify TrainModernFusionModel.m with these settings first)

%% Example 2: Multi-Model Ensemble
fprintf('Example 2: Multi-Model Ensemble\n');
fprintf('--------------------------------\n');
fprintf('Uses multiple CNNs for better performance\n\n');

% config.cnnModels = {'resnet50', 'densenet201'};
% config.fusionMethod = 'multimodal';  % Best for multiple models
% config.augmentation.method = 'borderline';  % Better than standard SMOTE

%% Example 3: Fast Training (No Augmentation)
fprintf('Example 3: Fast Training\n');
fprintf('------------------------\n');
fprintf('Skip augmentation for faster training\n\n');

% config.augmentation.enabled = false;
% config.classifier.optimizeHyperparams = false;  % Use default parameters

%% Example 4: High Performance
fprintf('Example 4: High Performance Setup\n');
fprintf('----------------------------------\n');
fprintf('Maximum performance (slower training)\n\n');

% config.cnnModels = {'resnet50', 'densenet201', 'mobilenetv2'};
% config.fusionMethod = 'bilinear';  % Richer features
% config.augmentation.method = 'adasyn';  % Adaptive sampling
% config.classifier.type = 'ensemble';  % Ensemble classifier

%% Example 5: Extract Features Only
fprintf('Example 5: Extract Features Only\n');
fprintf('---------------------------------\n');
fprintf('Extract features without training classifier\n\n');

% Load dataset
% imds = imageDatastore('kvasir-dataset', 'IncludeSubfolders', true, ...
%                      'LabelSource', 'foldernames');
% 
% % Extract handcrafted features
% hcFeatures = extractHandcraftedFeaturesModern(imds);
% 
% % Extract CNN features
% cnnFeatures = extractModernCNNFeatures(imds, 'resnet50');
% 
% % Fuse features
% fusedFeatures = fuseFeaturesModern({cnnFeatures}, hcFeatures, 'attention');
% 
% % Save features
% save('extracted_features.mat', 'fusedFeatures', 'hcFeatures', 'cnnFeatures');

%% Tips
fprintf('\n========================================\n');
fprintf('Tips for Best Results\n');
fprintf('========================================\n');
fprintf('1. Start with Example 1 for baseline\n');
fprintf('2. Use GPU if available (automatic detection)\n');
fprintf('3. For faster training: use single CNN model\n');
fprintf('4. For better accuracy: use multiple models + multimodal fusion\n');
fprintf('5. Borderline SMOTE often works better than standard SMOTE\n');
fprintf('6. Enable hyperparameter optimization for best SVM performance\n');
fprintf('7. Check results/ folder for saved features and models\n');
fprintf('========================================\n\n');

fprintf('To get started:\n');
fprintf('1. Update dataset path in TrainModernFusionModel.m\n');
fprintf('2. Choose your configuration (see examples above)\n');
fprintf('3. Run: TrainModernFusionModel\n');
fprintf('4. Check results/ folder for outputs\n\n');

