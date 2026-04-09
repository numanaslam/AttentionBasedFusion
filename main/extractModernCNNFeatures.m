function [features, modelInfo] = extractModernCNNFeatures(imds, modelType, layerName)
%EXTRACTMODERNCNNFEATURES Extract features from modern CNN architectures
%   [FEATURES, MODELINFO] = extractModernCNNFeatures(IMDS, MODELTYPE, LAYERNAME)
%   extracts deep features from images using state-of-the-art CNN models.
%
%   Inputs:
%       imds - Image datastore containing images
%       modelType - String specifying model: 'resnet50', 'resnet101', 
%                   'densenet201', 'efficientnetb0', 'mobilenetv2'
%       layerName - Layer name for feature extraction (optional)
%
%   Outputs:
%       features - Extracted features matrix (N x D)
%       modelInfo - Structure containing model information
%
%   Example:
%       imds = imageDatastore('path/to/images', 'IncludeSubfolders', true);
%       features = extractModernCNNFeatures(imds, 'resnet50');

% Default parameters
if nargin < 3
    layerName = [];
end

% Check for GPU availability
useGPU = canUseGPU();
if useGPU
    fprintf('Using GPU for feature extraction...\n');
    gpuDevice;  % Display GPU info
end

% Image preprocessing
inputSize = [224 224 3];
augimds = augmentedImageDatastore(inputSize, imds, 'ColorPreprocessing', 'gray2rgb');
% Set execution environment for datastore
if useGPU
    augimds.ExecutionEnvironment = 'gpu';
end

% Load appropriate model
fprintf('Loading %s model...\n', upper(modelType));
switch lower(modelType)
    case 'resnet50'
        net = resnet50('Weights', 'imagenet');
        if isempty(layerName)
            layerName = 'avg_pool';
        end
        modelInfo.modelName = 'ResNet-50';
        modelInfo.numParams = 25.6e6;
        
    case 'resnet101'
        net = resnet101('Weights', 'imagenet');
        if isempty(layerName)
            layerName = 'avg_pool';
        end
        modelInfo.modelName = 'ResNet-101';
        modelInfo.numParams = 44.5e6;
        
    case 'densenet201'
        net = densenet201('Weights', 'imagenet');
        if isempty(layerName)
            layerName = 'avg_pool';
        end
        modelInfo.modelName = 'DenseNet-201';
        modelInfo.numParams = 20.0e6;
        
    case 'mobilenetv2'
        net = mobilenetv2('Weights', 'imagenet');
        if isempty(layerName)
            layerName = 'global_average_pooling2d_1';
        end
        modelInfo.modelName = 'MobileNetV2';
        modelInfo.numParams = 3.5e6;
        
    case 'efficientnetb0'
        % EfficientNet may not be available in all MATLAB versions
        % Fallback to ResNet-50 if not available
        try
            net = efficientnetb0('Weights', 'imagenet');
            if isempty(layerName)
                layerName = 'top_activation';
            end
            modelInfo.modelName = 'EfficientNet-B0';
            modelInfo.numParams = 5.3e6;
        catch
            warning('EfficientNet not available. Using ResNet-50 instead.');
            net = resnet50('Weights', 'imagenet');
            layerName = 'avg_pool';
            modelInfo.modelName = 'ResNet-50 (EfficientNet fallback)';
            modelInfo.numParams = 25.6e6;
        end
        
    otherwise
        error('Unsupported model type: %s. Use: resnet50, resnet101, densenet201, mobilenetv2, or efficientnetb0', modelType);
end

modelInfo.layerName = layerName;
modelInfo.inputSize = inputSize;

% Move model to GPU if available
if useGPU
    try
        net = net.copy();
        % Set execution environment for activations
        fprintf('Model moved to GPU. Using GPU for feature extraction.\n');
    catch ME
        warning('Could not move model to GPU: %s. Using CPU.', ME.message);
        useGPU = false;
    end
end

% Extract features
fprintf('Extracting features from %d images...\n', numel(imds.Files));
tic;
if useGPU
    % Use larger batch size on GPU for better performance
    features = activations(net, augimds, layerName, 'OutputAs', 'rows', ...
                          'MiniBatchSize', 64, 'ExecutionEnvironment', 'gpu');
else
    features = activations(net, augimds, layerName, 'OutputAs', 'rows', ...
                          'MiniBatchSize', 32);
end
extractionTime = toc;

modelInfo.extractionTime = extractionTime;
modelInfo.featureDim = size(features, 2);
modelInfo.numImages = size(features, 1);

fprintf('Feature extraction completed in %.2f seconds.\n', extractionTime);
fprintf('Feature dimension: %d\n', modelInfo.featureDim);
end

function gpuAvailable = canUseGPU()
% Check if GPU is available
try
    gpuAvailable = canUseGPU && parallel.gpu.GPUDevice.isAvailable;
catch
    gpuAvailable = false;
end
end

