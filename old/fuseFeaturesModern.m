function [fusedFeatures, fusionInfo] = fuseFeaturesModern(cnnFeatures, handcraftedFeatures, fusionMethod, useGPU, labels)
%FUSEFEATURESMODERN Fuse CNN and handcrafted features using modern techniques
%   [FUSED, INFO] = fuseFeaturesModern(CNN, HC, METHOD, USEGPU, LABELS) fuses
%   deep CNN features with handcrafted features using various fusion strategies.
%
%   Inputs:
%       cnnFeatures - Cell array of CNN feature matrices or single matrix
%       handcraftedFeatures - Handcrafted feature matrix (N x D)
%       fusionMethod - Fusion strategy: 'concat', 'weighted', 'attention'
%                      - 'attention': Statistical adaptive weights from 
%                        variance + cross-modal correlation (NOT learnable)
%       useGPU - Optional: Use GPU for computations (default: false)
%       labels - Optional: Class labels for per-class weight computation
%
%   Outputs:
%       fusedFeatures - Fused feature matrix
%       fusionInfo - Structure with fusion details
%
%   Note: The 'attention' method computes weights from dataset-level statistics,
%   not via backpropagation. This provides transparent, reproducible feature weighting.

if nargin < 3
    fusionMethod = 'attention';
end
if nargin < 4
    useGPU = false;
end
if nargin < 5
    labels = [];  % Optional for per-class analysis
end

% Validate inputs
if isempty(cnnFeatures) || isempty(handcraftedFeatures)
    error('Input feature matrices cannot be empty');
end

% Convert single CNN features to cell array for uniform processing
if ~iscell(cnnFeatures)
    cnnFeatures = {cnnFeatures};
end

numModels = length(cnnFeatures);
numSamples = size(handcraftedFeatures, 1);

fprintf('Fusing features using %s method...\n', fusionMethod);

switch lower(fusionMethod)
    case 'concat'
        % Simple concatenation (baseline)
        allCNN = [];
        for i = 1:numModels
            allCNN = [allCNN, cnnFeatures{i}];
        end
        fusedFeatures = [allCNN, handcraftedFeatures];
        fusionInfo.method = 'Concatenation';
        fusionInfo.cnnDim = size(allCNN, 2);
        fusionInfo.hcDim = size(handcraftedFeatures, 2);
        
    case 'weighted'
        % Weighted fusion based on feature importance
        % Normalize each feature set first
        allCNN = [];
        for i = 1:numModels
            feat = cnnFeatures{i};
            feat = (feat - mean(feat, 1)) ./ (std(feat, [], 1) + eps);
            allCNN = [allCNN, feat];
        end
        
        hcNorm = (handcraftedFeatures - mean(handcraftedFeatures, 1)) ./ ...
                 (std(handcraftedFeatures, [], 1) + eps);
        
        % Learn weights using variance-based importance
        cnnVar = var(allCNN, [], 1);
        hcVar = var(hcNorm, [], 1);
        
        cnnWeight = mean(cnnVar) / (mean(cnnVar) + mean(hcVar) + eps);
        hcWeight = 1 - cnnWeight;
        
        fusedFeatures = [cnnWeight * allCNN, hcWeight * hcNorm];
        fusionInfo.method = 'Weighted Fusion';
        fusionInfo.cnnWeight = cnnWeight;
        fusionInfo.hcWeight = hcWeight;
        
    case 'attention'
        % Statistical attention-based fusion
        % Computes adaptive weights from variance + cross-modal correlation
        % NOTE: Weights are computed from dataset-level statistics, NOT learned via backprop
        
        % Normalize features
        allCNN = [];
        for i = 1:numModels
            feat = cnnFeatures{i};
            if useGPU
                feat = gpuArray(feat);
            end
            feat = (feat - mean(feat, 1)) ./ (std(feat, [], 1) + eps);
            allCNN = [allCNN, feat];
        end
        
        if useGPU
            hcNorm = gpuArray(handcraftedFeatures);
        else
            hcNorm = handcraftedFeatures;
        end
        hcNorm = (hcNorm - mean(hcNorm, 1)) ./ (std(hcNorm, [], 1) + eps);
        
        % Compute attention scores based on feature importance
        % Method 1: Variance-based importance (measures feature discriminativity)
        cnnVar = var(allCNN, [], 1);
        hcVar = var(hcNorm, [], 1);
        
        % Method 2: Cross-modal correlation (measures feature relevance)
        cnnMeanPerSample = mean(allCNN, 2);
        hcMeanPerSample = mean(hcNorm, 2);
        
        if useGPU
            correlation = corrcoef(gather(cnnMeanPerSample), gather(hcMeanPerSample));
        else
            correlation = corrcoef(cnnMeanPerSample, hcMeanPerSample);
        end
        crossModalCorr = abs(correlation(1, 2));
        
        % Combine variance and correlation for attention scores
        if useGPU
            cnnImportance = gather(mean(cnnVar)) * (1 + crossModalCorr);
            hcImportance = gather(mean(hcVar)) * (1 + crossModalCorr);
        else
            cnnImportance = mean(cnnVar) * (1 + crossModalCorr);
            hcImportance = mean(hcVar) * (1 + crossModalCorr);
        end
        
        % Compute attention weights using softmax normalization
        % This ensures weights sum to 1 and are interpretable (NOT learned via backprop)
        attentionScores = [cnnImportance, hcImportance];
        attentionWeights = exp(attentionScores) ./ (sum(exp(attentionScores)) + eps);
        
        alpha = attentionWeights(1);  % CNN attention weight (statistical)
        beta = attentionWeights(2);   % Handcrafted attention weight (statistical)
        
        % Apply attention weights to features
        cnnAttended = alpha * allCNN;
        hcAttended = beta * hcNorm;
        
        fusedFeatures = [cnnAttended, hcAttended];
        if useGPU
            fusedFeatures = gather(fusedFeatures);
        end
        
        fusionInfo.method = 'Statistical Attention-based Fusion';
        fusionInfo.alpha = alpha;
        fusionInfo.beta = beta;
        fusionInfo.crossModalCorr = crossModalCorr;
        fusionInfo.attentionScores = attentionScores;
        fusionInfo.note = 'Weights computed from dataset-level statistics, not learned via backpropagation';
        
    otherwise
        error('Unknown fusion method: %s', fusionMethod);
end

fusionInfo.finalDim = size(fusedFeatures, 2);
fprintf('Fusion completed. Final feature dimension: %d\n', fusionInfo.finalDim);
end

function y = softmax(x, dim)
% Softmax function
if nargin < 2
    dim = 1;
end
exp_x = exp(x - max(x, [], dim));
y = exp_x ./ (sum(exp_x, dim) + eps);
end