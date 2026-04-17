function [fusedFeatures, fusionInfo] = fuseFeaturesModern(cnnFeatures, handcraftedFeatures, fusionMethod, useGPU)
%FUSEFEATURESMODERN Fuse CNN and handcrafted features using modern techniques


if nargin < 3
    fusionMethod = 'varianceCorrelationWeighted';
end
if nargin < 4
    useGPU = false;
end

% === FIXED ROBUST METHOD NAME NORMALIZATION ===
method = lower(fusionMethod);

% Remove any suffix after underscore FIRST (this was the bug)
if contains(method, '_')
    method = extractBefore(method, '_');
end

% Now clean remaining characters
method = strrep(method, '-', '');
method = strrep(method, ' ', '');
method = strrep(method, '_', '');

% Convert single CNN features to cell array
if ~iscell(cnnFeatures)
    cnnFeatures = {cnnFeatures};
end

numModels = length(cnnFeatures);
numSamples = size(handcraftedFeatures, 1);

fprintf('Fusing features using %s method...\n', fusionMethod);

switch method
    case 'concat'
        allCNN = [];
        for i = 1:numModels
            allCNN = [allCNN, cnnFeatures{i}];
        end
        fusedFeatures = [allCNN, handcraftedFeatures];
        fusionInfo.method = 'Concatenation';

    case 'weighted'
        allCNN = [];
        for i = 1:numModels
            feat = cnnFeatures{i};
            feat = (feat - mean(feat,1)) ./ (std(feat,[],1) + eps);
            allCNN = [allCNN, feat];
        end
        hcNorm = (handcraftedFeatures - mean(handcraftedFeatures,1)) ./ ...
                 (std(handcraftedFeatures,[],1) + eps);
        cnnVar = var(allCNN,[],1);
        hcVar  = var(hcNorm,[],1);
        cnnWeight = mean(cnnVar) / (mean(cnnVar) + mean(hcVar) + eps);
        hcWeight  = 1 - cnnWeight;
        fusedFeatures = [cnnWeight*allCNN, hcWeight*hcNorm];
        fusionInfo.method = 'Weighted Fusion';

    case {'variancecorrelationweighted','variancecorrelation','vcweighted'}
        % === ADAPTIVE VARIANCE-CORRELATION WEIGHTED FUSION ===
        allCNN = [];
        for i = 1:numModels
            feat = cnnFeatures{i};
            if useGPU && canUseGPU()
                feat = gpuArray(feat);
            end
            feat = (feat - mean(feat,1)) ./ (std(feat,[],1) + eps);
            allCNN = [allCNN, feat];
        end

        if useGPU && canUseGPU()
            hcNorm = gpuArray(handcraftedFeatures);
        else
            hcNorm = handcraftedFeatures;
        end
        hcNorm = (hcNorm - mean(hcNorm,1)) ./ (std(hcNorm,[],1) + eps);

        cnnVar = var(allCNN,[],1);
        hcVar  = var(hcNorm,[],1);

        cnnMeanPerSample = mean(allCNN, 2);
        hcMeanPerSample  = mean(hcNorm,  2);
        correlation = corrcoef(cnnMeanPerSample, hcMeanPerSample);
        crossModalCorr = abs(correlation(1,2));

        cnnImportance = mean(cnnVar) * (1 + crossModalCorr);
        hcImportance  = mean(hcVar)  * (1 + crossModalCorr);

        attentionScores = [cnnImportance, hcImportance];
        attentionWeights = exp(attentionScores) ./ (sum(exp(attentionScores)) + eps);

        alpha = attentionWeights(1);
        beta  = attentionWeights(2);

        fusedFeatures = [alpha*allCNN, beta*hcNorm];
        if useGPU && canUseGPU()
            fusedFeatures = gather(fusedFeatures);
        end

        fusionInfo.method = 'Variance-Correlation Weighted Fusion';
        fusionInfo.alpha = alpha;
        fusionInfo.beta  = beta;
        fusionInfo.crossModalCorr = crossModalCorr;

    case 'bilinear'
        allCNN = [];
        for i = 1:numModels
            allCNN = [allCNN, cnnFeatures{i}];
        end
        if size(allCNN,2) > 512
            [~, score] = pca(allCNN, 'NumComponents',512);
            allCNN = score;
        end
        if size(handcraftedFeatures,2) > 64
            [~, score] = pca(handcraftedFeatures, 'NumComponents',64);
            hcReduced = score;
        else
            hcReduced = handcraftedFeatures;
        end
        bilinear = zeros(numSamples, size(allCNN,2)*size(hcReduced,2));
        for i = 1:numSamples
            bilinear(i,:) = reshape(allCNN(i,:)' * hcReduced(i,:), 1, []);
        end
        bilinear = sign(bilinear) .* sqrt(abs(bilinear));
        bilinear = bilinear ./ (sqrt(sum(bilinear.^2,2)) + eps);
        fusedFeatures = bilinear;
        fusionInfo.method = 'Bilinear Pooling';

    case 'multimodal'
        allCNN = [];
        for i = 1:numModels
            feat = cnnFeatures{i};
            feat = (feat - mean(feat,1)) ./ (std(feat,[],1) + eps);
            allCNN = [allCNN, feat];
        end
        hcNorm = (handcraftedFeatures - mean(handcraftedFeatures,1)) ./ ...
                 (std(handcraftedFeatures,[],1) + eps);
        attentionWeights  = softmax(allCNN * hcNorm', 2);
        cnnEnhanced = attentionWeights * hcNorm;
        attentionWeights2 = softmax(hcNorm * allCNN', 2);
        hcEnhanced = attentionWeights2 * allCNN;
        fusedFeatures = [allCNN + 0.3*cnnEnhanced, hcNorm + 0.3*hcEnhanced];
        fusionInfo.method = 'Multi-modal Cross-Attention';

    otherwise
        error('Unknown fusion method: %s\nSupported: concat, weighted, varianceCorrelationWeighted, bilinear, multimodal', fusionMethod);
end

fusionInfo.finalDim = size(fusedFeatures, 2);
fprintf('Fusion completed. Final feature dimension: %d\n', fusionInfo.finalDim);
end

%% Helper functions
function y = softmax(x, dim)
if nargin < 2, dim = 1; end
exp_x = exp(x - max(x,[],dim));
y = exp_x ./ (sum(exp_x,dim) + eps);
end

function available = canUseGPU()
persistent checked result
if isempty(checked)
    try
        result = parallel.gpu.GPUDevice.isAvailable;
    catch
        result = false;
    end
    checked = true;
end
available = result;
end
