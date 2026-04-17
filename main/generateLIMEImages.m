clear; clc; close all;

fprintf('=== Generating Custom LIME Explanations (One per Class) ===\n\n');

%% ====================== LOAD MODEL ======================
modelFile = 'C:\paper1\paper1_v1\results\trained_classifier.mat';
if ~exist(modelFile, 'file')
    error('Model not found: %s\nPlease run TrainModernFusionModel first.', modelFile);
end
data = load(modelFile);
model = data.classifier;
fprintf('✓ Ensemble model loaded successfully.\n');

%% ====================== LOAD TEST IMAGES & SELECT ONE PER CLASS ======================
imds = imageDatastore('kvasir-dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Keep only test set
[~, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Get unique classes
classes = unique(imdsTest.Labels);
numClasses = length(classes);

fprintf('Found %d classes. Selecting one image from each...\n', numClasses);

% Select exactly one image per class
selectedIdx = zeros(numClasses, 1);
for i = 1:numClasses
    idx = find(imdsTest.Labels == classes(i));
    selectedIdx(i) = idx(randi(length(idx)));  % random image from this class
end

fprintf('Selected images from classes:\n');
disp(classes');

%% ====================== MAIN LOOP (one image per class) ======================
for i = 1:numClasses
    idx = selectedIdx(i);
    img = readimage(imdsTest, idx);
    trueLabel = string(imdsTest.Labels(idx));
    
    fprintf('Processing image %d/%d (%s)... ', i, numClasses, trueLabel);
    
    % Original prediction
    origScores = predictSingle(img, model);
    [~, targetClass] = max(origScores);
    
    % Superpixels
    numSuperpixels = 60;
    [L, ~] = superpixels(img, numSuperpixels, 'Compactness', 10);
    
    % Compute importance
    weights = zeros(numSuperpixels, 1);
    for sp = 1:numSuperpixels
        perturbed = img;
        mask = (L == sp);
        for c = 1:3
            channel = perturbed(:,:,c);
            channel(mask) = mean(channel(:));
            perturbed(:,:,c) = channel;
        end
        s = predictSingle(perturbed, model);
        weights(sp) = origScores(targetClass) - s(targetClass);
    end
    
    % Normalize weights
    weights = weights - min(weights);
    if max(weights) > 0
        weights = weights / max(weights);
    end
    
    % Importance map
    importanceMap = zeros(size(L));
    for sp = 1:numSuperpixels
        importanceMap(L == sp) = weights(sp);
    end
    
    %% ====================== PLOT & SAVE ======================
    figure('Position', [100 100 1250 520]);
    
    subplot(1,2,1);
    imshow(img);
    title(sprintf('Original\nTrue: %s', trueLabel), 'FontSize', 13);
    
    subplot(1,2,2);
    imshow(img); hold on;
    h = imshow(importanceMap, []);
    colormap(jet);
    set(h, 'AlphaData', 0.65);
    title('LIME Explanation', 'FontSize', 13);
    colorbar;
    
    if ~exist('results','dir'), mkdir('results'); end
    outFile = sprintf('results/LIME_%02d_%s.png', i, trueLabel);
    saveas(gcf, outFile);
    close(gcf);
    
    fprintf('✓ Saved: %s\n', outFile);
end

fprintf('\n✅ All LIME explanations generated (ONE per class)!\n');
fprintf('Files saved in results/ folder.\n');

%% ====================== HELPER FUNCTIONS (unchanged) ======================
function scores = predictSingle(img, model)
    if size(img,3) == 1
        img = repmat(img, [1 1 3]);
    end
    resnetFeat  = extractModernCNNFeaturesSingle(img, 'resnet50');
    densenetFeat = extractModernCNNFeaturesSingle(img, 'densenet201');
    hcFeat      = extractHandcraftedFeaturesModernSingle(img);
    fused = fuseSingleSample({resnetFeat, densenetFeat}, hcFeat);
    [~, scores] = predict(model, fused(:)');
    scores = scores(:)';
end

function fused = fuseSingleSample(cnnFeatures, handcraftedFeatures)
    allCNN = [];
    for i = 1:length(cnnFeatures)
        allCNN = [allCNN, cnnFeatures{i}];
    end
    hc = handcraftedFeatures(:)';
    if length(hc) > 38, hc = hc(1:38); end
    if length(hc) < 38, hc = [hc, zeros(1,38-length(hc))]; end
    alpha = 0.65; beta = 0.35;
    fused = [alpha * allCNN, beta * hc];
end

function feat = extractModernCNNFeaturesSingle(img, modelType)
    if size(img,3)==1, img = repmat(img,[1 1 3]); end
    img = imresize(img,[224 224]);
    switch lower(modelType)
        case 'resnet50', net = resnet50('Weights','imagenet'); layer = 'avg_pool';
        case 'densenet201', net = densenet201('Weights','imagenet'); layer = 'avg_pool';
    end
    feat = activations(net, img, layer, 'OutputAs','rows');
end

function feat = extractHandcraftedFeaturesModernSingle(img)
    if size(img,3) == 1, img = repmat(img,[1 1 3]); end
    img = imresize(img,[256 256]);
    grayImg = im2gray(img); grayImg = im2double(grayImg);
    glcm = graycomatrix(grayImg, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
    stats = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});
    haralick = [mean(stats.Contrast), mean(stats.Correlation), ...
                mean(stats.Energy), mean(stats.Homogeneity)];
    texture = [mean(grayImg(:)), std(grayImg(:)), entropy(grayImg), ...
               var(grayImg(:)), kurtosis(grayImg(:)), skewness(grayImg(:))];
    bw = imbinarize(grayImg);
    props = regionprops(bw, 'Area','Perimeter','Eccentricity');
    if isempty(props)
        shape = zeros(1,6);
    else
        shape = [mean([props.Area]), mean([props.Perimeter]), ...
                 mean([props.Eccentricity]), std([props.Area]), ...
                 std([props.Perimeter]), numel(props)];
    end
    feat = [haralick, texture, shape];
end
