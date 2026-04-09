function features = extractHandcraftedFeatures(imds)
% Extract Haralick + Zernike features

numImages = numel(imds.Files);
features = zeros(numImages, 38); % 13 Haralick + 25 Zernike

for i = 1:numImages
    img = readimage(imds, i);
    grayImg = im2gray(imresize(img, [256 256]));

    % Haralick
    glcm = graycomatrix(grayImg, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
    stats = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});
    haralick = [mean(stats.Contrast), mean(stats.Correlation), ...
                mean(stats.Energy), mean(stats.Homogeneity)];
    haralick = repmat(haralick, 1, 3); % repeat to match 12 dims
    haralick = [haralick, mean(haralick)]; % add mean to make it 13 dims

    % Zernike (simple mock - actual implementation may vary)
    zm = double(grayImg(:))';
    zernike = zm(1:25); % simplified placeholder

    features(i, :) = [haralick, zernike];
end
end