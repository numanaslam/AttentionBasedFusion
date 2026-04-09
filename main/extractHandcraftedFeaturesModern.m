function features = extractHandcraftedFeaturesModern(imds)
%EXTRACTHANDCRAFTEDFEATURESMODERN Extract comprehensive Haralick and Zernike features
%   FEATURES = extractHandcraftedFeaturesModern(IMDS) extracts handcrafted
%   features including all 13 Haralick texture features and Zernike moments
%   up to order 8 (25 features).
%
%   Inputs:
%       imds - Image datastore containing images
%
%   Outputs:
%       features - Feature matrix (N x 38) where N is number of images
%                  Columns 1-13: Haralick features
%                  Columns 14-38: Zernike moments

numImages = numel(imds.Files);
numHaralick = 13;
numZernike = 25;
features = zeros(numImages, numHaralick + numZernike);

fprintf('Extracting handcrafted features from %d images...\n', numImages);

% Process images in batches for efficiency
batchSize = 100;
numBatches = ceil(numImages / batchSize);

for batchIdx = 1:numBatches
    startIdx = (batchIdx - 1) * batchSize + 1;
    endIdx = min(batchIdx * batchSize, numImages);
    
    for i = startIdx:endIdx
        % Read and preprocess image
        img = readimage(imds, i);
        
        % Convert to grayscale and resize
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        grayImg = imresize(grayImg, [256 256]);
        grayImg = double(grayImg);
        
        % Extract Haralick features (all 13 features)
        haralickFeatures = extractHaralickFeatures(grayImg);
        
        % Extract Zernike moments (order 0-8, 25 features)
        zernikeFeatures = extractZernikeMoments(grayImg, 8);
        
        % Combine features
        features(i, :) = [haralickFeatures, zernikeFeatures];
    end
    
    if mod(batchIdx, 10) == 0 || batchIdx == numBatches
        fprintf('Processed %d/%d images...\n', endIdx, numImages);
    end
end

fprintf('Handcrafted feature extraction completed.\n');
end

function haralick = extractHaralickFeatures(grayImg)
% Extract all 13 Haralick texture features from GLCM

% Quantize image to 8 bits (256 gray levels)
if max(grayImg(:)) > 255
    grayImg = uint8(255 * mat2gray(grayImg));
else
    grayImg = uint8(grayImg);
end

% Calculate GLCM for 4 directions (0°, 45°, 90°, 135°)
offsets = [0 1; -1 1; -1 0; -1 -1];
glcm_all = graycomatrix(grayImg, 'Offset', offsets, 'Symmetric', true, 'NumLevels', 8);

% Average GLCM across all directions
glcm = double(glcm_all);
if ndims(glcm) == 3
    % If 3D (multiple directions), average across third dimension
    glcm = mean(glcm, 3);
end

% Normalize GLCM
glcm = glcm ./ sum(glcm(:));

% Calculate all 13 Haralick features
haralick = zeros(1, 13);

% Feature 1: Angular Second Moment (Energy)
haralick(1) = sum(glcm(:).^2);

% Feature 2: Contrast
[N, ~] = size(glcm);
contrast = 0;
for i = 1:N
    for j = 1:N
        contrast = contrast + (i - j)^2 * glcm(i, j);
    end
end
haralick(2) = contrast;

% Feature 3: Correlation
[i, j] = meshgrid(1:N, 1:N);
mu_i = sum(i(:) .* glcm(:));
mu_j = sum(j(:) .* glcm(:));
sigma_i = sqrt(sum((i(:) - mu_i).^2 .* glcm(:)));
sigma_j = sqrt(sum((j(:) - mu_j).^2 .* glcm(:)));

if sigma_i * sigma_j > 0
    correlation = sum((i(:) - mu_i) .* (j(:) - mu_j) .* glcm(:)) / (sigma_i * sigma_j);
else
    correlation = 0;
end
haralick(3) = correlation;

% Feature 4: Sum of Squares (Variance)
haralick(4) = sum((i(:) - mu_i).^2 .* glcm(:));

% Feature 5: Inverse Difference Moment (Homogeneity)
homogeneity = 0;
for i = 1:N
    for j = 1:N
        homogeneity = homogeneity + glcm(i, j) / (1 + (i - j)^2);
    end
end
haralick(5) = homogeneity;

% Features 6-8: Sum statistics
px_plus_y = zeros(2*N, 1);
for k = 2:2*N
    for i = 1:N
        for j = 1:N
            if (i + j) == k
                px_plus_y(k) = px_plus_y(k) + glcm(i, j);
            end
        end
    end
end

% Feature 6: Sum Average
haralick(6) = sum((2:2*N)' .* px_plus_y(2:end));

% Feature 8: Sum Entropy (calculated before variance)
sum_entropy = -sum(px_plus_y(px_plus_y > 0) .* log2(px_plus_y(px_plus_y > 0)));
haralick(8) = sum_entropy;

% Feature 7: Sum Variance
haralick(7) = sum(((2:2*N)' - haralick(6)).^2 .* px_plus_y(2:end));

% Feature 9: Entropy
haralick(9) = -sum(glcm(glcm > 0) .* log2(glcm(glcm > 0)));

% Features 10-11: Difference statistics
px_minus_y = zeros(N, 1);
for k = 0:N-1
    for i = 1:N
        for j = 1:N
            if abs(i - j) == k
                px_minus_y(k+1) = px_minus_y(k+1) + glcm(i, j);
            end
        end
    end
end

% Feature 10: Difference Variance
diff_mean = sum((0:N-1)' .* px_minus_y);
haralick(10) = sum(((0:N-1)' - diff_mean).^2 .* px_minus_y);

% Feature 11: Difference Entropy
haralick(11) = -sum(px_minus_y(px_minus_y > 0) .* log2(px_minus_y(px_minus_y > 0)));

% Features 12-13: Information measures of correlation
px = sum(glcm, 2);
py = sum(glcm, 1);

HX = -sum(px(px > 0) .* log2(px(px > 0)));
HY = -sum(py(py > 0) .* log2(py(py > 0)));
HXY = haralick(9); % Entropy

HXY1 = 0;
HXY2 = 0;
for i = 1:N
    for j = 1:N
        if glcm(i, j) > 0
            HXY1 = HXY1 - glcm(i, j) * log2(px(i) * py(j));
            if px(i) * py(j) > 0
                HXY2 = HXY2 - px(i) * py(j) * log2(px(i) * py(j));
            end
        end
    end
end

% Feature 12: Information Measure of Correlation 1
if max(HX, HY) > 0
    haralick(12) = (HXY - HXY1) / max(HX, HY);
else
    haralick(12) = 0;
end

% Feature 13: Maximal Correlation Coefficient
if HXY2 - HXY > 0
    haralick(13) = sqrt(1 - exp(-2 * (HXY2 - HXY)));
else
    haralick(13) = 0;
end
end

function zernike = extractZernikeMoments(grayImg, maxOrder)
% Extract Zernike moments up to specified order
% Returns 25 features for order 0-8

% Normalize image to unit circle
[N, M] = size(grayImg);
x = linspace(-1, 1, M);
y = linspace(-1, 1, N);
[X, Y] = meshgrid(x, y);
[theta, rho] = cart2pol(X, Y);

% Only consider points inside unit circle
mask = rho <= 1;
rho = rho(mask);
theta = theta(mask);
imgValues = grayImg(mask);

% Initialize Zernike moments
zernike = [];
order = 0;
repetition = 0;

% Calculate moments for orders 0 to maxOrder
while order <= maxOrder
    % Calculate for positive and negative repetitions
    for rep = -order:2:order
        if abs(rep) <= order && mod(order - abs(rep), 2) == 0
            % Calculate Zernike polynomial
            V = zernikePolynomial(rho, theta, order, rep);
            
            % Calculate moment
            moment = sum(conj(V) .* imgValues) / sum(mask(:));
            zernike = [zernike, abs(moment)]; % Use magnitude
        end
    end
    order = order + 1;
end

% Ensure we have exactly 25 features (pad with zeros if needed)
if length(zernike) < 25
    zernike = [zernike, zeros(1, 25 - length(zernike))];
elseif length(zernike) > 25
    zernike = zernike(1:25);
end
end

function V = zernikePolynomial(rho, theta, n, m)
% Calculate Zernike polynomial V_n^m(rho, theta)
% n: order, m: repetition

% Radial polynomial
R = zeros(size(rho));
for s = 0:((n - abs(m)) / 2)
    coeff = ((-1)^s * factorial(n - s)) / ...
            (factorial(s) * factorial((n + abs(m))/2 - s) * ...
             factorial((n - abs(m))/2 - s));
    R = R + coeff * rho.^(n - 2*s);
end

% Angular component
if m >= 0
    V = R .* exp(1i * m * theta);
else
    V = R .* exp(-1i * abs(m) * theta);
end
end

