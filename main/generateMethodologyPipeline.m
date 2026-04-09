function generateMethodologyPipeline(outputPath)
%GENERATEMETHODOLOGYPIPELINE Generate the complete methodology pipeline figure
%   generateMethodologyPipeline(OUTPUTPATH) creates a publication-quality
%   figure showing the complete methodology pipeline from input images to
%   final predictions.
%
%   Input:
%       outputPath - Path to save the figure (default: 'results/Figure1_MethodologyPipeline.png')

if nargin < 1
    outputPath = fullfile('results', 'Figure1_MethodologyPipeline.png');
end

% Create figure
fig = figure('Position', [100, 100, 1400, 1000], 'Color', 'white');
hold on;
axis off;
axis equal;

% Define colors
colorInput = [225/255, 245/255, 255/255];      % Light blue
colorFeature = [255/255, 243/255, 205/255];     % Light yellow
colorAttention = [255/255, 193/255, 7/255];     % Amber
colorAugment = [248/255, 215/255, 218/255];     % Light red
colorClassify = [209/255, 236/255, 241/255];    % Light cyan
colorOutput = [212/255, 237/255, 218/255];      % Light green

% Define positions (normalized coordinates)
yStart = 0.95;
yStep = 0.12;
xCenter = 0.5;
xLeft = 0.15;
xRight = 0.85;

% Stage 1: Input
y = yStart;
rectangle('Position', [xCenter-0.08, y-0.04, 0.16, 0.08], ...
    'FaceColor', colorInput, 'EdgeColor', 'k', 'LineWidth', 2);
text(xCenter, y, {'Kvasir-v2 Dataset', '4000 Images, 8 Classes'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');

% Split
y = y - yStep;
arrow([xCenter, y+0.04], [xCenter, y-0.02], 'Color', 'k', 'LineWidth', 1.5);
text(xCenter, y, 'Train/Test Split (80/20)', ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

% Stage 2: Feature Extraction
y = y - yStep;
% CNN Features
rectangle('Position', [xLeft-0.06, y-0.04, 0.12, 0.08], ...
    'FaceColor', colorFeature, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xLeft, y, {'ResNet-50', '2048-D'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

rectangle('Position', [xLeft-0.06, y-0.15, 0.12, 0.08], ...
    'FaceColor', colorFeature, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xLeft, y-0.11, {'DenseNet-201', '1920-D'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

% Handcrafted Features
rectangle('Position', [xRight-0.06, y-0.04, 0.12, 0.08], ...
    'FaceColor', colorFeature, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xRight, y, {'Haralick', '13 Features'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

rectangle('Position', [xRight-0.06, y-0.15, 0.12, 0.08], ...
    'FaceColor', colorFeature, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xRight, y-0.11, {'Zernike', '25 Features'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

% Arrows from split
arrow([xCenter-0.02, y+0.12], [xLeft+0.06, y+0.04], 'Color', 'k', 'LineWidth', 1);
arrow([xCenter+0.02, y+0.12], [xRight-0.06, y+0.04], 'Color', 'k', 'LineWidth', 1);

% Stage 3: Normalization
y = y - 0.25;
text(xCenter, y, 'Z-score Normalization', ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontStyle', 'italic');
arrow([xLeft, y+0.08], [xCenter-0.08, y+0.02], 'Color', 'k', 'LineWidth', 1);
arrow([xRight, y+0.08], [xCenter+0.08, y+0.02], 'Color', 'k', 'LineWidth', 1);

% Stage 4: Concatenation
y = y - yStep;
rectangle('Position', [xCenter-0.1, y-0.04, 0.2, 0.08], ...
    'FaceColor', colorFeature, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xCenter, y, {'Concatenate CNN Features', '3968-D'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9);
arrow([xCenter, y+0.12], [xCenter, y+0.04], 'Color', 'k', 'LineWidth', 1.5);

% Stage 5: Attention Mechanism
y = y - yStep;
rectangle('Position', [xCenter-0.12, y-0.06, 0.24, 0.12], ...
    'FaceColor', colorAttention, 'EdgeColor', 'k', 'LineWidth', 2);
text(xCenter, y+0.02, {'Attention Mechanism', 'Compute α, β (Eq. 2-11)'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
text(xCenter, y-0.02, {'Variance + Correlation → Softmax'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 8, 'FontStyle', 'italic');
arrow([xCenter, y+0.12], [xCenter, y+0.06], 'Color', 'k', 'LineWidth', 1.5);

% Arrows to attention
arrow([xLeft, y+0.12], [xCenter-0.12, y+0.06], 'Color', 'k', 'LineWidth', 1);
arrow([xRight, y+0.12], [xCenter+0.12, y+0.06], 'Color', 'k', 'LineWidth', 1);

% Stage 6: Fusion
y = y - yStep;
rectangle('Position', [xCenter-0.1, y-0.04, 0.2, 0.08], ...
    'FaceColor', colorAttention, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xCenter, y, {'Fused Features', '4006-D'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
arrow([xCenter, y+0.12], [xCenter, y+0.04], 'Color', 'k', 'LineWidth', 1.5);

% Stage 7: SMOTE
y = y - yStep;
rectangle('Position', [xCenter-0.1, y-0.04, 0.2, 0.08], ...
    'FaceColor', colorAugment, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xCenter, y, {'SMOTE Augmentation', '6400 Samples'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9);
arrow([xCenter, y+0.12], [xCenter, y+0.04], 'Color', 'k', 'LineWidth', 1.5);

% Stage 8: Classification
y = y - yStep;
% Branch to classifiers
arrow([xCenter, y+0.12], [xCenter-0.08, y+0.04], 'Color', 'k', 'LineWidth', 1);
arrow([xCenter, y+0.12], [xCenter+0.08, y+0.04], 'Color', 'k', 'LineWidth', 1);

rectangle('Position', [xCenter-0.18, y-0.04, 0.16, 0.08], ...
    'FaceColor', colorClassify, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xCenter-0.1, y, {'SVM', '91.00%'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

rectangle('Position', [xCenter+0.02, y-0.04, 0.16, 0.08], ...
    'FaceColor', colorClassify, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(xCenter+0.1, y, {'Ensemble', '90.75%'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

% Stage 9: Output
y = y - yStep;
arrow([xCenter-0.08, y+0.12], [xCenter-0.02, y+0.04], 'Color', 'k', 'LineWidth', 1);
arrow([xCenter+0.08, y+0.12], [xCenter+0.02, y+0.04], 'Color', 'k', 'LineWidth', 1);

rectangle('Position', [xCenter-0.1, y-0.04, 0.2, 0.08], ...
    'FaceColor', colorOutput, 'EdgeColor', 'k', 'LineWidth', 2);
text(xCenter, y, {'Final Predictions', '8 Classes'}, ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');

% Add title
annotation('textbox', [0.1, 0.95, 0.8, 0.04], ...
    'String', 'Complete Methodology Pipeline: Attention-Based Multi-Modal Feature Fusion', ...
    'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', ...
    'EdgeColor', 'none');

% Save figure
[outputDir, ~, ~] = fileparts(outputPath);
if ~isempty(outputDir) && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

print(fig, outputPath, '-dpng', '-r300');
fprintf('Methodology pipeline figure saved to: %s\n', outputPath);

end

function arrow(start, stop, varargin)
% Simple arrow drawing function
p = inputParser;
addParameter(p, 'Color', 'k');
addParameter(p, 'LineWidth', 1);
parse(p, varargin{:});

dx = stop(1) - start(1);
dy = stop(2) - start(2);
len = sqrt(dx^2 + dy^2);

% Arrow line
line([start(1), stop(1)], [start(2), stop(2)], ...
    'Color', p.Results.Color, 'LineWidth', p.Results.LineWidth);

% Arrowhead
headLen = 0.015;
headAngle = pi/6;
angle = atan2(dy, dx);

x1 = stop(1) - headLen * cos(angle - headAngle);
y1 = stop(2) - headLen * sin(angle - headAngle);
x2 = stop(1) - headLen * cos(angle + headAngle);
y2 = stop(2) - headLen * sin(angle + headAngle);

line([stop(1), x1], [stop(2), y1], ...
    'Color', p.Results.Color, 'LineWidth', p.Results.LineWidth);
line([stop(1), x2], [stop(2), y2], ...
    'Color', p.Results.Color, 'LineWidth', p.Results.LineWidth);
end

