function augimds = applyImageAugmentation(imds, options)
%APPLYIMAGEAUGMENTATION Apply image-level data augmentation
%   AUGIMDS = applyImageAugmentation(IMDS, OPTIONS) creates an augmented
%   image datastore with various transformations applied.
%
%   Inputs:
%       imds - Image datastore
%       options - Structure with augmentation options:
%           .enabled - Enable augmentation (default: true)
%           .rotation - Rotation range in degrees (default: [-15, 15])
%           .translation - Translation range in pixels (default: [-10, 10])
%           .scale - Scale range (default: [0.9, 1.1])
%           .flip - 'horizontal', 'vertical', 'both', or 'none' (default: 'horizontal')
%           .brightness - Brightness range (default: [0.8, 1.2])
%           .contrast - Contrast range (default: [0.8, 1.2])
%           .saturation - Saturation range (default: [0.8, 1.2])
%           .inputSize - Output image size (default: [224, 224, 3])
%
%   Outputs:
%       augimds - Augmented image datastore

if nargin < 2
    options = struct();
end

% Default options
if ~isfield(options, 'enabled'), options.enabled = true; end
if ~isfield(options, 'rotation'), options.rotation = [-15, 15]; end
if ~isfield(options, 'translation'), options.translation = [-10, 10]; end
if ~isfield(options, 'scale'), options.scale = [0.9, 1.1]; end
if ~isfield(options, 'flip'), options.flip = 'horizontal'; end
if ~isfield(options, 'brightness'), options.brightness = [0.8, 1.2]; end
if ~isfield(options, 'contrast'), options.contrast = [0.8, 1.2]; end
if ~isfield(options, 'saturation'), options.saturation = [0.8, 1.2]; end
if ~isfield(options, 'inputSize'), options.inputSize = [224, 224, 3]; end

if ~options.enabled
    % Return non-augmented datastore
    augimds = augmentedImageDatastore(options.inputSize, imds, ...
        'ColorPreprocessing', 'gray2rgb');
    return;
end

% Create image data augmentation pipeline
imageAugmenter = imageDataAugmenter(...
    'RandRotation', options.rotation, ...
    'RandXTranslation', options.translation, ...
    'RandYTranslation', options.translation, ...
    'RandXReflection', strcmpi(options.flip, 'horizontal') || strcmpi(options.flip, 'both'), ...
    'RandYReflection', strcmpi(options.flip, 'vertical') || strcmpi(options.flip, 'both'), ...
    'RandXScale', options.scale, ...
    'RandYScale', options.scale, ...
    'RandBrightness', options.brightness, ...
    'RandContrast', options.contrast, ...
    'RandSaturation', options.saturation);

% Create augmented image datastore
augimds = augmentedImageDatastore(options.inputSize, imds, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

fprintf('Image augmentation enabled:\n');
fprintf('  Rotation: [%.1f, %.1f] degrees\n', options.rotation(1), options.rotation(2));
fprintf('  Translation: [%.1f, %.1f] pixels\n', options.translation(1), options.translation(2));
fprintf('  Scale: [%.2f, %.2f]\n', options.scale(1), options.scale(2));
fprintf('  Flip: %s\n', options.flip);
fprintf('  Brightness: [%.2f, %.2f]\n', options.brightness(1), options.brightness(2));
fprintf('  Contrast: [%.2f, %.2f]\n', options.contrast(1), options.contrast(2));

end

