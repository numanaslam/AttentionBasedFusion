%% runAblationStudy.m
% Updated ablation study with statistical validation and fine-tuning comparison

function runAblationStudy(config)
%RUNABLATIONSTUDY Systematic ablation with statistical validation
%   Runs all configurations from Table 4 with multi-seed validation

if nargin < 1
    config = loadDefaultConfig();
end

% Define ablation configurations
ablationConfigs = {
    % Name, CNN Models, Fusion Method, Classifier, Fine-tune
    'CNN Only (ResNet-50)', {'resnet50'}, 'none', 'svm', false;
    'CNN Only (DenseNet-201)', {'densenet201'}, 'none', 'svm', false;
    'Handcrafted Only', {}, 'none', 'svm', false;
    'Simple Concatenation', {'resnet50', 'densenet201'}, 'concat', 'svm', false;
    'Weighted Fusion', {'resnet50', 'densenet201'}, 'weighted', 'svm', false;
    'Attention-Based Fusion', {'resnet50', 'densenet201'}, 'attention', 'svm', false;
    'Attention + SMOTE', {'resnet50', 'densenet201'}, 'attention', 'svm', true;
    'Attention + SVM', {'resnet50', 'densenet201'}, 'attention', 'svm', false;
    'Attention + Ensemble', {'resnet50', 'densenet201'}, 'attention', 'ensemble_multi', false;
};

% Add fine-tuning ablation row if enabled
if config.fineTuningAblation.enabled
    ablationConfigs = [ablationConfigs; {
        'Fine-tuned CNN Features', {'resnet50', 'densenet201'}, 'attention', 'ensemble_multi', true;
    }];
end

% Results storage
resultsTable = table('Size', [size(ablationConfigs, 1), 4], ...
    'VariableTypes', {'string', 'double', 'double', 'string'}, ...
    'VariableNames', {'Configuration', 'TestAccuracy_Mean', 'TestAccuracy_Std', 'Notes'});

fprintf('Starting ablation study with %d configurations...\n\n', size(ablationConfigs, 1));

for cfgIdx = 1:size(ablationConfigs, 1)
    cfg = ablationConfigs{cfgIdx, :};
    configName = cfg{1};
    
    fprintf('\n=== Configuration %d/%d: %s ===\n', cfgIdx, size(ablationConfigs, 1), configName);
    
    % Update config for this ablation
    config.cnnModels = cfg{2};
    config.fusionMethod = cfg{3};
    config.classifier.type = cfg{4};
    config.augmentation.enabled = strcmp(cfg{5}, 'true') || strcmp(cfg{5}, true);
    config.fineTuningAblation.enabled = strcmp(cfg{6}, 'true') || strcmp(cfg{6}, true);
    
    % Run with statistical validation
    if config.statisticalValidation.enabled
        [meanAcc, stdAcc] = runConfigWithValidation(config, configName);
    else
        % Single split for faster testing
        [meanAcc, stdAcc] = runConfigSingleSplit(config, configName);
    end
    
    % Store results
    resultsTable.Configuration(cfgIdx) = configName;
    resultsTable.TestAccuracy_Mean(cfgIdx) = meanAcc;
    resultsTable.TestAccuracy_Std(cfgIdx) = stdAcc;
    resultsTable.Notes(cfgIdx) = cfg{7};
    
    fprintf('Result: %.2f%% ± %.2f%%\n', meanAcc*100, stdAcc*100);
end

% Save ablation results
save(fullfile(config.outputDir, 'ablation_results.mat'), 'resultsTable', '-v7.3');

% Generate ablation comparison figure
generateAblationFigure(resultsTable, config.outputDir);

fprintf('\nAblation study completed. Results saved.\n');
end

function [meanAcc, stdAcc] = runConfigWithValidation(config, configName)
% Helper: Run a single configuration with multi-seed validation
numSeeds = config.statisticalValidation.numSeeds;
accuracies = zeros(numSeeds, 1);

for seed = 1:numSeeds
    rng(seed);
    % Run pipeline for this seed (simplified)
    acc = runSingleExperiment(config, seed);  % Your actual pipeline call
    accuracies(seed) = acc;
    fprintf('  Seed %d: %.2f%%\n', seed, acc*100);
end

meanAcc = mean(accuracies);
stdAcc = std(accuracies);
end

function generateAblationFigure(resultsTable, outputDir)
% Generate ablation comparison bar chart with error bars
figure('Position', [100, 100, 1200, 700], 'Color', 'white');

configs = resultsTable.Configuration;
means = resultsTable.TestAccuracy_Mean * 100;
stds = resultsTable.TestAccuracy_Std * 100;

x = 1:height(resultsTable);
bar(x, means, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'none');
hold on;
errorbar(x, means, stds, 'k.', 'LineWidth', 1.5, 'CapSize', 8);

xlabel('Configuration', 'FontSize', 12);
ylabel('Test Accuracy (%)', 'FontSize', 12);
title('Ablation Study: Component Contributions', 'FontSize', 14, 'FontWeight', 'bold');
xticks(x);
xticklabels(configs, 'Interpreter', 'none');
xtickangle(45);
grid on;
ylim([0, 100]);

% Highlight best result
[~, bestIdx] = max(means);
text(bestIdx, means(bestIdx) + 2, sprintf('%.1f%%', means(bestIdx)), ...
    'HorizontalAlignment', 'center', 'Color', 'red', 'FontWeight', 'bold', 'FontSize', 11);

saveas(gcf, fullfile(outputDir, 'AblationStudy_Comparison.png'), 'png');
close(gcf);
end