function [metrics, stats] = evaluateMetrics(predictions, labels, scores, config)
%EVALUATEMETRICS Compute comprehensive evaluation metrics with statistical tests
%   [METRICS, STATS] = evaluateMetrics(PRED, LAB, SCORES, CONFIG) computes
%   accuracy, precision, recall, F1, MCC, ROC-AUC, and statistical tests.
%
%   Inputs:
%       predictions - Predicted class labels
%       labels - True class labels
%       scores - Classification scores/probabilities (for ROC analysis)
%       config - Configuration struct with options
%
%   Outputs:
%       metrics - Struct with all computed metrics
%       stats - Struct with statistical test results

% Basic metrics
confMat = confusionmat(labels, predictions);
accuracy = sum(diag(confMat)) / sum(confMat(:));

% Per-class metrics
classNames = categories(labels);
numClasses = length(classNames);
metrics.precision = zeros(numClasses, 1);
metrics.recall = zeros(numClasses, 1);
metrics.f1 = zeros(numClasses, 1);
metrics.fnr = zeros(numClasses, 1);  % False Negative Rate

for c = 1:numClasses
    TP = confMat(c, c);
    FP = sum(confMat(:, c)) - TP;
    FN = sum(confMat(c, :)) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;
    
    metrics.precision(c) = TP / (TP + FP + eps);
    metrics.recall(c) = TP / (TP + FN + eps);
    metrics.f1(c) = 2 * metrics.precision(c) * metrics.recall(c) / ...
                    (metrics.precision(c) + metrics.recall(c) + eps);
    metrics.fnr(c) = FN / (TP + FN + eps);  % False Negative Rate
end

% Macro-averaged metrics
metrics.macroPrecision = mean(metrics.precision);
metrics.macroRecall = mean(metrics.recall);
metrics.macroF1 = mean(metrics.f1);
metrics.macroFNR = mean(metrics.fnr);

% Matthews Correlation Coefficient (MCC) - robust for imbalanced data
if numClasses == 2
    % Binary case
    TP = confMat(2, 2);
    TN = confMat(1, 1);
    FP = confMat(1, 2);
    FN = confMat(2, 1);
    metrics.mcc = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + eps);
else
    % Multi-class MCC (simplified)
    metrics.mcc = computeMultiClassMCC(confMat);
end

% ROC-AUC (one-vs-rest for multi-class)
if exist('scores', 'var') && ~isempty(scores)
    metrics.rocAUC = computeMultiClassAUC(scores, labels, classNames);
else
    metrics.rocAUC = NaN;
end

% Statistical tests - NEW
if isfield(config, 'baselinePredictions') && ~isempty(config.baselinePredictions)
    % McNemar's test comparing two classifiers
    [stats.mcNemarP, stats.mcNemarStat] = mcnemarTest(predictions, labels, ...
        config.baselinePredictions);
    fprintf('McNemar''s test: p = %.4f\n', stats.mcNemarP);
else
    stats.mcNemarP = NaN;
    stats.mcNemarStat = NaN;
end

% Confidence intervals via bootstrap (optional)
if isfield(config, 'computeCI') && config.computeCI
    [metrics.accCI, metrics.f1CI] = computeBootstrapCI(predictions, labels, ...
        metrics.f1, config.numBootstraps);
else
    metrics.accCI = [accuracy, accuracy];
    metrics.f1CI = [metrics.macroF1, metrics.macroF1];
end

% Print summary
fprintf('\nEvaluation Metrics:\n');
fprintf('Accuracy: %.2f%% [%.2f%%, %.2f%%]\n', accuracy*100, ...
    metrics.accCI(1)*100, metrics.accCI(2)*100);
fprintf('Macro F1: %.2f%% [%.2f%%, %.2f%%]\n', metrics.macroF1*100, ...
    metrics.f1CI(1)*100, metrics.f1CI(2)*100);
fprintf('MCC: %.3f\n', metrics.mcc);
if ~isnan(metrics.rocAUC)
    fprintf('ROC-AUC: %.3f\n', metrics.rocAUC);
end
fprintf('Macro FNR: %.2f%%\n', metrics.macroFNR*100);

end

function mcc = computeMultiClassMCC(confMat)
% Simplified multi-class MCC computation
% Reference: Gorodkin 2004
K = size(confMat, 1);
t = sum(confMat, 2);  % True counts per class
p = sum(confMat, 1)'; % Predicted counts per class

c = sum(diag(confMat));  % Correct predictions
s = sum(confMat(:));     % Total samples

numerator = c*s - sum(t.*p);
denominator = sqrt(s^2 - sum(p.^2)) * sqrt(s^2 - sum(t.^2));

mcc = numerator / (denominator + eps);
end

function auc = computeMultiClassAUC(scores, labels, classNames)
% Compute macro-averaged ROC-AUC for multi-class
numClasses = length(classNames);
aucValues = zeros(numClasses, 1);

for c = 1:numClasses
    % One-vs-rest: class c vs. all others
    binaryLabels = double(labels == classNames(c));
    binaryScores = scores(:, c);
    
    % Compute ROC curve and AUC
    [fpr, tpr, ~] = perfcurve(binaryLabels, binaryScores, 1);
    aucValues(c) = trapz(fpr, tpr);
end

auc = mean(aucValues);  % Macro-average
end

function [pValue, stat] = mcnemarTest(pred1, labels, pred2)
% McNemar's test for paired classification results
% Tests if two classifiers have significantly different error rates

% Build contingency table
n00 = sum(pred1 == labels & pred2 == labels);  % Both correct
n01 = sum(pred1 == labels & pred2 ~= labels);  % Only pred1 correct
n10 = sum(pred1 ~= labels & pred2 == labels);  % Only pred2 correct
n11 = sum(pred1 ~= labels & pred2 ~= labels);  % Both wrong

% McNemar's chi-square statistic (with continuity correction)
if (n01 + n10) == 0
    stat = 0;
    pValue = 1;
else
    stat = (abs(n01 - n10) - 1)^2 / (n01 + n10);
    pValue = 1 - chi2cdf(stat, 1);
end
end

function [accCI, f1CI] = computeBootstrapCI(predictions, labels, f1Values, numBootstraps)
% Compute 95% confidence intervals via bootstrap resampling
if nargin < 4
    numBootstraps = 1000;
end

n = length(labels);
accBoot = zeros(numBootstraps, 1);
f1Boot = zeros(numBootstraps, 1);

for b = 1:numBootstraps
    % Resample with replacement
    idx = randsample(n, n, true);
    bootPred = predictions(idx);
    bootLabels = labels(idx);
    
    % Compute metrics for bootstrap sample
    accBoot(b) = mean(bootPred == bootLabels);
    
    % Simplified F1 for bootstrap (macro-average)
    confMat = confusionmat(bootLabels, bootPred);
    prec = diag(confMat) ./ (sum(confMat, 1)' + eps);
    rec = diag(confMat) ./ (sum(confMat, 2) + eps);
    f1 = 2 * prec .* rec ./ (prec + rec + eps);
    f1Boot(b) = mean(f1);
end

% 95% CI via percentile method
accCI = prctile(accBoot, [2.5, 97.5]);
f1CI = prctile(f1Boot, [2.5, 97.5]);
end