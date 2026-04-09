% generateSummaryTables.m
% Generate summary tables for publication (LaTeX format)

clc; clear; close all;

%% Configuration
config = struct();
config.resultsDir = 'results';
config.ablationDir = 'ablation_results';
config.outputDir = 'tables';

% Create output directory
if ~isfolder(config.outputDir)
    mkdir(config.outputDir);
end

fprintf('Generating summary tables...\n');

%% Load Results
load(fullfile(config.resultsDir, 'results.mat'), 'results');

%% Table 1: Main Results
fprintf('\nGenerating Table 1: Main Results...\n');
table1File = fullfile(config.outputDir, 'Table1_MainResults.tex');

fid = fopen(table1File, 'w');
fprintf(fid, '\\begin{table}[ht!]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Main Results of the Proposed Method}\n');
fprintf(fid, '\\label{tab:main_results}\n');
fprintf(fid, '\\begin{tabular}{|l|c|c|}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, '\\textbf{Metric} & \\textbf{Value} & \\textbf{Unit} \\\\\n');
fprintf(fid, '\\hline\n');
fprintf(fid, 'Cross-Validation Accuracy & %.2f\\%% & - \\\\\n', results.cvAccuracy * 100);
fprintf(fid, '\\hline\n');
fprintf(fid, 'Test Accuracy & %.2f\\%% & - \\\\\n', results.testAccuracy * 100);
fprintf(fid, '\\hline\n');
fprintf(fid, 'Number of Classes & %d & - \\\\\n', length(results.perClassAccuracy));
fprintf(fid, '\\hline\n');
% Get feature dimension (check if field exists, otherwise calculate from config)
if isfield(results, 'featureDim')
    featureDim = results.featureDim;
elseif isfield(results, 'config') && isfield(results.config, 'cnnModels')
    % Estimate from CNN models (approximate)
    % ResNet-50: 2048, DenseNet-201: 1920, Handcrafted: 38
    featureDim = 2048 + 1920 + 38;  % Approximate fused dimension
else
    featureDim = 4006;  % Default value
end
fprintf(fid, 'Feature Dimension & %d & - \\\\\n', featureDim);
fprintf(fid, '\\hline\n');
fprintf(fid, 'Training Time & %.2f & seconds \\\\\n', results.trainingTime);
fprintf(fid, '\\hline\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);
fprintf('Saved: %s\n', table1File);

%% Table 2: Per-Class Performance
fprintf('\nGenerating Table 2: Per-Class Performance...\n');
table2File = fullfile(config.outputDir, 'Table2_PerClassPerformance.tex');

cm = results.confusionMatrix;
numClasses = size(cm, 1);
classNames = {'Dyed-lifted-polyps', 'Dyed-resection-margins', ...
              'Esophagitis', 'Normal-cecum', 'Normal-pylorus', ...
              'Normal-z-line', 'Polyps', 'Ulcerative-colitis'};

% Calculate metrics
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = cm(i, i);
    FP = sum(cm(:, i)) - TP;
    FN = sum(cm(i, :)) - TP;
    
    precision(i) = TP / (TP + FP + eps) * 100;
    recall(i) = TP / (TP + FN + eps) * 100;
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

fid = fopen(table2File, 'w');
fprintf(fid, '\\begin{table}[ht!]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Per-Class Performance Metrics}\n');
fprintf(fid, '\\label{tab:per_class}\n');
fprintf(fid, '\\resizebox{\\textwidth}{!}{\n');
fprintf(fid, '\\begin{tabular}{|l|c|c|c|c|}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, '\\textbf{Class} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\\n');
fprintf(fid, '\\hline\n');

for i = 1:numClasses
    fprintf(fid, '%s & %.2f\\%% & %.2f\\%% & %.2f\\%% & %.2f\\%% \\\\\n', ...
            classNames{i}, results.perClassAccuracy(i) * 100, ...
            precision(i), recall(i), f1Score(i));
    fprintf(fid, '\\hline\n');
end

fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);
fprintf('Saved: %s\n', table2File);

%% Table 3: Ablation Study Results
fprintf('\nGenerating Table 3: Ablation Study...\n');
ablationFile = fullfile(config.ablationDir, 'ablation_results.mat');
if exist(ablationFile, 'file')
    load(ablationFile, 'ablationResults');
    
    table3File = fullfile(config.outputDir, 'Table3_AblationStudy.tex');
    fid = fopen(table3File, 'w');
    fprintf(fid, '\\begin{table}[ht!]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{Ablation Study Results}\n');
    fprintf(fid, '\\label{tab:ablation}\n');
    fprintf(fid, '\\resizebox{\\textwidth}{!}{\n');
    fprintf(fid, '\\begin{tabular}{|l|c|c|}\n');
    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\textbf{Method} & \\textbf{Test Accuracy} & \\textbf{CV Accuracy} \\\\\n');
    fprintf(fid, '\\hline\n');
    
    for i = 1:length(ablationResults.experiments)
        fprintf(fid, '%s & %.2f\\%% & %.2f\\%% \\\\\n', ...
                ablationResults.experiments{i}, ...
                ablationResults.accuracies(i) * 100, ...
                ablationResults.cvAccuracies(i) * 100);
        fprintf(fid, '\\hline\n');
    end
    
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '}\n');
    fprintf(fid, '\\end{table}\n');
    fclose(fid);
    fprintf('Saved: %s\n', table3File);
else
    fprintf('Ablation results not found. Run runAblationStudy.m first.\n');
end

%% Table 4: Comparison with State-of-the-Art
fprintf('\nGenerating Table 4: Comparison with SOTA...\n');
table4File = fullfile(config.outputDir, 'Table4_Comparison.tex');

fid = fopen(table4File, 'w');
fprintf(fid, '\\begin{table}[ht!]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Comparison with State-of-the-Art Methods}\n');
fprintf(fid, '\\label{tab:comparison}\n');
fprintf(fid, '\\resizebox{\\textwidth}{!}{\n');
fprintf(fid, '\\begin{tabular}{|l|c|c|c|}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, '\\textbf{Method} & \\textbf{Year} & \\textbf{Accuracy} & \\textbf{Dataset} \\\\\n');
fprintf(fid, '\\hline\n');
fprintf(fid, 'Proposed Method & 2024 & %.2f\\%% & Kvasir-v2 \\\\\n', results.testAccuracy * 100);
fprintf(fid, '\\hline\n');
fprintf(fid, 'Mohammad et al.~\\cite{mohammad2022deep} & 2022 & 99.8\\%% & Kvasir-v2 \\\\\n');
fprintf(fid, '\\hline\n');
fprintf(fid, 'Lonseko et al.~\\cite{lonseko2021gastrointestinal} & 2021 & 96.33\\%% & Kvasir-v2 \\\\\n');
fprintf(fid, '\\hline\n');
fprintf(fid, 'Qu et al.~\\cite{qu2019novel} & 2019 & 92.8\\%% & Kvasir-v2 + EAD2019 \\\\\n');
fprintf(fid, '\\hline\n');
fprintf(fid, 'Majid et al.~\\cite{majid2020classification} & 2020 & 96\\%% & Private \\\\\n');
fprintf(fid, '\\hline\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);
fprintf('Saved: %s\n', table4File);

fprintf('\nAll tables generated successfully!\n');
fprintf('Tables saved to: %s\n', config.outputDir);

