% generateAllPublicationFigures.m
% Master script to generate all publication figures and run ablation study
% Run this script to generate everything needed for publication
%
% NOTE: This script will train multiple classifiers for the ablation study.
% If you only want figures (no training), set runAblationStudy = false below.

clc; close all;

%% Configuration
runAblation = false;  % Set to true to run ablation study (trains 8+ models, takes 30-60 min)
                      % Set to false to skip ablation study and only generate figures

fprintf('========================================\n');
fprintf('Generating All Publication Materials\n');
fprintf('========================================\n\n');

%% Step 1: Generate Main Figures
fprintf('Step 1: Generating main publication figures...\n');
try
    % Save runAblation before calling generatePublicationFigures (which uses clear)
    shouldRunAblation = runAblation;
    generatePublicationFigures;
    % Restore runAblation after generatePublicationFigures
    runAblation = shouldRunAblation;
    fprintf('✓ Main figures generated successfully.\n\n');
catch ME
    % Restore runAblation in case of error
    if ~exist('runAblation', 'var')
        runAblation = false;
    end
    fprintf('✗ Error generating main figures: %s\n\n', ME.message);
end

%% Step 2: Run Ablation Study (Optional)
if exist('runAblation', 'var') && runAblation
    fprintf('Step 2: Running ablation study...\n');
    fprintf('⚠️  WARNING: This will train 8+ classifiers and may take 30-60 minutes!\n');
    fprintf('   The ablation study trains models with different configurations to compare them.\n');
    userInput = input('   Continue? (y/n): ', 's');
    if lower(userInput) == 'y' || lower(userInput) == 'yes'
        try
            runAblationStudy;  % Run the ablation study script
            fprintf('✓ Ablation study completed successfully.\n\n');
        catch ME
            fprintf('✗ Error running ablation study: %s\n\n', ME.message);
        end
    else
        fprintf('⚠️  Ablation study skipped by user.\n\n');
    end
else
    fprintf('Step 2: Skipping ablation study (runAblation = false).\n');
    fprintf('   To run ablation study, set runAblation = true in this script.\n');
    fprintf('   Or run runAblationStudy.m directly.\n\n');
end

%% Step 3: Generate Sample Images
fprintf('Step 3: Generating sample images...\n');
try
    generateSampleImages;
    fprintf('✓ Sample images generated successfully.\n\n');
catch ME
    fprintf('✗ Error generating sample images: %s\n\n', ME.message);
end

%% Step 4: Generate Summary Table
fprintf('Step 4: Generating summary tables...\n');
try
    generateSummaryTables;
    fprintf('✓ Summary tables generated successfully.\n\n');
catch ME
    fprintf('✗ Error generating summary tables: %s\n\n', ME.message);
end

fprintf('========================================\n');
fprintf('All Publication Materials Generated!\n');
fprintf('========================================\n');
fprintf('Check the following directories:\n');
fprintf('  - figures/ : Main publication figures\n');
if exist('runAblation', 'var') && runAblation
    fprintf('  - ablation_results/ : Ablation study results\n');
end
fprintf('  - figures/samples/ : Sample images\n');
fprintf('  - tables/ : Summary tables\n');
fprintf('========================================\n');
fprintf('\nNote: To generate ablation study results, run:\n');
fprintf('  runAblationStudy\n');
fprintf('Or set runAblation = true in this script.\n');

