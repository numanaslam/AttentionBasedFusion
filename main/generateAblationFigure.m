function generateAblationFigure(ablationResults, outputDir)
%GENERATEABLATIONFIGURE Generate visualization for ablation study results

    fig = figure('Position', [100, 100, 1200, 700]);
    
    numExps = length(ablationResults.experiments);
    x = 1:numExps;
    
    % Create grouped bar chart
    barData = [ablationResults.accuracies' * 100, ablationResults.cvAccuracies' * 100];
    b = bar(x, barData, 'grouped');
    b(1).FaceColor = [0.2 0.6 0.8];  % Test accuracy
    b(2).FaceColor = [0.8 0.4 0.2];  % CV accuracy
    
    set(gca, 'XTick', x, 'XTickLabel', ablationResults.experiments, 'FontSize', 9);
    xtickangle(45);
    ylabel('Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Ablation Study Results', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'Test Accuracy', 'CV Accuracy'}, 'Location', 'best', 'FontSize', 10);
    grid on;
    ylim([0, 100]);
    
    % Add value labels
    for i = 1:numExps
        text(i - 0.15, ablationResults.accuracies(i) * 100 + 2, ...
             sprintf('%.1f', ablationResults.accuracies(i) * 100), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Rotation', 90);
        text(i + 0.15, ablationResults.cvAccuracies(i) * 100 + 2, ...
             sprintf('%.1f', ablationResults.cvAccuracies(i) * 100), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Rotation', 90);
    end
    
    saveas(fig, fullfile(outputDir, 'AblationStudy_Results.png'), 'png');
    saveas(fig, fullfile(outputDir, 'AblationStudy_Results.fig'), 'fig');
    fprintf('Ablation study figure saved.\n');
end

