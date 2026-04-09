function testAcc = evaluateClassifier(classifier, featuresTest, labelsTest, trainMean, trainStd)
%EVALUATECLASSIFIER Helper function for ablation study
%   Evaluates a classifier on test data

    % Normalize test features using training statistics
    featuresTestNorm = (featuresTest - trainMean) ./ (trainStd + eps);
    
    % Clip extreme values (consistent with training)
    featuresTestNorm = max(min(featuresTestNorm, 5), -5);
    
    % Predict
    if isstruct(classifier) && isfield(classifier, 'classifiers')
        % Ensemble classifier
        predictions = predictEnsemble(classifier, featuresTestNorm);
    else
        % Single classifier
        predictions = predict(classifier, featuresTestNorm);
    end
    testAcc = mean(predictions == labelsTest);
end

