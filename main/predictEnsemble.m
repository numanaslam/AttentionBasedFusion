function predictions = predictEnsemble(ensembleClassifier, features)
%PREDICTENSEMBLE Predict using ensemble of classifiers
%   PREDICTIONS = predictEnsemble(ENSEMBLE, FEATURES) combines predictions
%   from multiple classifiers in an ensemble.
%
%   Inputs:
%       ensembleClassifier - Structure containing trained classifiers and weights
%       features - Feature matrix (N x D)
%
%   Outputs:
%       predictions - Combined predictions (N x 1 categorical)

numSamples = size(features, 1);
numClassifiers = length(ensembleClassifier.classifiers);
classNames = ensembleClassifier.classNames;
numClasses = length(classNames);

% Get predictions from all classifiers
allPredictions = cell(numClassifiers, 1);
allScores = zeros(numSamples, numClasses, numClassifiers);

for i = 1:numClassifiers
    try
        % Try to get probability scores
        [pred, scores] = predict(ensembleClassifier.classifiers{i}, features);
        allPredictions{i} = pred;
        
        % Convert scores to probability matrix if needed
        if size(scores, 2) == numClasses
            allScores(:, :, i) = scores;
        else
            % Convert class predictions to one-hot
            for j = 1:numSamples
                classIdx = find(strcmp(classNames, char(pred(j))));
                if ~isempty(classIdx)
                    allScores(j, classIdx, i) = 1.0;
                end
            end
        end
    catch
        % Fallback: use class predictions only
        pred = predict(ensembleClassifier.classifiers{i}, features);
        allPredictions{i} = pred;
        
        % Convert to one-hot
        for j = 1:numSamples
            classIdx = find(strcmp(classNames, char(pred(j))));
            if ~isempty(classIdx)
                allScores(j, classIdx, i) = 1.0;
            end
        end
    end
end

% Combine predictions based on voting method
switch lower(ensembleClassifier.votingMethod)
    case 'majority'
        % Majority voting
        voteMatrix = zeros(numSamples, numClasses);
        for i = 1:numClassifiers
            pred = allPredictions{i};
            for j = 1:numSamples
                classIdx = find(strcmp(classNames, char(pred(j))));
                if ~isempty(classIdx)
                    voteMatrix(j, classIdx) = voteMatrix(j, classIdx) + 1;
                end
            end
        end
        [~, maxIdx] = max(voteMatrix, [], 2);
        predictions = categorical(classNames(maxIdx), classNames);
        
    case {'weighted', 'stacking'}
        % Weighted voting using scores
        weightedScores = zeros(numSamples, numClasses);
        for i = 1:numClassifiers
            weightedScores = weightedScores + ...
                allScores(:, :, i) * ensembleClassifier.weights(i);
        end
        [~, maxIdx] = max(weightedScores, [], 2);
        predictions = categorical(classNames(maxIdx), classNames);
        
    otherwise
        % Default: majority voting
        voteMatrix = zeros(numSamples, numClasses);
        for i = 1:numClassifiers
            pred = allPredictions{i};
            for j = 1:numSamples
                classIdx = find(strcmp(classNames, char(pred(j))));
                if ~isempty(classIdx)
                    voteMatrix(j, classIdx) = voteMatrix(j, classIdx) + 1;
                end
            end
        end
        [~, maxIdx] = max(voteMatrix, [], 2);
        predictions = categorical(classNames(maxIdx), classNames);
end

end

