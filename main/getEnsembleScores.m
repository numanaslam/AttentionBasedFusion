function scores = getEnsembleScores(ensembleClassifier, features)
%GETENSEMBLESCORES Get prediction scores from ensemble classifier
%   SCORES = getEnsembleScores(ENSEMBLE, FEATURES) extracts scores from
%   an ensemble classifier using the same method as predictEnsemble.
%
%   Inputs:
%       ensembleClassifier - Structure containing trained classifiers and weights
%       features - Feature matrix (N x D)
%
%   Outputs:
%       scores - Combined scores (N x numClasses)

numSamples = size(features, 1);
numClassifiers = length(ensembleClassifier.classifiers);
classNames = ensembleClassifier.classNames;
numClasses = length(classNames);

% Get scores from all classifiers (same as predictEnsemble)
allScores = zeros(numSamples, numClasses, numClassifiers);

for i = 1:numClassifiers
    try
        % Try to get probability scores
        [pred, baseScores] = predict(ensembleClassifier.classifiers{i}, features);
        
        % Convert scores to probability matrix if needed
        if size(baseScores, 2) == numClasses
            allScores(:, :, i) = baseScores;
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
        
        % Convert to one-hot
        for j = 1:numSamples
            classIdx = find(strcmp(classNames, char(pred(j))));
            if ~isempty(classIdx)
                allScores(j, classIdx, i) = 1.0;
            end
        end
    end
end

% Combine scores based on voting method (same as predictEnsemble)
switch lower(ensembleClassifier.votingMethod)
    case 'majority'
        % For majority voting, use simple average of scores
        scores = mean(allScores, 3);
        
    case {'weighted', 'stacking'}
        % Weighted voting using scores
        scores = zeros(numSamples, numClasses);
        for i = 1:numClassifiers
            scores = scores + allScores(:, :, i) * ensembleClassifier.weights(i);
        end
        
    otherwise
        % Default: simple average
        scores = mean(allScores, 3);
end

end

