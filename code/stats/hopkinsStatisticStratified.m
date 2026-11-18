function H = hopkinsStatisticStratified(X, labels, mPerGroup)
% HOPKINSSTATISTICSTRATIFIED  Hopkins statistic con campionamento bilanciato:
%   estrae fino a mPerGroup punti da ciascun gruppo, cosi' il gruppo
%   minoritario non viene ignorato a causa dello sbilanciamento.
    groups = unique(labels);
    sampleIdx = [];
    for g = 1:numel(groups)
        idxG = find(labels == groups(g));
        nG = min(mPerGroup, numel(idxG));
        sampleIdx = [sampleIdx; idxG(randsample(numel(idxG), nG))]; %#ok<AGROW>
    end

    Xsample = X(sampleIdx, :);
    m = size(Xsample,1);
    d = size(X,2);

    Dreal = zeros(m,1);
    for i = 1:m
        dists = pdist2(Xsample(i,:), X);
        dists(dists==0) = Inf;
        Dreal(i) = min(dists);
    end

    minX = min(X); maxX = max(X);
    Xuniform = minX + rand(m,d) .* (maxX - minX);
    Dunif = zeros(m,1);
    for i = 1:m
        dists = pdist2(Xuniform(i,:), X);
        Dunif(i) = min(dists);
    end

    H = sum(Dunif) / (sum(Dreal) + sum(Dunif));
end
