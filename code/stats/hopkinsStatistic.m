function H = hopkinsStatistic(X, m)
% HOPKINSSTATISTIC  Tendenza naturale al clustering, senza guardare le etichette.
%   H = hopkinsStatistic(X, m) confronta le distanze al vicino piu' prossimo
%   di m punti reali con quelle di m punti uniformi nello stesso range.
%   H ~ 0.5: dati casuali; H > 0.75: forte tendenza al clustering.
    [n, d] = size(X);
    if nargin < 2, m = round(0.1*n); end % campione: 10% dei punti

    randIdx = randsample(n, m);
    Xsample = X(randIdx, :);

    % Distanza dai punti reali campionati al vicino piu' prossimo reale (escluso se stesso)
    Dreal = zeros(m,1);
    for i = 1:m
        dists = pdist2(Xsample(i,:), X);
        dists(dists==0) = Inf;
        Dreal(i) = min(dists);
    end

    % Punti generati uniformemente nello stesso range dei dati
    minX = min(X); maxX = max(X);
    Xuniform = minX + rand(m,d) .* (maxX - minX);
    Dunif = zeros(m,1);
    for i = 1:m
        dists = pdist2(Xuniform(i,:), X);
        Dunif(i) = min(dists);
    end

    H = sum(Dunif) / (sum(Dreal) + sum(Dunif));
end
