function [F, R2] = computeF(D, labels, groups, n)
% COMPUTEF  Pseudo-F e R^2 della PERMANOVA a partire dalla matrice D delle
%   distanze euclidee al quadrato (n x n). Usata da permanovaTestWithEffectSize.
%   SST = variabilita' totale, SSW = variabilita' entro i gruppi,
%   SSB = SST - SSW = variabilita' tra i gruppi, R^2 = SSB / SST.
    SST = sum(D(:)) / (2*n);
    SSW = 0;
    for g = 1:numel(groups)
        idx = labels == groups(g);
        ng = sum(idx);
        SSW = SSW + sum(sum(D(idx,idx))) / (2*ng);
    end
    SSB = SST - SSW;
    R2 = SSB / SST;
    dfB = numel(groups) - 1;
    dfW = n - numel(groups);
    F = (SSB/dfB) / (SSW/dfW);
end
