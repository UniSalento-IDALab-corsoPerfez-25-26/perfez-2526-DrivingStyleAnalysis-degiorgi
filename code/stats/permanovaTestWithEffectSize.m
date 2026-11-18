function [Fobs, pValue, R2] = permanovaTestWithEffectSize(Xz, groupLabels, nPerm)
% PERMANOVATESTWITHEFFECTSIZE  PERMANOVA su distanze euclidee al quadrato,
%   con p-value per permutazione delle etichette ed effect size R^2
%   (quota di variabilita' spiegata dai gruppi, indipendente dallo sbilanciamento).
%   Richiede computeF.m nella stessa cartella.
    groups = unique(groupLabels);
    n = size(Xz,1);
    D = pdist2(Xz, Xz).^2;

    [Fobs, R2] = computeF(D, groupLabels, groups, n);

    Fperm = zeros(nPerm,1);
    for p = 1:nPerm
        permLabels = groupLabels(randperm(n));
        Fperm(p) = computeF(D, permLabels, groups, n);
    end

    pValue = (sum(Fperm >= Fobs) + 1) / (nPerm + 1);
end
