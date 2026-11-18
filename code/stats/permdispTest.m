function [Fdisp, pDisp] = permdispTest(Xz, groupLabels, nPerm)
% PERMDISPTEST  Omogeneita' della dispersione tra gruppi: ANOVA sulle distanze
%   di ogni punto dal centroide del proprio gruppo, p-value per permutazione.
%   Serve a verificare che il risultato PERMANOVA non dipenda solo da
%   dispersioni diverse.
    groups = unique(groupLabels);
    centroidDist = zeros(size(Xz,1),1);

    for g = 1:numel(groups)
        idx = groupLabels == groups(g);
        centroid = mean(Xz(idx,:), 1);
        centroidDist(idx) = sqrt(sum((Xz(idx,:) - centroid).^2, 2));
    end

    [~, tbl] = anova1(centroidDist, groupLabels, 'off');
    Fdisp = tbl{2,5};

    n = numel(groupLabels);
    Fperm = zeros(nPerm,1);
    for p = 1:nPerm
        permLabels = groupLabels(randperm(n));
        [~, tblP] = anova1(centroidDist, permLabels, 'off');
        Fperm(p) = tblP{2,5};
    end
    pDisp = (sum(Fperm >= Fdisp) + 1) / (nPerm + 1);
end
