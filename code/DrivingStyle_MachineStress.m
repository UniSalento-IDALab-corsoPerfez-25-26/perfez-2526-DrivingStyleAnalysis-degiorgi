close all
clear all %#ok<CLALL>
clc

% Funzioni statistiche (PERMANOVA, PERMDISP, Hopkins) nella sottocartella stats
addpath(fullfile(fileparts(mfilename('fullpath')), 'stats'));

targetMission = 2;

%% ============================================================
%  STEP 1: CARICAMENTO DATI
% ============================================================

featNormal   = readtable('cycles_normal_features.csv');
featAggr     = readtable('cycles_aggressive_features.csv');
segNormal    = readtable('cycles_normal_signals.csv');
segAggr      = readtable('cycles_aggressive_signals.csv');

% Feature candidate (true = inclusa nel vettore di partenza)
featureFlags = struct(...
    'MeanSpeed',       true, ...
    'StdSpeed',        true, ...
    'MeanGearInSpeed', true, ...
    'StdGearInSpeed',  true, ...
    'MeanEngineSpeed', true, ...
    'MeanConvSlip',    true, ...
    'MeanThrottle',    true, ...
    'StdThrottle',     true, ...
    'MaxThrottle',     true, ...
    'RmsThrottle',     true, ...
    'JerkThrottle',    true, ...
    'IdleTimeRatio',   true, ...
    'MeanDecel',       true, ...
    'MaxDecel',        true, ...
    'RmsDecel',        true, ...
    'MeanInchPress',   true, ...
    'StdInchPress',    true);

allFeatureNames = fieldnames(featureFlags);
selectedMask    = structfun(@(x) x, featureFlags);
featureNames    = allFeatureNames(selectedMask);
featureNames    = featureNames(:)';

%% ============================================================
%  STEP 2: PULIZIA DATASET NORMAL
% ============================================================

dataMission     = featNormal(featNormal.Mission == targetMission, :);
isBadCycle_norm = dataMission.ZeroCrossingsBoom ~= 1;

fprintf('=== [DATASET GREZZO] Normal ===\n');
fprintf('Cicli scartati (segmentazione errata): %d su %d (%.1f%%)\n', ...
    sum(isBadCycle_norm), height(dataMission), 100*mean(isBadCycle_norm));

dataClean_norm = dataMission(~isBadCycle_norm, :);

Xnorm = dataClean_norm(:, [{'ID'}, featureNames]);
Xnorm = rmmissing(Xnorm);
Xnorm.Source = repmat({'Normal'}, height(Xnorm), 1);

%% ============================================================
%  STEP 3: PULIZIA DATASET AGGRESSIVE
% ============================================================

dataAggr        = featAggr;
isBadCycle_aggr = dataAggr.ZeroCrossingsBoom ~= 1;

fprintf('\n=== [DATASET GREZZO] Aggressive ===\n');
fprintf('Cicli scartati (segmentazione errata): %d su %d (%.1f%%)\n', ...
    sum(isBadCycle_aggr), height(dataAggr), 100*mean(isBadCycle_aggr));

dataClean_aggr = dataAggr(~isBadCycle_aggr, :);

Xaggr = dataClean_aggr(:, [{'ID'}, featureNames]);
Xaggr = rmmissing(Xaggr);

% Offset sugli ID aggressivi per evitare sovrapposizioni con i cicli Normal
Xaggr.ID     = Xaggr.ID + 1e6;
Xaggr.Source = repmat({'Aggressive'}, height(Xaggr), 1);

segAggr.ID = segAggr.ID + 1e6;

%% ============================================================
%  STEP 4: MERGE DATASET
% ============================================================

allSegments = [segNormal; segAggr];

Xall = [Xnorm; Xaggr];
Xall.Source = categorical(Xall.Source);

fprintf('\n[INTERO DATASET] Totale: Normal = %d | Aggressive = %d (rapporto %.1f:1)\n', ...
    sum(Xall.Source=='Normal'), sum(Xall.Source=='Aggressive'), ...
    sum(Xall.Source=='Normal')/sum(Xall.Source=='Aggressive'));

%% ============================================================
%  STEP 5: FEATURE DINAMICHE DI JERK
% ============================================================

uniqueIDs    = Xall.ID;
jerkSpeed_mean = nan(numel(uniqueIDs),1);
jerkSpeed_std  = nan(numel(uniqueIDs),1);

for i = 1:numel(uniqueIDs)
    seg = allSegments(allSegments.ID == uniqueIDs(i), :);
    if height(seg) > 2
        dSpeed = diff(seg.VehSpeed);
        jerkSpeed_mean(i) = mean(abs(dSpeed));
        jerkSpeed_std(i)  = std(dSpeed);
    end
end

Xall.JerkSpeed_mean = jerkSpeed_mean;
Xall.JerkSpeed_std  = jerkSpeed_std;

featureNames = [featureNames, {'JerkSpeed_mean','JerkSpeed_std'}];

Xall = rmmissing(Xall);

Xmat = Xall{:, featureNames};
Y    = Xall.Source;

%% ============================================================
%  STEP 6: STANDARDIZZAZIONE GLOBALE (SOLO ANALISI ESPLORATIVE)
% ============================================================

Xz = zscore(Xmat);

%% ============================================================
%  STEP 7: CLUSTERING DIAGNOSTICO K-MEANS + PCA [INTERO DATASET]
% ============================================================

rng(42);

clusterIdx = kmeans(Xz, 2, 'Replicates', 20);
Xall.ClusterCheck = categorical(clusterIdx);

fprintf('\n=== [INTERO DATASET] Verifica: il clustering ritrova la separazione? ===\n');
disp(crosstab(Xall.Source, Xall.ClusterCheck));

[~, scorePCA] = pca(Xz);

figure
gscatter(scorePCA(:,1), scorePCA(:,2), Xall.Source)
title('[DATASET] PCA for data separation')
xlabel('PC1')
ylabel('PC2')
grid on

%% ============================================================
%  STEP 8: TRAIN / TEST SPLIT STRATIFICATO
% ============================================================

cv = cvpartition(Y, 'HoldOut', 0.25);

XTrain = Xmat(training(cv), :);
YTrain = Y(training(cv));
XTest  = Xmat(test(cv), :);
YTest  = Y(test(cv));

fprintf('\n[TRAINING SET] %d cicli (Normal=%d, Aggressive=%d)\n', numel(YTrain), ...
    sum(YTrain=='Normal'), sum(YTrain=='Aggressive'));
fprintf('[TEST SET] %d cicli (Normal=%d, Aggressive=%d) -- NON bilanciato, riflette la realtà\n', ...
    numel(YTest), sum(YTest=='Normal'), sum(YTest=='Aggressive'));

%% ============================================================
%  STEP 9: STANDARDIZZAZIONE BASATA SOLO SUL TRAINING SET
% ============================================================

muTrain    = mean(XTrain,1);
sigmaTrain = std(XTrain,1);
sigmaTrain(sigmaTrain == 0) = 1;

XTrainZ = (XTrain - muTrain) ./ sigmaTrain;
XTestZ  = (XTest  - muTrain) ./ sigmaTrain;

%% ============================================================
%  STEP 10: FEATURE SELECTION CON MRMR [TRAINING SET]
% ============================================================

fprintf('\n=== [TRAINING SET] Feature Selection con MRMR ===\n');

[idxMRMR, scoresMRMR] = fscmrmr(XTrainZ, YTrain);

rankingTable = table( ...
    featureNames(idxMRMR)', ...
    scoresMRMR(idxMRMR)', ...
    'VariableNames', {'Feature', 'MRMR_Score'});

disp(rankingTable);

%% ============================================================
%  STEP 11: NUMERO DI FEATURE TRAMITE CV 5-FOLD [TRAINING SET]
% ============================================================

fprintf('\n=== [TRAINING SET - CV 5-fold] Selezione del numero ottimale di feature ===\n');

maxNumFeatures = numel(featureNames);
cvFS = cvpartition(YTrain, 'KFold', 5, 'Stratify', true);

balancedAccCV  = zeros(maxNumFeatures,1);
precisionAggCV = zeros(maxNumFeatures,1);
recallAggCV    = zeros(maxNumFeatures,1);

classOrder = categorical({'Normal','Aggressive'});

for nFeat = 1:maxNumFeatures

    selectedIdxTemp = idxMRMR(1:nFeat);

    foldBalancedAcc  = zeros(cvFS.NumTestSets,1);
    foldPrecisionAgg = zeros(cvFS.NumTestSets,1);
    foldRecallAgg    = zeros(cvFS.NumTestSets,1);

    for k = 1:cvFS.NumTestSets

        trIdx  = training(cvFS,k);
        valIdx = test(cvFS,k);

        XTr  = XTrainZ(trIdx, selectedIdxTemp);
        YTr  = YTrain(trIdx);
        XVal = XTrainZ(valIdx, selectedIdxTemp);
        YVal = YTrain(valIdx);

        costRatioK = sum(YTr=='Normal') / sum(YTr=='Aggressive');
        costK = [0 1; costRatioK 0];

        mdlK = fitcsvm(XTr, YTr, ...
            'KernelFunction', 'rbf', ...
            'Standardize', false, ...
            'ClassNames', classOrder, ...
            'Cost', costK, ...
            'BoxConstraint', 1, ...
            'KernelScale', 'auto');

        YPredVal = categorical(predict(mdlK, XVal));

        TP = sum(YPredVal=='Aggressive' & YVal=='Aggressive');
        FP = sum(YPredVal=='Aggressive' & YVal=='Normal');
        FN = sum(YPredVal=='Normal'     & YVal=='Aggressive');
        TN = sum(YPredVal=='Normal'     & YVal=='Normal');

        foldRecallAgg(k)    = TP / max(TP+FN, 1);
        foldPrecisionAgg(k) = TP / max(TP+FP, 1);
        foldBalancedAcc(k)  = 0.5 * (foldRecallAgg(k) + TN / max(TN+FP, 1));
    end

    balancedAccCV(nFeat)  = mean(foldBalancedAcc);
    precisionAggCV(nFeat) = mean(foldPrecisionAgg);
    recallAggCV(nFeat)    = mean(foldRecallAgg);
end

featureSelectionResults = table( ...
    (1:maxNumFeatures)', balancedAccCV, precisionAggCV, recallAggCV, ...
    'VariableNames', {'NumFeatures', 'BalancedAccuracy', 'PrecisionAggressive', 'RecallAggressive'});

disp(featureSelectionResults);

% Numero minimo di feature che raggiunge la migliore balanced accuracy
bestBalancedAcc = max(balancedAccCV);
threshold = 1 * bestBalancedAcc;
bestK = find(balancedAccCV >= threshold, 1, 'first');

selectedFeatureIdx   = idxMRMR(1:bestK);
selectedFeatureNames = featureNames(selectedFeatureIdx);

fprintf('\n[TRAINING SET - CV 5-fold] Best Balanced Accuracy CV: %.4f\n', bestBalancedAcc);
fprintf('[TRAINING SET - CV 5-fold] Numero minimo di feature selezionate: %d\n', bestK);
disp(selectedFeatureNames');

figure
plot(1:maxNumFeatures, balancedAccCV, '-o', 'LineWidth', 2)
xlabel('N selected feature')
ylabel('Balanced Accuracy in cross-validation')
title('[TRAINING SET - CV 5-fold] Feature Selection - MRMR')
grid on

%% ============================================================
%  STEP 12: TRAINING FINALE SVM COST-SENSITIVE [TRAINING SET]
% ============================================================

XTrainSel = XTrainZ(:, selectedFeatureIdx);
XTestSel  = XTestZ(:, selectedFeatureIdx);

costRatio  = sum(YTrain=='Normal') / sum(YTrain=='Aggressive');
costMatrix = [0 1; costRatio 0];

fprintf('\n[TRAINING SET] Cost ratio applicato (peso errore su Aggressive): %.1f\n', costRatio);

rng(42);

svmModel = fitcsvm(XTrainSel, YTrain, ...
    'KernelFunction', 'rbf', ...
    'Standardize', false, ...
    'ClassNames', classOrder, ...
    'Cost', costMatrix, ...
    'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct( ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 30, ...
        'Kfold', 5, ...
        'ShowPlots', false, ...
        'Verbose', 0));

%% ============================================================
%  STEP 13: PREDIZIONE SUL TEST SET + CALIBRAZIONE SIGMOIDE
% ============================================================

[YPred, rawScore] = predict(svmModel, XTestSel);
YPred = categorical(YPred);
YTest = categorical(YTest);

aggIdx      = find(string(svmModel.ClassNames) == "Aggressive");
rawAggScore = rawScore(:, aggIdx);

T = std(rawAggScore);
if T == 0
    T = 1;
end

drivingScore = 1 ./ (1 + exp(-rawAggScore / T));

%% ============================================================
%  STEP 14: VALUTAZIONE SUL TEST SET
% ============================================================

testAccuracy = mean(YPred == YTest);

TP = sum(YPred=='Aggressive' & YTest=='Aggressive');
FP = sum(YPred=='Aggressive' & YTest=='Normal');
FN = sum(YPred=='Normal'     & YTest=='Aggressive');
TN = sum(YPred=='Normal'     & YTest=='Normal');

precisionAgg = TP / max(TP+FP, 1);
recallAgg    = TP / max(TP+FN, 1);
f1Agg        = 2*precisionAgg*recallAgg / max(precisionAgg+recallAgg, eps);
balancedAcc  = 0.5*(TP/max(TP+FN,1) + TN/max(TN+FP,1));

fprintf('\n=== [TEST SET] Performance - SVM con feature selezionate ===\n');
fprintf('[TEST SET] Accuracy:          %.2f%%\n', testAccuracy*100);
fprintf('[TEST SET] Balanced Accuracy: %.2f%%\n', balancedAcc*100);
fprintf('[TEST SET] Precision Aggr:    %.3f\n', precisionAgg);
fprintf('[TEST SET] Recall Aggr:       %.3f\n', recallAgg);
fprintf('[TEST SET] F1-score Aggr:     %.3f\n', f1Agg);

figure
cm = confusionchart(YTest, YPred);
cm.Title = '[TEST SET] Confusion Matrix - SVM with feature vector';

[Xroc, Yroc, ~, AUC] = perfcurve(YTest, drivingScore, 'Aggressive');

figure
plot(Xroc, Yroc, 'LineWidth', 2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title(sprintf('[TEST SET] ROC Curve - AUC = %.3f', AUC))
grid on

%% ============================================================
%  STEP 15: CROSS-VALIDATION ROBUSTA [INTERO DATASET - CV 5-fold]
% ============================================================

cvFull = cvpartition(Y, 'KFold', 5, 'Stratify', true);

cvRecallAgg = zeros(5,1);
cvPrecAgg   = zeros(5,1);
cvBalAcc    = zeros(5,1);

for k = 1:5

    XTr  = Xmat(training(cvFull,k), :);
    YTr  = Y(training(cvFull,k));
    XVal = Xmat(test(cvFull,k), :);
    YVal = Y(test(cvFull,k));

    muK  = mean(XTr,1);
    sigK = std(XTr,1);
    sigK(sigK == 0) = 1;

    XTrSel  = (XTr  - muK) ./ sigK;  XTrSel  = XTrSel(:, selectedFeatureIdx);
    XValSel = (XVal - muK) ./ sigK;  XValSel = XValSel(:, selectedFeatureIdx);

    costRatioK = sum(YTr=='Normal') / sum(YTr=='Aggressive');
    costK = [0 1; costRatioK 0];

    mdlK = fitcsvm(XTrSel, YTr, ...
        'KernelFunction', 'rbf', ...
        'Standardize', false, ...
        'ClassNames', classOrder, ...
        'Cost', costK, ...
        'BoxConstraint', svmModel.ModelParameters.BoxConstraint, ...
        'KernelScale', svmModel.ModelParameters.KernelScale);

    YPredK = categorical(predict(mdlK, XValSel));

    TPk = sum(YPredK=='Aggressive' & YVal=='Aggressive');
    FPk = sum(YPredK=='Aggressive' & YVal=='Normal');
    FNk = sum(YPredK=='Normal'     & YVal=='Aggressive');
    TNk = sum(YPredK=='Normal'     & YVal=='Normal');

    cvRecallAgg(k) = TPk / max(TPk+FNk, 1);
    cvPrecAgg(k)   = TPk / max(TPk+FPk, 1);
    cvBalAcc(k)    = 0.5 * (cvRecallAgg(k) + TNk / max(TNk+FPk, 1));
end

fprintf('\n=== [INTERO DATASET - CV 5-fold, iperparametri fissi dal modello finale] ===\n');
fprintf('[CV 5-fold] Balanced Accuracy:    %.3f ± %.3f\n', mean(cvBalAcc), std(cvBalAcc));
fprintf('[CV 5-fold] Recall Aggressive:    %.3f ± %.3f\n', mean(cvRecallAgg), std(cvRecallAgg));
fprintf('[CV 5-fold] Precision Aggressive: %.3f ± %.3f\n', mean(cvPrecAgg), std(cvPrecAgg));

%% ============================================================
%  STEP 16: DISTRIBUZIONE SCORE SUL TEST SET
% ============================================================

figure
histogram(drivingScore(YTest=='Normal'),     'BinWidth',0.05, 'FaceAlpha',0.6)
hold on
histogram(drivingScore(YTest=='Aggressive'), 'BinWidth',0.05, 'FaceAlpha',0.6)
legend('Normal GT','Aggressive GT')
xlabel('Driving Score 0-1')
ylabel('Frequency')
title('[TEST SET] Score Distribution')
xlim([0 1])
grid on

%% ============================================================
%  STEP 17: DRIVING SCORE SU TUTTO IL DATASET
% ============================================================

XallZ   = (Xmat - muTrain) ./ sigmaTrain;
XallSel = XallZ(:, selectedFeatureIdx);

[~, rawScoreAll] = predict(svmModel, XallSel);
Xall.DrivingScore = 1 ./ (1 + exp(-rawScoreAll(:, aggIdx) / T));
Xall.OSS = 100 * Xall.DrivingScore;

fprintf('\n[INTERO DATASET] Driving Score calcolato per tutti i %d cicli\n', height(Xall));

%% ============================================================
%  STEP 18: MODELLO DI RIFERIMENTO CONSUMI (SOLO CICLI NORMAL)
% ============================================================

cycleFuel = nan(height(Xall),1);
for i = 1:height(Xall)
    seg = allSegments(allSegments.ID == Xall.ID(i), :);
    if ~isempty(seg)
        cycleFuel(i) = mean(seg.FuelRate, 'omitnan');
    end
end
Xall.MeanFuelCycle = cycleFuel;

normalMask = Xall.Source == "Normal";

fuelMeanNormal = Xall.MeanFuelCycle(normalMask);
fuelMeanNormal = fuelMeanNormal(~isnan(fuelMeanNormal));

muFuel    = mean(fuelMeanNormal);
sigmaFuel = std(fuelMeanNormal);
if sigmaFuel == 0
    sigmaFuel = 1;
end

fprintf('\nFuel reference model: Mean = %.3f | Std = %.3f\n', muFuel, sigmaFuel);

Xall.FuelZscore          = (Xall.MeanFuelCycle - muFuel) ./ sigmaFuel;
Xall.FuelPercentile      = 100*normcdf(Xall.FuelZscore);
Xall.FuelIncreasePercent = 100*(Xall.MeanFuelCycle - muFuel)/muFuel;

fuelClass = strings(height(Xall),1);
fuelClass(Xall.FuelPercentile <  50)                          = "Below average";
fuelClass(Xall.FuelPercentile >= 50 & Xall.FuelPercentile < 80) = "Normal";
fuelClass(Xall.FuelPercentile >= 80 & Xall.FuelPercentile < 95) = "High";
fuelClass(Xall.FuelPercentile >= 95 & Xall.FuelPercentile < 99) = "Very High";
fuelClass(Xall.FuelPercentile >= 99)                          = "Extreme";
Xall.FuelClass = fuelClass;

%% ============================================================
%  STEP 19: INDICI DI STRESS PER SOTTOSISTEMA
%  Ogni feature e' standardizzata rispetto ai soli cicli Normal:
%  z = (x - mu_Normal) / sigma_Normal
% ============================================================

Xall.EngineWearRaw = ...
      zFromNormal(Xall.MeanEngineSpeed, normalMask) ...
    + zFromNormal(Xall.MeanThrottle, normalMask) ...
    + zFromNormal(Xall.MaxThrottle,  normalMask);

Xall.TransmissionWearRaw = ...
      zFromNormal(Xall.MeanGearInSpeed,  normalMask) ...
    + zFromNormal(Xall.StdGearInSpeed,   normalMask) ...
    + zFromNormal(Xall.MeanInchPress, normalMask) ...
    + zFromNormal(Xall.StdInchPress,  normalMask);

Xall.BrakeWearRaw = ...
      zFromNormal(Xall.MeanDecel, normalMask) ...
    + zFromNormal(Xall.MaxDecel,  normalMask) ...
    + zFromNormal(Xall.RmsDecel,  normalMask);

% Ri-standardizzazione della somma rispetto ai soli cicli Normal
Xall.EngineWearZ       = zFromNormal(Xall.EngineWearRaw,       normalMask);
Xall.TransmissionWearZ = zFromNormal(Xall.TransmissionWearRaw, normalMask);
Xall.BrakeWearZ        = zFromNormal(Xall.BrakeWearRaw,        normalMask);

%% ============================================================
%  STEP 20: MACHINE STRESS SCORE (0-100)
% ============================================================

Xall.MachineStressScore = min( ...
      0.25 * stressDisplay(Xall.FuelZscore) ...
    + 0.25 * stressDisplay(Xall.EngineWearZ) ...
    + 0.25 * stressDisplay(Xall.TransmissionWearZ) ...
    + 0.25 * stressDisplay(Xall.BrakeWearZ), 100);

%% ============================================================
%  STEP 21: SALVATAGGIO RISULTATI
% ============================================================

writetable(Xall, 'DrivingScore_NormalVsAggressive_SelectedFeatures.csv');

save('DrivingScoreModel_SelectedFeatures.mat', ...
    'svmModel', 'muTrain', 'sigmaTrain', 'featureNames', ...
    'selectedFeatureIdx', 'selectedFeatureNames', ...
    'idxMRMR', 'scoresMRMR', 'featureSelectionResults', 'T');

fprintf('\nRisultati salvati in DrivingScore_NormalVsAggressive_SelectedFeatures.csv\n');
fprintf('Modello salvato in DrivingScoreModel_SelectedFeatures.mat\n');

%% ============================================================
%  STEP 22: TEST STATISTICI MULTIVARIATI [INTERO DATASET]
%  Funzioni in code/stats: permanovaTestWithEffectSize (+ computeF),
%  hopkinsStatistic, permdispTest, hopkinsStatisticStratified
% ============================================================

rng(1);
[Fobs, pValue, R2] = permanovaTestWithEffectSize(Xz, Xall.Source, 999);
fprintf('\n[INTERO DATASET] PERMANOVA: F = %.3f, p-value = %.4f, R2 = %.4f\n', Fobs, pValue, R2);

rng(2);
H = hopkinsStatistic(Xz, 200);
fprintf('[INTERO DATASET] Hopkins statistic: %.3f (>0.75 indica forte tendenza al clustering)\n', H);

rng(4);
[Fdisp, pDisp] = permdispTest(Xz, Xall.Source, 999);
fprintf('[INTERO DATASET] PERMDISP: F = %.3f, p-value = %.4f\n', Fdisp, pDisp);

rng(2);
Hstrat = hopkinsStatisticStratified(Xz, Xall.Source, 60);
fprintf('[INTERO DATASET] Hopkins stratificato: %.3f\n', Hstrat);

%% ============================================================
%  STEP 23: CONFRONTO DISTRIBUZIONI NORMAL vs AGGRESSIVE
% ============================================================

vars   = {'MeanFuelCycle','EngineWearRaw','TransmissionWearRaw','BrakeWearRaw','MachineStressScore','OSS'};
titles = {'Fuel Consumption','Engine Stress','Transmission Stress','Brake Stress','Machine Stress Score','Operator Style Score'};

figure
tiledlayout(2,3)

for k = 1:numel(vars)

    nexttile

    xN = Xall{Xall.Source=="Normal",     vars{k}};
    xA = Xall{Xall.Source=="Aggressive", vars{k}};

    [fN, xiN] = ksdensity(xN);
    [fA, xiA] = ksdensity(xA);

    plot(xiN, fN, 'b', 'LineWidth', 3)
    hold on
    plot(xiA, fA, 'r', 'LineWidth', 3)

    muN = mean(xN, 'omitnan');
    muA = mean(xA, 'omitnan');
    xline(muN, '--b', sprintf('N=%.2f', muN), 'LineWidth', 1.5);
    xline(muA, '--r', sprintf('A=%.2f', muA), 'LineWidth', 1.5);

    title(sprintf('%s (\\Delta = %.2f)', titles{k}, muA - muN))
    ylabel('Density')

    if any(strcmp(vars{k}, {'MachineStressScore','OSS'}))
        xlabel(titles{k})
        xlim([0 100])
    else
        xlabel('Index Value')
    end

    legend('Normal', 'Aggressive', 'Location', 'best')
    grid on
end

sgtitle('Distribution Comparison: Normal vs Aggressive')

%% ============================================================
%  STEP 24: CORRELAZIONE OSS - MACHINE STRESS SCORE
% ============================================================

rho_all  = corr(Xall.OSS, Xall.MachineStressScore, 'Type', 'Spearman');
rho_nor  = corr(Xall.OSS(normalMask), Xall.MachineStressScore(normalMask), 'Type', 'Spearman');
Sfuel    = stressDisplay(Xall.FuelZscore);
rho_fuel = corr(Xall.OSS(normalMask), Sfuel(normalMask), 'Type', 'Spearman');

q = [sum(Xall.OSS <  70 & Xall.MachineStressScore <  60), ...
     sum(Xall.OSS >= 70 & Xall.MachineStressScore <  60), ...
     sum(Xall.OSS <  70 & Xall.MachineStressScore >= 60), ...
     sum(Xall.OSS >= 70 & Xall.MachineStressScore >= 60)];

fprintf('\n[CORRELAZIONE] rho tutti = %.2f | rho Normal = %.2f | rho OSS-Fuel (Normal) = %.2f\n', ...
    rho_all, rho_nor, rho_fuel);
fprintf('[QUADRANTI] basso/basso = %d | stile senza stress = %d | stress senza stile = %d | alto/alto = %d\n', q);

%% ============================================================
%  STEP 25: GUI DI NAVIGAZIONE CICLI
% ============================================================

launchScoreGUI(Xall, allSegments, muFuel, sigmaFuel)
launchWearGUI(Xall)


%% ============================================================
%  FUNZIONI LOCALI
% ============================================================

function z = zFromNormal(values, normalMask)
% z-score rispetto alla distribuzione dei soli cicli Normal
mu    = mean(values(normalMask), 'omitnan');
sigma = std(values(normalMask),  'omitnan');
if sigma <= 0
    sigma = 1;
end
z = (values - mu) ./ sigma;
end

function score = stressDisplay(z)
% Mappatura z-score -> stress 0-100 (zero sotto la media Normal, saturazione per z grandi)
score = min(100*(1 - exp(-max(z,0)/2)), 100);
end

function txt = wearClass(val)
if val < 65
    txt = "GOOD";
elseif val < 85
    txt = "MODERATE";
elseif val < 95
    txt = "HIGH";
else
    txt = "SEVERE";
end
end

function drawRing(ax, r1, r2, color, alpha)
th = linspace(0, 2*pi, 400);
x  = [r2*cos(th) fliplr(r1*cos(th))];
y  = [r2*sin(th) fliplr(r1*sin(th))];
fill(ax, x, y, color, 'FaceAlpha', alpha, 'EdgeColor', 'none');
end

%% ------------------------------------------------------------
%  GUI 1: segnali del ciclo + posizione del consumo rispetto ai Normal
%% ------------------------------------------------------------
function launchScoreGUI(Xall, allSegments, muFuel, sigmaFuel)

state.X             = Xall;
state.segments      = allSegments;
state.currentSource = 'Normal';
state.currentIdx    = 1;
state.idsInGroup    = state.X.ID(state.X.Source == "Normal");

fig = uifigure('Name', 'Driving Score + Fuel Analysis', 'Position', [50 50 1400 920]);

uilabel(fig, 'Position', [20 885 120 22], 'Text', 'Group');
uidropdown(fig, 'Position', [80 885 180 22], 'Items', {'Normal','Aggressive'}, ...
    'ValueChangedFcn', @(dd,~) onGroupChanged(dd));
uibutton(fig, 'Text', '<< Previous', 'Position', [320 880 120 30], 'ButtonPushedFcn', @(~,~) onPrev());
uibutton(fig, 'Text', 'Next >>',     'Position', [450 880 120 30], 'ButtonPushedFcn', @(~,~) onNext());
lblInfo = uilabel(fig, 'Position', [600 875 780 40], 'Text', '', 'FontWeight', 'bold');

ax1 = uiaxes(fig, 'Position', [30 700 1320 140]);
ax2 = uiaxes(fig, 'Position', [30 520 1320 140]);
ax3 = uiaxes(fig, 'Position', [30 340 1320 140]);
ax4 = uiaxes(fig, 'Position', [30  30 1320 260]);

updatePlot();

    function onGroupChanged(dd)
        state.currentSource = dd.Value;
        state.idsInGroup    = state.X.ID(state.X.Source == dd.Value);
        state.currentIdx    = 1;
        updatePlot();
    end

    function onNext()
        n = numel(state.idsInGroup);
        if n == 0, return, end
        state.currentIdx = mod(state.currentIdx, n) + 1;
        updatePlot();
    end

    function onPrev()
        n = numel(state.idsInGroup);
        if n == 0, return, end
        state.currentIdx = mod(state.currentIdx - 2, n) + 1;
        updatePlot();
    end

    function updatePlot()
        n = numel(state.idsInGroup);
        if n == 0
            cla(ax1); cla(ax2); cla(ax3); cla(ax4);
            lblInfo.Text = 'No cycles available';
            return
        end

        currentID = state.idsInGroup(state.currentIdx);
        seg    = state.segments(state.segments.ID == currentID, :);
        rowIdx = state.X.ID == currentID;
        t      = (0:height(seg)-1)';

        plot(ax1, t, seg.VehSpeed, 'LineWidth', 1.5, 'Color', [0 0.447 0.741]);
        title(ax1, 'Vehicle Speed'); xlabel(ax1, 'Samples'); ylabel(ax1, 'km/h'); grid(ax1, 'on')

        plot(ax2, t, seg.BucketAngle, 'LineWidth', 1.5, 'Color', [0.85 0.33 0.1]);
        title(ax2, 'Bucket Angle'); xlabel(ax2, 'Samples'); ylabel(ax2, 'deg'); grid(ax2, 'on')

        plot(ax3, t, seg.BoomAngle, 'LineWidth', 1.5, 'Color', [0.47 0.67 0.19]);
        title(ax3, 'Boom Angle'); xlabel(ax3, 'Samples'); ylabel(ax3, 'deg'); grid(ax3, 'on')

        fuelVal = state.X.MeanFuelCycle(rowIdx);

        x = linspace(muFuel - 4*sigmaFuel, muFuel + 4*sigmaFuel, 1000);
        plot(ax4, x, normpdf(x, muFuel, sigmaFuel), 'k', 'LineWidth', 3);
        hold(ax4, 'on')
        xline(ax4, muFuel,               '--g', 'Mean');
        xline(ax4, muFuel +   sigmaFuel, '--y', '+1\sigma');
        xline(ax4, muFuel + 2*sigmaFuel, '--r', '+2\sigma');
        xline(ax4, muFuel + 3*sigmaFuel, '--m', '+3\sigma');
        yCurrent = normpdf(fuelVal, muFuel, sigmaFuel);
        if ~isnan(yCurrent)
            scatter(ax4, fuelVal, yCurrent, 180, 'r', 'filled');
        end
        hold(ax4, 'off')
        title(ax4, 'Fuel Consumption Relative To Normal Distribution')
        xlabel(ax4, 'Mean FuelRate'); ylabel(ax4, 'Probability Density'); grid(ax4, 'on')

        lblInfo.Text = sprintf(['%s | Cycle %d/%d | ID=%d | DrivingScore=%.3f | Fuel=%.2f | ' ...
            'FuelIncrease=%.1f%% | Z=%.2fσ | Percentile=%.1f%% | %s'], ...
            state.currentSource, state.currentIdx, n, currentID, ...
            state.X.DrivingScore(rowIdx), fuelVal, ...
            state.X.FuelIncreasePercent(rowIdx), state.X.FuelZscore(rowIdx), ...
            state.X.FuelPercentile(rowIdx), string(state.X.FuelClass(rowIdx)));
    end
end

%% ------------------------------------------------------------
%  GUI 2: profilo di stress per sottosistema + MSS + OSS
%% ------------------------------------------------------------
function launchWearGUI(Xall)

state.X             = Xall;
state.currentSource = 'Normal';
state.currentIdx    = 1;
state.idsInGroup    = state.X.ID(state.X.Source == "Normal");

fig = uifigure('Name', 'Machine Wear Analysis', 'Position', [100 100 1500 900]);

uidropdown(fig, 'Position', [20 850 180 25], 'Items', {'Normal','Aggressive'}, ...
    'ValueChangedFcn', @(dd,~) onGroupChanged(dd));
uibutton(fig, 'Position', [230 845 120 30], 'Text', '<< Previous', 'ButtonPushedFcn', @(~,~) onPrev());
uibutton(fig, 'Position', [370 845 120 30], 'Text', 'Next >>',     'ButtonPushedFcn', @(~,~) onNext());
lblInfo = uilabel(fig, 'Position', [520 835 950 40], 'FontWeight', 'bold', 'FontSize', 12);

axRadar = uiaxes(fig, 'Position', [20 220 650 580]);
axBar   = uiaxes(fig, 'Position', [760 430 650 320]);
axMSS = uiaxes(fig, 'Position', [760 180 650 170]);
axOSS   = uiaxes(fig, 'Position', [760  40 650 100]);

updatePlot();

    function onGroupChanged(dd)
        state.currentSource = dd.Value;
        state.idsInGroup    = state.X.ID(state.X.Source == dd.Value);
        state.currentIdx    = 1;
        updatePlot();
    end

    function onNext()
        n = numel(state.idsInGroup);
        if n == 0, return, end
        state.currentIdx = mod(state.currentIdx, n) + 1;
        updatePlot();
    end

    function onPrev()
        n = numel(state.idsInGroup);
        if n == 0, return, end
        state.currentIdx = mod(state.currentIdx - 2, n) + 1;
        updatePlot();
    end

    function updatePlot()
        currentID = state.idsInGroup(state.currentIdx);
        row = state.X(state.X.ID == currentID, :);

        vals = [stressDisplay(row.FuelZscore), ...
                stressDisplay(row.EngineWearZ), ...
                stressDisplay(row.TransmissionWearZ), ...
                stressDisplay(row.BrakeWearZ)];
        mss = row.MachineStressScore;
        oss   = row.OSS;

        % ---------------- RADAR ----------------
        cla(axRadar)
        hold(axRadar, 'on')
        drawRing(axRadar, 95, 100, [1 0 0],   0.45);
        drawRing(axRadar, 85,  95, [1 0.5 0], 0.40);
        drawRing(axRadar, 65,  85, [1 1 0],   0.35);
        drawRing(axRadar,  0,  65, [0 1 0],   0.30);

        for r = [20 40 60 65 85 95 100]
            th = linspace(0, 2*pi, 300);
            plot(axRadar, r*cos(th), r*sin(th), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
        end

        angles = linspace(0, 2*pi, 5);
        labels = {'Fuel', 'Engine', 'Transmission', 'Brake'};
        for k = 1:4
            plot(axRadar, [0 100*cos(angles(k))], [0 100*sin(angles(k))], 'k:');
            text(axRadar, 115*cos(angles(k)), 115*sin(angles(k)), labels{k}, ...
                'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        end

        x = vals .* cos(angles(1:end-1));  x = [x x(1)];
        y = vals .* sin(angles(1:end-1));  y = [y y(1)];
        fill(axRadar, x, y, [0.1 0.45 0.95], 'FaceAlpha', 0.55, 'EdgeColor', 'none');
        plot(axRadar, x, y, 'b-o', 'LineWidth', 4, 'MarkerSize', 8);

        axis(axRadar, 'equal')
        xlim(axRadar, [-130 130]); ylim(axRadar, [-130 130])
        axRadar.XTick = []; axRadar.YTick = [];
        title(axRadar, 'Machine Stress Profile')
        hold(axRadar, 'off')

        % ---------------- BAR CHART ----------------
        cla(axBar)
        b = bar(axBar, vals);
        colors = zeros(numel(vals), 3);
        for i = 1:numel(vals)
            if vals(i) < 65
                colors(i,:) = [0 0.75 0];
            elseif vals(i) < 85
                colors(i,:) = [1 0.85 0];
            elseif vals(i) < 95
                colors(i,:) = [1 0.5 0];
            else
                colors(i,:) = [0.9 0 0];
            end
            text(axBar, i, vals(i)+3, sprintf('%.0f%%', vals(i)), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
        b.FaceColor = 'flat';
        b.CData = colors;
        xticklabels(axBar, {'Fuel', 'Engine', 'Trans', 'Brake'});
        ylim(axBar, [0 100]); ylabel(axBar, 'Stress Score'); grid(axBar, 'on')
        title(axBar, 'Subsystem Stress Level')

        % ---------------- MACHINE STRESS SCORE ----------------
        cla(axMSS)
        if mss < 60
            mssColor = [0 0.75 0];
        elseif mss < 80
            mssColor = [1 0.85 0];
        else
            mssColor = [0.9 0 0];
        end
        barh(axMSS, mss, 'FaceColor', mssColor);
        hold(axMSS, 'on')
        xline(axMSS, 60, '--g'); xline(axMSS, 80, '--r');
        hold(axMSS, 'off')
        xlim(axMSS, [0 100]); grid(axMSS, 'on')
        title(axMSS, 'Machine Stress Score'); xlabel(axMSS, 'Machine Stress Score')
        text(axMSS, min(mss+2, 95), 1, sprintf('%.1f', mss), 'FontWeight', 'bold', 'FontSize', 14);

        % ---------------- OSS ----------------
        cla(axOSS)
        if oss < 40
            ossColor = [0 0.75 0];
        elseif oss < 70
            ossColor = [1 0.85 0];
        elseif oss < 90
            ossColor = [1 0.5 0];
        else
            ossColor = [0.9 0 0];
        end
        barh(axOSS, oss, 'FaceColor', ossColor);
        hold(axOSS, 'on')
        xline(axOSS, 40, '--g'); xline(axOSS, 70, '--y'); xline(axOSS, 90, '--r');
        hold(axOSS, 'off')
        xlim(axOSS, [0 100]); grid(axOSS, 'on')
        title(axOSS, 'Operator Style Score'); xlabel(axOSS, 'OSS')
        text(axOSS, min(oss+2, 95), 1, sprintf('%.1f', oss), 'FontWeight', 'bold', 'FontSize', 14);

        % ---------------- INFO ----------------
        lblInfo.Text = sprintf('ID=%d | OSS=%.1f | Fuel:%s | Engine:%s | Trans:%s | Brake:%s | MSS=%.1f', ...
            currentID, oss, wearClass(vals(1)), wearClass(vals(2)), ...
            wearClass(vals(3)), wearClass(vals(4)), mss);
    end
end
