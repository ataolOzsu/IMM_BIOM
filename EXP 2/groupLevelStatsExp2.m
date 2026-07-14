% Fits per-subject GLMs (full 2x2x2 factorial) and runs second-level
% one-sample t-tests. Mirrors the Experiment 1 pipeline, extended with the
% additional binary factor Congruency.
% Any questions / bugs: a.ozsu@ucl.ac.uk

clear; clc; close all

wanted = {'RT','ACC','Motion','Sync','congruency','Speed','Direction'};

% RT : Continuous
% ACC : Binary (1/0); 99 = missed, removed below
% Motion (MotionType)         : 1 Biological / 0 Scrambled
% Sync   (TemporalSynchrony)  : 1 Synchronous / 0 Asynchronous
% Congruency                  : 1 Congruent / 0 Incongruent
% Speed                       : 3 levels   ) dropped as nuisance regressors
% Direction                   : 2 levels   ) (re-check balance on Exp 2 data)

nSubjects = 33;
addpath('results\')
perSubjResults = cell(1, nSubjects);

termNames = { ...
    'MotionType',                              'Motion';
    'TemporalSynchrony',                       'Sync';
    'Congruency',                              'Cong';
    'MotionType:TemporalSynchrony',            'Motion x Sync';
    'MotionType:Congruency',                   'Motion x Cong';
    'TemporalSynchrony:Congruency',            'Sync x Cong';
    'MotionType:TemporalSynchrony:Congruency', 'Motion x Sync x Cong'};
nTerms = size(termNames,1);
beta = nan(nSubjects, nTerms, 2);

for iSubj = 3:nSubjects          
                                 
    fpath = fullfile('D:\BioMotion\EXP 2','results',[num2str(iSubj) '.mat']);
    if ~isfile(fpath), continue; end
    data = load(fpath);

    subjResults = [];
    for iBlock = 1:length(data.block)
        blockData = data.block{1, iBlock};
        blockResults = [];
        for iField = 1:numel(wanted)
            dat = blockData.(wanted{iField});
            if iscell(dat), dat = cell2mat(dat); end
            blockResults(iField,:) = dat(:)';   
        end
        subjResults = [subjResults, blockResults];
    end
    subjResults = subjResults';
    T = array2table(subjResults, 'VariableNames', ...
        {'RT','ACC','MotionType','TemporalSynchrony','Congruency','Speed','Direction'});

    T(T.ACC == 99, :) = [];              
    perSubjResults{iSubj} = T;           
    
    %% Fit GLM for reaction time

    try
        mdl_lin = fitglm(T, 'RT ~ MotionType*TemporalSynchrony*Congruency');
    catch ME
        warning('Subject %d RT GLM failed: %s', iSubj, ME.message); continue
    end
    for k = 1:nTerms
        beta(iSubj,k,1) = getBeta(mdl_lin.Coefficients, termNames{k,1});
    end

    %% Fit GLM for accuracy with logit link

    try
        mdlLog = fitglm(T, 'ACC ~ MotionType*TemporalSynchrony*Congruency', ...
            'Distribution','binomial', 'Link','logit', ...
            'LikelihoodPenalty','jeffreys-prior');
    catch ME
        warning('Subject %d ACC GLM failed: %s', iSubj, ME.message); continue
    end
    for k = 1:nTerms
        beta(iSubj,k,2) = getBeta(mdlLog.Coefficients, termNames{k,1});
    end
end


%% Second-level analyses
outcomeName = {'Reaction Time','Accuracy'};
for oc = 1:2
    valid  = ~isnan(beta(:,1,oc));          % all terms fill together per subject
    nValid = sum(valid);
    fprintf('\n========== %s  (N = %d) ==========\n', outcomeName{oc}, nValid);
    fprintf('=== One-sample t-tests on coefficients ===\n');

    figure('Color','w','Position',[100 100 1400 650])
    for k = 1:nTerms
        vals = beta(valid,k,oc);
        [~, p, ~, s] = ttest(vals);
        fprintf('%-22s mean=%7.3f  t(%d)=%7.3f  p=%.4f\n', ...
            termNames{k,2}, mean(vals), nValid-1, s.tstat, p);

        subplot(2,4,k); hold on
        scatter(ones(1,nValid), vals, 60, 'k', 'filled', 'MarkerFaceAlpha', 0.4)
        errorbar(1, mean(vals), std(vals)/sqrt(nValid), 'o', ...
            'Color',[0.2 0.6 0.9], 'MarkerFaceColor',[0.2 0.6 0.9], ...
            'LineWidth', 3, 'MarkerSize', 9, 'CapSize', 10)
        if p < 0.001, pStr = 'p < .001'; else, pStr = sprintf('p = %.3f', p); end
        yl = ylim; text(1, yl(2)-0.06*range(yl), pStr, ...
            'HorizontalAlignment','center', 'FontSize', 13, 'FontWeight','bold')
        yline(0,'k--'); xlim([0.5 1.5]); xticks([])
        title(termNames{k,2}, 'FontSize', 14); box off
        set(gca,'FontSize',12,'FontWeight','bold')
    end
    sgtitle(sprintf('Subject-level coefficients - %s', outcomeName{oc}), ...
        'FontWeight','bold', 'FontSize', 16)
end

%% ==================== SECOND-STAGE DIAGNOSTICS ======================
% Check if non-parametric and parametric agrees
fprintf('\n================ SECOND-STAGE DIAGNOSTICS ================\n');
for oc = 1:2
    fprintf('\n--- %s ---\n', outcomeName{oc});
    fprintf('%-22s %8s %9s %9s %11s %8s\n', ...
        'Coefficient','skew','LillieP','tP','signrankP','agree?');
    figure('Color','w','Position',[100 100 1400 650])
    for k = 1:nTerms
        v = beta(:,k,oc); v = v(~isnan(v));
        sk = skewness(v);
        if numel(v) > 4, [~, pL] = lillietest(v); else, pL = NaN; end
        [~, pT] = ttest(v);
        pW = signrank(v);                          % H0: median = 0
        agree = (pT<0.05) == (pW<0.05);
        fprintf('%-22s %8.3f %9.3f %9.4f %11.4f %8s\n', ...
            termNames{k,2}, sk, pL, pT, pW, string(agree));
        subplot(2,4,k); qqplot(v);
        title(termNames{k,2}); set(gca,'FontSize',10); box off
    end
    sgtitle(sprintf('Normality of subject-level betas - %s', outcomeName{oc}), ...
        'FontWeight','bold','FontSize',14)
end
fprintf('\nIf tP and signrankP disagree for a term, report both and lead with signed-rank.\n');


%% ===================== POST-HOC FIGURES (Exp 2) =====================

congRT  = nan(2, nSubjects);   congAcc = nan(2, nSubjects); 
syncRT  = nan(2, nSubjects);   motionRT = nan(2, nSubjects); 

aI = nan(1,nSubjects); aC = nan(1,nSubjects);  
sI = nan(1,nSubjects); sC = nan(1,nSubjects);   
bAI=nan(1,nSubjects); bAC=nan(1,nSubjects); bSI=nan(1,nSubjects); bSC=nan(1,nSubjects);
rAI=nan(1,nSubjects); rAC=nan(1,nSubjects); rSI=nan(1,nSubjects); rSC=nan(1,nSubjects);

for iSubj = 1:nSubjects
    Ts = perSubjResults{1, iSubj};
    if isempty(Ts), continue; end
    D  = table2array(Ts);
    rt = D(:,1); acc = D(:,2); M = D(:,3); S = D(:,4); C = D(:,5);

    % Main effects
    congRT(1,iSubj)  = mean(rt(C==0));   congRT(2,iSubj)  = mean(rt(C==1));   % Incong / Cong
    congAcc(1,iSubj) = mean(acc(C==0));  congAcc(2,iSubj) = mean(acc(C==1));

    syncRT(1,iSubj)  = mean(rt(S==0));   syncRT(2,iSubj)  = mean(rt(S==1)); 
    motionRT(1,iSubj)  = mean(rt(M==0));   motionRT(2,iSubj)  = mean(rt(M==1)); 

    % Sync x Congruency (RT)
    aI(iSubj)=mean(rt(S==0 & C==0));  aC(iSubj)=mean(rt(S==0 & C==1));
    sI(iSubj)=mean(rt(S==1 & C==0));  sC(iSubj)=mean(rt(S==1 & C==1));

    % Three-way finest cells (RT)
    bAI(iSubj)=mean(rt(M==1&S==0&C==0)); bAC(iSubj)=mean(rt(M==1&S==0&C==1));
    bSI(iSubj)=mean(rt(M==1&S==1&C==0)); bSC(iSubj)=mean(rt(M==1&S==1&C==1));
    rAI(iSubj)=mean(rt(M==0&S==0&C==0)); rAC(iSubj)=mean(rt(M==0&S==0&C==1));
    rSI(iSubj)=mean(rt(M==0&S==1&C==0)); rSC(iSubj)=mean(rt(M==0&S==1&C==1));
end

% Main effects (reuse your existing plotMainEffect)
plotMainEffect(congRT(2,:),  congRT(1,:),  'Congruent', 'Incongruent', ...
    'Main effect of Congruency on RT', nSubjects)
plotMainEffect(congAcc(2,:), congAcc(1,:), 'Congruent', 'Incongruent', ...
    'Main effect of Congruency on Accuracy', nSubjects)
plotMainEffect(motionRT(2,:),  motionRT(1,:),  'Biological', 'Scrambled', ...
    'Main effect of Motion Type on RT', nSubjects)
plotMainEffect(syncRT(2,:), syncRT(1,:), 'Synchronous', 'Asynchronous', ...
    'Main effect of Temporal Synchrony on RT', nSubjects)
% Two-way: Sync x Congruency on RT (grouped bars, congruency within synchrony)
plotInteraction2way(aI, aC, sI, sC, ...
    {'Asynchronous','Synchronous'}, {'Incongruent','Congruent'}, ...
    'Sync \times Congruency on RT', nSubjects)

intBio = (bSC - bSI) - (bAC - bAI);
intScr = (rSC - rSI) - (rAC - rAI);

plotThreeWayPanels(bAI,bAC,bSI,bSC, rAI,rAC,rSI,rSC, nSubjects)



function plotInteraction2way(g1b1, g1b2, g2b1, g2b2, groupLabels, ~, titleText, nSubjects)
% Grouped 2x2 bar: 2 groups on the x-axis, 2 bars within each group.
% g1b1/g1b2 = group-1 bar-1/bar-2 ; g2b1/g2b2 = group-2 bar-1/bar-2 (1 x nSubj)
sem = @(x) std(x,'omitnan')/sqrt(sum(~isnan(x)));
M  = [mean(g1b1,'omitnan') mean(g1b2,'omitnan');
      mean(g2b1,'omitnan') mean(g2b2,'omitnan')];
SE = [sem(g1b1) sem(g1b2); sem(g2b1) sem(g2b2)];

colB1 = [0.60 0.30 0.55];   % bar 1 (Incongruent) - purple
colB2 = [0.25 0.55 0.40];   % bar 2 (Congruent)   - green

figure; hold on
b = bar(M,'FaceColor','flat','FaceAlpha',0.6);
b(1).CData = repmat(colB1,2,1);
b(2).CData = repmat(colB2,2,1);
x1 = b(1).XEndPoints;   % x-pos of bar-1 in [group1 group2]
x2 = b(2).XEndPoints;   % x-pos of bar-2 in [group1 group2]

jit = 0.06*(rand(1,nSubjects)-0.5);      % shared per subject -> clean paired lines
% group 1: connect each subject's incong->cong
plot([x1(1)+jit; x2(1)+jit], [g1b1; g1b2], '-', 'Color',[0.6 0.6 0.6 0.35]);
scatter(x1(1)+jit, g1b1, 55, 'k','filled','MarkerFaceAlpha',0.4);
scatter(x2(1)+jit, g1b2, 55, 'k','filled','MarkerFaceAlpha',0.4);
% group 2
plot([x1(2)+jit; x2(2)+jit], [g2b1; g2b2], '-', 'Color',[0.6 0.6 0.6 0.35]);
scatter(x1(2)+jit, g2b1, 55, 'k','filled','MarkerFaceAlpha',0.4);
scatter(x2(2)+jit, g2b2, 55, 'k','filled','MarkerFaceAlpha',0.4);

errorbar([x1 x2], [M(:,1)' M(:,2)'], [SE(:,1)' SE(:,2)'], ...
    'k','linestyle','none','LineWidth',3,'CapSize',14);

set(gca,'XTick',1:2,'XTickLabel',groupLabels);
ylabel('RT'); title(titleText); box off
ax = gca; ax.FontSize = 20; ax.FontWeight = 'bold';
end


function plotThreeWayPanels(bAI,bAC,bSI,bSC, rAI,rAC,rSI,rSC, nSubjects)
figure('Color','w','Position',[100 100 1150 460]);
titles = {'Biological','Scrambled'};
data   = {bAI,bAC,bSI,bSC; rAI,rAC,rSI,rSC};
sem = @(x) std(x,'omitnan')/sqrt(sum(~isnan(x)));
axh = gobjects(1,2);
for pnl = 1:2
    axh(pnl) = subplot(1,2,pnl); hold on
    g1b1=data{pnl,1}; g1b2=data{pnl,2}; g2b1=data{pnl,3}; g2b2=data{pnl,4};
    M  = [mean(g1b1,'omitnan') mean(g1b2,'omitnan'); mean(g2b1,'omitnan') mean(g2b2,'omitnan')];
    SE = [sem(g1b1) sem(g1b2); sem(g2b1) sem(g2b2)];
    b = bar(M,'FaceColor','flat','FaceAlpha',0.6);
    b(1).CData = repmat([0.60 0.30 0.55],2,1);
    b(2).CData = repmat([0.25 0.55 0.40],2,1);
    x1 = b(1).XEndPoints; x2 = b(2).XEndPoints;
    jit = 0.06*(rand(1,nSubjects)-0.5);
    plot([x1(1)+jit; x2(1)+jit],[g1b1; g1b2],'-','Color',[0.6 0.6 0.6 0.3]);
    plot([x1(2)+jit; x2(2)+jit],[g2b1; g2b2],'-','Color',[0.6 0.6 0.6 0.3]);
    scatter([x1(1)+jit x2(1)+jit x1(2)+jit x2(2)+jit], ...
            [g1b1 g1b2 g2b1 g2b2], 35,'k','filled','MarkerFaceAlpha',0.35);
    errorbar([x1 x2],[M(:,1)' M(:,2)'],[SE(:,1)' SE(:,2)'], ...
        'k','linestyle','none','LineWidth',3,'CapSize',12);
    set(gca,'XTick',1:2,'XTickLabel',{'Async','Sync'});
    title(titles{pnl}); box off
    if pnl==1
        ylabel('RT');
    end
    ax = gca; ax.FontSize = 16; ax.FontWeight = 'bold';
end
linkaxes(axh,'y');
sgtitle('Three-way: Sync \times Congruency, split by Motion Type', ...
    'FontWeight','bold','FontSize',18)
end





%% ---- local functions -------------------------------------------------
function b = getBeta(coef, name)
    idx = strcmp(coef.Row, name);
    if any(idx), b = coef.Estimate(idx); else, b = NaN; end
end

function plotMainEffect(d1, d2, ax1, ax2, title_text, nSubjects)

rtBio = d1;   % Biological
rtScrambled  = d2;    % Scrambled

m2  = [mean(rtBio,'omitnan'),  mean(rtScrambled,'omitnan')];
se2 = [std(rtBio,'omitnan')/sqrt(sum(~isnan(rtBio))), ...
       std(rtScrambled,'omitnan')/sqrt(sum(~isnan(rtScrambled)))];

jit  = 0.08*(rand(1,nSubjects)-0.5);

figure;
b = bar(m2,'FaceColor','flat','FaceAlpha',0.6); hold on
b.CData(1,:) = [0.25 0.55 0.40];
b.CData(2,:) = [0.60 0.30 0.55];

plot([1+jit; 2+jit], [rtBio; rtScrambled], '-', 'Color',[0.6 0.6 0.6 0.4]);
scatter(1+jit, rtBio, 100,'k','filled','MarkerFaceAlpha',0.35);
scatter(2+jit, rtScrambled,  100,'k','filled','MarkerFaceAlpha',0.35);

errorbar(1:2, m2, se2, 'k','linestyle','none','LineWidth',4, "CapSize", 20);
set(gca,'XTick',1:2,'XTickLabel',{ax1,ax2});
box off
title(title_text)
ax = gca; ax.FontSize = 20; ax.FontWeight = 'bold';

end