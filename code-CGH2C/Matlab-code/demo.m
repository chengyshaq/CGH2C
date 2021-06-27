% Demo
clear;clc
%% Make experiments repeatedly
rng(1);

%% Add necessary pathes
addpath('datasets','Measures','functions');

%% Choose a dataset
datasets  =   'emotions';
load([datasets,'.mat']);

%% Set data
if exist('train_data','var')==1
    data    = [train_data;test_data];
    target  = [train_target,test_target];
    clear train_data test_data train_target test_target
end
lastcol = ones(size(data,1),1);
data = [data, lastcol];
target(target==0) = -1;
%% Set parameters
opts.lambda  = 0.0001;    
opts.C       = 2^1;       
opts.gamma   = 2^2;       

%% Perform n-fold cross validation
num_fold = 5; Results = zeros(9,num_fold);
indices = crossvalind('Kfold',size(data,1),num_fold);
for i = 1:5
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test;
    tic; [Outputs, Pre_Labels,test_Labels] = CGH2C_ELM(data(train,:),target(:,train),data(test,:),target(:,test),opts);
    Results(1,i) = toc;
    [RK_6M, IB_6M, MM_8M] = evaluation_20measures(test_Labels, Outputs, Pre_Labels);
    % ranking-based measures
    RankingLoss = RK_6M.RankingLoss; OneError = RK_6M.OneError;
    % instance-based measures
    HammingLoss = IB_6M.HammingLoss; SubsetAccuracy = IB_6M.SubsetAccuracy;
    % label-based measures
    MacroF1 = MM_8M.MacroF1; MicroF1 = MM_8M.MicroF1;
    MacroAUC = MM_8M.MacroAUC; MicroAUC = MM_8M.MicroAUC;
    Results(2:end,i) = [RankingLoss,OneError,HammingLoss,SubsetAccuracy,MacroF1,MicroF1,MacroAUC,MicroAUC];
end
meanResults = squeeze(mean(Results,2));
stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Show the experimental results
printmat([meanResults,stdResults],datasets,...
    'Time RankingLoss OneError HammingLoss SubsetAccuracy MacroF1 MicroF1 MacroAUC MicroAUC','Mean Std.');
