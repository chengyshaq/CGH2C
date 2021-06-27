function [Outputs,Pre_Labels,Labels] = CGH2C_ELM( train_data,train_target,test_data,test_target,opts )
% CGH2C: Cross-Granularity Hierarchy based on Huffman Coding for Label-Specific Features Learning
%% Set parameters
lambda = opts.lambda;
C      = opts.C;
gamma  = opts.gamma;

%% Transform Labels to Huffman Coding (L2HC)
[K,g_idx,g_hc] = L2HC(train_target);

%% Cross-Granularity Hierarchy based on Huffman Coding via PGD (CGH2C_PGD)
V = CGH2C_PGD( train_data,train_target,g_hc,K,g_idx,lambda);
V(abs(V)<=1e-4) = 0;
V(V~=0)=1;
train_target(train_target==0) = -1;

%% Build classifier via ELM
for j = 1:K 
    idx_feature = (V(:,j)~=0); idx_meta = (g_idx==j);
    meta_train_data = train_data(:,idx_feature);
    meta_test_data = test_data(:,idx_feature);
    meta_train_target = train_target(idx_meta,:)';
    meta_test_target = test_target(idx_meta,:)';
    [OutputWeight,Omega_test] = kelmtrain (meta_train_data, meta_train_target, meta_test_data, C, gamma);
    meta_Pre_Outputs = kelmpredict (OutputWeight,Omega_test);
    meta_Pre_Labels = sign(meta_Pre_Outputs);
    temp_Outputs{j,1} = meta_Pre_Outputs';
    temp_Pre_Labels{j,1} = meta_Pre_Labels';
    test_Labels{j,1} = meta_test_target';
end

Pre_Labels = cell2mat(temp_Pre_Labels);
Outputs = cell2mat(temp_Outputs);
Labels = cell2mat(test_Labels);
Labels(Labels==0) = -1;
end
