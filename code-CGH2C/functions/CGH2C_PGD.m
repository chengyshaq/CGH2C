function V = CGH2C_PGD( X,Y,g_hc,K,m,lambda)
% Cross-Granularity Hierarchy based on Huffman Coding (CGH2C) via PGD
%% Get the size of data
[num_data, num_feature] = size(X);

%% Transform Y into Z
Z = zeros(K, num_data);
for i = 1:K
    label_idx = find(m==i);
    meta_Z = Y(label_idx,:);
    for j = 1:num_data
        hc = [];
        id = find(meta_Z(:,j)==1);
        if isempty(id)
            hc = [hc,0];
        else
            hc = [hc,g_hc{label_idx(id),1}];
        end
        hc_label(j) = bi2de(hc);
    end
    meta_size = size(meta_Z,1);
    if meta_size > 1
        [~, ~, index] = unique(hc_label);
        index = index - 1;
%         if max(index) == 0
%             Z(i,:) = index./max(index+1);
%         else
            Z(i,:) = index./max(index);
%         end
        meta_size = size(meta_Z,1);
    else
        Z(i,:) = meta_Z;
    end
end

%% Solve Lasso by PGD

[V]= PGD(X,Z',lambda);
end
