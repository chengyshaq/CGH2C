function [K,g_idx,g_hc] = L2HC(train_target)
%% Transform Labels to Huffman Coding (L2HC)
[Num_Labels,Num_Samples] = size(train_target);

for i = 1:Num_Labels
    Num_Pos(i) = length(find(train_target(i,:)==1));
end

p = Num_Pos/sum(Num_Pos);
[h,e] = Huffman_code(p);
[q,idx_p] = sort(p);
strHC = cellstr(h);

for j = 1:Num_Labels
   temp = strHC{j};
   for k = 1:Num_Labels
       temp_HC(k) = str2double(temp(k));
   end
   temp_HC(isnan(temp_HC)) = [];
   HC{j,1} = temp_HC;
   len(j) = length(temp_HC);
end
idx = sort(unique(len),'descend');
K = length(idx); 
for i = 1:K
    code_length = idx(i);
    for j = 1:Num_Labels
        temp_code = HC{j,1};
        if length(temp_code) == code_length
            m_idx(j) = i;
        end
    end
end
for i = 1:Num_Labels
    temp_idx = find(idx_p == i);
    g_idx(i) = m_idx(temp_idx);
    g_hc{i,1} = HC{temp_idx,1};
end
