function [h,e] = Huffman_code(p)

if length(find(p<0))~=0
    error('p > 0£¡')
end
 
if abs(sum(p)-1)>10e-10
    error('sum(p) = 1£¡')
end

n=length(p);
p=sort(p);
q=p;
m=zeros(n-1,n);
for i=1:n-1
    [q,e]=sort(q); 
    m(i,:)=[e(1:n-i+1),zeros(1,i-1)];
    q=[q(1)+q(2),q(3:n),1]; 
end
 
for i=1:n-1
    c(i,1:n*n)=blanks(n*n); 
end
    c(n-1,n)='1';
    c(n-1,2*n)='0'; 
for i=2:n-1
    c(n-i,1:n-1)=c(n-i+1,n*(find(m(n-i+1,:)==1))-(n-2):n*(find(m(n-i+1,:)==1)));
    c(n-i,n)='0'; 
    c(n-i,n+1:2*n-1)=c(n-i,1:n-1);
    c(n-i,2*n)='1'; 
    for j=1:i-1
         c(n-i,(j+1)*n+1:(j+2)*n)=c(n-i+1,n*(find(m(n-i+1,:)==j+1)-1)+1:n*find(m(n-i+1,:)==j+1));
    end
end 
for i=1:n
    h(i,1:n)=c(1,n*(find(m(1,:)==i)-1)+1:find(m(1,:)==i)*n); %h is huffman coding
    len(i)=length(find(abs(h(i,:))~=32));
end
e=sum(p.*len);