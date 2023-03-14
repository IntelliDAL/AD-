clear;
clc;

%��������ϵ������PCC���㣬δ��һ������
load('data_815.mat');
data = data_815;
len = size(data,2);

corthreshold = 0.6;
%C��PCC���������������������
[H, E, Ecoef, Esign, C]=mygennetwork_f(data,corthreshold);

%ͳ�� δ��һ������ ����PCC����֮���star feature
C_1 = C - eye(len);     %��������Խ�����0

%��C_1���������ֵ�ͣ�������ÿ���������ӱ�Ȩ�صľ���ֵ��
sum_C_row = zeros(len,2);
for i=1:len
    sum_C_row(i,:) = [i,sum(abs(C_1(i,:)))];    
end

sum_C_row_des = sortrows(sum_C_row,-2);

%����feature���ӵı�����
num = zeros(len,1);
for i = 1:len
    
    for j = 1:len
        
        if C_1(i,j)>0
            num(i) = num(i)+1;
        end
    end
end

line_num = cat(2,(1:len)',num);
line_num_des = sortrows(line_num,-2);    
