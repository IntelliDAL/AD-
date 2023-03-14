clear;
clc;

%特征关联系数矩阵，PCC计算，未归一化数据
load('data_815.mat');
data = data_815;
len = size(data,2);

corthreshold = 0.6;
%C是PCC计算出来的特征关联矩阵
[H, E, Ecoef, Esign, C]=mygennetwork_f(data,corthreshold);

%统计 未归一化数据 特征PCC计算之后的star feature
C_1 = C - eye(len);     %关联矩阵对角线置0

%对C_1的行求绝对值和，即等于每个特征连接边权重的绝对值和
sum_C_row = zeros(len,2);
for i=1:len
    sum_C_row(i,:) = [i,sum(abs(C_1(i,:)))];    
end

sum_C_row_des = sortrows(sum_C_row,-2);

%计算feature连接的边数量
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
