clear;
clc;

load('ADNI_data_all.mat');

%刘筱力实验数据，共775个样本
data_1 = data_all{1,1};
%取出data_1中的X
data_X = data_1(:,27:end-5);

%求每个特征的标准差
std_f_row = std(data_X);    %写成行向量
std_f = zeros(319,2);       %改写成列向量
%第一列是特征序号，第二列是特征的标准差
for i=1:319
   std_f(i,1) = i;
   std_f(i,2) = std_f_row(i);
end

%按照特征的标准差 降序排序
%第一列是特征编号，对应刘筱力实验数据
std_f_descend_775 = sortrows(std_f,-2);

savefile = 'std_f_descend_775.mat';
save(savefile,'std_f_descend_775');


