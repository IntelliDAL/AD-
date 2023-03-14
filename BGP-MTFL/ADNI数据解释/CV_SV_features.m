%将特征里面的CV SA TA TS 分开分析
%提取区域的 CV 和 SV 特征

clear
clc

%改变输入时需要修改这两行
load('data_775.mat')
data = data_775;

length = size(data,1);
ind =  load('CV_SV_ind.txt');
data_CV_SV = zeros(length,1);

for i=1:113
    
    data_CV_SV = cat(2,data_CV_SV,data(:,ind(i)));
    
end

data_CV_SV(:,1) = [];   %删除初始0向量
data_CV_SV(:,10) = [];  %删除源数据量级太大的特征_总颅内体积

data_CV_SV_775 = data_CV_SV;
savefile = 'data_CV_SV_775.mat';
save(savefile, 'data_CV_SV_775');