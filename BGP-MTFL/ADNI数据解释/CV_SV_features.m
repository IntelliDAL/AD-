%�����������CV SA TA TS �ֿ�����
%��ȡ����� CV �� SV ����

clear
clc

%�ı�����ʱ��Ҫ�޸�������
load('data_775.mat')
data = data_775;

length = size(data,1);
ind =  load('CV_SV_ind.txt');
data_CV_SV = zeros(length,1);

for i=1:113
    
    data_CV_SV = cat(2,data_CV_SV,data(:,ind(i)));
    
end

data_CV_SV(:,1) = [];   %ɾ����ʼ0����
data_CV_SV(:,10) = [];  %ɾ��Դ��������̫�������_��­�����

data_CV_SV_775 = data_CV_SV;
savefile = 'data_CV_SV_775.mat';
save(savefile, 'data_CV_SV_775');