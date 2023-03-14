clear;
clc;

load('ADNI_data_all.mat');

%������ʵ�����ݣ���775������
data_1 = data_all{1,1};
%ȡ��data_1�е�X
data_X = data_1(:,27:end-5);

%��ÿ�������ı�׼��
std_f_row = std(data_X);    %д��������
std_f = zeros(319,2);       %��д��������
%��һ����������ţ��ڶ����������ı�׼��
for i=1:319
   std_f(i,1) = i;
   std_f(i,2) = std_f_row(i);
end

%���������ı�׼�� ��������
%��һ����������ţ���Ӧ������ʵ������
std_f_descend_775 = sortrows(std_f,-2);

savefile = 'std_f_descend_775.mat';
save(savefile,'std_f_descend_775');


