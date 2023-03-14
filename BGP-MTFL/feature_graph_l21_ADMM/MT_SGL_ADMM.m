clear;
clc;

% mex('dpotrf_.c', '-lmwblas','-lmwlapack');
% mex('dpotrs_.c', '-lmwblas','-lmwlapack');

% Data ADNI1 1.5T
load('Data_bl10');
Train = Data.Train;
Test = Data.Test;

lambda_1 = 0;
lambda_2 = 100;

Iters = length(Train);
FeaTask = 25;
Tasks_20 = 20;
Tasks = 18;
Ncross = 5;
% Dimen 320
Dimen = size(Train{1,1},2)-FeaTask-1-4;


ind = load('ind.txt');
ind = ind(1:length(ind)-4);
ind = cat(1,ind,ind(end)+1);
indt = ind;
for i = 1:Tasks-1
        indt = cat(1,indt,ind+Dimen*i);
end
ind = [0,indt'];

% Group Property
l = length(ind)-1;
for i = 1:l
    a = ind(i);
    b = ind(i+1);
    nn = b-a;
    temp = [a+1,b,sqrt(nn)];
    ind_t(:,i) = temp';
end
ind = ind_t;


% set opts
opts.Print = 0;
opts.MaxIter = 1000;
opts.Tol = 1e-3;
opts.rho = 10;
opts.Dimen = Dimen;
opts.ind = ind;
opts.Tasks = Tasks;

%Ϊ����Ԥ�����ڴ�
rmsei = zeros(Iters,Tasks);
cori = zeros(Iters,Tasks);
wri = zeros(1,Iters);
nmsei = zeros(1,Iters);
xi = cell(1,Iters);
gi = cell(1,Iters);
Ei = cell(1,Iters);
r1i = cell(1,Iters);
r2i = cell(1,Iters);
W_print = cell(1,Iters);
P_print = cell(1,Iters);
Q_print = cell(1,Iters);
R_print = cell(1,Iters);
lambda1 = cell(1,Iters);
lambda2 = cell(1,Iters);
lambda3 = cell(1,Iters);
%����Ԥ�����ڴ����?% lambda_1s = [1000 100 100 100 100 100 100 100 100 100]
%lambda_1s = [100 100 100 100 100 100 100 100 100 100]

%lambda_3s = [100 100 100 100 100 100 100 100 100 100]

corthreshold = 0;

for i = 1:Iters
    fprintf('Iters,%d\n',i);
    Traini = Train{1,i};
    Testi = Test{1,i};
    Xtrain = cat(2,Traini(:,FeaTask+2:end-5),Traini(:,end));
    Xtest = cat(2,Testi(:,FeaTask+2:end-5),Testi(:,end));
    Ytrain_all = Traini(:,2:FeaTask+1);
    Ytest_all = Testi(:,2:FeaTask+1);
    Ytrain_20 = Ytrain_all(:,1:Tasks_20);
    Ytest_20 = Ytest_all(:,1:Tasks_20);
    
    Ytrain = cat(2,Ytrain_20(:,1:9),Ytrain_20(:,12:20));
    
    Ytest = cat(2,Ytest_20(:,1:9),Ytest_20(:,12:20));
    
    %?????
%     [H, E, Ecoef, Esign, C] = mygennetwork_t(Ytrain,corthreshold);
    
%     h = size(E,1);    % n=0.7112, w=0.4185   
%     C = -C + eye (Tasks);
%     C = C + sum(abs(C(:)))*eye(Tasks);
%     D = C/h;%�����е�Z
    
    %?????
    %??X????????????b????????
    Xtrain_com_graph = Xtrain(:,1:319);
    [H_s, E_s, Ecoef_s, Esign_s, C_s] = mygennetwork_f(Xtrain_com_graph,corthreshold);
    C_s(:,320) = 0;
    C_s(320,:) = 0;
    h_s = size(E_s,1);    % n=0.7112, w=0.4185
    C_s = -C_s + eye(Dimen);
    C_s = C_s + sum(abs(C_s(:)))*eye(Dimen);
    D_s = C_s / h_s;%�Ƶ��е�S
    
    [lambda_1,lambda_3] = nest_cv_3(Xtrain,Ytrain,Ncross,D_s,opts); 
    %lambda_1 = s(i);
%     lambda_2 = lambda_2s(i);
    %lambda_3  =lambda_3s(i);
    lambda1{1,i} = lambda_1;
%     lambda2{1,i} = lambda_2s(i);
    lambda3{1,i} = lambda_3;
    [W,Q,R] = FLADMM(Xtrain, Ytrain, lambda_1, lambda_3,D_s, opts);
    W_print{1,i} = W;
%     P_print{1,i} = P;%����ͼ�ṹWZ
    Q_print{1,i} = Q;
    %R����ͼ�ṹ SW
    R_print{1,i} = R;
    [rmse,cor,nmse]= perf_regression(Xtest,Ytest,W);
    
    rmsei(i,:) = rmse;
    cori(i,:) = cor;
    [n,~] = size(Ytest);
    wri(i) = sum(cor*n)/(n*Tasks);
    nmsei(i) = sum(nmse)/(n*Tasks);  
    xi{i} = W;
%     gi{i} = D;
%     Ei{i} = E;
    r1i{i} = lambda_1;
    r2i{i} = lambda_3;
end
rmse_mean = mean(rmsei)
rmse_std = std(rmsei)
cor_mean = mean(cori)
cor_std = std(cori)
nmse_mean = mean(nmsei)
nmse_std = std(nmsei)
wr_mean = mean(wri)
wr_std = std(wri)

result.x = xi;
result.G = gi;
result.E = Ei;
result.r1 = r1i;
result.r2 = r2i;
result.rmse = rmsei;
result.cor = cori;
result.wr = wri;
result.nmse = nmsei;
result.W = W_print;
% result.P = P_print;
result.Q = Q_print;
result.R = R_print;
result.lambda1 = lambda1;
result.lambda2 = lambda2;
result.lambda3 = lambda3;
result.S = D_s;

savefile = 'feature_Graph_l21_indf-0.1-soft_18tasks.mat';
save(savefile, 'result');
save('W_featuregraph.mat','W')
save('lsmbds_1.mat','lambda1')
save('lsmbds_3.mat','lambda3')

    