function [rmse1,cor1,nmsek] = perf_regression( xtest, ytest, w)

    ntasks = size(ytest,2);
    yhat = xtest*w;
    rmse1 = rms(yhat - ytest);
	for j=1:ntasks,
        cor1(j) = corr(yhat(:,j),ytest(:,j));
        nmsek(j) = (norm(ytest(:,j)-yhat(:,j))^2)/std(ytest(:,j));
    end
