% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function distance = Metric_AIRM(X, Y)
distance = sqrt(sum(log(eig(X,Y)).^2));