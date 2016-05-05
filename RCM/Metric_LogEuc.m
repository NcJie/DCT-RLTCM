% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function distance = Metric_LogEuc(X, Y)
distance = norm((logm(X)-logm(Y)),'fro');