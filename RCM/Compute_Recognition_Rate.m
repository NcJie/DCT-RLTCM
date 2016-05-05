% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function outCRR = Compute_Recognition_Rate(trn_x, trn_y, tst_x, tst_y, Metric_Function)
    trn_y = trn_y(:);
    tst_y = tst_y(:);
    pair_dist = Compute_Metric(tst_x, trn_x, Metric_Function);
    
    [~,minIDX] = min(pair_dist);
    y_hat = trn_y(minIDX);
    outCRR = sum(tst_y == y_hat)/length(tst_y);
end