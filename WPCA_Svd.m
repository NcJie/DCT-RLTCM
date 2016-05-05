% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function projMat = WPCA_Svd(X, dimension)

   rglEpsilon = 0.001;
   
   X = bsxfun(@minus, X, mean(X, 2));
   [eigVectors_PCA, eigValues_PCA, ~] = svd(X, 0); 
  
   eigVectors_PCA = eigVectors_PCA(:, 1 : dimension);
   eigVectors_PCA = eigVectors_PCA';
   assert(size(eigVectors_PCA, 1) == dimension);
  
   eigValues_PCA = diag(eigValues_PCA.^2);
   eigValues_PCA = eigValues_PCA(1 : dimension);
  
   % Whitening PCA
   eigValues_wPCA = diag(1./sqrt(eigValues_PCA + rglEpsilon));
   projMat = eigValues_wPCA * eigVectors_PCA;

end