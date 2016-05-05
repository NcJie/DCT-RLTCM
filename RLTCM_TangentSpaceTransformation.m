% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function tangentFeas = RLTCM_TangentSpaceTransformation(covs, p)

    % Input
    %    Covs - Covariance Matrices size(Covs) => n x n x k
    %    p - projection matrix => n x n
    % 
    % Description
    %    Reference : Pennec, Xavier, Pierre Fillard, and Nicholas Ayache. “A Riemannian Framework for Tensor Computing.” International Journal of Computer Vision 66, no. 1 (January 2006): 41–66..
    %    Step 1. Flatten manifold at point p to obtain the point p tangent space
    %            s = p^0.5 * logm(p^-0.5 * c * p^-0.5) * p^0.5 
    %    Step 2. Vectorize s by obtaining only upper triangle / lower triangle elements of s  
    %            v = [s(1,1) sqrt(2)*s(1,2) sqrt(2)*s(1,3) ... sqrt(2)*s(n,n-2), sqrt(2)*s(n,n-1), s(n,n)]
    
    epsilon = 0; % 10e-6;
    Pa = p^-0.5;
    Pb = p^0.5;

    index = triu(ones(size(covs, 1)))==1;
    tangentFeas = [];
    
    for k = 1:size(covs, 3)
        c = covs(:,:,k);
        s = Pb*logm(Pa*(c + eye(size(c))*epsilon)*Pa)*Pb;
        v = sqrt(2)*triu(s,1)+diag(diag(s));
        tangentFeas = cat(3, tangentFeas, v(index)');
    end
end