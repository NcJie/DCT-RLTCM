% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function dctFilters = DCT_FilterBank(FilterSize, FilterCount)
    ps = FilterSize;
    nf = FilterCount;
    dctMatrix = DCT_Matrix(ps);
    diagIndex = AntiDiagonal_Index(ps, 1);
    dctMatrix(:, diagIndex) = dctMatrix(:,:);
    dctFilters = dctMatrix(:, 2:nf+1);
end

function dctMatrix = DCT_Matrix(N)
    dctMat = dctmtx(N);
    dctMatrix = zeros(N * N, N * N);

    for i = 1:N
        for j = 1:N
            M = dctMat(i,:)' * dctMat(j,:);
            dctMatrix(:, (i-1) * N + j) = M(:);
        end
    end
end

function indices = AntiDiagonal_Index(n, direction)
    % direction = 1 => north east
    % direction = 0 => south west
    indices = zeros(n, n);
    
    idx = 1;
    for i = 1:n
        u = i; v = 1;
        while u
            indices(u, v) = idx;
            indices(n - u + 1, n - v + 1) = n*n - idx + 1;
            idx = idx + 1;
            u = u - 1;
            v = v + 1;
        end
    end
    
    if direction
        indices = indices';
    end
end
