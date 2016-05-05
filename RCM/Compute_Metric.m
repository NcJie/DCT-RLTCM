% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function outMetric = Compute_Metric(Set_Cell_1 ,Set_Cell_2, Metric_Function )

% Set_Cell type is cell
%   Within each cell, there's a 3D matrix consists of cov for each image

l1 = size(Set_Cell_1,1);
l2 = size(Set_Cell_2,1);
outMetric = zeros(l2,l1);

for c1 = 1:l1
    X = Set_Cell_1{c1};
	
	if iscell(X)
		region = length(X);
	else
		[~,~,region] = size(X);
	end
	
    for c2 = 1:l2
        Y = Set_Cell_2{c2};
        distance = zeros(region,1);
		
        if iscell(Y)
            for r = 1:region
                distance(r) = Metric_Function(X{r}, Y{r});
            end
        else
            for r = 1:region
                distance(r) = Metric_Function(X(:,:,r), Y(:,:,r));
            end
        end
        
        score = sum(distance);
        
        if score < 1e-10
            score = 0.0;
        end
        outMetric(c2, c1) = score;
    end
	
	if mod(c1, 10) == 0
		fprintf('Computed Distance %d of %d\n', c1, l1);
        pause(0.001);
	end
end
        
end