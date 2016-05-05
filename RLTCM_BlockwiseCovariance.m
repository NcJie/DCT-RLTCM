% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function OutCovs = RLTCM_BlockwiseCovariance(InImgs, Params)
    
    blkSize = Params.BlockSize;
    stride = Params.Stride;
    
    tensors = {};
    
    % Crop to center, in case image size is not divisible by BlockSize
    for d = 1:length(InImgs)
        imgs = InImgs{d};
        stackImgs = [];
        
        for i = 1:length(imgs)
            stackImgs = cat(3, stackImgs, imgs{i});
        end
        
        [height, width, ~] = size(stackImgs);
        margin = [height width] - blkSize .* floor([height width] ./ blkSize);
        margin1 = round(margin / 2);
        margin2 = margin - margin1;
        
        stackImgs = stackImgs((margin1(1) + 1):(end - margin2(1)),(margin1(2) + 1):(end - margin2(2)),:);
        [height, width, ~] = size(stackImgs);
        
        tensors = cat(1, tensors, { stackImgs });
    end
    
    covs = [];
    
    dim = [height width];
    remain = blkSize - stride;
    p = ceil((dim - remain) ./ stride);
    newDim = p .* stride + remain;
    blocks = (newDim - blkSize) ./ stride + 1;
    rows = 1:blkSize(1); cols = 1:blkSize(2);
 
    for i = 1:length(tensors)
        stackImgs = tensors{i};

        temp = zeros(newDim(1), newDim(2), size(stackImgs,3));
        temp(1:dim(1),1:dim(2),:) = stackImgs;
        stackImgs = temp;
        
        for r=0:blocks(1)-1,
            for c=0:blocks(2)-1,
                block = stackImgs(r*stride(1) + rows, c*stride(2) + cols,:);
                [h, w, f] = size(block);
                block = reshape(block, [h * w f]);
                
                covs = cat(3, covs, cov(block, 1));
            end
        end
    end
    OutCovs = covs;
end
