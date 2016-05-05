% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function feature = RLTCM_FeaExtraction(InImgCell, Filters, Params)

    % InImgCell => Multi Channel Image
    
    %% Convolutions 
    filteredImgCells = RLTCM_Convolution({InImgCell}, Filters, Params, 1);

    %% Augmented Features (image cues)
    for i = 1:length(InImgCell)
        img = InImgCell{i};
        [h, w] = size(img);
        % [gx, gy] = gradient(img);
        
        filteredImgCells{i} = cat(1, ones(1,h)' * (1:w), filteredImgCells{i}); 
        filteredImgCells{i} = cat(1, (1:h)' * ones(1,w), filteredImgCells{i});
        % filteredImgCells{i} = cat(1, gx, filteredImgCells{i});
        % filteredImgCells{i} = cat(1, gy, filteredImgCells{i});
    end
    
    %% Block-wise Covariance
    blockwiseCovs = RLTCM_BlockwiseCovariance(filteredImgCells, Params);
    
    %% Tangent Space 
    if Params.LogTiedRankRCM == 0
        feature = blockwiseCovs;
    else
        p = eye(repmat(size(blockwiseCovs, 1), [1 2]));
        tangentFeas = RLTCM_TangentSpaceTransformation(blockwiseCovs, p);
        
        if Params.LogTiedRankRCM == 1
            dim = size(tangentFeas);
            for i = 1:size(tangentFeas, 1)
                tangentFea = reshape(tangentFeas(i,:,:), [dim(2:end) 1]);
                rankedFea = tiedrank(tangentFea);
                tangentFeas(i,:,:) = rankedFea;
            end
        end
        
        feature = tangentFeas(:)';
        
        % Remove potential complex number 
        feature = sign(real(feature)) .* abs(feature);
    end
end
