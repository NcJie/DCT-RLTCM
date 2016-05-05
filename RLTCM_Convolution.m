% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

function OutImgs = RLTCM_Convolution(InImgCells, Filters, Params, layer)
    
    OutImgs = {};
    numFilter = Params.NumFilters(layer);
    patchSize = Params.FilterSize(layer);
    filter = Filters{layer};
    mag = (patchSize-1)/2;
    
    for i = 1:length(InImgCells)
        imgCell = InImgCells{i};
        for j = 1:length(imgCell)
            [h, w] = size(imgCell{j});
            filteredImgs = {};
            
            % Zero Padding
            img = zeros(h+patchSize-1,w+patchSize-1, 1);
            img((mag+1):end-mag,(mag+1):end-mag,:) = imgCell{j};
            img = im2col(img, [patchSize patchSize]);

            % Cross-correlation as convolution 
            for p = 1:numFilter
                if isstruct(filter)
                    % complex-valued filter bank
                    freal = filter.Real(:,p)'*img;
                    fimag = filter.Img(:,p)'*img;
                    fImg = sqrt(freal.^2 + fimag.^2);
                else
                    % real-valued filter bank 
                    fImg = filter(:,p)'*img;
                    
                    % non-linear operations 
                    if Params.AbsoluteDCT == 1
                        tmean = mean(fImg);
                        tstd = std(fImg);
                        fImg = (fImg - tmean)/tstd;
                        fImg = sqrt(abs(fImg));
                    end
                end
                fImg = reshape(fImg, [h w]);
                filteredImgs = cat(1, filteredImgs, fImg);
            end
            OutImgs = cat(1, OutImgs, { filteredImgs });
        end
    end
end
