% Paper  : C. J. Ng, A. B. J. Teoh and C. Y. Low, "DCT BASED REGION LOG-TIEDRANK COVARIANCE MATRICES FOR FACE RECOGNITION" ICASSP, 2016. 

clear all;
addpath('RCM');

%% Parameters
Params.FilterSize = 11;           % Filter Size
Params.NumFilters = 30;           % Number of Filter for each layer
Params.BlockSize = [20 20];       % Block-Wise Size
Params.Stride = [20 20];          % Block Stride (overlapping pixels between regions)
Params.AbsoluteDCT = 1;           % 0 => Disable (DCT filter), 1 => Enable (Absolute DCT filter responses)
Params.LogTiedRankRCM = 0;        % 0 => Disable, 1 => Enable
Params.RCMMetric = 1;             % 1 => AIRM, 2 => LogEuclidean, Only applicable to LogTiedRankRCM = 0
Params.EnableWPCA = 0;            % 0 => Disable, 1 => Enable, only applicable to LogTiedRankRCM = 1
Params.WPCADim = 300;             % WPCA reduce dimension 

if Params.RCMMetric == 1
    Metric_Function = @Metric_AIRM;
elseif Params.RCMMetric == 2
    Metric_Function = @Metric_LogEuc;
end

fprintf('\n ====== Params Parameters ======= \n')
disp(Params)

%% Load Dataset 
% Load Face Data
load('FERET_B_128x128_(a,c,h,j,k)');

% Gallery ba, bj, bk (frontal faces)
trainData.X = [FERET_ba.X FERET_bj.X FERET_bk.X];
trainData.y = [FERET_ba.y FERET_bj.y FERET_bk.y];

% Probe bc, bh (+40, -40) degree in pose
testData = {FERET_bc, FERET_bh};
datasetNames = {'bc', 'bh'};

%% Initialize Filters
Filters = { DCT_FilterBank(Params.FilterSize, Params.NumFilters) };

%% Gallery Feature Extraction
fprintf('\n ====== Gallery Feature Extraction ======= \n')
tic;

if Params.LogTiedRankRCM == 0
    ftrain = {};
else
    ftrain = [];
end

for i = 1:length(trainData.y)
    img = reshape(trainData.X(:, i), [imgHeight imgWidth]);
    % Single Channel Image 
    ftrain = cat(1, ftrain, RLTCM_FeaExtraction({ img }, Filters, Params));
end

if Params.LogTiedRankRCM == 1 && Params.EnableWPCA
    meanTrn = mean(ftrain, 1);
    ftrain = bsxfun(@minus, ftrain, meanTrn);
    reduceMat = WPCA_Svd(ftrain', Params.WPCADim)';
    ftrain = ftrain * reduceMat;
end

fprintf('\n     Gallery Feature Extraction Time : %.2f secs.\n', toc);

%% Probe Set  
for i = 1:length(testData) 
    tic
    if Params.LogTiedRankRCM == 0
        ftest = {};
    else
        ftest = [];
    end
    
    for j = 1:length(testData{i}.y)
        imgCell = { reshape(testData{i}.X(:, j), [imgHeight imgWidth]) };
        ftest = cat(1, ftest, RLTCM_FeaExtraction(imgCell, Filters, Params));
    end
    
    if Params.LogTiedRankRCM == 1 && Params.EnableWPCA
        ftest = bsxfun(@minus, ftest, meanTrn);
        ftest = ftest * reduceMat;
    end

    FeaExtTime = toc;

    %% Recognition Rate
    tic
    fprintf('\n ===== Results of Params with NN classifier =====');
    fprintf('\n     Dataset %s', datasetNames{i});
    fprintf('\n     Feature Extraction Time: %.2f sec', FeaExtTime);
    
    if Params.LogTiedRankRCM == 1        
        pair_dist = zeros(length(trainData.y), length(testData{i}.y));
        for j = 1:size(ftrain, 3)
            trn_X = ftrain(:,:,j);
            tst_X = ftest(:,:,j);
            
            if Params.EnableWPCA == 1
                pair_dist = pair_dist + pdist2(trn_X, tst_X, 'cosine');
            else
                pair_dist = pair_dist + pdist2(trn_X, tst_X, 'euclidean');
            end
        end
        [~,minIDX] = min(pair_dist);
        outCRR = sum(testData{i}.y ==  trainData.y(minIDX))/length(testData{i}.y);
        fprintf('\n      Recognition Rate: %.6f', outCRR);
    else
        outCRR = Compute_Recognition_Rate(ftrain, trainData.y, ftest, testData{i}.y, Metric_Function);
        fprintf('\n     Recognition Rate: %.6f', outCRR);
    end
    fprintf('\n');
    toc
    pause(0.001);
end