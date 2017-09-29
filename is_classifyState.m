function is_classifyState(recPath,varargin)
% To classify brain states, this function has several steps: 
% Step 1: uses previously computed PCA eigenvector to
%         transform spectral data to state space. 
% Step 2: Data are then clustered in state space assuming between 2 to 8 clusters.
% Step 3: We then determine the  optimal number of clusters by computing the squared
%         euclidean distance between clusters. 
% Step 4: We re-cluster data multiple times to avoid any bias introduced by
%         random partitioning (initial case).
% Step 5: Interpolate state index vector to original sample rate. 
% I.S. 2016

set(0,'DefaultFigureVisible','off')
fprintf('Classifying state %s \n',recPath)

if isempty(varargin)
    ichan = 25; 
else
    ichan = varargin(1); 
end

% load in LFP data
[LFP,lfpFs] = is_load([recPath 'LFP/LFP_1_EU64L_' num2str(ichan)],'LFP','lfpFs');
sampVec = 1:numel(LFP);

%% Global mean rereferencing
nChans = 64;
lfpMat = nan(nChans,numel(LFP));
fprintf('Global mean rereferencing... \n')
for ichan = 1:nChans
    lfpChan = is_load([recPath 'LFP/LFP_1_EU64L_' num2str(ichan)],'LFP');
    lfpMat(ichan,:) = lfpChan;
end
% subtract global mean from channel of interest
LFP = LFP' - mean(lfpMat);

%% detect noise and reject 
rejThresh = 10;                                     % 10 standard deviation noise threshold
tcut    = 10;                                       % seconds to cut out around noisy epochs
delWin  = ones(1,round(lfpFs*tcut));                % 10s window to cut out
delInd  = (abs(zscore(LFP)) > rejThresh);           % Samples that surpass threshold
delVec  = (conv(double(delInd),delWin,'same') > 0); % Convolve to smooth window and detect non-zero samples
LFP(delVec) = nan;                                  % Noisy samples change to NaN's

%% define frequencies of interest and compute family of Morlet wavelets
lowFreq      = 0.3;
highFreq     = 100;
numFreqs     = 80;
foi          = logspace(log10(lowFreq),log10(highFreq),numFreqs);
wavs         = is_makeWavelet(foi,lfpFs); % subfunction to compute complex wavelets

c = nan(numel(foi),numel(LFP)); % initialize spectral matrix
f = 0;
while f < numFreqs
    f = f + 1;
    fprintf('Convolving wavelet freq = %f %i/%i \n',foi(f),f,numFreqs)
    c(f,:) = conv(LFP,wavs{f},'same'); % convolve LFP signal with wavelet at carrier frequency foi(f)
end

%% Compute analytic amplitude of signals at each carrier frequency and downsample
ampMat  = abs(c);                      % analytic amplitude
powMat  = ampMat.^2;                   % Power of signal
downFac = 100;                         % downsample factor
[dr,nr] = rat(lfpFs/(lfpFs/downFac));  % rational fraction of lfp sample rate and downsampled rate
am      = resample(ampMat',nr,dr)';    % downsample data to new sample rate (fs = lfpFs/downFac)

%% Compute gaussian function for smoothing analytic amplitude time series
Fs     = lfpFs/downFac;
w      = 3;                   % width of gaussian function in SD's
st     = 11;                  % standard deviation in the time domain
t      = -w*st:1/Fs:w*st;     % time vector for calculation
tap    = exp(-t.^2/(2*st^2)); % Gaussian function
g      = tap/sum(tap);        % Normalize sum to 1

% % plot gaussian function
% plot(t,g,'k','LineWidth',2);
% xlabel('Time (s)')
% ylabel('Amplitude')

%% Convolve analytic amplitude time series with Gaussuan kernel
gamp = conv2(am,g,'same'); % convolve amp matrix with gaussian kernel
fmat = repmat(1./foi',1,size(gamp,2)); % Frequency normalization matrix (correct for 1/f distribution)
tvec = (1:size(gamp,2))/Fs; % time vector for spectral data

%% Z-score transform power time series accounting for NaN's
x      = gamp.^2';
mu     = nanmean(x);
sigma  = nanstd(x);
sigma0 = sigma;
sigma0(sigma0==0) = 1; % control for dividing by 0
z      = bsxfun(@minus,x, mu); % subtract mean
z      = bsxfun(@rdivide, z, sigma0); % divide by standard deviation
zamp   = z;

%% Load PCA eigenvector coefficients and multiply with z-scored spectra to create state space vectors
coeff = is_load('D:\HH_ECOG\clusterAnalysis\Nola_4_coeff_25','coeff');
score  = zamp * coeff;

%% Determine the optimal number of clusters in state space

numClus = 2:8; % Let's assume between 2 to 8 brain state clusters
mnEUdist = nan(1,max(numClus)); % initialize 
mnBIC    = nan(1,max(numClus)); % initialize
for n = numClus
    % K-means method
    [IDX,~]     = kmeans(score(:,1:4),n); % K-means clustering assuming n clusters
    [silh,~]    = silhouette(score(~isnan(IDX),1:4),IDX(~isnan(IDX)),'sqeuclidean');
    mnEUdist(n) = mean(silh);
    
    % Gaussian mixture model method
    rejInd  = isnan(IDX);
    useInd  = ~isnan(IDX);
    fillInd = find(useInd>0); % only use non nan's
    
    % Mixed Gaussian model expectation maximization algorithm
    [label, model, llh] = mixGaussEm(score(useInd,1:4)', n);
    lx          = nan(size(IDX));
    lx(fillInd) = label;
    IDX         = lx; % reshape back to original size and reinstate nan's
    
    % Compute bayesian information criterion 
    [aic,bic] = aicbic(llh,numel(llh),size(score,1));
    mnBIC(n)  = bic(end);
end

% clear some variables to free up some memory
clear ampMat am fmat gamp x z c

%% 
% Perform k-means clustering 100 times to account for deviation in
% clustering performance based on arbitrary selection of initial cluster
% centroids
numReps   = 100;
numClus   = 3;
testIndex = nan(numReps,numel(LFP));
deltaFreq = 23; % ~1.5Hz
alphaFreq = 53; % ~13.7Hz
gammaFreq = 73; % ~60Hz
% cluster assuming 3 clustered (based on squared Euclidean distance)
for irep = 1:numReps
    [IDX,~] = kmeans(score(:,1:4),3); % K-means with 3 clusters
    intIDX  = round(interp1(tvec,IDX,sampVec/lfpFs)); % interpolate to original sample rate
    
    % Compute the power at slow wave, alpha, and gamma frequencies to determine
    % what cluster corresponds to each brain state
    for k = 1:numClus
        ind = (intIDX == k); tmpMn = mean(powMat(:,ind),2);
        lowPow(k) = tmpMn(deltaFreq);
        alpPow(k) = tmpMn(alphaFreq);
        gamPow(k) = tmpMn(gammaFreq);
    end
    
    stateIndex = nan(size(intIDX));
    
    % slow wave sleep
    [~,I] = sort(lowPow,'descend');
    stateIndex(intIDX == I(1)) = 1;
    
    % intermediate/awake
    stateIndex(intIDX == I(2)) = 2;
    
    % REM sleep
    stateIndex(intIDX == I(3)) = 3;
        
    testIndex(irep,:) = stateIndex;
end

% compute the median classification after 100 repetitions
stateIndex = nanmedian(testIndex);

% compute the power spectra of each clustered brain state
SWSpow = nanmean(powMat(:,stateIndex==1),2);
INTpow = nanmean(powMat(:,stateIndex==2),2);
REMpow = nanmean(powMat(:,stateIndex==3),2);

save([recPath 'stateIndex_EM'],'stateIndex','intIDX','mnBIC','mnEUdist','score','coeff','SWSpow','INTpow','REMpow','foi','zamp');


