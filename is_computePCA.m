glPath = 'D:\HH_ECOG\chronicECoG\';

% Define recording to use
animalNamke = 'Nola';
recNum      = '04';
recPath     = [glPath animalName '\recordingsMat\kjh_' animalName '_sleep_' recNum '\'];

% load in LFP data
ichan = 25;
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
gamp = conv2(am,g,'same');             % convolve amp matrix with gaussian kernel
fmat = repmat(1./foi',1,size(gamp,2)); % Frequency normalization matrix (correct for 1/f distribution)
tvec = (1:size(gamp,2))/Fs;            % time vector for spectral data

% plot spectrogram
imagesc(tvec/60,1:numFreqs,log10(gamp.^2./fmat))
y = colorbar;
ylabel(y,'Signal power log10(uV^2/Hz)')
% detect location of ticks
fois = [0.5 1 2 4 8 16 32 64];
tickLabel = {'0.5','1','2','4','8','16','32','64'};
tickLoc = nan(1,numel(fois));
for fi = 1:numel(fois)
    [bi,bb] = sort(abs(foi-fois(fi)));
    tickLoc(fi) = bb(1);
end
set(gca,'YDir','normal','YTick',tickLoc,'YTickLabel',tickLabel,'TickDir','out')
xlabel('Time (minutes)')
ylabel('Frequency (Hz)')
colormap(jet)
caxis([4.5 7.5])

%% Z-score transform power time series accounting for NaN's
x      = gamp.^2';
mu     = nanmean(x);
sigma  = nanstd(x);
sigma0 = sigma;
sigma0(sigma0==0) = 1; % control for dividing by 0
z      = bsxfun(@minus,x, mu); % subtract mean
z      = bsxfun(@rdivide, z, sigma0); % divide by standard deviation
zamp   = z;

%% Perform PCA on z-scored power spectrogram
[coeff,score,latent,tsquared,explained,mu] = pca(zamp);

%% Plot the amount of variance explained by each principal component
nComp = numel(explained);
figure
plot(1:nComp,cumsum(explained),'b.','MarkerSize',10); hold on
plot([0 nComp],ones(1,2)*80,'r--','LineWidth',1)
set(gca,'TickDir','out')
ylim([0 100])
xlim([0 20])
xlabel('Principal component')
ylabel('Percentage of explained variance')

%% Plot the loadings for each principal component
figure
semilogx(foi,coeff(:,1:4),'LineWidth',2); hold on
plot([foi(1) foi(end)],[0 0],'k--')
xlim([foi(1) foi(end)])
xlabel('Frequency (Hz)')
ylabel('Principal component loadings')
legend('PC 1', 'PC 2','PC 3','PC 4')
set(gca,'TickDir','out')

%% Plot projection into first 3 dimensions of 'state space'
figure
plotTime = 25; % time to plot in minutes
tvec = 1:(plotTime*60*Fs);
cline(score(tvec,1),score(tvec,2),score(tvec,3),tvec/(60*Fs))
xlabel('Principal component 1'); xlim([-10 10])
ylabel('Principal component 2'); ylim([-10 10])
zlabel('Principal component 3'); zlim([-5 10])
colormap(c_lut);
y = colorbar;
ylabel(y,'Time (minutes)')


