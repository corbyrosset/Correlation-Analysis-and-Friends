%% playXRMB will load the audio and articulatory data as well as the 
% transcription or text-alignment data for a given speaker for a given 
% task and plot the audio/artic data and play the audio waveform
%
% Usage: [X,fs,artic,tgrid]=playXRMB(spkr,task)
% Inputs: spkr - a string specifying the speaker name (e.g. 'JW11')
%         task - a string specifying the task name (e.g. '079')
% Outputs: X  - sampled audio waveform, vector of size 1xN (N samples) 
%          fs - sampling rate of the audio waveform
%          artic - articulatory data of size 17xM (first dimension is time, 
%                  remaining 16 dimensions are horizontal and vertical  
%                  displacement of 8 pellets on speaker's various 
%                  articulators, M samples)
%          tgrid - Text alignment data 
%
% Example: [X,fs,artic,tgrid]=playXRMB('JW11','079');
%%            

function [X,fs,artic,tgrid]=playXRMB(spkr,task)

%% Speaker data
datapath=sprintf('../DATA/XRMB/%s',spkr);

%% Task specific data files 
audiofile=sprintf('%s/downsampled/TP%s.wav',datapath,task);
gridfile=sprintf('%s/alignment/TP%s.WAV.TextGrid',datapath,task);
articfile=sprintf('%s/xy/TP%s.TXY',datapath,task);

if(~(exist(audiofile,'file') && exist(audiofile,'file') &&...
        exist(audiofile,'file')))
    warning('Data does not exist for speaker %s for task %s\n',spkr,task);
    X=[]; fs=[]; artic=[]; tgrid=[];
else
    %% Read data
    [X,fs]=audioread(audiofile);
    artic=load(articfile);
    artic(artic==1e6)=NaN;
    tgrid=textscan(gridfile,'%s','commentstyle','matlab',...
        'headerlines',6,'delimiter','\n','bufsize',4096000);
    
    %% Plot audio/artic data
    figure; taskstr=sprintf('%s, TP%s',spkr,task);
    subplot(2,1,1); plot(X); grid on; set(gca,'FontSize',20); 
    axis tight; title(sprintf('Audio waveform (%s)',taskstr)); 
    subplot(2,1,2); plot(artic(:,2:end)); grid on; set(gca,'FontSize',20); 
    axis tight; title(sprintf('Articulatory data (%s)',taskstr)); 
    
    %% Play audio file
    soundsc(X,fs);
end
end