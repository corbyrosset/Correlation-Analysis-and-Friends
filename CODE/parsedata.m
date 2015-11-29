%% Data for speaker JWxx in file JWxx[numfr1=s1,numfr2=s2,full].mat 
%  includes following information:
% 
%  MFCC - a d1xN dimensional matrix of mel-frequency cepstral coefficients,
%         where d = 39*s1 and N is the number of frames/samples. These are
%         13 dim MFCCs + deltas + double-deltas stacked over s1 frames.
% 
%  X    - a d2xN dimensional matrix of mel-frequency cepstral coefficients,
%         where d2 = 16*s2 and N is the number of frames/samples. These are
%         16 dim articulatory data (tracks of pellets on speaker's tongue, 
%         jaws and lips stacked over s2 consecutive frames. Some of the
%         pellets were mistracked during data collection at various points.
%         These are denoted by a value of 1e6 in the matrix. 
%
%  Phones - 1xN cell array of strings representing frame level phone labels.
%
%  P - 1xN integer-valued vector of frame level phone labels.
% 
%  indices - a mapping from string-valued 'Phones' to integer-valued 'P'. 
%
%  Words -  1xN cell array of strings representing frame level word labels.
%
%  Valid_Files - 1xM cell array of strings where the ith string gives the 
%                filename for the ith utterance. M is the total number
%                of utterances.
%
%  Frames - 1xM cell array of integers where the ith entry gives the 
%           total number of frames for the ith utterance (Valid_Files{i}). 
%
%  frame_locs -  1xM vector with the ith entry giving the index into 
%                MFCC/X/Phones/Words of the last frame from the ith 
%                utterance
%
%% Example:
clc, clear, close all;

%% Speaker and number of frames stacked
spkr='JW11';
s1=7; 
s2=7;

%% Load data and list the data variables 
fname=sprintf('../DATA/MAT/%s/%s[numfr1=%d,numfr2=%d,full].mat',...
    spkr,spkr,s1,s2);
load(fname);
whos;

%% Dimensionality of acoustic and articualtory features 
NMFCC=39; 
NARTIC=16; 
shft1=(NMFCC*(s1-1)/2)+1;
shft2=(NARTIC*(s2-1)/2)+1;

%% Sample a window
winstart=7500; 
winsize=1000;

%% Plot acoustic data
figure(1); clf; 
imagesc(MFCC(shft1:shft1+12,winstart:winstart+winsize));
set(gca,'XTickLabel',Phones(winstart:winstart+winsize));
colormap jet; colorbar; 
xlabel('Phone labels');
ylabel('13 dimensional MFCCs');

%% Plot articulatory data
figure(2); clf; 
imagesc(X(shft2:shft2+15,winstart:winstart+winsize));
set(gca,'XTickLabel',Phones(winstart:winstart+winsize));
colormap jet; colorbar
xlabel('Phone labels');
ylabel('16 dimensional articulatory measurments');
