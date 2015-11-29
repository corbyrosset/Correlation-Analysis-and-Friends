clear all;
close all;

%% Speaker and number of frames stacked
spkr='JW11';    %number of frames stacked = 7

%divide data into ~25k train, ~10k dev, ~15k test
train = 15948;  %25948; commented out numbers are what we will actually use
dev = 25948;    %40948;
test = 40948;   %50948; 

%% Load data and list the data variables 
path=sprintf('../DATA/MAT/%s[numfr1=7,numfr2=7]',spkr);
load(path, 'MFCC', 'X', 'P');
X1 = MFCC;        %273 x 50948 view1
X2 = X;           %112 x 50948 view2
n = size(X1, 2);

%”randomly” permute data by columns:
perm = randperm(n);
X1 = X1(:, perm);
X2 = X2(:, perm);
P = P(:, perm);

%data to be used
X1train = X1(:, 1:train);
X1dev = X1(:, train+1:dev);
X1test = X1(:, dev+1:test);
X2train = X2(:, 1:train);
ytrain = P(:, 1:train);
ydev = P(:, train+1:dev);
ytest = P(:, dev+1:test); 
baselineAcousticTrain = X1train(118:156, :)';
baselineAcousticDev = X1dev(118:156, :)';
baselineAcousticTest = X1test(118:156, :)';
display('loaded data');

%%center data
X1train = centerAndNormalize(X1train);
X2train = centerAndNormalize(X2train);
X1dev = centerAndNormalize(X1dev);
X1test = centerAndNormalize(X1test);

%%call training methods down here and such