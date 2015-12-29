function [dev, test, bestDevAccuracy, U, bestNeighbor,bestregX, bestregY ] = kcca()
clear all;
close all;

% Speaker and number of frames stacked
spkr='JW11';      %number of frames stacked = 7 

% Load data and list the data variables 
path=sprintf('../DATA/MAT/%s[numfr1=7,numfr2=7]',spkr);
load(path, 'MFCC', 'X', 'P');
X1 = MFCC;        %273 x 50948 view1
X2 = X;           %112 x 50948 view2
n = size(X1, 2);

%% data preprocessing and parameters definitions 
%randomly permute data:
perm = randperm(n);
X1 = X1(:, perm);
X2 = X2(:, perm);
P = P(:, perm);

%%center data
X1 = centerAndNormalize(X1);
X2 = centerAndNormalize(X2);

%choose size of train, dev, and test data. These numbers must be strictly
%increasing, refer to lines 28 to 34 below to see why. 
train = 2000; %use 25000
dev   = 4000; %use 40000
test  = 10000;%use 50000

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
X1_dev_hat = [];
X1_dev_hat_top2 = [];
display('loaded data');

%hyperparameters
D = [60]; %[10, 30, 50, 70, 90, 110]; %try these
neighbors = [4, 8, 12, 16];
counter = 0;


sigma1 = [1200]; %try others
sigma2 = [4800]; %try others
numSteps = length(D)*length(neighbors)*length(sigma1)*length(sigma2);
bstep = 100; %inconsequential, only used to calculate alpha*K_1 incrementally

%outputs
dev = [];
test = [];
bestDevAccuracy = 0;
bestAlpha = [];
bestNeighbor = 0;
bestd = 0;
bestSigma1 = 0; 
bestSigma2 = 0;
bestLearnedFeaturesTrain = [];
bestLearnedFeaturesTrainTop2 = [];

%% begin tuning parameters...
A = figure;
B = figure;
h = waitbar(0,'Please wait...');
for band1=1:length(sigma1)
    for band2=1:length(sigma2)
    
        for k = 1:length(D)
            fprintf('dim: %f , sigma1: %d, sigma2: %d\n', ...
                D(k), sigma1(band1), sigma2(band2));

            [alpha,learnedFeaturesTrain, learnedFeaturesTraintop2]...
            = scalableKCCA(X1train, X2train, D(k), sigma1(band1), sigma2(band2));

            X1_dev_hat = [];      %don't build up junk
            X1_dev_hat_top2 = [];
            for j = 1:bstep:(floor(size(X1dev, 2)/bstep)*bstep)
                K_temp = gram(X1train, X1dev, j, j+bstep-1, sigma1(band1));
                X1_dev_hat = [X1_dev_hat alpha'*K_temp];
                X1_dev_hat_top2 = [X1_dev_hat_top2 alpha(:, 1:2)'*K_temp];
            end

            if (size(X1_dev_hat, 2) ~= size(X1dev, 2))
                size(X1dev)
                size(X1_dev_hat)
                error('K^(dev) not right size');
            end
            X1_dev_hat_top2 = X1_dev_hat_top2'; %for easier plotting
            stackedTrain = [baselineAcousticTrain'; learnedFeaturesTrain];
            stackedDev = [baselineAcousticDev'; X1_dev_hat];
            display('computed learned features on dev set');

            %train  the KNN on training data, test on dev data
            for n = 1:length(neighbors)
                mdl = fitcknn(stackedTrain', ytrain, 'NumNeighbors', neighbors(n));

                %predict labels for dev data
                [labeldev, ~] = predict(mdl,stackedDev');

                %compute and store accuracy
                dev = [dev sum(ydev' == labeldev)/length(ydev)];

                %record best parameters and feature vectors
                if (dev(end) > bestDevAccuracy)
                    bestDevAccuracy = dev(end);
                    bestAlpha = alpha; %recall top_d = U(:, 1:D(k))
                    bestNeighbor = neighbors(n);
                    bestSigma1 = sigma1(band1);
                    bestSigma2 = sigma2(band2);
                    bestd = D(k);
                    bestLearnedFeaturesTrain = learnedFeaturesTrain;
                    bestLearnedFeaturesTrainTop2 = learnedFeaturesTraintop2;
                    %print progress
                    fprintf('found best:\nd: %f , sigma1: %d, sigma2: %d, numNeighbors: %d, devAcc: %f\n', ...
                        D(k), sigma1(band1), sigma2(band2), neighbors(n), dev(end));

                    %plot 2D dev and test clusters
                    figure(A);
                    gscatter(X1_dev_hat_top2(:, 1), X1_dev_hat_top2(:, 2), ydev');
                    s = sprintf('dev data: sig1=%d, sig2=%d, dim=%d', bestSigma1, bestSigma2, bestd);
                    title(s);
                    drawnow;
                    figure(B);
                    gscatter(bestLearnedFeaturesTrainTop2(:, 1), bestLearnedFeaturesTrainTop2(:, 2), ytrain');
                    s = sprintf('train data: sig1=%d, sig2=%d, dim=%d', bestSigma1, bestSigma2, bestd);
                    title(s);
                    drawnow;

                end
                waitbar(counter / numSteps);
                counter = counter +1;
            end
        end
    end
end
close(h);

% compute learned features on K^(test) using BEST alpha from training set:
% remember, K_x^(test) = kernel(x_i, x_j^(test))
learnedFeaturesTest = [];
learnedFeaturesTestTop2 = [];
for j = 1:bstep:(floor(size(X1test, 2)/bstep)*bstep)
    K_temp = gram(X1train, X1test, j, j+bstep-1, bestSigma1);
    learnedFeaturesTest = [learnedFeaturesTest bestAlpha'*K_temp];
    learnedFeaturesTestTop2 = [learnedFeaturesTestTop2 bestAlpha(:, 1:2)'*K_temp];
end
        
if (size(learnedFeaturesTest, 2) ~= size(X1test, 2))
    size(learnedFeaturesTest)
    error('K^(test) not right size');
end
learnedFeaturesTestTop2 = learnedFeaturesTestTop2'; %for easier plotting
stackedTrain = [baselineAcousticTrain'; bestLearnedFeaturesTrain];
stackedTest = [baselineAcousticTest'; learnedFeaturesTest];


mdl = fitcknn(stackedTrain', ytrain, 'NumNeighbors', bestNeighbor);
[labeltest, ~] = predict(mdl,stackedTest');
display('test accuracy');
test = sum(ytest' == labeltest)/length(ytest)
c = figure;
fprintf('bestDevAccuracy: %f, bestNeighbor: %d, bestKernelBandwidth1: %d, bestKernelBandwidth1: %d, bestDim: %d, testAccuracy, %f', ...
    bestDevAccuracy, bestNeighbor, bestSigma1, bestSigma2, bestd, test);
gscatter(learnedFeaturesTestTop2(:, 1), learnedFeaturesTestTop2(:, 2), ytest);
s = sprintf('test data: sig1=%d, sig2=%d, dim=%d', bestSigma1, bestSigma2, bestd);
title(s);
drawnow;

save('KCCAprojected_data', 'stackedTrain', 'bestAlpha', 'stackedTest', 'ytrain', 'ytest', 'bestd', 'bestSigma1', 'bestSigma2');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = plot2d(a, C, labels)
    figure(a);
    scatter(C(:, 1), C(:, 2), 9, labels)
end

function X = centerAndNormalize(X)
    mean = sum(X, 2)/size(X, 2);
    stdtrain = std(X');
    Xcenter = bsxfun(@minus, X, mean);
    X = bsxfun(@rdivide, Xcenter, stdtrain');
end

%% Radial Basis Function Kernel
function K = gram(X1, X2, start, stop, sigma)
    [d, n] = size(X1);
    K = zeros(n, (stop-start+1));
    for i = 1:n
        for j = 1:(stop-start+1)
            j_offset = j+start-1;
            a = exp(-1*(norm(X1(:, i) - X2(:, j_offset))^2)/(2*sigma^2));
            K(i, j) = a;
        end
    end

end

%% Polynomial kernel - performs better
% function K = gram(X1, X2, start, stop, p)
%     [d, n] = size(X1);
%     K = zeros(n, (stop-start+1));
%     for i = 1:n
%         for j = 1:(stop-start+1)
%             j_offset = j+start-1;
%             a = (X1(:, i)'*X2(:, j_offset) + 1)^p; 
%             %the +1 can be replaced by a variable...
%             K(i, j) = a;
%         end
%     end
% 
% end


%% hyperbolic tangent kernel 
% function K = gram(X1, X2, start, stop, p)
%     [d, n] = size(X1);
%     K = zeros(n, (stop-start+1));
%     for i = 1:n
%         for j = 1:(stop-start+1)
%             j_offset = j+start-1;
%             a = tanh((1/p)*(X1(:, i)'*X2(:, j_offset)) + 1); 
%             %the +1 can be replaced by a variable...
%             K(i, j) = a;
%         end
%     end
% 
% end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
