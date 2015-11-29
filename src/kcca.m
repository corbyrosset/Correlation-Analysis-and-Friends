function [dev, test, bestDevAccuracy, U, bestNeighbor,bestregX, bestregY ] = kcca()
clear all;
close all;

%% Speaker and number of frames stacked
spkr='JW11'; %number of frames stacked = 7
train = 2000; %25000;   %25948;
dev   = 3000; %40000;   %40948
test  = 5000; %50000;    %50948; 

%% Load data and list the data variables 
path=sprintf('../DATA/MAT/%s[numfr1=7,numfr2=7]',spkr);
load(path, 'MFCC', 'X', 'P');
X1 = MFCC;        %273 x 50948 view1
X2 = X;           %112 x 50948 view2
n = size(X1, 2);

%randomly permute data:
perm = randperm(n);
X1 = X1(:, perm);
X2 = X2(:, perm);
P = P(:, perm);

%% data to be used FIX FIX FIX FIX FIX FIX
X1train = X1(:, 1:train);
X1dev = X1(:, train+1:dev); %FIX FIX FIX FIX
X1test = X1(:, dev+1:test); %dev + 1
X2train = X2(:, 1:train);
ytrain = P(:, 1:train);
ydev = P(:, train+1:dev);
ytest = P(:, dev+1:test);  %dev +1
baselineAcousticTrain = X1train(118:156, :)';
baselineAcousticDev = X1dev(118:156, :)';
baselineAcousticTest = X1test(118:156, :)';
X1_dev_hat = [];
X1_dev_hat_top2 = [];
display('loaded data');

%%center data
X1train = centerAndNormalize(X1train);
X2train = centerAndNormalize(X2train);
X1dev = centerAndNormalize(X1dev);
X1test = centerAndNormalize(X1test);

%hyperparameters
D = [90, 110]; %[10, 30, 50, 70, 90, 110];
% regulars = [1E-6, 1E-4, 1E-2, 1E-1, 10];
neighbors = [4, 8, 12, 16];
counter = 0;
kernelBandwidth = [1]; %these will be squared! %8, 16, 32, 64];
numSteps = length(D)*length(neighbors)*length(kernelBandwidth);
bstep = 500; %inconsequential, only used to calculate alpha*K_1 incrementally
sigma1 = 20; %30
sigma2 = 20; %20

%outputs
dev = [];
test = [];
bestDevAccuracy = 0;
bestAlpha = [];
bestNeighbor = 0;
bestd = 0;
bestKernelBandwidth = sigma1; %!!!!!!!!
bestLearnedFeaturesTrain = [];
bestLearnedFeaturesTrainTop2 = [];




A = figure;
B = figure;
h = waitbar(0,'Please wait...');
for band=1:length(kernelBandwidth)
    
    for k = 1:length(D)
        fprintf('dim: %f , sigma: %d\n', D(k), kernelBandwidth(band));

%%%%%%%%%%%%
        [alpha,learnedFeaturesTrain, learnedFeaturesTraintop2]...
        = scalableKCCA(X1train, X2train, D(k), sigma1, sigma2); %kernelBandwidth(b)); %K_1, K_2, D(k), kernelBandwidth(b));

        
        
        % compute learned features on K^(dev) using alpha from training set:
        % remember, K_x^(dev) = kernel(x_i, x_j^(dev))
        X1_dev_hat = [];
        X1_dev_hat_top2 = [];
        for j = 1:bstep:(floor(size(X1dev, 2)/bstep)*bstep)
            K_temp = gram(X1train, X1dev, j, j+bstep-1, sigma1);
            X1_dev_hat = [X1_dev_hat alpha'*K_temp];
            X1_dev_hat_top2 = [X1_dev_hat_top2 alpha(:, 1:2)'*K_temp];
        end
        %check if the kernel matrices match
%         X1_hat_other = alpha'*gram(X1train, X1dev, 1, size(X1dev, 2), sigma1);
%         match = sum(sum(X1_hat_other == X1_dev_hat));
%         [b1, b2] = size(X1_hat_other);
%         if (match ~= b1*b2)
%             match
%             b1*b2
%             error('do not match')
%         end
        
        if (size(X1_dev_hat, 2) ~= size(X1dev, 2))
            size(X1dev)
            size(X1_dev_hat)
            error('K^(dev) not right size');
        end
        X1_dev_hat_top2 = X1_dev_hat_top2'; %for easier plotting
        stackedTrain = [baselineAcousticTrain'; learnedFeaturesTrain];
        stackedDev = [baselineAcousticDev'; X1_dev_hat];
        display('computed learned features on dev set');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% added

%         learnedFeaturesTrain = [];
%         learnedFeaturesTraintop2 = [];
%         for j = 1:b:(floor(size(X1train, 2)/b)*b)
%             K_temp = gram(X1train, X1train, j, j+b-1, kernelBandwidth(b));
%             learnedFeaturesTrain = [learnedFeaturesTrain alpha(:, 1:D(k))'*K_temp];
%             learnedFeaturesTraintop2 = [learnedFeaturesTraintop2 alpha(:, 1:2)'*K_temp];
%         end
%         if (size(learnedFeaturesTrain, 2) ~= size(X1train, 2))
%             size(learnedFeaturesTrain)
%             error('projected kernel onto alpha is not right size')
%         end
%         learnedFeaturesTraintop2 = (1/norm(learnedFeaturesTraintop2))*learnedFeaturesTraintop2'; %for easier plotting
% 
%         X1_dev_hat = [];
%         X1_dev_hat_top2 = [];
%         for j = 1:b:(floor(size(X1dev, 2)/b)*b)
%             K_temp = gram(X1train, X1dev, j, j+b-1, kernelBandwidth(b));
%             X1_dev_hat = [X1_dev_hat alpha(:, 1:D(k))'*K_temp];
%             X1_dev_hat_top2 = [X1_dev_hat_top2 alpha(:, 1:2)'*K_temp];
%         end
%         if (size(X1_dev_hat, 2) ~= size(X1dev, 2))
%             size(X1_dev_hat)
%             error('K^(dev) not right size');
%         end
%         X1_dev_hat_top2 = (1/norm(X1_dev_hat_top2))*X1_dev_hat_top2'; %for easier plotting
%         stackedTrain = [baselineAcousticTrain'; learnedFeaturesTrain];
%         stackedDev = [baselineAcousticDev'; X1_dev_hat];
%         display('computed learned features on dev set');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %train and classify the KNN
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
                bestNeighbor = n;
                bestKernelBandwidth = kernelBandwidth(band);
                bestd = D(k);
                bestLearnedFeaturesTrain = learnedFeaturesTrain;
                bestLearnedFeaturesTrainTop2 = learnedFeaturesTraintop2;
                %print progress
                fprintf('found best:\nd: %f , sigma: %d, numNeighbors: %d, devAcc: %f\n', ...
                    D(k), kernelBandwidth(band), neighbors(n), dev(end));
                % save kernel and alpha matrices?:
%                 str = sprintf('kernel_matrices_15948_sigma_%d_dim_%d', bestKernelBandwidth, D(k));
%                 save(str, 'K_1', 'K_2');
%                 str = sprintf('R2_alpha_sigma_%d_dim_%d', kernelBandwidth(b), D(k));
%                 save(str, 'alpha', 'learnedFeaturesTrain', 'learnedFeaturesTraintop2');
                
                %plot 2D dev and test clusters
                figure(A);
                gscatter(X1_dev_hat_top2(:, 1), X1_dev_hat_top2(:, 2), ydev');
                drawnow;
                figure(B);
                gscatter(bestLearnedFeaturesTrainTop2(:, 1), bestLearnedFeaturesTrainTop2(:, 2), ytrain');
                drawnow;


            end
            waitbar(counter / numSteps);
            counter = counter +1;
        end
    end
end
close(h);

% compute learned features on K^(test) using BEST alpha from training set:
% remember, K_x^(test) = kernel(x_i, x_j^(test))
learnedFeaturesTest = [];
learnedFeaturesTestTop2 = [];
for j = 1:bstep:(floor(size(X1test, 2)/bstep)*bstep)
    K_temp = gram(X1train, X1test, j, j+bstep-1, bestKernelBandwidth);
    learnedFeaturesTest = [learnedFeaturesTest bestAlpha'*K_temp];
    learnedFeaturesTestTop2 = [learnedFeaturesTestTop2 bestAlpha(:, 1:2)'*K_temp];
end
%IF NUMBER OF EXAMPLES SMALL ENOUGH: you may check if matrices match
% learnedFeaturesOther = bestAlpha'*gram(X1train, X1test, 1, size(X1test, 2), bestKernelBandwidth); %!!!!!!
% match = sum(sum(learnedFeaturesOther == learnedFeaturesTest));
% [b1, b2] = size(learnedFeaturesOther);
% if (match ~= b1*b2)
%     match
%     b1*b2
%     error('do not match')
% end
        
if (size(learnedFeaturesTest, 2) ~= size(X1test, 2))
    size(learnedFeaturesTest)
    error('K^(test) not right size');
end
learnedFeaturesTestTop2 = learnedFeaturesTestTop2'; %for easier plotting

% remember, K_x^(test) = kernel(x_i, x_j^(test))

stackedTrain = [baselineAcousticTrain'; bestLearnedFeaturesTrain];
stackedTest = [baselineAcousticTest'; learnedFeaturesTest];
mdl = fitcknn(stackedTrain', ytrain, 'NumNeighbors', bestNeighbor);
[labeltest, ~] = predict(mdl,stackedTest');
display('test accuracy');
test = sum(ytest' == labeltest)/length(ytest)
c = figure;
fprintf('bestDevAccuracy: %f, bestNeighbor: %d, bestKernelBandwidth: %d, bestDim: %d, testAccuracy, %f', ...
    bestDevAccuracy, bestNeighbor, bestKernelBandwidth, bestd, test);
gscatter(learnedFeaturesTestTop2(:, 1), learnedFeaturesTestTop2(:, 2), ytest);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
