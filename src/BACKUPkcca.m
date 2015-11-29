function [dev, test, bestDevAccuracy, U, bestNeighbor,bestregX, bestregY ] = BACKUPkcca()
clear all;
close all;

%% Speaker and number of frames stacked
spkr='JW11'; %number of frames stacked = 7
train = 15948; %25948;
dev = 25948; %40948;
test = 40948; %50948; 

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

% mean = sum(X1train, 2)/size(X1train, 2);
% X1train = bsxfun(@minus, X1train, mean);
% mean = sum(X2train, 2)/size(X2train, 2);
% X2train = bsxfun(@minus, X2train, mean);
% mean = sum(X1dev, 2)/size(X1dev, 2);
% X1dev = bsxfun(@minus, X1dev, mean);
% mean = sum(X1test, 2)/size(X1test, 2);
% X1test = bsxfun(@minus, X1test, mean);

%hyperparameters
D = [10, 30, 50, 70, 90, 110];
% regulars = [1E-8, 1E-6, 1E-4, 1E-2, 1E-1, 10];
neighbors = [4, 8, 12, 16];
counter = 0;
kernelBandwidth = [4, 2, 8]; %these will be squared! %8, 16, 32, 64];
numSteps = length(D)*length(neighbors)*length(kernelBandwidth);

%outputs
dev = [];
test = [];
bestDevAccuracy = 0;
bestAlpha = [];
bestNeighbor = 0;
% bestregX = 0;
% bestregY = 0;
bestd = 0;
bestKernelBandwidth = 0;




% a = figure;
% b = figure;
h = waitbar(0,'Please wait...');
for b=1:length(kernelBandwidth)
    %calculate kernel matrices
    K_1 = gram(X1train, kernelBandwidth(b)); %myKernelMatrix(X1, X1, 2, sigma2 );
    K_2 = gram(X2train, kernelBandwidth(b)); %myKernelMatrix(X2, X2, 2, sigma2 );

    %fix the kernel matrices so matlab knows they are symmetric
    K_1 = (K_1+K_1')/2;
    K_2 = (K_2+K_2')/2;

    % ATTEMPT 3
    % use incremental kpca on each view to learn U1_k for X1 and U2_k for X2
    % then do cca on (U1_k^T)(U1_k * S * U1_k^T) /approx= (U1_k^T)K_1
    
    for k = 1:length(D)
        fprintf('dim: %f , sigma: %d\n', D(k), kernelBandwidth(b));

        [alpha,learnedFeaturesTrain, learnedFeaturesTraintop2]...
        = scalableKCCA(K_1, K_2, D(k), kernelBandwidth(b));


%       [alphaDev,learnedFeaturesDev]  = scalableKCCA(X1dev, X2dev, max(D), kernelBandwidth(b));
%       save('alphaDev.mat', 'alphaDev');
        if (size(alpha) ~= [n D(k)])
            size(alpha)
            error('alpha not n by 110');
        end

        %now stack the 39 x n baseline acoustic feature vector onto proj
        stackedTrain = [baselineAcousticTrain'; learnedFeaturesTrain]; %matrix (39 + d) by n
        %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        %go back and 
        stackedDev = stackedTrain; %%%%!!!!!![baselineAcousticDev'; learnedFeaturesDev];
        ydev = ytrain; %!!!!!!!!!!!!!!!!!!!!!
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
                bestKernelBandwidth = kernelBandwidth(b);
                bestd = D(k);
                %print progress
                fprintf('found best:\nd: %f , sigma: %d, numNeighbors: %d, devAcc: %f\n', ...
                    D(k), kernelBandwidth(b), neighbors(n), dev(end));
                %save kernel and alpha matrices:
%                 str = sprintf('kernel_matrices_15948_sigma_%d_dim_%d', bestKernelBandwidth, D(k));
%                 save(str, 'K_1', 'K_2');
                str = sprintf('R2_alpha_sigma_%d_dim_%d', kernelBandwidth(b), D(k));
                save(str, 'alpha', 'learnedFeaturesTrain', 'learnedFeaturesTraintop2');
%                         figure(a)
%                         gscatter((alpha(:, 1)'*X1train)', (alpha(:, 2)'*X1train)', ytrain);
%                         gscatter(learnedFeaturesTraintop2(:, 1),...
%                             learnedFeaturesTraintop2(:, 1), ytrain);
%                         drawnow
%                         figure(b)
%                         gscatter(alpha(:, 1)'*X1test, alpha(:, 2)'*X1test, ytest);
%                         drawnow

            end
            waitbar(counter / numSteps);
            counter = counter +1;
        end
    end
end
close(h);

%now try on the test data:
top_d = bestAlpha(:, 1:bestd); %273 by d matrix
learnedFeaturesTrain = top_d'*X1train; %d by n, n = 8527
learnedFeaturesTest = top_d'*X1test;
stackedTrain = [baselineAcousticTrain'; learnedFeaturesTrain];
stackedTest = [baselineAcousticTest'; learnedFeaturesTest];

mdl = fitcknn(stackedTrain', ytrain, 'NumNeighbors', bestNeighbor);
[labeltest, ~] = predict(mdl,stackedTest');
display('test accuracy');
test = sum(ytest' == labeltest)/length(ytest)
c = figure;
fprintf('bestDevAccuracy: %f, bestNeighbor: %d, bestKernelBandwidth: %d, bestDim: %d, testAccuracy, %f', ...
    bestDevAccuracy, bestNeighbor, bestKernelBandwidth, bestd, test);
gscatter(bestAlpha(:, 1)'*X1test, bestAlpha(:, 2)'*X1test, ytest);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function alpha = getSubspace(X1, X2, regx, regy, k, sigma)
    [alpha, ~] = naive_kcca(X1, X2, regx, regy, k, sigma);
    
    %other better implementations

end

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

function K = gram(X, sigma)
    [d, n] = size(X);
    K = zeros(n);
    for i = 1:n
        for j = 1:i
            a = (-1*exp(norm(X(:, i) - X(:, j))))/(sigma^2);
            K(i, j) = a;
            K(j, i) = a;
        end
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [alpha, beta] = calc_kcca_also_Bad(X1,X2,kernel1,kernel2,reg1, reg2,decomp,lrank)
% KM_KCCA performs kernel canonical correlation analysis.
% Input:	- X1, X2: data matrices containing one datum per ROW!!!
%			- kernel1, kernel2: structures containing kernel type (e.g. 
%			'gauss') and kernel paraemters for each data set
%			- kernelpar: kernel parameter value
%			- reg1, reg2: regularization for each view
%			- decomp: low-rank decomposition technique (e.g. 'ICD')
%			- lrank: target rank of decomposed matrices
% Output:	- y1, y2: nonlinear projections of X1 and X2 (estimates of the
%			latent variable)
%			- beta: first canonical correlation
%           - alpha an N by lrank matrix of projections...
% USAGE: [y1,y2,beta] = km_kcca(X1,X2,kernel1,kernel2,reg,decomp,lrank)
%
% Author: Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2012.
% Id: km_kcca.m v1.1 2012/04/08
% This file is part of the Kernel Methods Toolbox (KMBOX) for MATLAB.
% http://sourceforge.net/p/kmbox
%
% The algorithm in this file is based on the following publications:
% D. R. Hardoon, S. Szedmak and J. Shawe-Taylor, "Canonical Correlation
% Analysis: An Overview with Application to Learning Methods", Neural
% Computation, Volume 16 (12), Pages 2639--2664, 2004.
% F. R. Bach, M. I. Jordan, "Kernel Independent Component Analysis", Journal
% of Machine Learning Research, 3, 1-48, 2002.
%
% This program is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation, version 3 (http://www.gnu.org/licenses).

N = size(X1,1);	% number of data

if nargin<6
	decomp = 'full';
end

switch decomp
	case 'ICD' % incomplete Cholesky decomposition

		% get incompletely decomposed kernel matrices. K1 \approx G1*G1'
		G1 = km_kernel_icd(X1,kernel1{1},kernel1{2},lrank);
		G2 = km_kernel_icd(X2,kernel2{1},kernel2{2},lrank);

		% remove mean. avoid standard calculation N0 = eye(N)-1/N*ones(N);
 		G1 = G1-repmat(mean(G1),N,1);
 		G2 = G2-repmat(mean(G2),N,1);

		% ones and zeros
		N1 = size(G1,2); N2 = size(G2,2);
		Z11 = zeros(N1); Z22 = zeros(N2); Z12 = zeros(N1,N2);
		I11 = eye(N1); I22 = eye(N2);

		% 3 GEV options, all of them are fairly equivalent

		% % option 1: standard Hardoon
		% R = [Z11, G1'*G1*G1'*G2; G2'*G2*G2'*G1, Z22];
		% D = [G1'*G1*G1'*G1+reg1*I11, Z12; Z12', G2'*G2*G2'*G2+reg2*I22];

		% option 2: simplified Hardoon
		R = [Z11, G1'*G2; G2'*G1 Z22];
		D = [G1'*G1+reg1*I11 Z12; Z12' G2'*G2+reg2*I22];

		% % option 3: Kettenring-like generalizable formulation
		% R = 1/2*[G1'*G1 G1'*G2; G2'*G1 G2'*G2];
		% D = [G1'*G1+reg1*I11 Z12; Z12' G2'*G2+reg2*I22];

		% solve generalized eigenvalue problem
        % NOTE repetition in eigenvalues: 
        % {?1, ??1, . . . , ?p, ??p, 0, . . . , 0}.
		[alphas,betas] = eig(R,D);
		[betas,ind] = sort(real(diag(betas)), 'descend');
        alphas = alphas(:, ind);
        if (rank(alphas) < lrank)
            warning('alpha has fewer components than lrank');
            size(alphas)
            return;
        end
        % because eigenvalues repeat, take top half...
        alpha = alphas(:, 1:lrank);
        beta = diag(betas(1:lrank));
%         if (norm(alphas(:, 1:lrank) + sort(alphas(:, (lrank+1):end), 'ascend')) > 0.0000001)
%             warning('symmetry of eigenvalues in GEP is breaking');
%         end
        % because of structure of generalized eigenvalue problem, the 
        % eigenvectors alpha for view one are the top 1/2 of alpha...
        alpha = alpha((1+lrank):end, :); %%%!!!!!!!!!!!!!!!
        
        
        
% 		alpha = alphas(:,ind(end)); alpha = alpha/norm(alpha);
% 		beta = betass(end);
% 
% 		% expansion coefficients
% 		alpha1 = alpha(1:N1);
% 		alpha2 = alpha(N1+1:end);
% 
% 		% estimates of latent variable
% 		y1 = G1*alpha1;
% 		y2 = G2*alpha2;

	case 'full' % no kernel matrix decomposition (full KCCA)
        error('do not call full');

		I = eye(N); Z = zeros(N);
		N0 = eye(N)-1/N*ones(N);

		% get kernel matrices
		K1 = N0*km_kernel(X1,X1,kernel1{1},kernel1{2})*N0;
		K2 = N0*km_kernel(X2,X2,kernel2{1},kernel2{2})*N0;

		% 3 GEV options, all of them are fairly equivalent

		% % option 1: standard Hardoon
		% R = [Z K1*K2; K2*K1 Z];
		% D = 1/2*[K1*(K1+reg1*I) Z; Z K2*(K2+reg2*I)];
		% % R = R/2+R'/2;   % avoid numerical problems
		% % D = D/2+D'/2;   % avoid numerical problems

		% option 2: simplified Hardoon
		% R = [Z K2; K1 Z];
		% D = [K1+reg1*I Z; Z K2+reg2*I];
		% % R = R/2+R'/2;   % avoid numerical problems
		% % D = D/2+D'/2;   % avoid numerical problems

		% % option 3: Kettenring-like generalizable formulation
		R = 1/2*[K1 K2; K1 K2];
		D = [K1+reg1*I Z; Z K2+reg2*I];

		% solve generalized eigenvalue problem
		[alphas,betas] = eig(R,D);
% 		[betass,ind] = sort(real(diag(betas)));
        [betas,ind] = sort(real(diag(betas)), 'descend');
        alphas = alphas(ind);
        alpha = alphas(:, (lrank+1):end);
        beta = betas(:, 1:lrank);
        
% 		alpha = alphas(:,ind(end)); alpha = alpha/norm(alpha);
% 		beta = betass(end);
% 
% 		% expansion coefficients
% 		alpha1 = alpha(1:N);
% 		alpha2 = alpha(N+1:end);
% 
% 		% estimates of latent variable
% 		y1 = K1*alpha1;
% 		y2 = K2*alpha2;

    end
end

function [W_x, W_y, corr_list] = calc_kcca_bad(K_X, K_Y, options)
% KCCA: Kernel Canonical Correlation Analysis
%
%
% Usage:
%     [W_x, W_y] = KCCA(K_X, K_Y, options)
%     [W_x, W_y] = KCCA(K_X, K_Y)
% 
%    Input:
%        K_X       - Kernel matrix for X. 
%        K_Y       - Kernel matrix for Y. 
%       
%        options.PrjX    = 1: The projection for K_X is requried.
%                             = 0: The projeciton for K_X is not required.
%                                    The default value is 1 (required). 
%        options.PrjY    = 1: The projection for K_Y is requried.
%                             = 0: The projeciton for K_Y is not required.
%                                    The default value is 1 (required).  
%        options.RegX  - The regularization parameter for X. The default value is 0.
%        options.RegY  - The regularization parameter for Y. The default value is 0.
% 
%
%    Output:
%        W_x: each column is a projection vector for K_X. If the projection
%                 for X is not required, an empty matrix is returned.
%        W_y: each column is a projection vector for K_Y. If the projection
%                 for Y is not required, an empty matrix is returned.
%        corr_list: the list of correlation coefficients in the projected space.
% 
%    Examples:
%        X = rand(7,10); K_X = X' * X;
%        Y = rand(15, 10); K_Y = Y' * Y;
%        options.PrjX = 1;
%        options.PrjY = 0;
%        options.RegX = 0.5;
%        options.RegY = 1;
%        [W_x, W_y] = CCA(K_X, K_Y, options);
% 
% ? 2008 Liang Sun (sun.liang@asu.edu), Arizona State University
% 

    % Extract the input parameters, and set the values of parameters.
    if (~exist('options','var'))
       options = [];
    end
    PrjX = 1;
    PrjY = 1;
    if isfield(options, 'PrjX')
        PrjX = options.PrjX;
    end
    if isfield(options, 'PrjY')
        PrjY = options.PrjY;
    end
    RegX = 0;
    RegY = 0;
    if isfield(options, 'RegX')
        RegX = options.RegX;
    end
    if isfield(options, 'RegY')
        RegY = options.RegY;
    end


    W_x = [];
    W_y = [];
    corr_list = [];
    % Preprocess the input 
    if size(K_X, 1) ~= size(K_Y, 1)
        disp('The numbers of samples in X and Y are not equal!');
        return;
    end
    if issparse(K_X)
        K_X = full(K_X);
    end
    if issparse(K_Y)
        K_Y = full(K_Y);
    end

    % Compute the SVD of K_X
    [X_U, X_Sigma, X_V] = svd(K_X, 'econ');
    X_rank = rank(X_Sigma);
    X_U = X_U(:, 1:X_rank);
    X_Sigma = X_Sigma(1:X_rank, 1:X_rank);
    X_sigma = diag(X_Sigma);
    X_V = X_V(:, 1:X_rank);

    % Compute the SVD of K_Y
    [Y_U, Y_Sigma, Y_V] = svd(K_Y, 'econ');
    rank_Y = rank(Y_Sigma);
    Y_U = Y_U(:, 1:rank_Y);
    Y_Sigma = Y_Sigma(1:rank_Y, 1:rank_Y);
    Y_sigma = diag(Y_Sigma);
    Y_V = Y_V(:, 1:rank_Y);

    % Compute the projection vectors
    if PrjX == 1    
        % Compute the Projection for X
        % 
        % Construct matrix H
        if RegY == 0
            % No regularization for Y is considered
            H = Y_U;
        else
            % We consider the regularization for Y
            Y_sigma_reg = Y_sigma + RegY;        
            Y_sigma_reg = Y_sigma ./ Y_sigma_reg;
            Y_sigma_reg = sqrt(Y_sigma_reg);
            H = Y_U * diag(Y_sigma_reg);
        end

        X_sigma_reg = X_sigma + RegX;
        X_sigma_reg = X_sigma ./ X_sigma_reg;    
        X_sigma_reg = sqrt(X_sigma_reg);
        B = diag(X_sigma_reg) * X_U' * H;
        [P, B_Sigma, Q] = svd(B, 'econ');
        clear Q;
        rank_B = rank(B_Sigma);
        P = P(:, 1:rank_B);
        B_sigma = diag(B_Sigma);
        B_sigma = B_sigma(1:rank_B);
        corr_list = B_sigma;
        W_x = X_U * diag(1./sqrt((X_sigma + RegX) .* X_sigma)) * P;    

        % Compute the Projection for Y
        if PrjY == 1
            if RegY == 0
                W_y = Y_U * diag(1./Y_sigma) * Y_V' * K_X * W_x;
            else
                W_y = Y_U * diag(1 ./ (Y_sigma + RegY)) * Y_V' * K_X * W_x;
            end
            % Normalize W_y
            W_y = W_y ./ repmat(sqrt(sum(W_y.^2)), size(W_y, 1), 1);
        end

    else
        % If the projection for X is not required, we only focus on the
        % projection for Y
        if PrjY == 1
            % Construct matrix H, then we can call the general module to solve this
            % eigenvalue problem.
            if RegX == 0
                % No regularization for Y is considered
                H = X_U;
            else
                % We consider the regularization for X
                X_sigma_reg = X_sigma + RegX;        
                X_sigma_reg = X_sigma ./ X_sigma_reg;
                X_sigma_reg = sqrt(X_sigma_reg);
                H = X_U * diag(X_sigma_reg);
            end

            Y_sigma_reg = Y_sigma + RegY;
            Y_sigma_reg = Y_sigma ./ Y_sigma_reg;
            Y_sigma_reg = sqrt(Y_sigma_reg);
            B = diag(Y_sigma_reg) * Y_U' * H;
            [P, B_Sigma, Q] = svd(B, 'econ');
            clear Q;
            rank_B = rank(B_Sigma);
            P = P(:, 1:rank_B);
            B_sigma = diag(B_Sigma);
            B_sigma = B_sigma(1:rank_B);
            corr_list = B_sigma;
            W_y = Y_U * diag(1./sqrt((Y_sigma + RegY) .* Y_sigma)) * P;
        end
    end
end

