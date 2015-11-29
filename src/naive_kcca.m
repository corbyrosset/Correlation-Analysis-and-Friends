function  [U, V, corr_list] = naive_kcca(X1, X2, regx, regy, k, sigma)
%     clear all;
%     close all;
% 
%     %% Speaker and number of frames stacked
%     spkr='JW11'; %number of frames stacked = 7
%     train = 15948; %25948;
%     dev = 25948; %40948;
%     test = 40948; %50948; 
% 
%     %% Load data and list the data variables 
%     path=sprintf('../DATA/MAT/%s[numfr1=7,numfr2=7]',spkr);
%     load(path, 'MFCC', 'X', 'P');
%     X1 = MFCC;        %273 x 50948 view1
%     X2 = X;           %112 x 50948 view2
%     n = size(X1, 2);
%     d = size(X1, 1);
% 
%     %randomly permute data:
%     perm = randperm(n);
%     X1 = X1(:, perm);
%     X2 = X2(:, perm);
%     P = P(:, perm);
% 
%     %data to be used
%     X1train = X1(:, 1:train);
%     X1dev = X1(:, train+1:dev);
%     X1test = X1(:, dev+1:test);
%     X2train = X2(:, 1:train);
%     ytrain = P(:, 1:train);
%     ydev = P(:, train+1:dev);
%     ytest = P(:, dev+1:test); 
%     baselineAcousticTrain = X1train(118:156, :)';
%     baselineAcousticDev = X1dev(118:156, :)';
%     baselineAcousticTest = X1test(118:156, :)';
% 
%     %parameters
%     n_test = size(ytest, 2);
%     n = size(X1train, 2);
%     sigma2 = 16;
%     regulars = [1E-8, 1E-6, 1E-4, 1E-2, 1E-1, 10];

    n = size(X1, 2);
    d = size(X1, 1);
    display('beginning naive kcca');

    %%center data
    X1 = centerAndNormalize(X1);
    X2 = centerAndNormalize(X2);

    %calculate Gram matrix for training data
    K_X = myKernelMatrix( X1, X1, 2, sigma ); %gram(X1, sigma2);
    K_Y = myKernelMatrix( X2, X2, 2, sigma ); %gram(X2, sigma2);
    display('calculated gram matrices naive kcca');

    
    options = struct('RegX', regx, 'RegY', regy);
    [U, V, corr_list] = boring_kcca(K_X, K_Y, options);
    [corr_list, ind] = sort(corr_list, 'descend');
    U = U(:, ind);
    U = U(:, 1:k);
    display('done with naive kcca');


%     %center kernel matrices
%     K_cent_tr = K_tr - one_tr*K_tr - K_tr*one_tr + one_tr*K_tr*one_tr;
%     K_cent_test = K_test - K_tr*one_test - one_tr*K_test + one_tr*K_tr*one_test;
%     cov_X = K_cent_tr;
%     cov_Y = K_cent_test;   
% 
%     %take eigendecomposition of training covariance
%     [V, D] = eigs(cov_X);  
%     eigenvalues = diag(D);
% 
%     %take top two principle components and normalize
%     U_proj = V(:, 1:topk);
%     U_proj(:, 1) = (1/eigenvalues(1))*U_proj(:, 1);
%     U_proj(:, 2) = (1/eigenvalues(1))*U_proj(:, 2);
% 
%     %project training X and testing Y data onto first two PCs
%     X_proj = U_proj'*K_cent_tr;  
%     Y_proj = U_proj'*K_cent_test;
% 
% 
%     K = 200;
%     [U,S,V] = SEQKL(A, K);

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
            a = (-1*exp(norm(X(:, i) - X(:, j))))/sigma;
            K(i, j) = a;
            K(j, i) = a;
        end
    end

end

function [W_x, W_y, corr_list] = boring_kcca(K_X, K_Y, options)

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
% © 2008 Liang Sun (sun.liang@asu.edu), Arizona State University
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


