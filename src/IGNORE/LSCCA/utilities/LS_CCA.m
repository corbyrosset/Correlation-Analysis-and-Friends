function W_x = LS_CCA(X, Y, options)

% LS_CCA: Least Squares formulation of Canonical Correlation Analysis
% Only the projection for X is computed. 
%
%
% Usage:
%     W_x = LS_CCA(X, Y, options)
%     W_x = LS_CCA(X, Y)
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point.
%        Y       - Data matrix Y. Each column of Y is a data point.
%       
%        options.RegX      - The regularization parameter for X. The default value is 0.
%        optoins.RegType = 0: No regularization is imposed.
%                                 = 1: 1-norm regularized least squares (lasso).
%                                 = 2: 2-norm regularized least squres (ridge regression)    
%                                     The default value is 0 (No regularization)
%
%    Output:
%        W_x: each column is a projection vector for X. 
% 
%    Examples:
%        X = rand(15,10);
%        Y = rand(4, 10);
%        options.RegX = 0.5;
%        options.RegType = 2;
%        W_x = CCA(X, Y, options);
% 
% © 2008 Liang Sun (sun.liang@asu.edu), Arizona State University
% 


% Step 1. Preprocess
% Extract the input parameters, and set the values of parameters.
if (~exist('options','var'))
   options = [];
end
RegX = 0;
RegType = 0;
if isfield(options, 'RegX')
    RegX = options.RegX;
end
if isfield(options, 'RegType')
    RegType = options.RegType;
end



% Step 2. Compute the target matrix from Y, and solve the least squares
% formulation
W_x = [];
[d1, n1] = size(X);
[d2, n2] = size(Y);
if n1 ~= n2
    disp('The numbers of samples in X and Y are not equal!');
    return;    
end
n = n1;
if issparse(X)
    X = full(X);
end
if issparse(Y)
    Y = full(Y);
end
% Make sure X and Y are both column-centered, i.e., Xe=0, Ye=0;
if norm(X * ones(n, 1)) > 1e-4
    X = colCenter(X);
end
if norm(Y * ones(n, 1)) > 1e-4
    Y = colCenter(Y);
end


% Compute the SVD of Y
[Y_U, Y_Sigma, Y_V] = svd(Y, 'econ');
rank_Y = rank(Y_Sigma);
Y_U = Y_U(:, 1:rank_Y);
Y_V = Y_V(:, 1:rank_Y);
% Compute the target for least squares
Target = Y_U * Y_V';

% Call least squares and its variants to solve the problem
lsOptions.RegX = RegX;
lsOptions.RegType = RegType;
W_x = LS(X, Target, lsOptions);



