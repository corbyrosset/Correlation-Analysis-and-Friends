function [W_x, eigv_list] =OPLS(X, Y, options)

% OPLS: Orthonormalized Partial Least Squares
% Only the projection for X is computed.
%
%
% Usage:
%     [W_x, eigv_list] = OPLS(X, Y, options)
%     [W_x, eigv_list] = OPLS(X, Y)
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point.
%        Y       - Data matrix Y. Each column of Y is a data point.
%       
%        options.RegX  - The regularization parameter for X. The default value is 0.
% 
%
%    Output:
%        W_x: each column is a projection vector for X. If the projection
%                 for X is not required, an empty matrix is returned.
%        eigv_list: the list of eigenvalues in the resulting generalized eigenvalue problem.
% 
%    Examples:
%        X = rand(15,10);
%        Y = rand(3, 10);
%        options.RegX = 0.5;
%        [W_x, eigv_list] = OPLS(X, Y, options);
% 
% © 2008 Liang Sun (sun.liang@asu.edu), Arizona State University
% 


% Extract the input parameters, and set the values of parameters.
if (~exist('options','var'))
   options = [];
end
RegX = 0;
if isfield(options, 'RegX')
    RegX = options.RegX;
end


W_x = [];
eigv_list = [];
% Preprocess the input 
if size(X, 2) ~= size(Y, 2)
    disp('The numbers of samples in X and Y are not equal!');
    return;
end
if issparse(X)
    X = full(X);
end
if issparse(Y)
    Y = full(Y);
end

% Compute the SVD of X
[X_U, X_Sigma, X_V] = svd(X, 'econ');
X_rank = rank(X_Sigma);
X_U = X_U(:, 1:X_rank);
X_Sigma = X_Sigma(1:X_rank, 1:X_rank);
X_V = X_V(:, 1:X_rank);

% Compute the Projection for X
% 
% Construct matrix H, then we can call the general module to solve this
% eigenvalue problem.
H = Y';

[W_x, eigv_list] = solve_eigen(X_U, X_Sigma, X_V, H, RegX);