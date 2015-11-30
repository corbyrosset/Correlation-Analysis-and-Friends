function W = LS(X, T, options)

% LS: Least Squares
%
%
% Usage:
%     W = CCA(X, T, options)
%     W = CCA(X, T)
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point.
%        T       - Target matrix T. Each column of T is a data point.
%       
%        options.RegX      - The regularization parameter for X. The default value is 0.
%        optoins.RegType = 0: No regularization is imposed.
%                                 = 1: 1-norm regularized least squares (lasso).
%                                 = 2: 2-norm regularized least squres (ridge regression)    
%                                     The default value is 0 (No regularization)
% 
%
%    Output:
%        W: each column is a projection vector for X. 
% 
%    Examples:
%        X = rand(15,10);
%        T = rand(4, 10);
%        options.RegX = 0.5;
%        options.RegType = 2;
%        W_x = CCA(X, T, options);
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


% Step 2. Solve the least squares formulation
W = [];
[d1, n1] = size(X);
[d2, n2] = size(T);
if n1 ~= n2
    disp('The numbers of samples in X and T are not equal!');
    return;    
end
n = n1;
if issparse(X)
    X = full(X);
end
if issparse(T)
    T = full(T);
end
% Make sure X and T are both column-centered, i.e., Xe=0, Te=0;
if norm(X * ones(n, 1)) > 1e-4
    X = colCenter(X);
end
if norm(T * ones(n, 1)) > 1e-4
    T = colCenter(T);
end

switch RegType
    case 1
        % Lasso
        W = lasso_func(X, T, RegX);
    case 2
        % Ridge Regression
        [U, Sigma, V] = svd(X, 'econ');
        r = rank(Sigma);
        U1 = U(:, 1:r);
        V1 = V(:, 1:r);
        sigma_r = diag(Sigma(1:r, 1:r));
        D = sigma_r.^2 + RegX;
        D = sigma_r ./ D;
        W = U1 * diag(D) * V1' * T';        
    otherwise
        % Default: No regularization
        [U, Sigma, V] = svd(X, 'econ');
        r = rank(Sigma);
        U1 = U(:, 1:r);
        V1 = V(:, 1:r);
        sigma_r = diag(Sigma(1:r, 1:r));        
        W = U1 * diag(1./sigma_r) * V1' * T';
end