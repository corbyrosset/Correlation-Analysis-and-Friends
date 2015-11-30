function W = KLS(K_X, T, options)

% KLS: Kernel Least Squares
%
%
% Usage:
%     W = CCA(K_X, T, options)
%     W = CCA(K_X, T)
% 
%    Input:
%        K_X       - Kernel matrix for X. 
%        T           - Target matrix T. Each column of T corresponds to a data point.
%       
%        options.RegX      - The regularization parameter for K_X. The default value is 0.
%        optoins.RegType = 0: No regularization is imposed.
%                                 = 2: 2-norm regularized least squres (ridge regression)    
%                                     The default value is 0 (No regularization)
% 
%
%    Output:
%        W: each column is a projection vector for K_X. 
% 
%    Examples:
%        X = rand(15,10); 
%        K_X = X' * K; 
%        T = rand(4, 10);
%        options.RegX = 0.5;
%        options.RegType = 2;
%        W_x = KLS(K_X, T, options);
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
[n0, n1] = size(K_X);
[d2, n2] = size(T);
if n0 ~= n1
    disp('The input for X is not a kernel!');
    return;
end
if n1 ~= n2
    disp('The numbers of samples in X and T are not equal!');
    return;    
end
n = n1;
if issparse(K_X)
    K_X = full(K_X);
end
if issparse(T)
    T = full(T);
end
% Make sure X and T are both column-centered, i.e., Xe=0, Te=0;
% In kernel space, we must have e'K=0, Ke=0;
if norm(K_X * ones(n, 1)) > 1e-4
    K_X = colCenter(K_X);
end
if norm(ones(1, n) * K_X) > 1e-4
    K_X = rowCenter(K_X);
end
if norm(T * ones(n, 1)) > 1e-4
    T = colCenter(T);
end

switch RegType
    case 2
        % Ridge Regression
        [U, Sigma, V] = svd(K_X, 'econ');
        r = rank(Sigma);
        U1 = U(:, 1:r);
        V1 = V(:, 1:r);
        sigma_r = diag(Sigma(1:r, 1:r));
        D = 1 ./ (sigma_r + RegX);
        W = U1 * diag(D) * V1' * T';        
    otherwise
        % Default: No regularization
        [U, Sigma, V] = svd(K_X, 'econ');
        r = rank(Sigma);
        U1 = U(:, 1:r);
        V1 = V(:, 1:r);
        sigma_r = diag(Sigma(1:r, 1:r));        
        W = U1 * diag(1./sigma_r) * V1' * T';
end