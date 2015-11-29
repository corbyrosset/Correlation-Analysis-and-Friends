function [W_x, W_y, corr_list] = CCA(X, Y, options)

% CCA: Canonical Correlation Analysis
%
%
% Usage:
%     [W_x, W_y] = CCA(X, Y, options)
%     [W_x, W_y] = CCA(X, Y)
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point.
%        Y       - Data matrix Y. Each column of Y is a data point.
%       
%        options.PrjX    = 1: The projection for X is requried.
%                             = 0: The projeciton for X is not required.
%                                    The default value is 1 (required). 
%        options.PrjY    = 1: The projection for Y is requried.
%                             = 0: The projeciton for Y is not required.
%                                    The default value is 1 (required).  
%        options.RegX  - The regularization parameter for X. The default value is 0.
%        options.RegY  - The regularization parameter for Y. The default value is 0.
% 
%
%    Output:
%        W_x: each column is a projection vector for X. If the projection
%                 for X is not required, an empty matrix is returned.
%        W_y: each column is a projection vector for Y. If the projection
%                 for Y is not required, an empty matrix is returned.
%        corr_list: the list of correlation coefficients in the projected spaces of X and Y. 
% 
%    Examples:
%        X = rand(7,10);
%        Y = rand(15, 10);
%        options.PrjX = 1;
%        options.PrjY = 0;
%        options.RegX = 0.5;
%        [W_x, W_y] = CCA(X, Y, options);
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

% Compute the SVD of Y
[Y_U, Y_Sigma, Y_V] = svd(Y, 'econ');
rank_Y = rank(Y_Sigma);
Y_U = Y_U(:, 1:rank_Y);
Y_Sigma = Y_Sigma(1:rank_Y, 1:rank_Y);
Y_V = Y_V(:, 1:rank_Y);

% Compute the projection vectors
if PrjX == 1    
    % Compute the Projection for X
    % 
    % Construct matrix H, then we can call the general module to solve this
    % eigenvalue problem.    
    if RegY == 0
        % No regularization for Y is considered
        H = Y_V * Y_U';
    else
        % We consider the regularization for Y
        Y_sigma = diag(Y_Sigma);
        Y_sigma_reg = Y_sigma.^2 + RegY;
        Y_sigma_reg = sqrt(Y_sigma_reg);
        Y_sigma_reg = Y_sigma ./ Y_sigma_reg;
        H = Y_V * diag(Y_sigma_reg) * Y_U';
    end

    % Call the general function to solve the resulting eigenvalue problem
    [W_x, eigenV_list_x] = solve_eigen(X_U, X_Sigma, X_V, H, RegX);    
    corr_list = sqrt(eigenV_list_x);
    
    % Compute the Projection for Y
    if PrjY == 1
        if RegY == 0
            W_y = Y_U * diag(1./diag(Y_Sigma)) * Y_V' * X' * W_x;
        else
            Y_sigma = diag(Y_Sigma);
            Y_sigma_reg = Y_sigma.^2 + RegY;
            Y_sigma_reg = Y_sigma ./ Y_sigma_reg;
            W_y = Y_U * diag(Y_sigma_reg) * Y_V' * X' * W_x;
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
            H = X_V * X_U';
        else
            % We consider the regularization for Y
            X_sigma = diag(X_Sigma);
            X_sigma_reg = X_sigma.^2 + RegX;
            X_sigma_reg = sqrt(X_sigma_reg);
            X_sigma_reg = X_sigma ./ X_sigma_reg;
            H = X_V * diag(X_sigma_reg) * X_U';
        end

        % Call the general function to solve the resulting eigenvalue problem
        [W_y, eigenV_list_x] = solve_eigen(Y_U, Y_Sigma, Y_V, H, RegY);
        corr_list = sqrt(eigenV_list_x);        
    end
end