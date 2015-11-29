function W = lasso_func(X, T, RegX)

% lasso_func: This function solves the LASSO problem. Note that this function
% acts as an interface to call LASSO. The users can choose different
% implementations of LASSO if the input/output is consistent with this
% function. In this version, we use l1_ls function provided by Kwangmoo
% Koh, Seung-Jean Kim, and Stephen Boyd
% (http://www.stanford.edu/~boyd/l1_ls/).
% Faster implementations of lasso include Fix-point algorithm 
% (http://www.caam.rice.edu/~optimization/L1/fpc/) and SLEPl1 algorithm
% (http://www.public.asu.edu/~jye02/Software/slepl1/).
%
%
% Usage:
%     W = lasso_func(X, T, RegX)
%     It solves the following optimization problem:
%     min_W ||W' * X - T||_F^2 + RegX ||W||_1,
%     where the 1-norm of W is defined for each column of W separately.
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point.
%        T       - Target matrix T. Each column of T is a data point.
%        RegX - The regularization parameter in lasso.
% 
%
%    Output:
%        W: each column is a projection vector for X. 
% 
%    Examples:
%        X = rand(15,10);
%        T = rand(4, 10);
%        RegX = 0.5;
%        W = lasso_func(X, T, RegX);
% 
% © 2008 Liang Sun (sun.liang@asu.edu), Arizona State University
% 

W = [];
[d1, n1] = size(X);
[d2, n2] = size(T);
if n1 ~= n2
    disp('The numbers of samples in X and T are not equal!');
    return;    
end
if RegX < 0
    disp('The regularization parameter is negative!');
    return;
end
n = n1;
if issparse(X)
    X = full(X);
end
if issparse(T)
    T = full(T);
end

for c = 1:d2
    t = T(c, :)';
    % ** Users can change this line to use different implementations of lasso **
    w = l1_ls(X', t, RegX, 1e-4, true);
    % ** Users can change this line to use different implementations of lasso **
    W(:, c) = w;
end