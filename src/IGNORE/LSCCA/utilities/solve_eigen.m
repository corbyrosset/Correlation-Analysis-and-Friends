function [W, eigenV_list] = solve_eigen(X_U, X_Sigma, X_V, H, X_reg)
% 
% This function  solves the eigenvalue problem in the following form:
% (XX')^\dag X HH' X' w = a w, where a is the eigenvalue, w is the
% eigenvector. When regularization on X is considered, the eigenvalue
% problme to be solved is (XX' + X_reg * I)^{-1} X HH' X' w = a w.
% 
% ************ Output Description *******************
% The eigenvectors corresponding to nonzero eigenvalues are returned
% Each column of W is an eigenvector; eigenV_list is a vector, which
% contains the list of eigenvalues.
%
% ************ Input Description   *******************
% we have two views: 
%  X: d-by-n matrix
%  H: n-by-k matrix
% where d, and k are the dimensions of X and Y, respectively, and n is the
% number of data points
% X_reg: regularization paramter for X
% Note that we do not need to consider the regularization for Y, since it
% only changes H. We can investigate the regularization on Y in the loop
% outside this function.
% W: the projection matrix returned by OPLS algorithm
% Note that we consider regularization for both X and Y.
% Note that for this function, we require that the thin SVD of X. In other
% words, the thin SVD of X is:
% X = X_U * X_Sigma * X_V';
% 
% © 2008 Liang Sun (sun.liang@asu.edu), Arizona State University
%
if nargin < 5
    X_reg = 0;
end

X_sigma = diag(X_Sigma);
X_sigma_reg = X_sigma.^2 + X_reg;
X_sigma_reg = sqrt(X_sigma_reg);

X_sigma_B = X_sigma ./ X_sigma_reg;
B = diag(X_sigma_B) * X_V' * H;
[P, B_Sigma, Q] = svd(B, 'econ');
clear Q;
rank_B = rank(B);
P = P(:, 1:rank_B);
B_sigma = diag(B_Sigma);
B_sigma = B_sigma(1:rank_B);

eigenV_list = B_sigma.^2;
W = X_U * diag(1./X_sigma_reg) * P;



