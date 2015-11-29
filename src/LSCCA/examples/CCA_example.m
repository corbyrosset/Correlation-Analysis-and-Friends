
% CCA example
% This script demonstrate how to use the CCA function using an example
clear all; clc;
% We generate 200 samples, with dimensionality 10 and 100
d1 = 10;
d2 = 100;
n = 200;
X = rand(d1, n);
Y = rand(d2, n);

% ********** Examples of using CCA *********
options.PrjX = 1;
options.PrjY = 1;
options.RegX = 0.1;
options.RegY = 1;
[W_x, W_y, corr_list] = CCA(X, Y, options);
X_p = W_x' * X;
Y_p = W_y' * Y;

% We verify the correlation coefficient in the projected space
c_list = zeros(size(W_x, 2), 1);
for i = 1:size(X_p, 1)
    x_p = X_p(i, :)';    
    y_p = Y_p(i, :)';
    % We consider regularization for both X and Y.
    denom1 = x_p' * x_p + options.RegX * W_x(:, i)' * W_x(:, i);
    denom2 = y_p' * y_p + options.RegY * W_y(:, i)' * W_y(:, i);
    c = x_p' * y_p / sqrt(denom1) / sqrt(denom2);
    c_list(i) = c;
end


