
% CCA example
% This script demonstrate how to use the CCA function using an example
clear all; clc;
% We generate 50 samples, with dimensionality 10 and 100
d1 = 10;
d2 = 100;
n = 50;
X = rand(d1, n);
Y = rand(d2, n);
X = colCenter(X);
Y = colCenter(Y);
K_X = X' * X;
K_Y = Y' * Y;

% ********** Examples of using CCA *********
options.PrjX = 1;
options.PrjY = 1;
options.RegX = 0.5;
options.RegY = 0.6;
[W_x, W_y, corr_list] = KCCA(K_X, K_Y, options);

% We verify the correlation coefficient in the projected space
c_list = zeros(size(W_x, 2), 1);
for i = 1:size(W_x, 2)
    w_x_p = W_x(:, i);
    w_y_p = W_y(:, i);
    x_p = w_x_p' * K_X;
    y_p = w_y_p' * K_Y;
    denom1 = w_x_p' * (K_X * K_X + options.RegX * K_X) * w_x_p;
    denom2 = w_y_p' * (K_Y * K_Y + options.RegY * K_Y) * w_y_p;
    c = x_p * y_p' / sqrt(denom1) / sqrt(denom2);
    c_list(i) = c;
end


