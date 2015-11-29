
% LS_CCA example
% This script demonstrate how to use the LS_CCA function using an example
clear all; clc;
% We generate 50 samples, with dimensionality 10 and 100
d1 = 100;
d2 = 10;
n = 50;
X = rand(d1, n);
Y = rand(d2, n);
X = colCenter(X);
Y = colCenter(Y);

% ********** Examples of using LS_CCA *********
options0.RegType = 0;
W0 = LS_CCA(X, Y, options0);

options1.RegX = 0.1;
options1.RegType = 1;
W1 = LS_CCA(X, Y, options1);

options2.RegX = 0.1;
options2.RegType = 2;
W2 = LS_CCA(X, Y, options2);

% ********* Examples of using CCA **********
% We want to verify the equivalence between CCA and LS-CCA
options.PrjX = 1;
options.PrjY = 1;
options.RegX = 0;
options.RegY = 0;
[W_x, W_y, corr_list] = CCA(X, Y, options);

% ********* To show the equivalence between CCA and LS-CCA
WW_d = W0 * W0' - W_x * W_x';
disp('the difference between CCA and LS_CCA is');
disp(norm(WW_d))

