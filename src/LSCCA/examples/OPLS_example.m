
% CCA example
% This script demonstrate how to use the CCA function using an example
clear all; clc;
% We generate 200 samples, with dimensionality 10 and 100
d1 = 100;
d2 = 10;
n = 200;
X = rand(d1, n);
Y = rand(d2, n);

% ********** Examples of using OPLS *********
options.RegX = 0.1;
[W, eigv_list] =OPLS(X, Y, options);

% ******** Verify the equivlance between CCA and OPLS *******
opt_cca.RegX = 0.1;
opt_cca.PrjX = 1;
[W_x, W_y, corr_list] = CCA(X, Y, opt_cca);

WW_d = W * W' - W_x * W_x';
disp('the difference between CCA and OPLS is');
norm(WW_d)


opt2_cca.RegX = 0.1;
opt2_cca.RegY = 4;
opt2_cca.PrjX = 1;
[W2_x, W2_y, corr_list] = CCA(X, Y, opt2_cca);

WW2_d = W * W' - W2_x * W2_x';
disp('the difference between CCA and OPLS is');
norm(WW2_d)
