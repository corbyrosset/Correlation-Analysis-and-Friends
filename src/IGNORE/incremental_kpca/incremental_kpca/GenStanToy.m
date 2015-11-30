function [ A ] = GenStanToy( P )
%GENSTANTOY Generate points for standard toy problem.
% Parameter list:
% P = Number of points.

x = rand(P,1)*2 - ones(P,1);

n = randn(P,1)*0.2;
 
y = (x.^2) + n;

A = ([x y]');

% x = (rand(P,1)*2 - ones(P,1))*0.4;
% y = -0.75*x + 0.5;
% y = y + rand(P,1)*0.2;
% 
% A = ([x y]');