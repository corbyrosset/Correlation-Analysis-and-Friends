% This function centers each column of matrix M
function M = colCenter(M, colMean)

if nargin == 1
    colMean = mean(M, 2);
end
for c = 1:size(M,2)
    M(:, c) = M(:, c) - colMean;
end
