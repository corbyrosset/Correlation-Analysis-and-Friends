% This function centers each row of matrix M
function M = rowCenter(M, rowMean)
if nargin == 1
    rowMean = mean(M);
end
for r = 1:size(M,1)
    M(r, :) = M(r, :) - rowMean;
end
