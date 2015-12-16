perp = 100;
theta = 0.5;

load CCAprojected_data.mat
% mappedXtestCCA = tsne(stackedTest', ytest, 2, bestd);
map = fast_tsne(stackedTrain', 2, bestd, perp, theta);
f1 = figure
gscatter(map(:,1), map(:,2), ytrain);

% load KCCAprojected_data.mat
% % mappedXtestKCCA = tsne(stackedTest', ytest, 2, bestd);
% map2 = fast_tsne(stackedTrain', 2, bestd, perp, theta);
% figure
% gscatter(map2(:,1), map2(:,2), ytrain);
