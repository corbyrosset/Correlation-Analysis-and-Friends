load CCAprojected_data.mat
mappedXtestCCA = tsne(stackedTest', ytest, 2, bestd);

load KCCAprojected_data.mat
mappedXtestKCCA = tsne(stackedTest', ytest, 2, bestd);
