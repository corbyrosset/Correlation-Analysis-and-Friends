perp = 100;
theta = 0.5;

% load CCAprojected_data.mat
% map = fast_tsne(stackedTrain', 2, bestd, perp, theta);
% f1 = figure
% gscatter(map(:,1), map(:,2), ytrain);
% 
% map2 = fast_tsne(stackedTest', 2, bestd, perp, theta);
% f2 = figure
% gscatter(map2(:,1), map2(:,2), ytest);

%kcca
% load KCCAprojected_data.mat
% map3 = fast_tsne(stackedTrain', 2, bestd, perp, theta);
% f3 = figure
% gscatter(map3(:,1), map3(:,2), ytrain);
% 
% map4 = fast_tsne(stackedTest', 2, bestd, perp, theta);
% f4 = figure
% gscatter(map4(:,1), map4(:,2), ytest);

% dcca
load dcca_projected_data.mat
map5 = fast_tsne(dataTr, 2, 89, perp, theta);
f5 = figure
gscatter(map5(:,1), map5(:,2), PhonesTr);

map6 = fast_tsne(dataTest, 2, 89, perp, theta);
f6 = figure
gscatter(map6(:,1), map6(:,2), PhonesTest);
