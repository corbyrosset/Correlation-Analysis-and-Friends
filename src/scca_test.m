function result=scca_test(X,Y)
p=size(X,2);
q=size(Y,2);
n=size(X,1);

% K-folder cross validation
k=5;
% sparsity para of u and v, since u and v will be normalized, the upper
% bound can be 2
lambda_u_seq=0:0.02:0.2;
lambda_v_seq=0:0.02:0.2;

% num of paras to be tuned
num_lambdas_u=length(lambda_u_seq);
num_lambdas_v=length(lambda_v_seq);

% sparseness parameter combinations matrix for correlation
corr_mat=zeros(num_lambdas_u,num_lambdas_v);

% Cross-validation to select optimal combination of sparseness parameters
indices = crossvalind('Kfold', n, k);
for iter=1:k
    testing=(indices==iter);
    training=~testing;
    K=covariance(X(training,:),Y(training,:));
    
    % Get starting values for singular vectors as column and row means from matrix K
    U_init=K*(ones(p,1)/p);
    U.init = U.init /norm(U_init);
    V_init=K'*(ones(q,1)/q);
    V.init = V.init /norm(V_init);
    
    % Standardize testing data
    X_test=X(testing,:);Y_test=Y(testing,:);
    X_test=X_test-repmat(mean(X_test),[size(X_test,1),1]);
    Y_test=Y_test-repmat(mean(Y_test),[size(Y_test,1),1]);
    sigma1=var(X_test);
    sigma2=var(Y_test);
    X_test=X_test*diag(1./sqrt(sigma1));
    Y_test=Y_test*diag(1./sqrt(sigma2));
    
    % Loops for sparseness parameter combinations
    for i =1:num_lambdas_u
        flag_nan = 0;
        for j=1:num_lambdas_v
            lambda_u = lambda_u_seq(i);	% sparseness parameter for X
            lambda_v = lambda_v_seq(j);	% sparseness parameter for Y
            if flag_nan == 0
                [U,V]=scca(K,U_init,V_init,lambda_u,lambda_v);
                corr_mat(i,j)=corr_mat(i,j)+ abs(corrcoef(X_test*U,Y_test*V));
                if isnan(corr_mat(i,j)), flag_nan = 1;end
            else
                corr_mat(i,j:end)=corr_mat(i,j:end)+NaN;
                break;
            end
        end
    end 
end
% Identify optimal sparseness parameter combination
corr_mat(isnan(corr_mat))=0;
corr_mat=corr_mat./k;
max_corr=max(max(abs(corr_mat)));
[best_i,best_j] = find(abs(corr_mat)==max_corr,1);
best_lambda_u=lambda_u_seq(best_i);
best_lambda_v=lambda_v_seq(best_j);

% Compute singular vectors using the optimal sparseness parameter combination for the whole data
K=covariance(X,Y);
% Get starting values for singular vectors as column and row means from matrix K
U_init=K*(ones(p,1)/p);
U.init = U.init /norm(U_init);
V_init=K'*(ones(q,1)/q);
V.init = V.init /norm(V_init);
[U,V]=scca(K,U_init,V_init,best_lambda_u,best_lambda_v);
result=abs(corrcoef(X*U,Y*V));
end

function c=covariance(X,Y)
X=X-repmat(mean(X),[size(X,1),1]);
Y=Y-repmat(mean(Y),[size(Y,1),1]);
sigma1=var(X);
sigma2=var(Y);

X=X*diag(1./sqrt(sigma1));
Y=Y*diag(1./sqrt(sigma2));

c=cov(X,Y);
end