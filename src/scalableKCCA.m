% function [alpha, X1_hat, X1_hat_top2] = scalableKCCA(K_1, K_2, k, sigma2)
function [alpha, X1_hat, X1_hat_top2] = scalableKCCA(X_1, X_2, k, sigma1, sigma2)

    % input:
    % X_1 a d_1 by n matrix of examples from first view
    % X_2 a d_2 by n matrix of examples from second view
    % k the target dimensionality of the subspace alpha: n by k
    % sigma2 the squared kernel bandwidth
    
    % output
    % alpha the n by k subspace on which to project the K_1 matrix
    % X1_hat the projection of K_1 onto alpha; a k by n matrix
    % X1_hat_top2 the top2 principles directions, a n by 2 matrix
    
    % parameters
    % b the blocksize with which to incrementally calculate the SVD of K
    % p the number of iterations over all examples in kernel matrices 
    %   so as to converge to true singular value decomposition for K

    global b;             %blocksize
    global K;             %target dimension
    K = k;
    b = 500;              %also tune this parameter?
    p = 1;                %numbe of iterations?
    [d1, n1] = size(X_1);
    [d2, n2] = size(X_2);
    if (n1 ~= n2)
        size(X_1)
        size(X_2)
        error('dimensions of inputs do not match');
    end
    
    %output
    X1_hat = [];
    X1_hat_top2 = [];
    
    U_1 = zeros(n1, K);           %K_1 ~= U1*S1*U1^T
    U_2 = zeros(n1, K);           %K_2 ~= U2*S2*U2^T
    S_1 = zeros(K, K);
    S_2 = zeros(K, K);
    V_1 = zeros(n1, K);
    V_2 = zeros(n1, K);
    
    for i = 1:p
        for j = 1:b:(floor(n1/b)*b)
            C_1 = gram(X_1, j, j+b-1, sigma1); %K_1(:,j:(j+b-1));  %%columns of K_1 
            [U_1, S_1, V_1] = isvd(U_1,S_1,V_1,C_1,K);%incrSVD(C_1, U_1, S_1, V_1);
            C_2 = gram(X_2, j, j+b-1, sigma2); %K_2(:,j:(j+b-1)); % %%columns of K_1
            [U_2, S_2, V_2] = isvd(U_2,S_2,V_2,C_2,K);%incrSVD(C_2, U_2, S_2, V_2);
        end
        display('done computing eigendecomposition');
        F_hat = sqrtm(S_1)*U_1';
        if (~isreal(F_hat))
            error('F_hat not real');
        end
        
        G_hat = sqrtm(S_2)*U_2';
        C_ff = F_hat*F_hat';
        C_ff = (C_ff+C_ff')/2;    %force symmetry
        if (~isreal(C_ff))
            error('C_ff not real');
        end
        C_fg = F_hat*G_hat';
        C_gf = G_hat*F_hat';
        C_gg = G_hat*G_hat';
        C_gg = (C_gg+C_gg')/2;    %force symmetry
        if (~isreal(C_gg))
            error('C_gg not real');
        end
        
        [alpha_hat, evalues] = eig(inv(C_ff + 0.0002*eye(K))*C_fg*inv(C_gg + 0.0002*eye(K))*C_gf);
%         diag(evalues)
        
        if (~isreal(alpha_hat))
            error('alpha_hat not real');
        end
        %sort alpha by eigenvalues
%         [evalues, ind] = sort(diag(evalues), 'descend');
%         alpha_hat = (1/evalues(1))*alpha_hat(:, ind);
            
        FF = F_hat'*F_hat;
        FF = (FF + FF')/2;
%         [~, vals] = eig(inv(FF));
%         diag(vals)
        alpha = (inv(FF)*F_hat')*alpha_hat; 
%         norm(alpha)
        alpha = (1/norm(alpha))*alpha;
        if (~isreal(alpha))
            error('alpha not real');
        end
        %Beta = inv(C_gg)*C_gf*alpha_hat;
    end
    display('done computing alpha');
    
    %compute projection of kernel matrices onto alpha
    %X1_hat = alpha'*K_1;
    %X2_hat = alpha'*K_2;
    %X1_hat_top2 = alpha(:, 1:2)'*K_1;
    for j = 1:b:(floor(n1/b)*b)
        K_temp = gram(X_1, j, j+b-1, sigma2);
        X1_hat = [X1_hat alpha'*K_temp];
        X1_hat_top2 = [X1_hat_top2 alpha(:, 1:2)'*K_temp];
    end
%     X1_hat_other = alpha'*gram(X_1, 1, size(X_1, 2), sigma2);
%     match = sum(sum(X1_hat_other == X1_hat));
%     if (match ~= K*size(X_1, 2))
%         error('do not match')
%     end
    
    
    
    if (size(X1_hat, 2) ~= n1)
        size(X1_hat)
        error('projected kernel onto alpha is not right size')
    end
    X1_hat_top2 = X1_hat_top2'; %for easier plotting
    display('done with scalable KCCA');
  



end
%% isvd Incremental Singular Value Decomposition. 
% [U,S,V]=isvd(U,S,V,C,r) performs incremental singular value 
% decomposition. At each iteration, isvd processes a new batch of samples
% in nxp data matrix C, and updates the previous estimates of singular 
% vectors and values. The rank of the SVD is provided as an input argument.
% 
% Inputs:
%    U - nxr matrix of left singular vectors
%    V - mxr matrix of right singular vectors
%    S - rxr diagonal matrix of singular values  
%    C - nxp matrix of new batch of samples 
%    r - desired rank of the primal singular subspace
%
% Outputs:
%    U - updated nxr matrix of left singular vectors
%    V - updated mxr matrix of right singular vectors
%    S - updated rxr diagonal matrix of singular values  
%--------------------------------------------------------------------------
% Author:        Raman Arora
% E-mail:        arora@ttic.edu
% Affiliation:   Toyota Technological Institute at Chicago
% Version:       0.1, created 10/19/11
%--------------------------------------------------------------------------
%%

function [U,S,V]=isvd(U,S,V,C,RANK) 
    TOLERANCE=1e-06;
    [~,p]=size(C);  % Number of new columns
    [~,r]=size(U);  % rank-r SVD given by USV'
    [m,~]=size(V);  % rank-r SVD given by USV'
    L=U'*C;         % rxp matrix (projection of C onto U)
    H=C-U*L;        % nxp matrix (projection of C onto U_perp)
    [J,Kt]=qr(H,0);  % J is nxn unitary matrix, Kt is nxp upper triangular matrix
    v=sqrt(det(Kt'*Kt));      % Volume of C orthogonal to U
    %ERROR: for some reason sqrt(det(Kt'*Kt)) evaluated to zero?????
%     if(v < TOLERANCE)
%         Q=[S L];                % Equivalently Kt is 0
%     else
        Q=[S L; zeros(size(Kt,1),r) Kt];%(n+r)x(p+r) matrix-blocks [rxr rxp; nxr nxp]
%     end
    [U2,S2,V2]=svd(Q,'econ');     % U2 is (n+r)xr, S2 is rxr, V2 is (p+r)xr
%     if(v < TOLERANCE)
%         U=U*U2;                % [U J] is of size nx(n+r). U2 is of size (n+r)xr. U is nxr.
%     else
    U=[U J]*U2;    % [U J] is of size nx(n+r). U2 is of size (n+r)xr. U is nxr.
%     end
    S=S2;          % S is rxr
    % Old V is mxr.  [V 0; 0 I] is (m+p)x(p+r). V2 is (p+r)xr. New V is (m+p)xr
    V=[V zeros(m,p); zeros(p,r) eye(p)]*V2; 

    if(nargin>4)
        if(length(diag(S>0))>RANK)      % Truncate if the rank increases 
            dS=diag(S);                 % Extract the new singular values
            [dS,I]=sort(dS,'descend');   % Sort the new singular values
            I=I(1:RANK);                % Indices of Top-r singular values 
            U=U(:,I);                   % Fetch Top-r left singular vectors 
            V=V(:,I);                   % Fetch Top-r right singular vectors 
            S=diag(dS(I));              % Updated rxr matrix of singular values
        end 
    end
end

function Kb = gram(X, start, stop, sigma)
    [d, n] = size(X);
    Kb = zeros(n, (stop-start+1));
    for i = 1:n
        for j = 1:(stop-start+1)
            j_offset = j+start-1;
            e = (norm(X(:, i) - X(:, j_offset))^2)/(2*sigma^2);
            a = exp(-1*e);
            if (e > 7)
                e
                error('expnent too large, make sigma larger');
            end

            Kb(i, j) = a;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U, S, V] = incrSVD(C, U_est, S_est, V_est)
    %U_est 15948 by 110
    %S_est 110 by 110
    %V_est 15948 by 110
    %S = 120 by 120
    %U = 15948 by 120
    %W = 10 by 10
    %J = 15948 by 10
    V = [];
    if (size(C, 2) ~= b)
        size(C)
        error('C not right size');
    end
    
    if (size(U_est) ~= [n K])
        size(U_est);
        error('U not right size)');
    end
    L = U_est'*C;
    H = C - U_est*L;
    [J,W] = qr(H,0);
    Q = [S_est, L; zeros(b, K), W];
    size(Q);
    
    [U_tilde, S_tilde, V_tilde] = svd(Q);
    U = [U_est J]*U_tilde;
    S = S_tilde;
%???     V = [V_est zeros(size(V_est, 1), b); zeros(size(V_est, 1), b), W]*V_tilde;
    
    %sort eigenvectors by eigenvalues
    [S, ind] = sort(diag(S), 'descend');
    S = diag(S);
    S = S(:, ind);
    
    U = U(:, ind);
%     V = V(:, ind);
    S = S(1:K, 1:K);
    U = U(:, 1:K);
%     V = V(:, 1:K);
end
