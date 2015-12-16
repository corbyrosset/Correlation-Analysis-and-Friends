% function [alpha, X1_hat, X1_hat_top2] = scalableKCCA(K_1, K_2, k, sigma2)
function [alpha, X1_hat, X1_hat_top2] = scalableKCCA(X_1, X_2, k, sigma1, sigma2)

    % input:
    % X_1 a d_1 by n matrix of examples from first view
    % X_2 a d_2 by n matrix of examples from second view
    % k the target dimensionality of the subspace alpha: n by k
    % sigma1 and sigma2, the un-squared kernel bandwidths
    
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
    
    %output variables
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
            fprintf('%d of %d\n', j-1, (floor(n1/b)*b));
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
        C_ff = (C_ff+C_ff')/2;    %force symmetry to make it real
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
        
        %should we regularize? a little?
        [alpha_hat, evalues] = eig(inv(C_ff + 0.02*eye(K))*C_fg*inv(C_gg + 0.02*eye(K))*C_gf);
        
        if (~isreal(alpha_hat))
            error('alpha_hat not real');
        end
            
        FF = F_hat'*F_hat;
        FF = (FF + FF')/2;
        alpha = (inv(FF)*F_hat')*alpha_hat; 
        % QUESTION: DO WE NORMALIZE alpha?? IF NOT, PROJECTIONS ARE 
        % WAY TOO BIG (on order of >>10^16)
        alpha = (1/norm(alpha))*alpha;
        if (~isreal(alpha))
            error('alpha not real');
        end
        %Beta = inv(C_gg)*C_gf*alpha_hat; %we don't need this
    end
    display('done computing alpha');
    
    %compute projection of kernel matrices onto alpha incrementally
    %equivalent of doing 
    %X1_hat = alpha'*K_1;
    %X2_hat = alpha'*K_2;
    %X1_hat_top2 = alpha(:, 1:2)'*K_1;
    %but we don't have kernel matrices bc they too big...
    for j = 1:b:(floor(n1/b)*b)
        K_temp = gram(X_1, j, j+b-1, sigma2);
        X1_hat = [X1_hat alpha'*K_temp];
        X1_hat_top2 = [X1_hat_top2 alpha(:, 1:2)'*K_temp];
    end
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
%

% CORBY: to comment out some parts. Apparently something funny with Kt
% made it seem as though C was never orthogonal to U??
%
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

% note this gram function only takes in ONE data matrix, unlike the one in
% kcca.m
% function Kb = gram(X, start, stop, sigma)
%     [d, n] = size(X);
%     Kb = zeros(n, (stop-start+1));
%     for i = 1:n
%         for j = 1:(stop-start+1)
%             j_offset = j+start-1;
%             e = (norm(X(:, i) - X(:, j_offset))^2)/(2*sigma^2);
%             a = exp(-1*e);
%             Kb(i, j) = a;
%         end
%     end
% end

function K = gram(X1, start, stop, p)
    [d, n] = size(X1);
    K = zeros(n, (stop-start+1));
    for i = 1:n
        for j = 1:(stop-start+1)
            j_offset = j+start-1;
            a = (100*X1(:, i)'*X1(:, j_offset) + 1)^p; 
            %the +1 can be replaced by a variable, usually and integer...
            K(i, j) = a;
        end
    end

end

% function K = gram(X1, start, stop, p)
%     [d, n] = size(X1);
%     K = zeros(n, (stop-start+1));
%     for i = 1:n
%         for j = 1:(stop-start+1)
%             j_offset = j+start-1;
%             a = tanh((1/p)*(X1(:, i)'*X1(:, j_offset)) + 1); 
%             %the +1 can be replaced by a variable...
%             K(i, j) = a;
%         end
%     end
% 
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
