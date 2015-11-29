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
[J,K]=qr(H,0);  % J is nxn unitary matrix, K is nxp upper triangular matrix
v=sqrt(det(K'*K));      % Volume of C orthogonal to U
if(v < TOLERANCE)
    Q=[S L];                % Equivalently K is 0
else
    Q=[S L; zeros(size(K,1),r) K];%(n+r)x(p+r) matrix-blocks [rxr rxp; nxr nxp]
end
[U2,S2,V2]=svd(Q,'econ');     % U2 is (n+r)xr, S2 is rxr, V2 is (p+r)xr
if(v < TOLERANCE)
    U=U*U2;                % [U J] is of size nx(n+r). U2 is of size (n+r)xr. U is nxr.
else
U=[U J]*U2;    % [U J] is of size nx(n+r). U2 is of size (n+r)xr. U is nxr.
end
S=S2;          % S is rxr
% Old V is mxr.  [V 0; 0 I] is (m+p)x(p+r). V2 is (p+r)xr. New V is (m+p)xr
V=[V zeros(m,p); zeros(p,r) eye(p)]*V2; 
if(nargin>4)
    if(length(diag(S>0))>RANK)      % Truncate if the rank increases 
        dS=diag(S);                 % Extract the new singular values
        [~,I]=sort(dS,'descend');   % Sort the new singular values
        I=I(1:RANK);                % Indices of Top-r singular values 
        U=U(:,I);                   % Fetch Top-r left singular vectors 
        V=V(:,I);                   % Fetch Top-r right singular vectors 
        S=diag(dS(I));              % Updated rxr matrix of singular values
    end 
end
end
