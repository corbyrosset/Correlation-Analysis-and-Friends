function [ MODEL ] = MyKPCA( A, OPTIONS )
%DOKPCA Perform kernel PCA via kernel SVD.
%----------------
% Parameter list:
%----------------
% A             = Data matrix.
% OPTIONS       = Training options.
%   .R          = Subspace dimension to keep.
%   .KTYPE      = Kernel function type.
%   .KPARAM     = Parameter for KTYPE kernel function.
%   .DISP       = Display result in figure (for 2D data only).
%----------------------
% Output argument list:
%----------------------
% MODEL         = Output data structure.
%   .VECT       = Expansion vectors for kernel principal components.
%   .ALPHA      = Expansion coefficients for kernel principal components.
%   .NU         = Expansion coefficients for data mean.
%   .SIGMA      = Singular values.
%   .OPTIONS    = Original training options.

%--------------------------
% Process input parameters.
%--------------------------
if (nargin < 2)||...
   (~isfield(OPTIONS,'R'))||...
   (~isfield(OPTIONS,'KTYPE'))||...
   (~isfield(OPTIONS,'KPARAM'))||...
   (~isfield(OPTIONS,'DISP'))
        fprintf(1,'Required parameters missing.\n');
        OPTIONS = CreateOpt(1);
end

%----------------------
% Get training options.
%----------------------
R = OPTIONS.R;
KTYPE = OPTIONS.KTYPE;
KPARAM = OPTIONS.KPARAM;
DISP = OPTIONS.DISP;

%----------------------
% Get number of points.
%----------------------
[ d n ] = size(A);

%-------------
% Start timer.
%-------------
tic;

%-------------
% Center data.
%-------------
NU = (1/n)*ones(n,1);
nuprime = eye(n) - NU*ones(1,n);

%----------------------
% Create kernel matrix.
%----------------------
M = (nuprime')*myKernelMatrix( A, A, KTYPE, KPARAM )*nuprime;

%----------
% EVD of M.
%----------
[ Q D ] = eig(M);
D = real(D);
[sorted sortind] = sort(diag(D),1,'descend');
D = diag(sorted);
Q = Q(:,sortind);

%--------------------------
% Check rank of input data.
%--------------------------
if R > rank(D)
    R = rank(D);
    OPTIONS.R = rank(D);
    warning('Rank of data is less than R.');
end

%-------------------------------
% Retain R principal components.
%-------------------------------
ALPHA = nuprime*Q(:,1:R)*diag(1./sqrt(diag(D(1:R,1:R))));

%-------------------------
% Singular values of ksvd.
%-------------------------
SIGMA = diag(sqrt(diag(D(1:R,1:R))));

%------------------
% Get elapsed time.
%------------------
fprintf(1,'Elapsed time is %f.\n',toc);

%---------------------------------
% Project and display if required.
%---------------------------------
if ((DISP == 1)&&(d == 2))

    fprintf(1,'Plotting results...');
    
    % Get test points.
    [X Y] = meshgrid(-1:0.05:1, -0.5:0.05:1.5);    
    [rX cX] = size(X);    
    Pts_x = reshape(X,1,rX*cX);
    Pts_y = reshape(Y,1,rX*cX);
    Pts = [ Pts_x ; Pts_y ];  
    
    % Number of data points.
    [ dPts nPts ] = size(Pts);

    % Mean adjust data.
    muadj = [ (-NU)*ones(1,nPts) ; eye(nPts) ];

    % Project onto subspaces.
    proj = (ALPHA(:,1:R)')*myKernelMatrix(A,[A Pts],KTYPE,KPARAM)*muadj;            

    for pc = 1:R
        
        % Display input points.
        figure;
        set(gcf, 'Renderer', 'zbuffer');          
        axis([-1 1 -0.5 1.5]);

        % Contour values.
        Z = reshape(proj(pc,:),rX,cX);        

        % Plot.
        colormap(gray);
        hold on;
        box on;
        pcolor(X, Y, Z);
        shading interp;
        contour(X, Y, Z, 9, 'b');    
        plot(A(1,:), A(2,:), 'r.');
        
        drawnow;
        
    end

    fprintf(1,'Done.\n');
 
end

%----------------------------
% Construct return structure.
%----------------------------
MODEL = struct(...
'VECT',A,...
'ALPHA',ALPHA,...
'NU',NU,...
'SIGMA',SIGMA,...
'OPTIONS',OPTIONS...
);